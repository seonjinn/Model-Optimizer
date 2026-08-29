# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the stager's parallel byte-range download against a local server.

The staged corpus is dominated by a few very large single files, and a single
HTTP stream measured 775 KB/s against the dataset CDN - 5.4 hours for the 15 G
shard. The stager therefore fetches large files as independent byte ranges,
which is only safe if a partial or misbehaving transfer can never be mistaken
for a complete one. That is what these tests pin down: reassembly is exact, a
server that ignores Range is refused rather than trusted, an interrupted run
resumes from the chunks already on disk, and nothing lands at the target path
unless the whole file is there at the length the revision claims.

No token and no network: the fixture serves bytes from a local socket.
"""

from __future__ import annotations

import http.server
import os
import re
import socket
import socketserver
import subprocess
import threading
import time
from pathlib import Path

import pytest

STAGER = Path(__file__).resolve().parents[1] / "common" / "specdec" / "stage_ptv3.sh"
CHUNK = 1 << 20
PAYLOAD_BYTES = 10 * CHUNK + 12345


def _handler(path: Path, honor_range: bool) -> type[http.server.BaseHTTPRequestHandler]:
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

        def do_GET(self) -> None:
            data = path.read_bytes()
            requested = self.headers.get("Range")
            match = re.match(r"bytes=(\d+)-(\d+)", requested) if requested else None
            if match and honor_range:
                start, end = int(match.group(1)), int(match.group(2))
                body = data[start : end + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
            else:
                body = data
                self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _serve(path: Path, honor_range: bool) -> tuple[str, _Server]:
    server = _Server(("127.0.0.1", 0), _handler(path, honor_range))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = str(server.server_address[0]), int(server.server_address[1])
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            socket.create_connection((host, port), 0.2).close()
            break
        except OSError:
            time.sleep(0.02)
    else:
        raise RuntimeError("test server did not start")
    return f"http://{host}:{port}/payload", server


def _run(workdir: Path, script: str) -> subprocess.CompletedProcess[str]:
    """Source the stager's helper block, then run script against it.

    The helpers are lifted out of the shipped file rather than duplicated, so
    the test cannot drift away from the code it is meant to cover.
    """
    text = STAGER.read_text()
    start = text.index("# Token on a pipe")
    end = text.index("for entry in $MANIFEST")
    helpers = text[start:end]
    preamble = "\n".join(
        [
            "set -uo pipefail",
            helpers,
            "HFTOK=test-token-not-a-secret",
            f"CHUNK_BYTES={CHUNK}",
            "STREAMS=4",
            "RANGE_MIN_BYTES=1",
        ]
    )
    return subprocess.run(
        ["bash", "-c", preamble + "\n" + script],
        cwd=workdir,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def payload(tmp_path: Path) -> Path:
    source = tmp_path / "payload"
    source.write_bytes(os.urandom(PAYLOAD_BYTES))
    return source


def test_ranged_download_reproduces_the_source_exactly(tmp_path: Path, payload: Path) -> None:
    url, server = _serve(payload, honor_range=True)
    try:
        result = _run(tmp_path, f"fetch_ranged {url} out.bin {PAYLOAD_BYTES}")
    finally:
        server.shutdown()
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "out.bin").read_bytes() == payload.read_bytes()


def test_chunk_directory_is_removed_once_the_file_is_whole(tmp_path: Path, payload: Path) -> None:
    url, server = _serve(payload, honor_range=True)
    try:
        _run(tmp_path, f"fetch_ranged {url} out.bin {PAYLOAD_BYTES}")
    finally:
        server.shutdown()
    assert not (tmp_path / "out.bin.parts").exists()


def test_a_server_that_ignores_range_is_refused(tmp_path: Path, payload: Path) -> None:
    """Each range is checked against the length that range should hold.

    A server answering every request with the whole object would otherwise
    concatenate into a file many times too long, or - worse, if the checks were
    only on the total - into one of the right length made of the wrong bytes.
    """
    url, server = _serve(payload, honor_range=False)
    try:
        result = _run(tmp_path, f"fetch_ranged {url} out.bin {PAYLOAD_BYTES}")
    finally:
        server.shutdown()
    assert result.returncode != 0
    assert not (tmp_path / "out.bin").exists()


def test_an_interrupted_run_resumes_from_the_chunks_on_disk(tmp_path: Path, payload: Path) -> None:
    data = payload.read_bytes()
    parts = tmp_path / "out.bin.parts"
    parts.mkdir()
    starts = list(range(0, len(data), CHUNK))
    for index, start in enumerate(starts):
        if index == 2:
            continue  # never fetched
        chunk = data[start : start + CHUNK]
        if index == 5:
            chunk = chunk[:100]  # died mid-transfer
        (parts / f"{start:020d}").write_bytes(chunk)

    url, server = _serve(payload, honor_range=True)
    try:
        result = _run(tmp_path, f"fetch_ranged {url} out.bin {PAYLOAD_BYTES}")
    finally:
        server.shutdown()
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "out.bin").read_bytes() == data


def test_a_size_the_source_cannot_satisfy_produces_no_target(tmp_path: Path, payload: Path) -> None:
    url, server = _serve(payload, honor_range=True)
    try:
        result = _run(tmp_path, f"fetch_ranged {url} out.bin {PAYLOAD_BYTES + 5000}")
    finally:
        server.shutdown()
    assert result.returncode != 0
    assert not (tmp_path / "out.bin").exists()


@pytest.mark.parametrize(
    ("contents", "expected"),
    [(b"", "0"), (b"x" * 7, "7"), (None, "-1")],
)
def test_filesize_reports_a_bare_number_and_minus_one_when_absent(
    tmp_path: Path, contents: bytes | None, expected: str
) -> None:
    """Wc pads its output on some platforms, and a missing file is not a size.

    Both would turn a numeric comparison into a silently false one, which in
    this script means a truncated shard counted as complete.
    """
    target = tmp_path / "probe"
    if contents is not None:
        target.write_bytes(contents)
    result = _run(tmp_path, "filesize probe")
    assert result.stdout.strip() == expected


def test_the_manifest_override_replaces_the_pinned_default(tmp_path: Path) -> None:
    """PTv2 rides the stager through STAGE_MANIFEST rather than a second copy of it.

    An override that merely *appended* would silently re-stage the 125 G PTv3
    core every time PTv2 was asked for, so what matters is that the default is
    gone, not just that the override is present. Both names here are repos that
    do not exist, which resolves the same way with or without a network: the
    tree fetch fails and the loop records TREE_FAILED, having proved it read
    the override.
    """
    dest = tmp_path / "staged"
    result = subprocess.run(
        ["bash", str(STAGER), str(dest)],
        input="test-token-not-a-secret\n",
        env={**os.environ, "STAGE_MANIFEST": "not-a-repo-alpha:aaaaaaaa not-a-repo-beta:bbbbbbbb"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    status = (dest / "stage_ptv3_status.txt").read_text()
    assert "not-a-repo-alpha sha=aaaaaaaa TREE_FAILED" in status
    assert "not-a-repo-beta sha=bbbbbbbb TREE_FAILED" in status
    assert "Nemotron" not in status
