"""Compare staged PTv3 shards against the file list at each pinned revision.

The stager writes a status file, but a restarted run truncates it and the sizes
it records disagree with disk, so it cannot be used as the completeness record.
The revision tree is the only authority: a repo is staged when every file the
pinned SHA lists is present at its published size.
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(sys.argv[1])
TOKEN = os.environ["HF_TOKEN"]
MANIFEST = {
    "Nemotron-SFT-Math-v4": "84d42ad0",
    "Nemotron-RL-Math-v2": "804418c1",
    "Nemotron-SFT-SWE-v3": "3f73de64",
    "Nemotron-SFT-SWE-v3.5": "ad641292",
    "Nemotron-SFT-SWE-v2": "bd151f3f",
    "Nemotron-SWE-v1": "0fe17a96",
    "Open-SWE-Traces": "c2114fc8",
    "Nemotron-Agentic-v1": "650d5909",
    "Nemotron-SFT-Agentic-v2": "7c804833",
    "Nemotron-RL-Lightning-Training-Blend": "262eb58c",
}

# The stager's own bookkeeping, written inside each repo directory and belonging
# to no revision: a cached tree and a log of transfers that failed.
STAGER_ARTIFACTS = {".tree.json", ".errors"}

# What the row counter would read as data. An unlisted file matching one of
# these is the case that matters -- the counters glob the directory rather than
# replaying the revision manifest, so such a file is counted as rows that no
# revision vouches for. An unlisted file that the counter would skip (a .part
# from an interrupted transfer, a stray note) is worth reporting but is not a
# threat to the totals.
COUNTED_SUFFIXES = (".jsonl", ".jsonl.gz", ".json.gz", ".parquet")


def _next_page(link_header):
    """Return the rel="next" URL from an RFC 5988 Link header, or None."""
    if not link_header:
        return None
    for part in link_header.split(","):
        segments = part.split(";")
        if len(segments) < 2:
            continue
        if any(seg.strip().replace(" ", "") in ('rel="next"', "rel=next") for seg in segments[1:]):
            return segments[0].strip().lstrip("<").rstrip(">")
    return None


def tree(repo, sha):
    """List every file at a revision, following the API's pagination.

    The tree endpoint caps a response at a fixed page size and advertises the
    rest through a Link header. Reading only the first page would drop the tail
    of a large repo's manifest silently -- and because completeness is judged
    against this list, a dropped tail reads as COMPLETE rather than as an error.
    """
    url = f"https://huggingface.co/api/datasets/nvidia/{repo}/tree/{sha}?recursive=1"
    entries = []
    while url:
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})
        with urllib.request.urlopen(req, timeout=180) as r:
            entries.extend(json.load(r))
            url = _next_page(r.headers.get("Link"))
    return [x for x in entries if x.get("type") == "file"]


def local_files(out):
    """Every staged file under a repo, relative to it, minus stager bookkeeping."""
    if not out.is_dir():
        return set()
    return {
        str(p.relative_to(out))
        for p in out.rglob("*")
        if p.is_file() and p.name not in STAGER_ARTIFACTS
    }


failures = 0
for repo, sha in MANIFEST.items():
    out = ROOT / repo
    try:
        files = tree(repo, sha)
    except Exception as exc:
        print(f"{repo:38s} TREE_FAILED {exc}")
        failures += 1
        continue
    missing = short = unsized = 0
    have_bytes = want_bytes = 0
    for entry in files:
        # Distinguish "the revision reports no size" from "the size is zero":
        # collapsing them into 0 would skip the comparison below and let a
        # truncated file pass as complete.
        want = entry.get("size")
        if want is None:
            unsized += 1
        else:
            want_bytes += want
        target = out / entry["path"]
        if not target.exists():
            missing += 1
            continue
        actual = target.stat().st_size
        have_bytes += actual
        if want is not None and actual != want:
            short += 1
    # Files on disk that the revision does not list, split by whether the row
    # counter would read them. Only the counted kind can move a total.
    unlisted = sorted(local_files(out) - {e["path"] for e in files})
    stray = [f for f in unlisted if f.endswith(COUNTED_SUFFIXES)]
    ok = files and missing == 0 and short == 0 and unsized == 0 and not stray
    if not ok:
        failures += 1
    print(f"{repo:38s} {'COMPLETE' if ok else 'INCOMPLETE':10s} files={len(files):4d} "
          f"missing={missing:4d} size_mismatch={short:3d} unsized={unsized:3d} "
          f"stray={len(stray):3d} unlisted={len(unlisted):3d} "
          f"bytes={have_bytes}/{want_bytes}")
    for f in unlisted:
        print(f"{'':38s}   unlisted{' STRAY' if f in stray else ''}: {f}")

# Exit non-zero so a caller can gate on staging without parsing this output.
sys.exit(1 if failures else 0)
