# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the matched Q30 Base DFlash2 step-4166 evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import pytest
from common.specdec.dflash2_runtime_contract import artifact_tree_sha256, write_artifact_receipt
from common.specdec.dflash2_speculators_eval import (
    DATASET_REVISION,
    PROBE_ROWS,
    STANDARD_SUBSETS,
    analyze_divergence_probe,
    build_artifact_identity,
    build_target_control_allocation_receipt,
    build_target_control_receipt,
    build_tie_aware_pilot_receipt,
    capture_divergence_probe,
    capture_outputs,
    capture_tie_aware_pilot,
    classify_tie_aware_rows,
    compute_prompt_set,
    materialize_prompt_set,
    summarize_pair,
    summarize_target_control,
    summarize_tie_aware_pilot,
    validate_milestone_export,
    validate_output_equivalence,
    validate_target_control_allocation_receipt,
    validate_target_control_receipt,
    validate_tie_aware_pilot_receipt,
)
from common.specdec.dflash2_target_contract import dflash2_target_spec

_LAUNCHER = Path(__file__).resolve().parents[1]
_WRAPPER = _LAUNCHER / "common/specdec/run_speculators_eval.sh"
_PAIR = _LAUNCHER / "common/specdec/run_speculators_eval_pair.sbatch"
_SERVER_STAGER = _LAUNCHER / "common/specdec/stage_speculators_eval_server_runtime.sh"
_HELPER = _LAUNCHER / "common/specdec/dflash2_speculators_eval.py"


def test_target_control_allocation_receipt_rejects_spoofed_topology(tmp_path: Path) -> None:
    """The diagnostic receipt rejects a self-rehashed non-four-GPU allocation."""
    receipt = tmp_path / "allocation.json"
    payload = build_target_control_allocation_receipt(
        receipt,
        slurm_job_id="12345",
        slurm_job_num_nodes=1,
        slurm_job_nodelist="lyris0092",
        gpu_count=4,
    )

    assert payload["cell_visible_devices"] == {"left": "0,1", "right": "2,3"}
    forged = json.loads(receipt.read_text())
    forged["gpu_count"] = 8
    unsigned = {key: value for key, value in forged.items() if key != "receipt_sha256"}
    forged["receipt_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    receipt.write_text(json.dumps(forged) + "\n")
    with pytest.raises(ValueError, match="allocation"):
        validate_target_control_allocation_receipt(receipt)


def test_divergence_probe_identifies_first_invalid_dflash2_token(tmp_path: Path) -> None:
    """A target replay at the common prefix decides whether the draft token was valid."""
    baseline = tmp_path / "baseline.json"
    draft = tmp_path / "draft.json"
    target_records = []
    draft_records = []
    for subset, index in PROBE_ROWS:
        target_records.append(
            {
                "subset": subset,
                "index": index,
                "prompt_sha256": "a" * 64,
                "request_sha256": "b" * 64,
                "token_ids": [10, 11, 12],
                "output_text": "abc",
                "replays": [
                    {
                        "position": 0,
                        "expected_token_id": 10,
                        "token_id": 10,
                        "top_logprobs": {"token_id:10": -0.1},
                    },
                    {
                        "position": 1,
                        "expected_token_id": 11,
                        "token_id": 11,
                        "top_logprobs": {"token_id:11": -0.1},
                    },
                    {
                        "position": 2,
                        "expected_token_id": 12,
                        "token_id": 12,
                        "top_logprobs": {"token_id:99": -9.0, "token_id:12": -0.1},
                    },
                ],
            }
        )
        draft_records.append(
            {
                "subset": subset,
                "index": index,
                "prompt_sha256": "a" * 64,
                "request_sha256": "b" * 64,
                "token_ids": [10, 11, 99] if (subset, index) == PROBE_ROWS[0] else [10, 11, 12],
                "output_text": "abx",
                "replays": [],
            }
        )

    def write_probe(path: Path, method: str, records: list[dict[str, Any]]) -> None:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "producer": "q30-dflash2-first-token-probe-v1",
            "method": method,
            "model": "target",
            "dataset_prompt_sha256": "c" * 64,
            "sampling": {"temperature": 0, "top_p": 1, "seed": 42, "logprobs": 20},
            "records": records,
        }
        payload["receipt_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        path.write_text(json.dumps(payload) + "\n")

    write_probe(baseline, "baseline", target_records)
    write_probe(draft, "dflash2", draft_records)

    report = analyze_divergence_probe(baseline, draft)

    row = report["records"][0]
    assert row["first_divergence_position"] == 2
    assert row["target_token_id"] == 12
    assert row["dflash2_token_id"] == 99
    assert row["dflash2_token_is_target_argmax"] is False
    assert row["dflash2_token_target_rank"] == 2
    assert row["verdict"] == "target-invalid"

    tied = json.loads(baseline.read_text())
    tied["records"][0]["replays"][2]["token_id"] = 99
    tied["records"][0]["replays"][2]["top_logprobs"] = {
        "token_id:99": -0.1,
        "token_id:12": -0.1,
    }
    unsigned = {key: value for key, value in tied.items() if key != "receipt_sha256"}
    tied["receipt_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    baseline.write_text(json.dumps(tied) + "\n")
    tied_row = analyze_divergence_probe(baseline, draft)["records"][0]
    assert tied_row["dflash2_token_is_target_argmax"] is True
    assert tied_row["dflash2_token_target_rank"] == 1
    assert tied_row["replay_token_id"] == 99
    assert tied_row["verdict"] == "target-valid-argmax"

    tied["records"][1]["replays"][2]["token_id"] = 77
    tied["records"][1]["replays"][2]["top_logprobs"] = {
        "token_id:77": -0.1,
        "token_id:12": -1.0,
        "token_id:99": -1.0,
    }
    unsigned = {key: value for key, value in tied.items() if key != "receipt_sha256"}
    tied["receipt_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    baseline.write_text(json.dumps(tied) + "\n")
    changed_draft = json.loads(draft.read_text())
    changed_draft["records"][1]["token_ids"] = [10, 11, 99]
    unsigned = {
        key: value for key, value in changed_draft.items() if key != "receipt_sha256"
    }
    changed_draft["receipt_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    draft.write_text(json.dumps(changed_draft) + "\n")
    inconclusive = analyze_divergence_probe(baseline, draft)["records"][1]
    assert inconclusive["target_token_is_replay_argmax"] is False
    assert inconclusive["verdict"] == "replay-inconclusive"


def test_capture_divergence_probe_uses_pinned_vllm_token_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The probe requests and consumes vLLM token IDs and top-logprob fields exactly."""
    manifest, hf_home = _write_prompt_snapshot(tmp_path / "prompts")
    calls: list[dict[str, Any]] = []

    def fake_completion(_endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
        calls.append(body)
        if body["max_tokens"] == 64:
            token_ids = [10, 11]
            prompt_ids = [1, 2]
            top = [{"token_id:10": -0.1}, {"token_id:11": -0.1}]
            text = "ab"
        else:
            position = len(body["prompt"]) - 2
            token_ids = [10 + position]
            prompt_ids = list(body["prompt"])
            top = [{f"token_id:{10 + position}": -0.1}]
            text = "a"
        return {
            "choices": [
                {
                    "text": text,
                    "token_ids": token_ids,
                    "prompt_token_ids": prompt_ids,
                    "logprobs": {"top_logprobs": top},
                }
            ]
        }

    monkeypatch.setattr(
        "common.specdec.dflash2_speculators_eval._post_completion", fake_completion
    )
    output = tmp_path / "probe.json"
    payload = capture_divergence_probe(
        manifest,
        hf_home,
        output,
        endpoint="http://target/v1",
        model="target",
        method="baseline",
    )

    assert len(payload["records"]) == 5
    assert calls[0]["logprobs"] == 20
    assert calls[0]["return_tokens_as_token_ids"] is True
    assert calls[0]["return_token_ids"] is True
    assert calls[1]["prompt"] == [1, 2]
    assert calls[2]["prompt"] == [1, 2, 10]


def test_tie_aware_classifier_is_fail_closed_for_top20_and_termination() -> None:
    """The pilot accepts only exact rows or a DFlash token tied at target maximum."""

    def row(tokens: list[int], finish: str, top: list[dict[str, float]]) -> dict[str, Any]:
        return {
            "subset": "HumanEval",
            "index": 0,
            "source_row": 0,
            "prompt_sha256": "a" * 64,
            "request_core_sha256": "b" * 64,
            "prompt_token_ids": [1, 2],
            "token_ids": tokens,
            "output_text": "out",
            "finish_reason": finish,
            "top_logprobs": top,
        }

    target_top = [
        {"token_id:10": -0.1},
        {"token_id:11": -0.1, "token_id:99": -0.1, "token_id:77": -3.0},
    ]
    target = row([10, 11], "length", target_top)

    assert classify_tie_aware_rows(target, row([10, 11], "length", []))["class"] == "exact"
    tied = classify_tie_aware_rows(target, row([10, 99], "length", []))
    assert tied["class"] == "tied-target-valid"
    assert tied["dflash2_token_target_rank"] == 1
    assert classify_tie_aware_rows(target, row([10, 77], "length", []))["class"] == (
        "target-invalid"
    )
    assert classify_tie_aware_rows(target, row([10, 88], "length", []))["class"] == (
        "unresolved-top20"
    )
    assert classify_tie_aware_rows(target, row([10], "length", []))["class"] == (
        "unresolved-termination"
    )
    assert classify_tie_aware_rows(target, row([10, 11], "stop", []))["class"] == (
        "unresolved-termination"
    )
    malformed = row([10, 11], "length", [])
    malformed["top_logprobs"] = [{"token_id:10": -0.1}, {"token_id:11": float("nan")}]
    assert classify_tie_aware_rows(target=malformed, dflash2=row([10, 11], "length", []))[
        "class"
    ] == "unresolved-target-evidence"
    later_malformed = row(
        [10, 11, 12],
        "length",
        [
            {"token_id:10": -0.1},
            {"token_id:11": -0.1, "token_id:99": -0.1},
            {"token_id:12": float("nan")},
        ],
    )
    earlier_tie = row([10, 99, 55], "length", [])
    assert classify_tie_aware_rows(later_malformed, earlier_tie)["class"] == (
        "tied-target-valid"
    )


def test_tie_aware_pilot_streams_exact_humaneval_schedule_and_online_target_logits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Capture cycles the authenticated source rows and preserves online target logits."""
    manifest, hf_home = _write_prompt_snapshot(tmp_path / "prompts", rows=164)
    matched_hf = tmp_path / "matched-hf"
    matched_manifest = tmp_path / "matched-manifest.json"
    materialize_prompt_set(manifest, hf_home, matched_hf, matched_manifest)
    calls: list[dict[str, Any]] = []

    def fake_completion(_endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
        calls.append(body)
        return {
            "choices": [
                {
                    "text": "x",
                    "finish_reason": "length",
                    "token_ids": [10],
                    "prompt_token_ids": [1, 2],
                    "logprobs": {"top_logprobs": [{"token_id:10": -0.1}]},
                }
            ]
        }

    monkeypatch.setattr(
        "common.specdec.dflash2_speculators_eval._post_completion", fake_completion
    )
    target_path = tmp_path / "target.jsonl"
    capture_tie_aware_pilot(
        matched_manifest,
        matched_hf,
        target_path,
        endpoint="http://target/v1",
        model="target",
        role="target",
    )
    rows = [json.loads(line) for line in target_path.read_text().splitlines()]

    assert len(rows) == 200
    assert rows[164]["source_row"] == 0
    assert rows[164]["prompt_sha256"] == rows[0]["prompt_sha256"]
    assert all(call["logprobs"] == 20 for call in calls)
    assert all(call["return_token_ids"] is True for call in calls)
    assert rows[0]["top_logprobs"] == [{"token_id:10": -0.1}]
    with pytest.raises(FileExistsError):
        capture_tie_aware_pilot(
            matched_manifest,
            matched_hf,
            target_path,
            endpoint="http://target/v1",
            model="target",
            role="target",
        )


def test_tie_aware_pilot_summary_fails_closed_on_unresolved_rows(tmp_path: Path) -> None:
    """Any invalid or unresolved occurrence prevents a passing pilot receipt."""
    target_path = tmp_path / "target.jsonl"
    draft_path = tmp_path / "draft.jsonl"

    def record(index: int, role: str, token: int) -> dict[str, Any]:
        top = [{"token_id:10": -0.1, "token_id:99": -0.1}] if role == "target" else []
        extracted = {
            "output_text": "x",
            "token_ids": [token],
            "prompt_token_ids": [1, 2],
            "finish_reason": "length",
            "top_logprobs": top,
        }
        return {
            "schema_version": 1,
            "producer": "q30-tie-aware-online-row-v1",
            "role": role,
            "subset": "HumanEval",
            "index": index,
            "source_row": index % 164,
            "prompt_sha256": f"{index:064x}",
            "request_core_sha256": "a" * 64,
            "request_sha256": "b" * 64,
            **extracted,
            "response_sha256": hashlib.sha256(
                json.dumps(extracted, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }

    target_rows = [record(index, "target", 10) for index in range(200)]
    draft_rows = [record(index, "dflash2", 99) for index in range(200)]
    target_path.write_text("".join(json.dumps(row) + "\n" for row in target_rows))
    draft_path.write_text("".join(json.dumps(row) + "\n" for row in draft_rows))
    passed = summarize_tie_aware_pilot(target_path, draft_path)
    assert passed["status"] == "passed"
    assert passed["counts"] == {"tied-target-valid": 200}
    assert passed["occurrences"] == 200
    assert passed["unique_source_rows"] == 164

    draft_rows[0]["token_ids"] = [88]
    draft_rows[0]["response_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "output_text": "x",
                "token_ids": [88],
                "prompt_token_ids": [1, 2],
                "finish_reason": "length",
                "top_logprobs": [],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    draft_path.write_text("".join(json.dumps(row) + "\n" for row in draft_rows))
    failed = summarize_tie_aware_pilot(target_path, draft_path)
    assert failed["status"] == "failed"
    assert failed["counts"]["unresolved-top20"] == 1


def test_tie_aware_receipt_replays_descriptors_and_rejects_job_or_port_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Receipt composition binds rows, manifests, ports, job allocation, and replay bytes."""
    prompt_file = tmp_path / "HumanEval.jsonl"
    prompts = [
        {"prompt": f"prompt {index % 164}", "_specdec_source_row": index % 164}
        for index in range(200)
    ]
    prompt_file.write_text("".join(json.dumps(row) + "\n" for row in prompts))
    schedule = [
        {
            "subset": "HumanEval",
            "index": index,
            "source_row": index % 164,
            "prompt_sha256": hashlib.sha256(f"prompt {index % 164}".encode()).hexdigest(),
        }
        for index in range(200)
    ]
    identity_payload = {
        "target": {"path": "target"},
        "dataset": {
            "files": {"HumanEval": {"path": str(prompt_file)}},
            "ordered_prompts": schedule,
        },
    }
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(json.dumps(identity_payload) + "\n")
    target_rows = tmp_path / "target.jsonl"
    draft_rows = tmp_path / "draft.jsonl"

    def write_rows(path: Path, role: str) -> None:
        with path.open("w") as stream:
            for index, prompt_row in enumerate(prompts):
                prompt = prompt_row["prompt"]
                core = {
                    "model": "target",
                    "prompt": prompt,
                    "max_tokens": 64,
                    "temperature": 0,
                    "top_p": 1,
                    "seed": 42,
                    "request_id": f"specdec-tie-pilot-HumanEval-{index}",
                }
                body = {
                    **core,
                    "logprobs": 20 if role == "target" else 0,
                    "return_tokens_as_token_ids": True,
                    "return_token_ids": True,
                }
                extracted = {
                    "output_text": "x",
                    "token_ids": [10],
                    "prompt_token_ids": [1, 2],
                    "finish_reason": "length",
                    "top_logprobs": [{"token_id:10": -0.1}] if role == "target" else [],
                }
                row = {
                    "schema_version": 1,
                    "producer": "q30-tie-aware-online-row-v1",
                    "role": role,
                    "subset": "HumanEval",
                    "index": index,
                    "source_row": index % 164,
                    "prompt_sha256": schedule[index]["prompt_sha256"],
                    "request_core_sha256": hashlib.sha256(
                        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "request_sha256": hashlib.sha256(
                        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest(),
                    **extracted,
                    "response_sha256": hashlib.sha256(
                        json.dumps(extracted, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest(),
                }
                stream.write(json.dumps(row) + "\n")

    write_rows(target_rows, "target")
    write_rows(draft_rows, "dflash2")
    target_manifest = tmp_path / "target-manifest.json"
    draft_manifest = tmp_path / "draft-manifest.json"
    target_fingerprint = tmp_path / "target-fingerprint.json"
    draft_fingerprint = tmp_path / "draft-fingerprint.json"
    target_launcher = tmp_path / "target-launcher.yaml"
    draft_launcher = tmp_path / "draft-launcher.yaml"
    for path in (
        target_manifest,
        draft_manifest,
        target_fingerprint,
        draft_fingerprint,
        target_launcher,
        draft_launcher,
    ):
        path.write_text(path.name + "\n")
    target_manifest_payload = {
        "slurm_job_id": "12345",
        "server_args": ["serve", "target", "--tensor-parallel-size", "2", "--port", "8000"],
    }
    draft_manifest_payload = {
        "slurm_job_id": "12345",
        "server_args": [
            "serve",
            "target",
            "--tensor-parallel-size",
            "2",
            "--port",
            "8010",
            "--speculative-config",
            '{"method":"dflash"}',
        ],
    }
    monkeypatch.setattr(
        "common.specdec.dflash2_speculators_eval._validate_artifact_identity",
        lambda _path: identity_payload,
    )
    monkeypatch.setattr(
        "common.specdec.dflash2_speculators_eval._validate_target_control_manifest",
        lambda *_args: (
            target_manifest_payload,
            target_fingerprint,
            {"inputs": {}},
            target_launcher,
        ),
    )
    monkeypatch.setattr(
        "common.specdec.dflash2_speculators_eval._validate_dflash2_probe_manifest",
        lambda *_args: (draft_manifest_payload, draft_launcher, draft_fingerprint),
    )
    current = {
        "slurm_job_id": "12345",
        "slurm_job_num_nodes": 1,
        "slurm_job_nodelist": "lyris0001",
        "gpu_count": 4,
        "cell_visible_devices": {"left": "0,1", "right": "2,3"},
    }
    monkeypatch.setattr(
        "common.specdec.dflash2_speculators_eval._query_current_allocation", lambda: current
    )
    allocation = tmp_path / "allocation.json"
    build_target_control_allocation_receipt(
        allocation,
        slurm_job_id="12345",
        slurm_job_num_nodes=1,
        slurm_job_nodelist="lyris0001",
        gpu_count=4,
    )
    receipt = tmp_path / "receipt.json"
    build_tie_aware_pilot_receipt(
        target_rows,
        draft_rows,
        target_manifest,
        draft_manifest,
        identity_path,
        allocation,
        receipt,
    )
    assert validate_tie_aware_pilot_receipt(receipt)["status"] == "passed"

    original_rows = target_rows.read_text()
    target_rows.write_text(original_rows.replace('"output_text": "x"', '"output_text": "y"', 1))
    with pytest.raises(ValueError, match="descriptor"):
        validate_tie_aware_pilot_receipt(receipt)
    target_rows.write_text(original_rows)

    draft_manifest_payload["server_args"][5] = "8000"
    with pytest.raises(ValueError, match="ports"):
        build_tie_aware_pilot_receipt(
            target_rows,
            draft_rows,
            target_manifest,
            draft_manifest,
            identity_path,
            allocation,
            tmp_path / "same-port.json",
        )
    draft_manifest_payload["server_args"][5] = "8010"
    draft_manifest_payload["slurm_job_id"] = "99999"
    with pytest.raises(ValueError, match="job"):
        build_tie_aware_pilot_receipt(
            target_rows,
            draft_rows,
            target_manifest,
            draft_manifest,
            identity_path,
            allocation,
            tmp_path / "wrong-job.json",
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _init_git_repo(path: Path) -> str:
    path.mkdir()
    (path / "tracked.txt").write_text("tracked\n")
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "add", "tracked.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
    )
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_prompt_snapshot(tmp_path: Path, *, rows: int = 200) -> tuple[Path, Path]:
    hf_home = tmp_path / "hf"
    snapshot = (
        hf_home / "hub/datasets--RedHatAI--speculator_benchmarks/snapshots" / DATASET_REVISION
    )
    snapshot.mkdir(parents=True)
    files: dict[str, dict[str, str]] = {}
    for subset in STANDARD_SUBSETS:
        path = snapshot / f"{subset}.jsonl"
        path.write_text(
            "".join(
                json.dumps({"prompt": f"{subset} prompt {index}"}, sort_keys=True) + "\n"
                for index in range(rows)
            )
        )
        files[subset] = {"path": str(path), "sha256": _sha256(path)}
    manifest = tmp_path / "dataset-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_id": "RedHatAI/speculator_benchmarks",
                "revision": DATASET_REVISION,
                "hf_home": str(hf_home),
                "files": files,
            },
            sort_keys=True,
        )
        + "\n"
    )
    return manifest, hf_home


def _write_ledger(path: Path, *, mutate: tuple[str, int] | None = None) -> None:
    with path.open("w") as stream:
        for subset in STANDARD_SUBSETS:
            for index in range(200):
                output = f"answer:{subset}:{index}"
                if mutate == (subset, index):
                    output += ":wrong"
                record = {
                    "subset": subset,
                    "index": index,
                    "source_row": index,
                    "prompt_sha256": hashlib.sha256(
                        f"{subset} prompt {index}".encode()
                    ).hexdigest(),
                    "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                    "output_text": output,
                    "output_tokens": list(output),
                    "request_sha256": hashlib.sha256(
                        f"request:{subset}:{index}".encode()
                    ).hexdigest(),
                    "seed": 42,
                    "finish_reason": "stop",
                }
                stream.write(json.dumps(record, sort_keys=True) + "\n")


def _write_perf(path: Path, output_tps: float, latency: float) -> None:
    fields = (
        "subset",
        "strategy",
        "max_concurrency",
        "request_budget",
        "completed_requests",
        "rps_median",
        "latency_median_s",
        "itl_median_ms",
        "ttft_median_ms",
        "output_tps_median",
        "total_output_tokens",
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for subset in STANDARD_SUBSETS:
            writer.writerow(
                {
                    "subset": subset,
                    "strategy": "throughput",
                    "max_concurrency": 32,
                    "request_budget": 200,
                    "completed_requests": 200,
                    "rps_median": 1,
                    "latency_median_s": latency,
                    "itl_median_ms": 2,
                    "ttft_median_ms": 10,
                    "output_tps_median": output_tps,
                    "total_output_tokens": 2000,
                }
            )


def _write_acceptance(path: Path) -> None:
    positions = tuple(f"acceptance_at_pos_{index}" for index in range(7))
    fields = (
        "subset",
        "num_drafts",
        "num_draft_tokens",
        "num_accepted_tokens",
        "acceptance_length",
        *positions,
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for subset in STANDARD_SUBSETS:
            writer.writerow(
                {
                    "subset": subset,
                    "num_drafts": 100,
                    "num_draft_tokens": 700,
                    "num_accepted_tokens": 350,
                    "acceptance_length": 4.5,
                    **dict.fromkeys(positions, 0.5),
                }
            )


def test_prompt_digest_binds_exact_revision_nine_subsets_and_200_rows(tmp_path: Path) -> None:
    """The benchmark prompt set is an exact ordered 9x200 prefix."""
    manifest, hf_home = _write_prompt_snapshot(tmp_path)

    provenance = compute_prompt_set(manifest, hf_home, requests_per_subset=200)

    assert provenance["revision"] == DATASET_REVISION
    assert provenance["subsets"] == list(STANDARD_SUBSETS)
    assert provenance["requests_per_subset"] == 200
    assert provenance["total_requests"] == 1800
    assert len(str(provenance["prompt_sha256"])) == 64


def test_prompt_digest_cycles_short_subsets_to_exact_request_budget(tmp_path: Path) -> None:
    """A finite benchmark subset is replayed deterministically to the 200-request budget."""
    manifest, hf_home = _write_prompt_snapshot(tmp_path, rows=3)

    provenance = compute_prompt_set(manifest, hf_home, requests_per_subset=200)

    assert provenance["total_requests"] == 1800
    assert all(entry["available_rows"] == 3 for entry in provenance["files"].values())


def test_materialized_prompt_set_is_exact_200_row_authenticated_cycle(tmp_path: Path) -> None:
    """GuideLLM and correctness consume the same source-bound 200-row files."""
    manifest, hf_home = _write_prompt_snapshot(tmp_path, rows=3)
    output_home = tmp_path / "matched-hf"
    output_manifest = tmp_path / "matched-manifest.json"

    materialize_prompt_set(manifest, hf_home, output_home, output_manifest)
    provenance = compute_prompt_set(output_manifest, output_home)

    assert provenance["total_requests"] == 1800
    assert provenance["ordered_prompts"][3]["source_row"] == 0
    for entry in provenance["files"].values():
        assert sum(1 for _ in Path(entry["path"]).open()) == 200

    payload = json.loads(output_manifest.read_text())
    source = Path(payload["derivation"]["source_files"][STANDARD_SUBSETS[0]]["path"])
    source.write_text(source.read_text().replace("prompt 0", "tampered", 1))
    with pytest.raises(ValueError):
        compute_prompt_set(output_manifest, output_home)


@pytest.mark.parametrize("mutation", ["revision", "empty-subset", "prompt"])
def test_prompt_digest_rejects_unmatched_inputs(tmp_path: Path, mutation: str) -> None:
    """Wrong revisions, empty subsets, and mutated bytes fail closed."""
    manifest, hf_home = _write_prompt_snapshot(
        tmp_path, rows=0 if mutation == "empty-subset" else 200
    )
    if mutation == "revision":
        payload = json.loads(manifest.read_text())
        payload["revision"] = "0" * 40
        manifest.write_text(json.dumps(payload))
    elif mutation == "prompt":
        payload = json.loads(manifest.read_text())
        path = Path(payload["files"][STANDARD_SUBSETS[0]]["path"])
        path.write_text(path.read_text().replace("prompt 0", "tampered", 1))

    with pytest.raises(ValueError):
        compute_prompt_set(manifest, hf_home, requests_per_subset=200)


def test_step4166_milestone_binds_exact_export_bytes(tmp_path: Path) -> None:
    """The evaluator consumes the lifecycle-preserved step-4166 export only."""
    output_root = tmp_path / "train"
    export = output_root / "export-4166"
    export.mkdir(parents=True)
    (export / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DFlash2DraftModel"],
                "block_size": 8,
                "num_attention_heads": 32,
                "num_key_value_heads": 4,
                "head_dim": 128,
                "intermediate_size": 6144,
                "dflash_config": {"projector_type": "dflash2"},
            }
        )
        + "\n"
    )
    (export / "model.safetensors").write_bytes(b"draft-weights")
    milestone = output_root / "milestones/step-004166"
    milestone.mkdir(parents=True)
    (milestone / "exact-model").symlink_to(export)
    hashes = {
        str(path.relative_to(export)): _sha256(path)
        for path in sorted(export.rglob("*"))
        if path.is_file()
    }
    manifest = milestone / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "exact_model_path": "../../export-4166",
                "exact_model_step": 4166,
                "exact_model_sha256": hashes,
                "resume_checkpoint_path": "resume-checkpoint-004166",
                "resume_checkpoint_step": 4166,
                "resume_checkpoint_sha256": {"trainer_state.json": "f" * 64},
            }
        )
    )

    result = validate_milestone_export(manifest, export, expected_step=4166)

    assert result["step"] == 4166
    assert result["export_path"] == str(export.resolve())
    assert len(str(result["export_tree_sha256"])) == 64
    (export / "model.safetensors").write_bytes(b"mutated-weights")
    with pytest.raises(ValueError, match="export file hashes"):
        validate_milestone_export(manifest, export, expected_step=4166)


def test_output_equivalence_is_a_fail_closed_gate(tmp_path: Path) -> None:
    """Any deterministic target-output mismatch blocks a speedup claim."""
    baseline = tmp_path / "baseline.jsonl"
    draft = tmp_path / "draft.jsonl"
    _write_ledger(baseline)
    _write_ledger(draft)

    receipt = validate_output_equivalence(baseline, draft, requests_per_subset=200)

    assert receipt["status"] == "passed"
    assert receipt["matched_requests"] == 1800
    expected_schedule = [
        {
            "subset": subset,
            "index": index,
            "source_row": index,
            "prompt_sha256": hashlib.sha256(f"{subset} prompt {index}".encode()).hexdigest(),
        }
        for subset in STANDARD_SUBSETS
        for index in range(200)
    ]
    validate_output_equivalence(
        baseline,
        draft,
        requests_per_subset=200,
        expected_prompt_schedule=expected_schedule,
    )
    expected_schedule[0]["source_row"] = 1
    with pytest.raises(ValueError, match="does not match artifact identity"):
        validate_output_equivalence(
            baseline,
            draft,
            requests_per_subset=200,
            expected_prompt_schedule=expected_schedule,
        )
    _write_ledger(draft, mutate=(STANDARD_SUBSETS[-1], 199))
    with pytest.raises(ValueError, match="target output mismatch"):
        validate_output_equivalence(baseline, draft, requests_per_subset=200)
    rows = baseline.read_text().splitlines()
    first = json.loads(rows[0])
    first["source_row"] = True
    rows[0] = json.dumps(first, sort_keys=True)
    baseline.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="ledger row schema"):
        validate_output_equivalence(baseline, draft, requests_per_subset=200)


def test_capture_outputs_records_seed_text_and_selected_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The diagnostic ledger preserves replayable request and token evidence."""
    manifest, hf_home = _write_prompt_snapshot(tmp_path)
    requests: list[dict[str, object]] = []

    class _Response:
        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        @staticmethod
        def read() -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {
                            "text": "answer",
                            "finish_reason": "stop",
                            "logprobs": {"tokens": ["ans", "wer"]},
                        }
                    ]
                }
            ).encode()

    def _urlopen(request: Any, *, timeout: int) -> _Response:
        assert timeout == 600
        body = json.loads(request.data)
        requests.append(body)
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    output = tmp_path / "ledger.jsonl"

    capture_outputs(manifest, hf_home, output, endpoint="http://server/v1", model="target")

    first = json.loads(output.open().readline())
    assert first["seed"] == 42
    assert first["output_text"] == "answer"
    assert first["output_tokens"] == ["ans", "wer"]
    assert requests[0]["seed"] == 42
    assert requests[0]["logprobs"] == 0
    assert requests[0]["return_tokens_as_token_ids"] is True
    assert requests[0]["request_id"] == "specdec-correctness-HumanEval-0"


def test_target_control_quantifies_cross_and_repeat_divergence(tmp_path: Path) -> None:
    """A target-only control reports nondeterminism without weakening the production gate."""
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    _write_ledger(left)
    _write_ledger(right)
    right_rows = right.read_text().splitlines()
    changed = json.loads(right_rows[1])
    changed["output_text"] = "answer:changed"
    changed["output_tokens"] = ["answer", ":", "changed"]
    changed["output_sha256"] = hashlib.sha256(changed["output_text"].encode()).hexdigest()
    right_rows[1] = json.dumps(changed, sort_keys=True)
    right.write_text("\n".join(right_rows) + "\n")

    report = summarize_target_control(left, right, requests_per_subset=200)

    assert report["status"] == "diagnostic-only"
    assert report["requests"] == 1800
    assert report["cross_server"]["exact_text_matches"] == 1799
    assert report["cross_server"]["exact_token_matches"] == 1799
    assert report["cross_server"]["mean_common_token_prefix"] >= 0
    assert report["within_server"]["left"]["repeat_groups"] == 0
    assert report["claim_scope"] == "no training-quality or speedup claim"

    repeated_left = tmp_path / "repeated-left.jsonl"
    repeated_right = tmp_path / "repeated-right.jsonl"
    _write_ledger(repeated_left)
    rows = [json.loads(line) for line in repeated_left.read_text().splitlines()]
    for subset_offset in range(0, len(rows), 200):
        for index in range(100, 200):
            source = rows[subset_offset + index - 100]
            repeated = rows[subset_offset + index]
            for key in ("source_row", "prompt_sha256", "output_sha256", "output_text", "output_tokens"):
                repeated[key] = source[key]
    rows[100]["output_text"] = "different"
    rows[100]["output_tokens"] = ["different"]
    rows[100]["output_sha256"] = hashlib.sha256(b"different").hexdigest()
    repeated_left.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    repeated_right.write_text(repeated_left.read_text())

    repeated = summarize_target_control(repeated_left, repeated_right)

    assert repeated["within_server"]["left"]["repeat_groups"] == 900
    assert repeated["within_server"]["left"]["divergent_text_repeat_groups"] == 1
    assert repeated["within_server"]["left"]["mean_common_token_prefix"] >= 0


def test_target_control_receipt_rejects_minimal_fabricated_provenance(tmp_path: Path) -> None:
    """Self-hashed ledgers cannot substitute for authenticated job provenance."""
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    _write_ledger(left)
    _write_ledger(right)
    manifest_payload = {
        "method": "baseline",
        "config_sha256": {"launcher": "a" * 64, "target": "b" * 64},
        "evaluation": {
            "max_concurrency": 1,
            "max_requests": 200,
            "temperature": 0,
            "top_p": 1,
            "tensor_parallel_size": 2,
        },
        "server_args": ["serve", "target", "--tensor-parallel-size", "2"],
    }
    manifests = [tmp_path / "left-manifest.json", tmp_path / "right-manifest.json"]
    for manifest in manifests:
        manifest.write_text(json.dumps(manifest_payload, sort_keys=True) + "\n")
    identity = tmp_path / "identity.json"
    identity.write_text(
        json.dumps(
            {
                "dataset": {
                    "ordered_prompts": [
                        {
                            "subset": subset,
                            "index": index,
                            "source_row": index,
                            "prompt_sha256": hashlib.sha256(
                                f"{subset} prompt {index}".encode()
                            ).hexdigest(),
                        }
                        for subset in STANDARD_SUBSETS
                        for index in range(200)
                    ]
                }
            },
            sort_keys=True,
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="artifact identity"):
        build_target_control_receipt(
            left,
            right,
            manifests[0],
            manifests[1],
            identity,
            tmp_path / "allocation.json",
            tmp_path / "control.json",
        )


def test_target_control_receipt_replays_authenticated_job_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A genuine target/runtime/dataset/job closure replays and detects ledger tamper."""
    target = tmp_path / "target"
    target.mkdir()
    spec = dflash2_target_spec("q30-base")
    (target / "snapshot-manifest.json").write_text(
        json.dumps({"source_identity": spec.revision}) + "\n"
    )
    (target / "config.json").write_text(
        json.dumps(
            {
                "num_attention_heads": spec.num_attention_heads,
                "num_key_value_heads": spec.num_key_value_heads,
                "head_dim": spec.head_dim,
                "intermediate_size": spec.intermediate_size,
            }
        )
        + "\n"
    )
    target_receipt = tmp_path / "target-receipt.json"
    write_artifact_receipt(target_receipt, target, kind="target")

    export = tmp_path / "export"
    export.mkdir()
    (export / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DFlash2DraftModel"],
                "block_size": 8,
                "num_attention_heads": 32,
                "num_key_value_heads": 4,
                "head_dim": 128,
                "intermediate_size": 6144,
                "dflash_config": {"projector_type": "dflash2"},
            }
        )
        + "\n"
    )
    (export / "model.safetensors").write_bytes(b"weights")
    milestone = tmp_path / "milestone"
    milestone.mkdir()
    (milestone / "exact-model").symlink_to(export)
    milestone_manifest = milestone / "manifest.json"
    milestone_manifest.write_text(
        json.dumps(
            {
                "exact_model_path": str(export),
                "exact_model_step": 4166,
                "exact_model_sha256": {
                    str(path.relative_to(export)): _sha256(path)
                    for path in sorted(export.iterdir())
                },
                "resume_checkpoint_path": "checkpoint-4166",
                "resume_checkpoint_step": 4166,
                "resume_checkpoint_sha256": {"trainer_state.json": "f" * 64},
            }
        )
        + "\n"
    )
    dataset_manifest, hf_home = _write_prompt_snapshot(tmp_path / "prompts")
    client_runtime = tmp_path / "client.tar"
    server_runtime = tmp_path / "server.tar"
    client_runtime.write_bytes(b"client")
    server_runtime.write_bytes(b"server")
    identity_payload = build_artifact_identity(
        target_path=target,
        target_sha256=artifact_tree_sha256(target),
        target_receipt_path=target_receipt,
        target_receipt_sha256=_sha256(target_receipt),
        export_path=export,
        milestone_manifest_path=milestone_manifest,
        dataset_manifest_path=dataset_manifest,
        hf_home=hf_home,
        client_runtime_archive=client_runtime,
        client_runtime_archive_sha256=_sha256(client_runtime),
        server_runtime_archive=server_runtime,
        server_runtime_archive_sha256=_sha256(server_runtime),
        server_runtime_receipt_sha256="e" * 64,
    )
    identity = tmp_path / "identity.json"
    identity.write_text(json.dumps(identity_payload, indent=2, sort_keys=True) + "\n")

    image = tmp_path / "image.sqsh"
    image.write_bytes(b"image")
    image_identity = tmp_path / "image-identity.json"
    image_identity.write_text(
        json.dumps(
            {"path": str(image), "sha256": _sha256(image), "size_bytes": image.stat().st_size}
        )
        + "\n"
    )
    dataset = identity_payload["dataset"]
    container = {
        "identity_path": str(image_identity),
        "identity_sha256": _sha256(image_identity),
        "path": str(image),
        "sha256": _sha256(image),
        "size_bytes": image.stat().st_size,
    }
    modelopt_repo = tmp_path / "modelopt-repo"
    speculators_repo = tmp_path / "speculators-repo"
    source = {
        "modelopt_sha": _init_git_repo(modelopt_repo),
        "speculators_sha": _init_git_repo(speculators_repo),
    }
    runtimes = {"client": "/scratch/client", "server": "/scratch/server"}
    manifests: list[Path] = []
    for label, port in (("left", "8000"), ("right", "8010")):
        run = tmp_path / label
        run.mkdir()
        launcher_config = run / "resolved-launcher.yaml"
        launcher_config.write_text(
            "pipeline:\n  task_0:\n    slurm_config:\n"
            f"      container: {image.resolve()}\n"
        )
        config_sha256 = {
            "launcher": _sha256(launcher_config),
            "target": _sha256(target / "config.json"),
        }
        manifest = run / "manifest.json"
        server_args = [
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            str(target),
            "--tensor-parallel-size",
            "2",
            "--port",
            port,
        ]
        producer = _LAUNCHER / "common/specdec/speculators_eval_artifacts.py"
        subprocess.run(
            [
                "python3",
                str(producer),
                "manifest",
                "--output",
                str(manifest),
                "--status",
                "success",
                "--method",
                "baseline",
                "--block-size",
                "0",
                "--num-speculative-tokens",
                "0",
                "--target-model",
                str(target),
                "--draft-model",
                "",
                "--speculators-repo",
                str(speculators_repo),
                "--speculators-sha",
                source["speculators_sha"],
                "--modelopt-repo",
                str(modelopt_repo),
                "--modelopt-sha",
                source["modelopt_sha"],
                "--modelopt-dirty",
                "false",
                "--client-runtime",
                runtimes["client"],
                "--server-runtime",
                runtimes["server"],
                "--artifact-identity",
                str(identity),
                "--artifact-identity-sha256",
                _sha256(identity),
                "--container-image",
                str(image),
                "--container-identity",
                str(image_identity),
                "--dataset-manifest",
                str(dataset_manifest),
                "--hf-home",
                str(hf_home),
                "--slurm-job-id",
                "12345",
                "--launcher-config",
                str(launcher_config),
                "--python-version",
                "3.12.0",
                "--vllm-version",
                "0.27.1",
                "--guidellm-version",
                "0.4.0",
                "--max-concurrency",
                "1",
                "--max-requests",
                "200",
                "--evaluation-mode",
                "throughput",
                "--tensor-parallel-size",
                "2",
                *[f"--server-arg={value}" for value in server_args],
            ],
            check=True,
        )
        inputs = {
            "target_config_sha256": config_sha256["target"],
            "draft_config_sha256": None,
            "dataset": {
                "revision": dataset["revision"],
                "manifest_sha256": dataset["manifest_sha256"],
            },
            "image": {
                "sha256": container["sha256"],
                "identity_sha256": container["identity_sha256"],
            },
            "runtimes": {
                "client": {"sha256": _sha256(client_runtime)},
                "server": {"sha256": _sha256(server_runtime), "receipt_sha256": "e" * 64},
            },
            "artifact_identity_sha256": _sha256(identity),
            "source": source,
            "launcher_config_sha256": config_sha256["launcher"],
            "evaluation": {
                "method": "baseline",
                "block_size": 0,
                "num_speculative_tokens": 0,
                "max_concurrency": 1,
                "max_requests": 200,
                "mode": "throughput",
                "tensor_parallel_size": 2,
            },
        }
        (run / "input-fingerprint.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "sha256": hashlib.sha256(
                        json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "inputs": inputs,
                },
                sort_keys=True,
            )
            + "\n"
        )
        manifests.append(manifest)

    ledgers: list[Path] = []
    for label in ("left", "right"):
        ledger = tmp_path / f"{label}.jsonl"
        with ledger.open("w") as stream:
            for subset in STANDARD_SUBSETS:
                for index in range(200):
                    prompt = f"{subset} prompt {index}"
                    body = json.dumps(
                        {
                            "model": str(target),
                            "prompt": prompt,
                            "max_tokens": 64,
                            "temperature": 0,
                            "top_p": 1,
                            "seed": 42,
                            "logprobs": 0,
                            "return_tokens_as_token_ids": True,
                            "request_id": f"specdec-correctness-{subset}-{index}",
                        }
                    ).encode()
                    output = f"answer:{subset}:{index}"
                    stream.write(
                        json.dumps(
                            {
                                "subset": subset,
                                "index": index,
                                "source_row": index,
                                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                                "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                                "output_text": output,
                                "output_tokens": list(output),
                                "request_sha256": hashlib.sha256(body).hexdigest(),
                                "seed": 42,
                                "finish_reason": "stop",
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
        ledgers.append(ledger)
    receipt = tmp_path / "control.json"
    allocation = tmp_path / "allocation.json"
    current_allocation = {
        "slurm_job_id": "12345",
        "slurm_job_num_nodes": 1,
        "slurm_job_nodelist": "lyris0092",
        "gpu_count": 4,
        "cell_visible_devices": {"left": "0,1", "right": "2,3"},
    }
    monkeypatch.setattr(
        "common.specdec.dflash2_speculators_eval._query_current_allocation",
        lambda: current_allocation,
    )
    build_target_control_allocation_receipt(
        allocation,
        slurm_job_id="12345",
        slurm_job_num_nodes=1,
        slurm_job_nodelist="lyris0092",
        gpu_count=4,
    )
    build_target_control_receipt(
        ledgers[0], ledgers[1], manifests[0], manifests[1], identity, allocation, receipt
    )

    assert validate_target_control_receipt(receipt)["status"] == "diagnostic-only"
    monkeypatch.setattr(
        "common.specdec.dflash2_speculators_eval._query_current_allocation",
        lambda: {**current_allocation, "slurm_job_id": "99999"},
    )
    with pytest.raises(ValueError, match="live allocation mismatch"):
        build_target_control_receipt(
            ledgers[0],
            ledgers[1],
            manifests[0],
            manifests[1],
            identity,
            allocation,
            tmp_path / "forged-control.json",
        )
    monkeypatch.setattr(
        "common.specdec.dflash2_speculators_eval._query_current_allocation",
        lambda: current_allocation,
    )
    (modelopt_repo / "untracked.txt").write_text("dirty\n")
    with pytest.raises(ValueError, match="ModelOpt source checkout mismatch"):
        validate_target_control_receipt(receipt)
    (modelopt_repo / "untracked.txt").unlink()
    image.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="container"):
        validate_target_control_receipt(receipt)
    image.write_bytes(b"image")
    ledgers[1].write_text(ledgers[1].read_text().replace("answer:HumanEval:0", "tampered", 1))
    with pytest.raises(ValueError, match="descriptor"):
        validate_target_control_receipt(receipt)


def test_target_control_phase_is_baseline_only_and_never_runs_speed_cells() -> None:
    """The diagnostic phase is isolated from the production correctness/performance path."""
    pair = _PAIR.read_text()

    assert "diagnostic:1:200:2" in pair
    assert '"${method_a}:${method_b}" == baseline:baseline' in pair
    diagnostic = pair.index('if [[ "${PAIR_PHASE}" == diagnostic ]]')
    capture = pair.index("run_pair_cells 1", diagnostic)
    analyze = pair.index("analyze-control", capture)
    verify = pair.index("verify-control", analyze)
    stop = pair.index("exit 0", verify)
    performance = pair.index("run_pair_cells 0", stop)
    assert diagnostic < capture < analyze < verify < stop < performance
    assert 'readonly MATCHED_HF_HOME="${RESULT_ROOT}/matched-hf"' in pair
    assert '"${STUDY_HELPER}" allocation-receipt' in pair
    assert '--allocation-receipt "${ALLOCATION_RECEIPT_PATH}"' in pair
    assert '"SLURM_JOB_NUM_NODES"' in _HELPER.read_text()
    assert '"nvidia-smi", "--query-gpu=index", "--format=csv,noheader"' in _HELPER.read_text()


def test_first_divergence_phase_is_bounded_and_never_runs_speed_cells() -> None:
    """The mismatch phase captures fixed probes, verifies a receipt, and exits."""
    pair = _PAIR.read_text()
    wrapper = _WRAPPER.read_text()

    branch = pair.index('if [[ "${PAIR_PHASE}" == mismatch ]]')
    probe = pair.index("run_pair_cells 1", branch)
    analyze = pair.index("analyze-divergence", probe)
    verify = pair.index("verify-divergence", analyze)
    stop = pair.index("exit 0", verify)
    performance = pair.index("run_pair_cells 0", stop)
    assert branch < probe < analyze < verify < stop < performance
    assert 'DIVERGENCE_PROBE="${DIVERGENCE_PROBE_MODE}"' in pair
    assert "capture-probe" in wrapper
    assert '"${DIVERGENCE_PROBE:-0}" == 1' in wrapper


def test_tie_aware_pilot_is_humaneval_c1_only_and_exits_before_speed() -> None:
    """The pilot is a bounded diagnostic gate that cannot enter performance cells."""
    pair = _PAIR.read_text()
    wrapper = _WRAPPER.read_text()

    assert "tie-pilot:1:200:2" in pair
    branch = pair.index('if [[ "${PAIR_PHASE}" == tie-pilot ]]')
    capture = pair.index("run_pair_cells 1", branch)
    analyze = pair.index("analyze-tie-pilot", capture)
    verify = pair.index("verify-tie-pilot", analyze)
    status = pair.index('receipt.get("status") != "passed"', verify)
    stop = pair.index("exit 0", status)
    speed = pair.index("run_pair_cells 0", stop)
    assert branch < capture < analyze < verify < status < stop < speed
    assert 'TIE_AWARE_PILOT="${TIE_AWARE_PILOT_MODE}"' in pair
    assert '"${SCRIPT_DIR}/dflash2_speculators_eval.py" capture-tie-pilot' in wrapper


def test_pair_summary_reports_per_gpu_speed_latency_and_acceptance(tmp_path: Path) -> None:
    """The report contains matched performance and speculative-quality metrics."""
    baseline = tmp_path / "baseline"
    draft = tmp_path / "draft"
    baseline.mkdir()
    draft.mkdir()
    _write_perf(baseline / "perf_results.csv", output_tps=100, latency=0.8)
    _write_perf(draft / "perf_results.csv", output_tps=160, latency=0.5)
    _write_acceptance(draft / "acceptance.csv")

    report = summarize_pair(baseline, draft, tensor_parallel_size=2)

    assert report["aggregate"]["output_tps_per_gpu_baseline"] == 50
    assert report["aggregate"]["output_tps_per_gpu_dflash2"] == 80
    assert report["aggregate"]["output_tps_speedup"] == 1.6
    assert report["aggregate"]["latency_speedup"] == 1.6
    assert report["aggregate"]["acceptance_rate"] == 0.5
    assert report["aggregate"]["mean_accepted_length"] == 4.5
    assert len(report["subsets"]) == 9

    acceptance_path = draft / "acceptance.csv"
    acceptance_path.write_text(acceptance_path.read_text().replace(",350,4.5,", ",0,1.0,"))
    zero_report = summarize_pair(baseline, draft, tensor_parallel_size=2)
    assert zero_report["aggregate"]["acceptance_rate"] == 0


def test_split_runtime_pair_wiring_is_mandatory_for_matched_dflash2() -> None:
    """Server and GuideLLM client runtimes are independently pinned."""
    pair = _PAIR.read_text()
    wrapper = _WRAPPER.read_text()

    for required in (
        "CLIENT_RUNTIME_ARCHIVE",
        "CLIENT_RUNTIME_ARCHIVE_SHA256",
        "SERVER_RUNTIME_ARCHIVE",
        "SERVER_RUNTIME_ARCHIVE_SHA256",
        "SERVER_RUNTIME_RECEIPT_SHA256",
        "dflash2:8:7",
        "correctness:1:200:2|performance:32:200:2",
    ):
        assert required in pair
    assert 'SPECULATORS_CLIENT_RUNTIME="${JOB_CLIENT_RUNTIME}"' in pair
    assert 'VLLM_SERVER_RUNTIME="${JOB_SERVER_RUNTIME}"' in pair
    assert '"${SERVER_PYTHON}" "${SERVER_ARGS[@]}"' in wrapper
    assert '"${SPECULATORS_CLIENT_RUNTIME}/bin/python3" "${subset_args[@]}"' in wrapper
    assert '[[ "${SPEC_METHOD}" == dflash2 ]] && printf dflash' in wrapper
    assert "export VLLM_USE_V2_MODEL_RUNNER=1" in wrapper


def test_exact_matched_matrix_requires_c1_c32_200_and_tp2() -> None:
    """Only the approved C1/C32 TP2+TP2 matrix is accepted."""
    pair = _PAIR.read_text()

    assert 'case "${PAIR_PHASE}:${concurrency}:${max_requests}:${tp_size}"' in pair
    assert "correctness:1:200:2|performance:32:200:2" in pair
    assert "baseline:0:0" in pair
    assert "dflash2:8:7" in pair
    assert "c1-performance-report.json" in pair
    assert "c32-performance-report.json" in pair
    assert '"${STUDY_HELPER}" materialize-prompts' in pair
    assert 'DATASET_MANIFEST_PATH="${MATCHED_DATASET_MANIFEST_PATH}"' in pair
    assert 'HF_HOME_DURABLE="${MATCHED_HF_HOME}"' in pair
    equivalence_run = pair.index("run_pair_cells 1")
    correctness_gate = pair.index("compare-outputs", equivalence_run)
    performance_run = pair.index("run_pair_cells 0", correctness_gate)
    assert equivalence_run < correctness_gate < performance_run
    assert "--concurrency 1" in pair
    assert "--concurrency 32" in pair


def test_server_runtime_stager_binds_archive_and_embedded_receipt(tmp_path: Path) -> None:
    """The server venv is extracted once with both immutable markers."""
    source = tmp_path / "source"
    (source / "bin").mkdir(parents=True)
    python = source / "bin/python"
    python.write_text("#!/bin/bash\nexit 0\n")
    python.chmod(0o755)
    old_venv = "/durable/source-runtime"
    (source / "bin/activate").write_text(f'export VIRTUAL_ENV="{old_venv}"\n')
    (source / "pyvenv.cfg").write_text(f"home = {old_venv}\n")
    receipt = source / "dflash2-vllm-runtime-receipt.json"
    receipt.write_text('{"fixture":true}\n')
    archive = tmp_path / "server.tar"
    with tarfile.open(archive, "w") as stream:
        for path in source.rglob("*"):
            stream.add(path, arcname=path.relative_to(source))
    scratch = tmp_path / "scratch"
    destination = scratch / "123/server"
    env = {
        **os.environ,
        "SERVER_RUNTIME_ARCHIVE": str(archive),
        "SERVER_RUNTIME_ARCHIVE_SHA256": _sha256(archive),
        "SERVER_RUNTIME_RECEIPT_SHA256": _sha256(receipt),
        "VLLM_SERVER_RUNTIME": str(destination),
        "MARS_SCRATCH_ROOT": str(scratch),
        "SLURM_JOB_ID": "123",
    }

    first = subprocess.run(
        ["bash", str(_SERVER_STAGER)], env=env, capture_output=True, text=True, check=False
    )
    second = subprocess.run(
        ["bash", str(_SERVER_STAGER)], env=env, capture_output=True, text=True, check=False
    )

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert (destination / ".archive.sha256").read_text().strip() == _sha256(archive)
    assert _sha256(destination / "dflash2-vllm-runtime-receipt.json") == _sha256(receipt)
    assert str(destination) in (destination / "bin/activate").read_text()


def test_server_runtime_stager_rejects_receipt_tamper(tmp_path: Path) -> None:
    """An archive cannot be relabelled with an unrelated runtime receipt."""
    source = tmp_path / "source"
    (source / "bin").mkdir(parents=True)
    python = source / "bin/python"
    python.write_text("#!/bin/bash\nexit 0\n")
    python.chmod(0o755)
    (source / "bin/activate").write_text('export VIRTUAL_ENV="/old/runtime"\n')
    (source / "pyvenv.cfg").write_text("home = /old/runtime\n")
    (source / "dflash2-vllm-runtime-receipt.json").write_text("{}\n")
    archive = tmp_path / "server.tar"
    with tarfile.open(archive, "w") as stream:
        for path in source.rglob("*"):
            stream.add(path, arcname=path.relative_to(source))
    scratch = tmp_path / "scratch"
    result = subprocess.run(
        ["bash", str(_SERVER_STAGER)],
        env={
            **os.environ,
            "SERVER_RUNTIME_ARCHIVE": str(archive),
            "SERVER_RUNTIME_ARCHIVE_SHA256": _sha256(archive),
            "SERVER_RUNTIME_RECEIPT_SHA256": "0" * 64,
            "VLLM_SERVER_RUNTIME": str(scratch / "123/server"),
            "MARS_SCRATCH_ROOT": str(scratch),
            "SLURM_JOB_ID": "123",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
