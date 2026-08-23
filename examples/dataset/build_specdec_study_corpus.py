# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build deterministic assistant-token-budgeted SpecDec study manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer_targets(total: int, weights: dict[str, float]) -> dict[str, int]:
    raw = {key: total * float(weight) for key, weight in weights.items()}
    targets = {key: int(value) for key, value in raw.items()}
    remainder = total - sum(targets.values())
    order = sorted(weights, key=lambda key: (-(raw[key] - targets[key]), key))
    for key in order[:remainder]:
        targets[key] += 1
    return targets


def _rank(row: dict[str, Any], seed: int) -> str:
    identity = "\0".join(
        str(row[key]) for key in ("pool", "category", "context_bucket", "prompt_id")
    )
    return hashlib.sha256(f"{seed}\0{identity}".encode()).hexdigest()


def _validate_config(config: dict[str, Any], arm: str) -> None:
    if arm not in config["arms"]:
        raise ValueError(f"unknown arm {arm!r}")
    if abs(sum(config["arms"][arm].values()) - 1.0) > 1e-9:
        raise ValueError(f"arm {arm} pool weights must sum to 1")
    for pool, pool_config in config["pools"].items():
        if abs(sum(pool_config["categories"].values()) - 1.0) > 1e-9:
            raise ValueError(f"pool {pool} category weights must sum to 1")
    if int(config.get("training_seq_len", 0)) < 1:
        raise ValueError("training_seq_len must be positive")
    if re.fullmatch(r"[0-9a-f]{64}", str(config.get("tokenizer_sha256", ""))) is None:
        raise ValueError("tokenizer_sha256 must be an exact lowercase digest")


def _validate_candidates(
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
    held_out_prompt_ids: set[str],
) -> None:
    seen: set[str] = set()
    allowed_buckets = set(config["context_buckets"])
    for row in candidates:
        missing = {
            "prompt_id",
            "pool",
            "category",
            "context_bucket",
            "assistant_tokens",
            "source_id",
            "source_revision",
            "license",
            "source_manifest_sha256",
            "source_file_path",
            "source_file_sha256",
            "full_token_count",
            "response_source",
            "tool_lane",
        } - row.keys()
        if missing:
            raise ValueError(f"candidate missing required fields: {sorted(missing)}")
        prompt_id = str(row["prompt_id"])
        if prompt_id in seen:
            raise ValueError(f"duplicate prompt_id {prompt_id}")
        seen.add(prompt_id)
        if prompt_id in held_out_prompt_ids:
            raise ValueError(f"held-out contamination detected for prompt_id {prompt_id}")
        if int(row["assistant_tokens"]) <= 0:
            raise ValueError(f"prompt {prompt_id} has no trainable assistant tokens")
        if not isinstance(row["source_id"], str) or not row["source_id"].strip():
            raise ValueError(f"prompt {prompt_id} has no source identity")
        if not isinstance(row["license"], str) or not row["license"].strip():
            raise ValueError(f"prompt {prompt_id} has no source license")
        for digest_field in ("source_manifest_sha256", "source_file_sha256"):
            if re.fullmatch(r"[0-9a-f]{64}", str(row[digest_field])) is None:
                raise ValueError(f"prompt {prompt_id} has no verified {digest_field}")
        if not isinstance(row["source_file_path"], str) or not row["source_file_path"]:
            raise ValueError(f"prompt {prompt_id} has no verified source file path")
        revision = row["source_revision"]
        if (
            not isinstance(revision, str)
            or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision) is None
        ):
            raise ValueError(f"prompt {prompt_id} requires a pinned source revision")
        token_ids = row.get("input_ids", row.get("token_ids"))
        loss_mask = row.get("loss_mask")
        if (
            not isinstance(token_ids, list)
            or not token_ids
            or not all(
                isinstance(token, int) and not isinstance(token, bool) for token in token_ids
            )
            or not isinstance(loss_mask, list)
            or len(loss_mask) != len(token_ids)
            or any(value not in (0, 1) for value in loss_mask)
        ):
            raise ValueError(f"prompt {prompt_id} requires aligned token IDs and binary loss_mask")
        if int(row["assistant_tokens"]) != sum(loss_mask):
            raise ValueError(f"prompt {prompt_id} assistant_tokens does not match loss_mask")
        if len(token_ids) > int(config["training_seq_len"]):
            raise ValueError(f"prompt {prompt_id} exceeds the training sequence length")
        if row.get("tokenizer_sha256") != config["tokenizer_sha256"]:
            raise ValueError(f"prompt {prompt_id} tokenizer identity mismatch")
        if row["context_bucket"] not in allowed_buckets:
            raise ValueError(f"prompt {prompt_id} has invalid context bucket")
        if "full_token_count" in row:
            full_token_count = int(row["full_token_count"])
            derived_bucket = (
                "le4k"
                if full_token_count <= 4096
                else "4k_16k"
                if full_token_count <= 16384
                else "16k_32k"
                if full_token_count <= 32768
                else None
            )
            if row["context_bucket"] != derived_bucket:
                raise ValueError(f"prompt {prompt_id} has an invalid derived context bucket")


def _trim_assistant_tokens(row: dict[str, Any], keep: int) -> dict[str, Any]:
    """Return a copy supervising exactly the first ``keep`` assistant tokens."""
    if keep <= 0 or keep > int(row["assistant_tokens"]):
        raise ValueError("assistant-token trim must retain a positive in-range count")
    trimmed = deepcopy(row)
    remaining = keep
    mask = []
    for value in row["loss_mask"]:
        retained = int(value == 1 and remaining > 0)
        mask.append(retained)
        remaining -= retained
    trimmed["loss_mask"] = mask
    trimmed["assistant_tokens"] = keep
    return trimmed


def _take_until_tokens(
    ordered: Iterable[dict[str, Any]],
    *,
    target_tokens: int,
    selected_ids: set[str],
    selected: list[dict[str, Any]],
) -> int:
    tokens = sum(int(row["assistant_tokens"]) for row in selected)
    if tokens > target_tokens:
        raise ValueError("configured prompt minimum exceeds assistant-token quota")
    for row in ordered:
        if tokens >= target_tokens:
            break
        prompt_id = str(row["prompt_id"])
        if prompt_id in selected_ids:
            continue
        remaining = target_tokens - tokens
        selected.append(
            row
            if int(row["assistant_tokens"]) <= remaining
            else _trim_assistant_tokens(row, remaining)
        )
        selected_ids.add(prompt_id)
        tokens += min(int(row["assistant_tokens"]), remaining)
    return tokens


def select_arm(
    candidates: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    arm: str,
    target_assistant_tokens: int,
    held_out_prompt_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    _validate_config(config, arm)
    held_out = held_out_prompt_ids or set()
    _validate_candidates(candidates, config, held_out)
    seed = int(config["seed"])
    ranked = sorted(candidates, key=lambda row: (_rank(row, seed), str(row["prompt_id"])))
    pool_targets = _integer_targets(target_assistant_tokens, config["arms"][arm])
    all_selected: list[dict[str, Any]] = []

    for pool, pool_target in pool_targets.items():
        if pool_target == 0:
            continue
        pool_config = config["pools"][pool]
        category_targets = _integer_targets(pool_target, pool_config["categories"])
        for category, category_target in category_targets.items():
            if category_target == 0:
                continue
            available = [
                row for row in ranked if row["pool"] == pool and row["category"] == category
            ]
            if not available:
                raise ValueError(f"no candidates for {pool}/{category}")
            selected: list[dict[str, Any]] = []
            selected_ids: set[str] = set()
            cell_minimum = int(config.get("minimum_prompts_per_populated_cell", 0))
            if cell_minimum:
                for bucket in config["context_buckets"]:
                    cell = [row for row in available if row["context_bucket"] == bucket]
                    if cell and len(cell) < cell_minimum:
                        raise ValueError(
                            f"{pool}/{category}/{bucket} has {len(cell)} prompts, "
                            f"below minimum {cell_minimum}"
                        )
                    for row in cell[:cell_minimum]:
                        selected.append(row)
                        selected_ids.add(str(row["prompt_id"]))

            category_minimum = int(config.get("minimum_prompts_per_category", 0))
            for row in available:
                if len(selected) >= category_minimum:
                    break
                if str(row["prompt_id"]) not in selected_ids:
                    selected.append(row)
                    selected_ids.add(str(row["prompt_id"]))
            if len(selected) < category_minimum:
                raise ValueError(
                    f"{pool}/{category} has {len(selected)} prompts, below minimum "
                    f"{category_minimum}"
                )

            actual = _take_until_tokens(
                available,
                target_tokens=category_target,
                selected_ids=selected_ids,
                selected=selected,
            )
            if actual < category_target:
                raise ValueError(
                    f"{pool}/{category} has {actual} assistant tokens, below target "
                    f"{category_target}"
                )
            all_selected.extend(selected)

    return sorted(all_selected, key=lambda row: (_rank(row, seed), str(row["prompt_id"])))


def token_totals_by(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    totals: dict[str, int] = defaultdict(int)
    for row in rows:
        totals[str(row[key])] += int(row["assistant_tokens"])
    return dict(sorted(totals.items()))


def select_ptv23_arm(
    candidates: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    arm: str,
    target_assistant_tokens: int,
    prior_prompt_ids: set[str] | None = None,
    held_out_prompt_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Select exact assistant-loss-token B/C/D domain and replay-lane quotas."""
    if arm not in {"B", "C", "D"}:
        raise ValueError(f"unknown arm {arm!r}")
    exclusions = (prior_prompt_ids or set()) | (held_out_prompt_ids or set())
    seen: set[str] = set()
    eligible = []
    for row in candidates:
        prompt_id = str(row["prompt_id"])
        if prompt_id in seen:
            raise ValueError(f"duplicate prompt_id {prompt_id}")
        seen.add(prompt_id)
        if prompt_id in exclusions:
            continue
        if arm == "B":
            if row["pool"] != "ptv2":
                continue
            if row.get("language") == "de":
                continue
            if row["category"] == "swe":
                continue
        eligible.append(row)

    weights = config["b_domains" if arm == "B" else "cd_domains"]
    domain_targets = _integer_targets(target_assistant_tokens, weights)
    seed = int(config["seed"])
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    for domain, target in domain_targets.items():
        if domain == "swe":
            lanes = config["c_swe_lanes" if arm == "C" else "d_swe_lanes"]
            lane_targets = {
                lane: int(target_assistant_tokens * float(weight)) for lane, weight in lanes.items()
            }
            if sum(lane_targets.values()) != target:
                raise ValueError("SWE lane quotas do not reconcile with the top-level quota")
        else:
            lane_targets = {"target-synth": target}
        for lane, lane_target in lane_targets.items():
            available = [
                row
                for row in eligible
                if row["category"] == domain and row.get("lane", "target-synth") == lane
            ]
            available.sort(
                key=lambda row: hashlib.sha256(
                    "\0".join(
                        (
                            arm,
                            domain,
                            str(row["context_bucket"]),
                            str(row["source_id"]),
                            str(row["prompt_id"]),
                            str(seed),
                        )
                    ).encode()
                ).hexdigest()
            )
            lane_selected: list[dict[str, Any]] = []
            actual = _take_until_tokens(
                available,
                target_tokens=lane_target,
                selected_ids=selected_ids,
                selected=lane_selected,
            )
            if actual != lane_target:
                raise ValueError(f"quota shortfall for {domain}/{lane}: {actual} < {lane_target}")
            selected.extend(lane_selected)
    return selected


def build_arm_manifest(
    selected: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    arm: str,
    target_assistant_tokens: int,
    exposure_epochs: int,
) -> dict[str, Any]:
    maximum_epochs = int(config["maximum_epochs"])
    if not 1 <= exposure_epochs <= maximum_epochs:
        raise ValueError(f"exposure_epochs exceeds maximum_epochs={maximum_epochs}")
    unique_tokens = sum(int(row["assistant_tokens"]) for row in selected)
    sources: dict[str, dict[str, Any]] = {}
    source_files: set[tuple[str, str]] = set()
    for row in selected:
        source_id = str(row["source_id"])
        identity = {
            "revision": str(row["source_revision"]),
            "license": str(row["license"]),
            "manifest_sha256": row.get("source_manifest_sha256"),
        }
        previous = sources.setdefault(source_id, identity)
        if previous != identity:
            raise ValueError(f"source identity changed within manifest: {source_id}")
        if row.get("source_file_path") and row.get("source_file_sha256"):
            source_files.add((str(row["source_file_path"]), str(row["source_file_sha256"])))
    category_totals: dict[str, int] = defaultdict(int)
    for row in selected:
        category_totals[f"{row['pool']}/{row['category']}"] += int(row["assistant_tokens"])
    return {
        "schema_version": 1,
        "arm": arm,
        "seed": int(config["seed"]),
        "target_assistant_tokens": target_assistant_tokens,
        "unique_assistant_tokens": unique_tokens,
        "exposure_epochs": exposure_epochs,
        "exposure_assistant_tokens": unique_tokens * exposure_epochs,
        "prompt_count": len(selected),
        "pool_assistant_tokens": token_totals_by(selected, "pool"),
        "category_assistant_tokens": dict(sorted(category_totals.items())),
        "context_assistant_tokens": token_totals_by(selected, "context_bucket"),
        "source_revisions": dict(sorted(sources.items())),
        "source_files": [{"path": path, "sha256": digest} for path, digest in sorted(source_files)],
        "selected_prompt_ids_sha256": hashlib.sha256(
            "\n".join(str(row["prompt_id"]) for row in selected).encode()
        ).hexdigest(),
    }


def materialize_parquet(
    selected: list[dict[str, Any]],
    output_dir: Path,
    *,
    rows_per_shard: int,
    tokenizer_sha256: str,
    selection_manifest: dict[str, Any] | None = None,
    selection_manifest_name: str = "SELECTION.json",
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not selected:
        raise ValueError("cannot materialize an empty corpus")
    if rows_per_shard <= 0:
        raise ValueError("rows_per_shard must be positive")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    partial = output_dir.with_name(f".{output_dir.name}.partial-{os.getpid()}")
    if partial.exists():
        raise FileExistsError(f"partial directory already exists: {partial}")
    partial.mkdir()
    try:
        schema = pa.schema(
            [
                pa.field("prompt_id", pa.string(), nullable=False),
                pa.field("prompt_uuid", pa.string()),
                pa.field("arm", pa.string()),
                pa.field("pool", pa.string(), nullable=False),
                pa.field("category", pa.string(), nullable=False),
                pa.field("domain", pa.string()),
                pa.field("lane", pa.string()),
                pa.field("language", pa.string()),
                pa.field("context_bucket", pa.string(), nullable=False),
                pa.field("assistant_tokens", pa.int64(), nullable=False),
                pa.field("full_token_count", pa.int64()),
                pa.field("full_assistant_tokens", pa.int64()),
                pa.field("source_id", pa.string(), nullable=False),
                pa.field("source_revision", pa.string(), nullable=False),
                pa.field("license", pa.string(), nullable=False),
                pa.field("source_manifest_sha256", pa.string()),
                pa.field("source_file_path", pa.string()),
                pa.field("source_file_sha256", pa.string()),
                pa.field("source_row_index", pa.int64()),
                pa.field("candidate_rank", pa.int64()),
                pa.field("selection_index", pa.int64()),
                pa.field("selection_status", pa.string()),
                pa.field("response_source", pa.string()),
                pa.field("tool_lane", pa.string()),
                pa.field("tokenizer_sha256", pa.string(), nullable=False),
                pa.field("input_ids", pa.list_(pa.int64()), nullable=False),
                pa.field("loss_mask", pa.list_(pa.int8()), nullable=False),
                pa.field("messages", pa.large_string()),
                pa.field("tools", pa.large_string()),
                pa.field("canonical_prompt", pa.large_string()),
            ]
        )
        normalized = []
        for row in selected:
            value = {field.name: row.get(field.name) for field in schema}
            value["domain"] = row.get("domain", row.get("category"))
            value["selection_status"] = row.get("selection_status", row.get("status"))
            value["full_token_count"] = row.get("full_token_count", len(row["input_ids"]))
            value["full_assistant_tokens"] = row.get(
                "full_assistant_tokens", row["assistant_tokens"]
            )
            value["messages"] = (
                json.dumps(row["messages"], ensure_ascii=False, sort_keys=True)
                if row.get("messages") is not None
                else None
            )
            value["tools"] = (
                json.dumps(row["tools"], ensure_ascii=False, sort_keys=True)
                if row.get("tools") is not None
                else None
            )
            value["canonical_prompt"] = (
                json.dumps(row["canonical_prompt"], ensure_ascii=False, sort_keys=True)
                if row.get("canonical_prompt") is not None
                else None
            )
            normalized.append(value)
        files = []
        shard_count = (len(selected) + rows_per_shard - 1) // rows_per_shard
        for shard_index, start in enumerate(range(0, len(selected), rows_per_shard)):
            rows = normalized[start : start + rows_per_shard]
            name = f"train-{shard_index:05d}-of-{shard_count:05d}.parquet"
            path = partial / name
            pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")
            files.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        manifest = {
            "schema_version": 1,
            "format": "parquet",
            "compression": "zstd",
            "row_count": len(selected),
            "unique_assistant_tokens": sum(int(row["assistant_tokens"]) for row in selected),
            "tokenizer_sha256": tokenizer_sha256,
            "file_count": len(files),
            "files": files,
        }
        (partial / "MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if selection_manifest is not None:
            if Path(selection_manifest_name).name != selection_manifest_name:
                raise ValueError("selection manifest name must be a basename")
            published_selection = dict(selection_manifest)
            published_selection["corpus_manifest_path"] = str(
                (output_dir / "MANIFEST.json").resolve()
            )
            published_selection["corpus_manifest_sha256"] = sha256_file(partial / "MANIFEST.json")
            published_selection["tokenizer_sha256"] = tokenizer_sha256
            (partial / selection_manifest_name).write_text(
                json.dumps(published_selection, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        os.rename(partial, output_dir)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return manifest


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial-{os.getpid()}")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=["B", "C", "D"], required=True)
    parser.add_argument("--target-assistant-tokens", type=int, required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--exposure-epochs", type=int, default=1)
    parser.add_argument("--held-out-prompt-ids", type=Path)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-corpus", type=Path, required=True)
    parser.add_argument("--rows-per-shard", type=int, default=10_000)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config["tokenizer_sha256"] = args.tokenizer_sha256
    candidates = _load_jsonl(args.inventory)
    held_out = (
        set(args.held_out_prompt_ids.read_text(encoding="utf-8").splitlines())
        if args.held_out_prompt_ids
        else set()
    )
    selected = select_arm(
        candidates,
        config=config,
        arm=args.arm,
        target_assistant_tokens=args.target_assistant_tokens,
        held_out_prompt_ids=held_out,
    )
    manifest = build_arm_manifest(
        selected,
        config=config,
        arm=args.arm,
        target_assistant_tokens=args.target_assistant_tokens,
        exposure_epochs=args.exposure_epochs,
    )
    if args.output_manifest.parent.resolve(strict=False) != args.output_corpus.resolve(
        strict=False
    ):
        raise ValueError("output manifest must be published inside output corpus")
    materialize_parquet(
        selected,
        args.output_corpus,
        rows_per_shard=args.rows_per_shard,
        tokenizer_sha256=args.tokenizer_sha256,
        selection_manifest=manifest,
        selection_manifest_name=args.output_manifest.name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
