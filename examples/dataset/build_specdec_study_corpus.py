# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

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
    for row in selected:
        source_id = str(row["source_id"])
        identity = {
            "revision": str(row["source_revision"]),
            "license": str(row["license"]),
        }
        previous = sources.setdefault(source_id, identity)
        if previous != identity:
            raise ValueError(f"source identity changed within manifest: {source_id}")
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
        files = []
        shard_count = (len(selected) + rows_per_shard - 1) // rows_per_shard
        for shard_index, start in enumerate(range(0, len(selected), rows_per_shard)):
            rows = selected[start : start + rows_per_shard]
            name = f"train-{shard_index:05d}-of-{shard_count:05d}.parquet"
            path = partial / name
            pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
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
    materialize_parquet(
        selected,
        args.output_corpus,
        rows_per_shard=args.rows_per_shard,
        tokenizer_sha256=args.tokenizer_sha256,
    )
    corpus_manifest_path = args.output_corpus / "MANIFEST.json"
    manifest["corpus_manifest_path"] = str(corpus_manifest_path.resolve())
    manifest["corpus_manifest_sha256"] = sha256_file(corpus_manifest_path)
    manifest["tokenizer_sha256"] = args.tokenizer_sha256
    _write_json_atomic(args.output_manifest, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
