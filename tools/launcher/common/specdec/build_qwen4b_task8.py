# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize and publish one authenticated Qwen3-4B PTV2 Task8 arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import stat
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import yaml

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_RESPONSE_KEYS = frozenset(
    {
        "schema_version",
        "role",
        "source_commit",
        "selection_sha256",
        "source_response_root_sha256",
        "occurrence_count",
        "files",
        "receipt_sha256",
    }
)
_REJECTION_KEYS = frozenset(
    {
        "schema_version",
        "role",
        "source_commit",
        "selection_sha256",
        "occurrence_count",
        "rejection_count",
        "reason_counts",
        "files",
        "receipt_sha256",
    }
)


class Task8BuildError(ValueError):
    """A Task8 input or publication contract is incomplete or invalid."""


def publication_rows_per_shard(occurrence_count: int) -> int:
    """Return the stable row width that yields at most 201 source-order shards."""
    if isinstance(occurrence_count, bool) or not isinstance(occurrence_count, int) or occurrence_count < 1:
        raise Task8BuildError("publication occurrence count must be positive")
    return (occurrence_count + 200) // 201


@dataclass(frozen=True)
class _BoundTokenizer:
    tokenizer: Any
    tokenizer_sha256: str
    chat_template: str
    assistant_loss_target_sha256: str

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> Any:
        return self.tokenizer.apply_chat_template(messages, **kwargs)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _require_digest(value: str, label: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise Task8BuildError(f"{label} must be a lowercase SHA-256")


def _read_receipt(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    _require_digest(expected_sha256, f"{label} receipt SHA-256")
    if path.is_symlink() or not path.is_file():
        raise Task8BuildError(f"{label} receipt must be a no-follow regular file")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise Task8BuildError(f"{label} receipt SHA-256 mismatch")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Task8BuildError(f"{label} receipt is not JSON") from error
    if not isinstance(payload, dict):
        raise Task8BuildError(f"{label} receipt must be an object")
    return payload


def _authenticate_declared_files(
    receipt_path: Path, payload: Mapping[str, Any], label: str
) -> None:
    descriptors = payload.get("files")
    if not isinstance(descriptors, list) or not descriptors:
        raise Task8BuildError(f"{label} receipt has no declared files")
    for descriptor in descriptors:
        if not isinstance(descriptor, Mapping):
            raise Task8BuildError(f"{label} declared file descriptor is malformed")
        _authenticate_descriptor(receipt_path, descriptor, f"{label} declared file")


def _authenticate_descriptor(
    receipt_path: Path, descriptor: Mapping[str, Any], label: str
) -> Path:
    raw_relative = descriptor.get("path", descriptor.get("staged_path"))
    relative = PurePosixPath(str(raw_relative or ""))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise Task8BuildError(f"{label} path is unsafe")
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        parent_descriptor = os.open(receipt_path.parent, directory_flags)
    except OSError as error:
        raise Task8BuildError(f"{label} parent is unsafe") from error
    file_descriptor = -1
    try:
        for component in relative.parts[:-1]:
            try:
                child = os.open(component, directory_flags, dir_fd=parent_descriptor)
            except OSError as error:
                raise Task8BuildError(
                    f"{label} contains a symlink or non-directory"
                ) from error
            os.close(parent_descriptor)
            parent_descriptor = child
        try:
            file_descriptor = os.open(relative.parts[-1], file_flags, dir_fd=parent_descriptor)
        except OSError as error:
            raise Task8BuildError(f"{label} is missing or unsafe") from error
        observed = os.fstat(file_descriptor)
        digest = descriptor.get("sha256")
        content_digest = hashlib.sha256()
        with os.fdopen(os.dup(file_descriptor), "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                content_digest.update(block)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_size != descriptor.get("bytes")
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            or content_digest.hexdigest() != digest
        ):
            raise Task8BuildError(f"{label} authentication failed")
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        os.close(parent_descriptor)
    return receipt_path.parent.joinpath(*relative.parts)


def _identity(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = payload.get("identity")
    return nested if isinstance(nested, Mapping) else payload


def validate_required_role_receipts(
    *,
    view: Any,
    source_commit: str,
    response_receipt: Path,
    response_receipt_sha256: str,
    rejection_receipt: Path,
    rejection_receipt_sha256: str,
) -> None:
    """Fail before tokenization unless genuine response and rejection roots are pinned."""
    response_payload = _read_receipt(
        Path(response_receipt), response_receipt_sha256, "response"
    )
    rejection_payload = _read_receipt(
        Path(rejection_receipt), rejection_receipt_sha256, "rejection"
    )
    _authenticate_declared_files(Path(response_receipt), response_payload, "response")
    _authenticate_declared_files(Path(rejection_receipt), rejection_payload, "rejection")
    response = _identity(response_payload)
    rejection = _identity(rejection_payload)
    selection_sha256 = getattr(view, "selection_sha256", None)
    response_root = getattr(view, "source_response_root_sha256", None)
    response_claimed = response_payload.get("receipt_sha256")
    rejection_claimed = rejection_payload.get("receipt_sha256")
    response_without_claim = {
        key: value for key, value in response_payload.items() if key != "receipt_sha256"
    }
    rejection_without_claim = {
        key: value for key, value in rejection_payload.items() if key != "receipt_sha256"
    }
    reason_counts = rejection.get("reason_counts")
    rejection_count = rejection.get("rejection_count")
    if (
        set(response_payload) != _RESPONSE_KEYS
        or response.get("schema_version") != 1
        or response.get("role") != "response"
        or response.get("source_commit") != source_commit
        or _COMMIT.fullmatch(source_commit) is None
        or response.get("occurrence_count") != getattr(view, "occurrence_count", None)
        or response_claimed != hashlib.sha256(_canonical_json(response_without_claim)).hexdigest()
    ):
        raise Task8BuildError("response receipt schema or identity is invalid")
    if (
        set(rejection_payload) != _REJECTION_KEYS
        or rejection.get("schema_version") != 1
        or rejection.get("role") != "rejection"
        or rejection.get("source_commit") != source_commit
        or rejection.get("occurrence_count") != getattr(view, "occurrence_count", None)
        or isinstance(rejection_count, bool)
        or not isinstance(rejection_count, int)
        or rejection_count < 0
        or not isinstance(reason_counts, Mapping)
        or any(
            not isinstance(reason, str)
            or not reason
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
            for reason, count in reason_counts.items()
        )
        or sum(reason_counts.values()) != rejection_count
        or rejection_claimed
        != hashlib.sha256(_canonical_json(rejection_without_claim)).hexdigest()
    ):
        raise Task8BuildError("rejection receipt schema or identity is invalid")
    if response.get("selection_sha256") != selection_sha256:
        raise Task8BuildError("response receipt is not bound to the Task9 selection")
    if response.get("source_response_root_sha256") != response_root:
        raise Task8BuildError("response receipt does not bind Task9 source-native responses")
    if rejection.get("selection_sha256") != selection_sha256:
        raise Task8BuildError("rejection receipt is not bound to the Task9 selection")


def _publication_rows(view: Any, tokenized_database: Path) -> Iterator[dict[str, Any]]:
    selection = sqlite3.connect(f"file:{Path(view.index_path)}?mode=ro", uri=True)
    tokenized = sqlite3.connect(f"file:{tokenized_database}?mode=ro", uri=True)
    try:
        selected_rows = selection.execute(
            "SELECT occurrences.ordinal,occurrences.prompt_uuid,occurrences.cell,"
            "source_rows.language FROM occurrences JOIN source_rows "
            "ON occurrences.source_identity_sha256=source_rows.source_identity_sha256 "
            "AND occurrences.source_row=source_rows.source_row "
            "WHERE occurrences.strategy=? ORDER BY occurrences.ordinal",
            (view.strategy,),
        )
        token_rows = tokenized.execute(
            "SELECT ordinal,prompt_uuid,input_ids_json,loss_mask_json,assistant_tokens "
            "FROM records ORDER BY ordinal"
        )
        count = 0
        sentinel = object()
        for selected, tokens in zip_longest(selected_rows, token_rows, fillvalue=sentinel):
            if selected is sentinel or tokens is sentinel:
                raise Task8BuildError("selection and tokenized row counts differ")
            assert isinstance(selected, tuple) and isinstance(tokens, tuple)
            ordinal, prompt_uuid, cell, language = selected
            token_ordinal, token_prompt_uuid, ids_json, mask_json, assistant_tokens = tokens
            if (ordinal, prompt_uuid) != (token_ordinal, token_prompt_uuid):
                raise Task8BuildError("selection and tokenized row identities differ")
            input_ids = json.loads(ids_json)
            loss_mask = [bool(value) for value in json.loads(mask_json)]
            count += 1
            yield {
                "prompt_uuid": prompt_uuid,
                "domain": cell,
                "lane": "source-native",
                "context_bucket": "ptv2-one-pass",
                "language": language,
                "input_ids": input_ids,
                "loss_mask": loss_mask,
                "assistant_tokens": assistant_tokens,
                "rejection_reason": None,
            }
        if count != view.occurrence_count:
            raise Task8BuildError("published row count does not match Task9 selection")
    finally:
        tokenized.close()
        selection.close()


def materialize_task8_publication(
    *,
    view: Any,
    tokenizer: Any,
    tokenizer_sha256: str,
    chat_template_sha256: str,
    assistant_loss_target_sha256: str,
    training_config_sha256: str,
    source_receipt: Path,
    source_receipt_sha256: str,
    selection_receipt: Path,
    selection_receipt_sha256: str,
    response_receipt: Path,
    response_receipt_sha256: str,
    rejection_receipt: Path,
    rejection_receipt_sha256: str,
    work_root: Path,
    materialization_root: Path,
    publication_root: Path,
    source_commit: str,
    job_id: str,
    workers: int = 96,
) -> Any:
    """Build token/exposure artifacts and atomically publish all six authenticated roles."""
    from build_assistant_token_views import (  # pyright: ignore[reportMissingImports]
        _materialize_ptv2_exposure,
        derive_ptv2_one_pass_corpus,
    )
    from specdec_publication import (  # pyright: ignore[reportMissingImports]
        CorpusBundle,
        InputArtifact,
        publish_bundle,
    )

    source_payload = _read_receipt(Path(source_receipt), source_receipt_sha256, "source")
    _authenticate_declared_files(Path(source_receipt), source_payload, "source")
    _read_receipt(Path(selection_receipt), selection_receipt_sha256, "selection")
    validate_required_role_receipts(
        view=view,
        source_commit=source_commit,
        response_receipt=response_receipt,
        response_receipt_sha256=response_receipt_sha256,
        rejection_receipt=rejection_receipt,
        rejection_receipt_sha256=rejection_receipt_sha256,
    )
    for digest, label in (
        (tokenizer_sha256, "tokenizer"),
        (chat_template_sha256, "chat template"),
        (assistant_loss_target_sha256, "assistant loss target"),
        (training_config_sha256, "training config"),
        (source_receipt_sha256, "source receipt"),
        (selection_receipt_sha256, "selection receipt"),
    ):
        _require_digest(digest, label)
    if _COMMIT.fullmatch(source_commit) is None:
        raise Task8BuildError("source commit must be exact")
    if workers < 1:
        raise Task8BuildError("Task8 workers must be positive")
    work_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    corpus = derive_ptv2_one_pass_corpus(
        view,
        tokenizer_sha256=tokenizer_sha256,
        chat_template_sha256=chat_template_sha256,
        assistant_loss_target_sha256=assistant_loss_target_sha256,
        training_config_sha256=training_config_sha256,
        tokenizer=tokenizer,
        output_root=materialization_root,
        work_root=work_root,
        workers=workers,
        source_commit=source_commit,
    )
    _materialize_ptv2_exposure(corpus, corpus.assistant_tokens)
    exposure_root = (
        Path(corpus.tokenized_path).parent.parent / f"{corpus.strategy.lower()}-exposures"
    )
    exposure_receipt = (
        exposure_root / f"{corpus.strategy.lower()}-{corpus.assistant_tokens}-assistant-tokens.json"
    )
    tokenized_receipt = Path(corpus.receipt_path)
    artifacts = (
        InputArtifact("source", Path(source_receipt), source_receipt_sha256),
        InputArtifact("selection", Path(selection_receipt), selection_receipt_sha256),
        InputArtifact("response", Path(response_receipt), response_receipt_sha256),
        InputArtifact("tokenized", tokenized_receipt, _sha256_file(tokenized_receipt)),
        InputArtifact("exposure", exposure_receipt, _sha256_file(exposure_receipt)),
        InputArtifact("rejection", Path(rejection_receipt), rejection_receipt_sha256),
    )
    return publish_bundle(
        CorpusBundle(
            artifacts=artifacts,
            rows=lambda: _publication_rows(view, Path(corpus.tokenized_path)),
            prompt_count=corpus.occurrence_count,
            assistant_token_count=corpus.assistant_tokens,
            quarantine_count=0,
            selection_manifest_sha256=selection_receipt_sha256,
            artifact_source_commit=source_commit,
        ),
        publication_root,
        job_id,
        rows_per_shard=publication_rows_per_shard(corpus.occurrence_count),
    )


def _validate_typed_task9_receipt(
    receipt_path: Path, payload: Mapping[str, Any], source_commit: str
) -> None:
    from specdec_publication import (  # pyright: ignore[reportMissingImports]
        PublicationError,
        _role_file_descriptors,
        _validate_ptv2_selection_policy,
        _validate_task9_execution_receipt,
    )

    claimed_root = payload.get("root_sha256")
    root_body = {key: value for key, value in payload.items() if key != "root_sha256"}
    if claimed_root != hashlib.sha256(_canonical_json(root_body)).hexdigest():
        raise Task8BuildError("Task9 selection root digest is invalid")
    try:
        descriptors = _role_file_descriptors("selection", payload)
        files: list[tuple[str, Path, int, str]] = []
        for descriptor in descriptors:
            if not isinstance(descriptor, Mapping):
                raise Task8BuildError("Task9 declared file descriptor is malformed")
            path = _authenticate_descriptor(receipt_path, descriptor, "Task9 declared file")
            relative = str(descriptor.get("path", descriptor.get("staged_path", "")))
            size = descriptor.get("bytes")
            digest = descriptor.get("sha256")
            if not isinstance(size, int) or isinstance(size, bool) or not isinstance(digest, str):
                raise Task8BuildError("Task9 declared file identity is malformed")
            files.append((relative, path, size, digest))
        _validate_ptv2_selection_policy(payload, files)
        _validate_task9_execution_receipt(payload, files)
    except PublicationError as error:
        raise Task8BuildError("Task9 typed selection authentication failed") from error
    descriptor = payload["execution_receipt"]
    execution_path = receipt_path.parent / str(descriptor["path"])
    execution = _read_receipt(
        execution_path,
        str(descriptor["sha256"]),
        "Task9 execution",
    )
    if execution.get("source_commit") != source_commit:
        raise Task8BuildError("Task9 execution source commit does not match Task8")


def load_task9_selection(receipt_path: Path, receipt_sha256: str, source_commit: str) -> Any:
    """Authenticate an exact schema-v3 Task9 selection and reconstruct its materialized view."""
    payload = _read_receipt(receipt_path, receipt_sha256, "selection")
    identity = payload.get("selection_identity")
    descriptor = payload.get("index")
    if (
        payload.get("schema_version") != 3
        or payload.get("strategy") not in {"A-repair", "B-balanced"}
        or payload.get("occurrence_count") != 2_000_000
        or not isinstance(payload.get("execution_receipt"), Mapping)
        or not isinstance(identity, dict)
        or not isinstance(descriptor, dict)
    ):
        raise Task8BuildError("Task8 requires an exact 2M schema-v3 Task9 selection")
    index_path = _authenticate_descriptor(receipt_path, descriptor, "Task9 selection index")
    policy_descriptor = payload.get("policy")
    if not isinstance(policy_descriptor, dict):
        raise Task8BuildError("Task9 selection policy descriptor is missing")
    policy_path = _authenticate_descriptor(receipt_path, policy_descriptor, "Task9 policy")
    try:
        policy = yaml.safe_load(policy_path.read_bytes())
    except (OSError, yaml.YAMLError) as error:
        raise Task8BuildError("Task9 selection policy is unreadable") from error
    if not isinstance(policy, dict) or policy.get("trainer_epochs") != 1:
        raise Task8BuildError("Task9 selection must bind one trainer epoch")
    _validate_typed_task9_receipt(receipt_path, payload, source_commit)
    connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    try:
        constructed = int(
            connection.execute(
                "SELECT count(*) FROM occurrences WHERE strategy=? AND reuse_index>0",
                (payload["strategy"],),
            ).fetchone()[0]
        )
    finally:
        connection.close()
    unique = identity.get("unique_prompt_count")
    if not isinstance(unique, int) or isinstance(unique, bool):
        raise Task8BuildError("Task9 unique prompt count is invalid")
    natural = payload["occurrence_count"] - unique - constructed
    if natural < 0:
        raise Task8BuildError("Task9 multiplicity counts do not reconcile")
    return SimpleNamespace(
        strategy=payload["strategy"],
        occurrence_count=payload["occurrence_count"],
        trainer_epochs=1,
        unique_prompt_count=unique,
        natural_duplicate_count=natural,
        constructed_repeat_count=constructed,
        index_path=index_path,
        ordered_occurrences_sha256=identity.get("ordered_occurrences_sha256"),
        source_response_root_sha256=identity.get("source_response_root_sha256"),
        selection_sha256=payload.get("selection_sha256"),
    )


def _load_bound_tokenizer(
    receipt_path: Path,
    receipt_sha256: str,
    assistant_loss_target_sha256: str,
) -> tuple[_BoundTokenizer, str, str]:
    from build_specdec_inventory import (  # pyright: ignore[reportMissingImports]
        load_tokenizer_snapshot,
    )
    from transformers import AutoTokenizer  # pyright: ignore[reportMissingImports]

    snapshot = load_tokenizer_snapshot(receipt_path, receipt_sha256)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot.root, local_files_only=True, trust_remote_code=False
    )
    return (
        _BoundTokenizer(
            tokenizer,
            snapshot.tokenizer_sha256,
            snapshot.chat_template_canonical_json,
            assistant_loss_target_sha256,
        ),
        snapshot.tokenizer_sha256,
        snapshot.chat_template_sha256,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source-receipt",
        "source-receipt-sha256",
        "selection-receipt",
        "selection-receipt-sha256",
        "response-receipt",
        "response-receipt-sha256",
        "rejection-receipt",
        "rejection-receipt-sha256",
        "tokenizer-receipt",
        "tokenizer-receipt-sha256",
        "assistant-loss-target-sha256",
        "training-config-sha256",
        "work-root",
        "materialization-root",
        "publication-root",
        "source-commit",
        "job-id",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--workers", type=int, default=96)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the fail-closed Task8 producer from command-line arguments."""
    args = _parser().parse_args(argv)
    selection_receipt = Path(args.selection_receipt)
    view = load_task9_selection(
        selection_receipt, args.selection_receipt_sha256, args.source_commit
    )
    validate_required_role_receipts(
        view=view,
        source_commit=args.source_commit,
        response_receipt=Path(args.response_receipt),
        response_receipt_sha256=args.response_receipt_sha256,
        rejection_receipt=Path(args.rejection_receipt),
        rejection_receipt_sha256=args.rejection_receipt_sha256,
    )
    tokenizer, tokenizer_sha256, template_sha256 = _load_bound_tokenizer(
        Path(args.tokenizer_receipt),
        args.tokenizer_receipt_sha256,
        args.assistant_loss_target_sha256,
    )
    receipt = materialize_task8_publication(
        view=view,
        tokenizer=tokenizer,
        tokenizer_sha256=tokenizer_sha256,
        chat_template_sha256=template_sha256,
        assistant_loss_target_sha256=args.assistant_loss_target_sha256,
        training_config_sha256=args.training_config_sha256,
        source_receipt=Path(args.source_receipt),
        source_receipt_sha256=args.source_receipt_sha256,
        selection_receipt=selection_receipt,
        selection_receipt_sha256=args.selection_receipt_sha256,
        response_receipt=Path(args.response_receipt),
        response_receipt_sha256=args.response_receipt_sha256,
        rejection_receipt=Path(args.rejection_receipt),
        rejection_receipt_sha256=args.rejection_receipt_sha256,
        work_root=Path(args.work_root),
        materialization_root=Path(args.materialization_root),
        publication_root=Path(args.publication_root),
        source_commit=args.source_commit,
        job_id=args.job_id,
        workers=args.workers,
    )
    print(json.dumps(receipt.__dict__, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
