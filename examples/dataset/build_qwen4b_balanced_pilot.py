# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministically select and publish the Qwen3-4B paired PTV2 pilot."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import uuid
from collections.abc import Iterable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HISTORICAL_PROPORTION = "historical-proportion"
BALANCED = "balanced"
SEED = 20_260_822


class PilotError(ValueError):
    """The pilot's caller-pinned data or publication contract is invalid."""


@dataclass(frozen=True)
class PilotQuota:
    """An exact row target from one source split for one pilot arm."""

    category: str
    split: str
    rows: int


@dataclass(frozen=True)
class PilotConfig:
    """Immutable scientific identity and exact selection quotas for the pilot."""

    seed: int = SEED
    workers: int = 1
    minimum_assistant_tokens: int = 16_000_000
    source_repository: str = "nvidia/Nemotron-Post-Training-Dataset-v2"
    source_revision: str = ""
    historical_quotas: tuple[PilotQuota, ...] = (
        PilotQuota("chat", "chat", 48_286),
        PilotQuota("math", "math", 18_420),
        PilotQuota("code", "code", 13_462),
        PilotQuota("stem", "stem", 19_832),
    )
    balanced_quotas: tuple[PilotQuota, ...] = (
        PilotQuota("math", "math", 25_000),
        PilotQuota("code", "code", 20_000),
        PilotQuota("stem", "stem", 25_000),
        PilotQuota("chat", "chat", 20_000),
        PilotQuota("multilingual", "multilingual_de", 2_000),
        PilotQuota("multilingual", "multilingual_es", 2_000),
        PilotQuota("multilingual", "multilingual_fr", 2_000),
        PilotQuota("multilingual", "multilingual_it", 2_000),
        PilotQuota("multilingual", "multilingual_ja", 2_000),
    )

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise PilotError("seed must be an integer")
        if isinstance(self.workers, bool) or not isinstance(self.workers, int) or self.workers < 1:
            raise PilotError("workers must be a positive integer")
        _validate_quotas(HISTORICAL_PROPORTION, self.historical_quotas)
        _validate_quotas(BALANCED, self.balanced_quotas)
        if sum(quota.rows for quota in self.historical_quotas) != sum(
            quota.rows for quota in self.balanced_quotas
        ):
            raise PilotError("both pilot arms must have the same row quota")
        if (
            isinstance(self.minimum_assistant_tokens, bool)
            or not isinstance(self.minimum_assistant_tokens, int)
            or self.minimum_assistant_tokens < 1
        ):
            raise PilotError("minimum_assistant_tokens must be a positive integer")


@dataclass(frozen=True)
class PilotRow:
    """A normalized, selected source row with a canonical prompt identity."""

    prompt_uuid: str
    split: str
    category: str
    messages_json: str

    @property
    def messages(self) -> list[dict[str, Any]]:
        value = json.loads(self.messages_json)
        if not isinstance(value, list):  # Defensive: this value is created below.
            raise PilotError("stored pilot messages are malformed")
        return value


@dataclass(frozen=True)
class PilotSelection:
    """The two paired deterministic selections and their filtering evidence."""

    historical_proportion: tuple[PilotRow, ...]
    balanced: tuple[PilotRow, ...]
    excluded_held_out: int
    excluded_invalid: int
    excluded_duplicate_candidates: int

    def rows_for(self, arm: str) -> tuple[PilotRow, ...]:
        if arm == HISTORICAL_PROPORTION:
            return self.historical_proportion
        if arm == BALANCED:
            return self.balanced
        raise PilotError(f"unknown pilot arm: {arm!r}")


@dataclass(frozen=True)
class PilotArmCompletion:
    """Receipt-bound immutable completion information for one published arm."""

    arm: str
    row_count: int
    assistant_tokens: int
    manifest_sha256: str


@dataclass(frozen=True)
class PilotCompletion:
    """Receipt-bound aggregate completion for the two-arm publication root."""

    output_root: Path
    historical_proportion: PilotArmCompletion
    balanced: PilotArmCompletion
    complete_sha256: str


def canonical_json(value: object) -> str:
    """Return the unambiguous canonical JSON representation used for identities."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def prompt_uuid_from_messages(messages: object) -> str:
    """Hash only the prompt context, removing exactly the terminal assistant turn."""
    normalized = _normalize_messages(messages)
    if normalized[-1]["role"] != "assistant":
        raise PilotError("row must have a terminal assistant response")
    return hashlib.sha256(canonical_json(normalized[:-1]).encode("utf-8")).hexdigest()


def load_held_out_prompt_uuids(path: Path) -> set[str]:
    """Load the caller-provided held-out prompt UUID JSON array fail-closed."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PilotError("held-out UUID receipt must be a JSON UUID array") from error
    if not isinstance(payload, list):
        raise PilotError("held-out UUID receipt must be a JSON UUID array")
    return _validate_held_out(payload)


def select_pilot_rows(
    split_rows: Mapping[str, Iterable[Mapping[str, object]]],
    *,
    config: PilotConfig | None = None,
    held_out_prompt_uuids: Iterable[str] = (),
    scratch_root: Path | None = None,
) -> PilotSelection:
    """Select exact paired quotas from a private SQLite spool without corpus materialization."""
    config = config or PilotConfig()
    if not isinstance(split_rows, Mapping):
        raise PilotError("split_rows must map PTV2 split names to iterables")
    held_out = _validate_held_out(held_out_prompt_uuids)
    scratch_parent = Path(scratch_root) if scratch_root is not None else Path(tempfile.gettempdir())
    if not scratch_parent.is_dir():
        raise PilotError("selection scratch root must be a directory")
    spool_root = Path(tempfile.mkdtemp(prefix=".qwen4b-selection-", dir=scratch_parent))
    os.chmod(spool_root, 0o700)
    database = spool_root / "selection.sqlite"
    try:
        connection = sqlite3.connect(database)
        try:
            _prepare_selection_database(connection)
            invalid, held_out_count = _spool_candidates(connection, split_rows, config, held_out)
            duplicate_count = _duplicate_count(connection)
            historical = _select_arm_from_spool(
                connection, HISTORICAL_PROPORTION, config.historical_quotas, config
            )
            balanced = _select_arm_from_spool(connection, BALANCED, config.balanced_quotas, config)
        finally:
            connection.close()
        return PilotSelection(
            historical_proportion=historical,
            balanced=balanced,
            excluded_held_out=held_out_count,
            excluded_invalid=invalid,
            excluded_duplicate_candidates=duplicate_count,
        )
    finally:
        shutil.rmtree(spool_root, ignore_errors=True)


def _validate_quotas(arm: str, quotas: tuple[PilotQuota, ...]) -> None:
    if not quotas:
        raise PilotError(f"{arm} quotas cannot be empty")
    if any(
        not isinstance(quota, PilotQuota)
        or isinstance(quota.rows, bool)
        or not isinstance(quota.rows, int)
        or quota.rows < 1
        for quota in quotas
    ):
        raise PilotError(f"{arm} quota rows must be positive integers")
    if len({quota.split for quota in quotas}) != len(quotas):
        raise PilotError(f"{arm} cannot declare a split more than once")


def _known_splits(config: PilotConfig) -> set[str]:
    return {quota.split for quota in config.historical_quotas + config.balanced_quotas}


def _validate_held_out(values: Iterable[str]) -> set[str]:
    try:
        result = set(values)
    except TypeError as error:
        raise PilotError("held-out UUIDs must be iterable strings") from error
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in result
    ):
        raise PilotError("held-out UUIDs must be SHA-256 strings")
    return result


def _normalize_row(split: str, raw_row: Mapping[str, object]) -> PilotRow:
    if not isinstance(raw_row, Mapping):
        raise PilotError("PTV2 row must be a mapping")
    if "tools" in raw_row or "tool_calls" in raw_row:
        raise PilotError("tool/tool-call rows are excluded from the raw pilot")
    messages = _normalize_messages(raw_row.get("messages"))
    messages_json = canonical_json(messages)
    return PilotRow(
        prompt_uuid=prompt_uuid_from_messages(messages),
        split=split,
        category="multilingual" if split.startswith("multilingual_") else split,
        messages_json=messages_json,
    )


def _try_normalize_row(split: str, raw_row: Mapping[str, object]) -> PilotRow | None:
    try:
        return _normalize_row(split, raw_row)
    except PilotError:
        return None


def _normalize_messages(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise PilotError("row messages must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    for message in value:
        if not isinstance(message, Mapping):
            raise PilotError("each message must be a mapping")
        role = message.get("role")
        if not isinstance(role, str) or role not in {"system", "user", "assistant"}:
            raise PilotError("tool or malformed message role is excluded")
        if "tool_calls" in message or "tool_call_id" in message:
            raise PilotError("tool/tool-call rows are excluded from the raw pilot")
        content = message.get("content")
        if not isinstance(content, str):
            raise PilotError("message content must be a string")
        normalized.append({"role": role, "content": content})
    if normalized[-1]["role"] != "assistant" or not normalized[-1]["content"]:
        raise PilotError("row must have a non-empty terminal assistant response")
    return normalized


def _deduplicate_candidates(rows: list[PilotRow]) -> tuple[list[PilotRow], int]:
    by_identity: dict[str, PilotRow] = {}
    for row in rows:
        previous = by_identity.get(row.prompt_uuid)
        if previous is None or (row.split, row.messages_json) < (
            previous.split,
            previous.messages_json,
        ):
            by_identity[row.prompt_uuid] = row
    return list(by_identity.values()), len(rows) - len(by_identity)


def _rank(row: PilotRow, seed: int) -> str:
    return hashlib.sha256(
        canonical_json([seed, row.split, row.prompt_uuid]).encode("utf-8")
    ).hexdigest()


def _prepare_selection_database(connection: sqlite3.Connection) -> None:
    connection.execute(
        "CREATE TABLE candidates (prompt_uuid TEXT NOT NULL, split TEXT NOT NULL, "
        "category TEXT NOT NULL, messages_json TEXT NOT NULL, rank TEXT NOT NULL)"
    )
    connection.execute("CREATE INDEX candidates_split_rank ON candidates(split, rank, prompt_uuid)")


def _spool_candidates(
    connection: sqlite3.Connection,
    split_rows: Mapping[str, Iterable[Mapping[str, object]]],
    config: PilotConfig,
    held_out: set[str],
) -> tuple[int, int]:
    invalid = 0
    held_out_count = 0
    for split, rows in split_rows.items():
        if not isinstance(split, str) or split not in _known_splits(config):
            raise PilotError(f"unapproved PTV2 split: {split!r}")
        try:
            iterator = iter(rows)
        except TypeError as error:
            raise PilotError(f"split {split!r} is not iterable") from error
        batch: list[tuple[str, str, str, str, str]] = []
        for raw_row in iterator:
            normalized = _try_normalize_row(split, raw_row)
            if normalized is None:
                invalid += 1
                continue
            if normalized.prompt_uuid in held_out:
                held_out_count += 1
                continue
            batch.append(
                (
                    normalized.prompt_uuid,
                    normalized.split,
                    normalized.category,
                    normalized.messages_json,
                    _rank(normalized, config.seed),
                )
            )
            if len(batch) == 1_024:
                connection.executemany("INSERT INTO candidates VALUES (?, ?, ?, ?, ?)", batch)
                batch.clear()
        if batch:
            connection.executemany("INSERT INTO candidates VALUES (?, ?, ?, ?, ?)", batch)
    connection.commit()
    return invalid, held_out_count


def _duplicate_count(connection: sqlite3.Connection) -> int:
    total = int(connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])
    unique = int(
        connection.execute("SELECT COUNT(DISTINCT prompt_uuid) FROM candidates").fetchone()[0]
    )
    return total - unique


def _select_arm_from_spool(
    connection: sqlite3.Connection,
    arm: str,
    quotas: tuple[PilotQuota, ...],
    config: PilotConfig,
) -> tuple[PilotRow, ...]:
    selected: list[PilotRow] = []
    selected_uuids: set[str] = set()
    for quota in quotas:
        cursor = connection.execute(
            "SELECT prompt_uuid, split, category, messages_json FROM candidates "
            "WHERE split = ? ORDER BY rank, prompt_uuid",
            (quota.split,),
        )
        filled = 0
        for prompt_uuid, split, category, messages_json in cursor:
            if prompt_uuid in selected_uuids:
                continue
            selected.append(PilotRow(prompt_uuid, split, category, messages_json))
            selected_uuids.add(prompt_uuid)
            filled += 1
            if filled == quota.rows:
                break
        if filled != quota.rows:
            raise PilotError(
                f"{arm}/{quota.category}/{quota.split} has {filled} unique eligible rows, "
                f"below exact quota {quota.rows}"
            )
    return tuple(sorted(selected, key=lambda row: (_rank(row, config.seed), row.prompt_uuid)))


def _select_arm(
    arm: str,
    candidates: list[PilotRow],
    quotas: tuple[PilotQuota, ...],
    config: PilotConfig,
) -> tuple[PilotRow, ...]:
    selected: list[PilotRow] = []
    selected_uuids: set[str] = set()
    for quota in quotas:
        available = sorted(
            (row for row in candidates if row.split == quota.split),
            key=lambda row: (_rank(row, config.seed), row.prompt_uuid),
        )
        filled = 0
        for row in available:
            if row.prompt_uuid in selected_uuids:
                continue
            selected.append(row)
            selected_uuids.add(row.prompt_uuid)
            filled += 1
            if filled == quota.rows:
                break
        if filled != quota.rows:
            raise PilotError(
                f"{arm}/{quota.category}/{quota.split} has {filled} unique eligible rows, "
                f"below exact quota {quota.rows}"
            )
    return tuple(sorted(selected, key=lambda row: (_rank(row, config.seed), row.prompt_uuid)))


def count_assistant_tokens(tokenizer: Any, messages: list[dict[str, Any]]) -> int:
    """Count only the token positions marked trainable by Qwen's chat template."""
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
    except Exception as error:
        raise PilotError("token worker failed while applying the chat template") from error
    if not isinstance(encoded, Mapping):
        raise PilotError("tokenizer did not return a token mapping")
    input_ids = encoded.get("input_ids")
    mask = encoded.get("assistant_masks", encoded.get("assistant_tokens_mask"))
    if (
        not isinstance(input_ids, list)
        or not isinstance(mask, list)
        or len(input_ids) != len(mask)
        or any(value not in (0, 1) for value in mask)
    ):
        raise PilotError("tokenizer did not return an aligned exact assistant mask")
    assistant_tokens = sum(mask)
    if assistant_tokens < 1:
        raise PilotError("row has no trainable assistant tokens")
    return assistant_tokens


def build_pilot_bundles(
    split_rows: Mapping[str, Iterable[Mapping[str, object]]],
    *,
    config: PilotConfig | None = None,
    held_out_prompt_uuids: Iterable[str] = (),
    tokenizer: Any,
    tokenizer_path: str | Path,
    tokenizer_sha256: str,
    output_root: Path,
    producer_source_commit: str,
    workers: int | None = None,
    scratch_root: Path | None = None,
    execution: Mapping[str, object] | None = None,
) -> PilotCompletion:
    """Publish only the frozen 100K-row, 16M-token Qwen3-4B pilot identity."""
    config = config or PilotConfig()
    _validate_production_identity(config)
    _verify_tokenizer_artifact(Path(tokenizer_path), tokenizer_sha256)
    tokenizer = _load_verified_qwen_tokenizer(Path(tokenizer_path))
    return _build_pilot_bundles_for_test(
        split_rows,
        config=config,
        held_out_prompt_uuids=held_out_prompt_uuids,
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        tokenizer_sha256=tokenizer_sha256,
        output_root=output_root,
        producer_source_commit=producer_source_commit,
        workers=workers,
        scratch_root=scratch_root,
        execution=execution,
    )


def _build_pilot_bundles_for_test(
    split_rows: Mapping[str, Iterable[Mapping[str, object]]],
    *,
    config: PilotConfig | None = None,
    held_out_prompt_uuids: Iterable[str] = (),
    tokenizer: Any,
    tokenizer_path: str | Path,
    tokenizer_sha256: str,
    output_root: Path,
    producer_source_commit: str,
    workers: int | None = None,
    scratch_root: Path | None = None,
    execution: Mapping[str, object] | None = None,
) -> PilotCompletion:
    """Test-only generic publication helper; production callers use ``build_pilot_bundles``."""
    config = config or PilotConfig()
    output_root = Path(output_root)
    _validate_build_inputs(
        config=config,
        tokenizer_path=tokenizer_path,
        tokenizer_sha256=tokenizer_sha256,
        producer_source_commit=producer_source_commit,
        output_root=output_root,
        workers=workers,
    )
    if output_root.exists():
        raise PilotError(f"publication destination already exists: {output_root}")
    scratch_parent = Path(scratch_root) if scratch_root is not None else output_root.parent
    if not scratch_parent.is_dir():
        raise PilotError(f"scratch parent is not a directory: {scratch_parent}")
    selected = select_pilot_rows(
        split_rows,
        config=config,
        held_out_prompt_uuids=held_out_prompt_uuids,
        scratch_root=scratch_parent,
    )
    effective_workers = workers if workers is not None else config.workers
    tokens_by_arm = {
        arm: _count_selected_rows(tokenizer, selected.rows_for(arm), effective_workers)
        for arm in (HISTORICAL_PROPORTION, BALANCED)
    }
    template_sha256 = hashlib.sha256(
        str(getattr(tokenizer, "chat_template", "")).encode("utf-8")
    ).hexdigest()
    for arm, counts in tokens_by_arm.items():
        if sum(counts) < config.minimum_assistant_tokens:
            raise PilotError(
                f"{arm} assistant-token minimum is {config.minimum_assistant_tokens}, "
                f"but selected rows contain {sum(counts)}"
            )

    partial = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.scratch-", dir=scratch_parent))
    os.chmod(partial, 0o700)
    local_partial: Path | None = None
    try:
        arms = _write_bundle_contents(
            partial,
            selected=selected,
            tokens_by_arm=tokens_by_arm,
            config=config,
            tokenizer_path=tokenizer_path,
            tokenizer_sha256=tokenizer_sha256,
            tokenizer_template_sha256=template_sha256,
            producer_source_commit=producer_source_commit,
            effective_workers=effective_workers,
            execution=execution,
        )
        _verify_bundle_root(partial)
        local_partial = output_root.parent / f".{output_root.name}.publish-{uuid.uuid4().hex}"
        shutil.copytree(partial, local_partial)
        os.chmod(local_partial, 0o700)
        _verify_bundle_root(local_partial)
        if output_root.exists():
            raise PilotError(f"publication destination already exists: {output_root}")
        try:
            _rename_noreplace(local_partial, output_root)
        except OSError as error:
            if error.errno == errno.EEXIST:
                raise PilotError(
                    f"publication destination already exists: {output_root}"
                ) from error
            raise PilotError("atomic pilot publication failed") from error
        local_partial = None
        _fsync_directory(output_root.parent)
        _verify_bundle_root(output_root)
        complete = _read_json(output_root / "COMPLETE.json")
        return PilotCompletion(
            output_root=output_root,
            historical_proportion=arms[HISTORICAL_PROPORTION],
            balanced=arms[BALANCED],
            complete_sha256=str(complete["complete_sha256"]),
        )
    finally:
        if partial.exists():
            shutil.rmtree(partial)
        if local_partial is not None and local_partial.exists():
            shutil.rmtree(local_partial)


def verify_pilot_completion(output_root: Path) -> PilotCompletion:
    """Replay every file descriptor and receipt in an installed pilot root."""
    output_root = Path(output_root)
    arms = _verify_bundle_root(output_root)
    complete = _read_json(output_root / "COMPLETE.json")
    return PilotCompletion(
        output_root=output_root,
        historical_proportion=arms[HISTORICAL_PROPORTION],
        balanced=arms[BALANCED],
        complete_sha256=str(complete["complete_sha256"]),
    )


def _validate_build_inputs(
    *,
    config: PilotConfig,
    tokenizer_path: str | Path,
    tokenizer_sha256: str,
    producer_source_commit: str,
    output_root: Path,
    workers: int | None,
) -> None:
    if not str(tokenizer_path):
        raise PilotError("tokenizer path is required")
    if len(config.source_revision) not in (40, 64) or any(
        char not in "0123456789abcdef" for char in config.source_revision
    ):
        raise PilotError("source_revision must be an immutable hexadecimal commit")
    if len(tokenizer_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in tokenizer_sha256
    ):
        raise PilotError("tokenizer_sha256 must be an exact lowercase SHA-256")
    if len(producer_source_commit) not in (40, 64) or any(
        char not in "0123456789abcdef" for char in producer_source_commit
    ):
        raise PilotError("producer_source_commit must be an immutable hexadecimal commit")
    if output_root.parent == output_root or not output_root.parent.is_dir():
        raise PilotError("output root must have an existing parent directory")
    if workers is not None and (
        isinstance(workers, bool) or not isinstance(workers, int) or workers < 1
    ):
        raise PilotError("workers must be a positive integer")


def _validate_production_identity(config: PilotConfig) -> None:
    approved = PilotConfig()
    if (
        config.seed != SEED
        or config.minimum_assistant_tokens < 16_000_000
        or config.source_repository != approved.source_repository
        or config.historical_quotas != approved.historical_quotas
        or config.balanced_quotas != approved.balanced_quotas
        or len(config.source_revision) not in (40, 64)
        or any(char not in "0123456789abcdef" for char in config.source_revision)
    ):
        raise PilotError("public builder requires the frozen production identity")
    if not sys.platform.startswith("linux"):
        raise PilotError("public builder requires Linux atomic no-replace publication")


def _verify_tokenizer_artifact(path: Path, expected_sha256: str) -> None:
    if not path.exists() or path.is_symlink():
        raise PilotError("tokenizer path must be an existing non-symlink artifact")
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
    elif path.is_dir():
        for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
            digest.update(child.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(child.read_bytes())
    else:
        raise PilotError("tokenizer path must be a file or directory")
    if digest.hexdigest() != expected_sha256:
        raise PilotError("tokenizer artifact does not match the pinned SHA-256")


def _load_verified_qwen_tokenizer(path: Path) -> Any:
    try:
        from transformers import AutoTokenizer  # pyright: ignore[reportMissingImports]

        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    except Exception as error:
        raise PilotError("unable to load the pinned Qwen3-4B tokenizer") from error
    if not isinstance(getattr(tokenizer, "chat_template", None), str):
        raise PilotError("pinned Qwen3-4B tokenizer has no chat template")
    return tokenizer


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Install a directory without replacement; Linux gets the kernel primitive."""
    if sys.platform.startswith("linux"):
        import ctypes

        renameat2 = ctypes.CDLL(None, use_errno=True).syscall
        result = renameat2(316, -100, os.fsencode(source), -100, os.fsencode(destination), 1)
        if result != 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number), destination)
        return
    if destination.exists():
        raise OSError(errno.EEXIST, "destination exists", destination)
    os.rename(source, destination)


def _count_selected_rows(
    tokenizer: Any, rows: tuple[PilotRow, ...], workers: int
) -> tuple[int, ...]:
    def count(row: PilotRow) -> int:
        return count_assistant_tokens(tokenizer, row.messages)

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            limit = max(1, workers * 2)
            pending: dict[Future[int], int] = {}
            results = [0] * len(rows)
            next_index = 0
            while next_index < len(rows) or pending:
                while next_index < len(rows) and len(pending) < limit:
                    future = executor.submit(count, rows[next_index])
                    pending[future] = next_index
                    next_index += 1
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    index = pending.pop(future)
                    results[index] = future.result()
            return tuple(results)
    except PilotError:
        raise
    except Exception as error:
        raise PilotError("token worker failed") from error


def _write_bundle_contents(
    root: Path,
    *,
    selected: PilotSelection,
    tokens_by_arm: Mapping[str, tuple[int, ...]],
    config: PilotConfig,
    tokenizer_path: str | Path,
    tokenizer_sha256: str,
    tokenizer_template_sha256: str,
    producer_source_commit: str,
    effective_workers: int,
    execution: Mapping[str, object] | None,
) -> dict[str, PilotArmCompletion]:
    arms: dict[str, PilotArmCompletion] = {}
    for arm in (HISTORICAL_PROPORTION, BALANCED):
        arm_root = root / arm
        arm_root.mkdir(mode=0o700)
        rows = selected.rows_for(arm)
        data_path = arm_root / "data.jsonl"
        with data_path.open("w", encoding="utf-8", newline="\n") as destination:
            for row in rows:
                destination.write(
                    canonical_json(
                        {
                            "prompt_uuid": row.prompt_uuid,
                            "split": row.split,
                            "category": row.category,
                            "messages": row.messages,
                        }
                    )
                    + "\n"
                )
            destination.flush()
            os.fsync(destination.fileno())
        execution_payload = {
            **(dict(execution) if execution is not None else {}),
            "requested_workers": config.workers,
            "effective_workers": effective_workers,
        }
        _write_json(arm_root / "EXECUTION.json", execution_payload)
        manifest = {
            "schema_version": "qwen3-4b-balanced-pilot-v1",
            "arm": arm,
            "selection_mode": "pilot-row-quota-v1",
            "source": {
                "repository": config.source_repository,
                "revision": config.source_revision,
            },
            "seed": config.seed,
            "quotas": _quota_counts(config, arm),
            "row_count": len(rows),
            "category_counts": _category_counts(rows),
            "prompt_uuid_sha256": _prompt_digest(rows),
            "data_file": {
                "path": "data.jsonl",
                "bytes": data_path.stat().st_size,
                "sha256": _sha256_file(data_path),
            },
            "tokenizer": {
                "path": str(tokenizer_path),
                "sha256": tokenizer_sha256,
                "chat_template_sha256": tokenizer_template_sha256,
            },
            "assistant_tokens": sum(tokens_by_arm[arm]),
            "minimum_assistant_tokens": config.minimum_assistant_tokens,
            "exclusions": {
                "held_out": selected.excluded_held_out,
                "invalid": selected.excluded_invalid,
                "duplicate_candidates": selected.excluded_duplicate_candidates,
            },
            "producer_source_commit": producer_source_commit,
            "execution_sha256": _sha256_file(arm_root / "EXECUTION.json"),
        }
        manifest["manifest_sha256"] = _self_hash(manifest, "manifest_sha256")
        _write_json(arm_root / "MANIFEST.json", manifest)
        arms[arm] = PilotArmCompletion(
            arm=arm,
            row_count=len(rows),
            assistant_tokens=sum(tokens_by_arm[arm]),
            manifest_sha256=str(manifest["manifest_sha256"]),
        )
    complete = {
        "schema_version": "qwen3-4b-balanced-pilot-v1",
        "arms": {
            arm: {
                "manifest_path": f"{arm}/MANIFEST.json",
                "manifest_sha256": arms[arm].manifest_sha256,
            }
            for arm in (HISTORICAL_PROPORTION, BALANCED)
        },
    }
    complete["complete_sha256"] = _self_hash(complete, "complete_sha256")
    _write_json(root / "COMPLETE.json", complete)
    return arms


def _verify_bundle_root(root: Path) -> dict[str, PilotArmCompletion]:
    complete = _read_json(root / "COMPLETE.json")
    if complete.get("complete_sha256") != _self_hash(complete, "complete_sha256"):
        raise PilotError("aggregate completion self-hash does not reconcile")
    arms: dict[str, PilotArmCompletion] = {}
    descriptors = complete.get("arms")
    if not isinstance(descriptors, Mapping) or set(descriptors) != {
        HISTORICAL_PROPORTION,
        BALANCED,
    }:
        raise PilotError("aggregate completion arm descriptors are invalid")
    for arm, descriptor in descriptors.items():
        if (
            not isinstance(descriptor, Mapping)
            or descriptor.get("manifest_path") != f"{arm}/MANIFEST.json"
        ):
            raise PilotError("aggregate completion manifest descriptor is invalid")
        manifest_path = root / str(descriptor["manifest_path"])
        manifest = _read_json(manifest_path)
        if manifest.get("manifest_sha256") != _self_hash(manifest, "manifest_sha256"):
            raise PilotError(f"{arm} manifest self-hash does not reconcile")
        if descriptor.get("manifest_sha256") != manifest.get("manifest_sha256"):
            raise PilotError(f"{arm} manifest descriptor does not reconcile")
        data = manifest.get("data_file")
        if not isinstance(data, Mapping) or data.get("path") != "data.jsonl":
            raise PilotError(f"{arm} data file descriptor is invalid")
        data_path = manifest_path.parent / "data.jsonl"
        if data.get("bytes") != data_path.stat().st_size or data.get("sha256") != _sha256_file(
            data_path
        ):
            raise PilotError(f"{arm} data file descriptor does not reconcile")
        execution_path = manifest_path.parent / "EXECUTION.json"
        if manifest.get("execution_sha256") != _sha256_file(execution_path):
            raise PilotError(f"{arm} execution receipt does not reconcile")
        rows = _read_data_rows(data_path)
        if manifest.get("row_count") != len(rows):
            raise PilotError(f"{arm} row count does not reconcile")
        quotas = manifest.get("quotas")
        split_counts: dict[str, int] = {}
        for row in rows:
            split_counts[row.split] = split_counts.get(row.split, 0) + 1
        if not isinstance(quotas, Mapping) or dict(sorted(quotas.items())) != dict(
            sorted(split_counts.items())
        ):
            raise PilotError(f"{arm} exact split quotas do not reconcile")
        if len({row.prompt_uuid for row in rows}) != len(rows):
            raise PilotError(f"{arm} global prompt UUID uniqueness does not reconcile")
        if manifest.get("category_counts") != _category_counts(rows):
            raise PilotError(f"{arm} category counts do not reconcile")
        if manifest.get("prompt_uuid_sha256") != _prompt_digest(rows):
            raise PilotError(f"{arm} prompt UUID digest does not reconcile")
        source = manifest.get("source")
        if (
            not isinstance(source, Mapping)
            or source.get("repository") != "nvidia/Nemotron-Post-Training-Dataset-v2"
            or not isinstance(source.get("revision"), str)
            or len(source["revision"]) not in (40, 64)
            or any(character not in "0123456789abcdef" for character in source["revision"])
        ):
            raise PilotError(f"{arm} pinned source identity does not reconcile")
        tokenizer = manifest.get("tokenizer")
        if (
            not isinstance(tokenizer, Mapping)
            or not isinstance(tokenizer.get("sha256"), str)
            or not isinstance(tokenizer.get("chat_template_sha256"), str)
            or any(
                len(str(tokenizer[field])) != 64
                or any(character not in "0123456789abcdef" for character in str(tokenizer[field]))
                for field in ("sha256", "chat_template_sha256")
            )
        ):
            raise PilotError(f"{arm} tokenizer/template identity does not reconcile")
        if manifest.get("assistant_tokens", 0) < manifest.get("minimum_assistant_tokens", 1):
            raise PilotError(f"{arm} assistant-token minimum does not reconcile")
        arms[arm] = PilotArmCompletion(
            arm=arm,
            row_count=len(rows),
            assistant_tokens=int(manifest["assistant_tokens"]),
            manifest_sha256=str(manifest["manifest_sha256"]),
        )
    return arms


def _read_data_rows(path: Path) -> tuple[PilotRow, ...]:
    rows: list[PilotRow] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            try:
                payload = json.loads(line)
                row = _normalize_row(str(payload["split"]), payload)
            except (KeyError, TypeError, json.JSONDecodeError, PilotError) as error:
                raise PilotError("data JSONL row is malformed") from error
            if row.category != payload.get("category") or row.prompt_uuid != payload.get(
                "prompt_uuid"
            ):
                raise PilotError("data JSONL identity does not reconcile")
            rows.append(row)
    return tuple(rows)


def _quota_counts(config: PilotConfig, arm: str) -> dict[str, int]:
    quotas = config.historical_quotas if arm == HISTORICAL_PROPORTION else config.balanced_quotas
    return {quota.split: quota.rows for quota in quotas}


def _category_counts(rows: Iterable[PilotRow]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for row in rows:
        totals[row.category] = totals.get(row.category, 0) + 1
    return dict(sorted(totals.items()))


def _prompt_digest(rows: Iterable[PilotRow]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(row.prompt_uuid for row in rows)).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        destination.write(canonical_json(payload) + "\n")
        destination.flush()
        os.fsync(destination.fileno())


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PilotError(f"unable to read receipt: {path}") from error
    if not isinstance(payload, dict):
        raise PilotError(f"receipt must be a JSON object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _self_hash(payload: Mapping[str, object], field: str) -> str:
    stripped = dict(payload)
    stripped.pop(field, None)
    return hashlib.sha256(canonical_json(stripped).encode("utf-8")).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
