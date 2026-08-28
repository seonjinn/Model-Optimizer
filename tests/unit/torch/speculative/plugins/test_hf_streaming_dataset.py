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

"""Tests for the map-style StreamingDataset.

The dataset is a plain ``torch.utils.data.Dataset``: DDP sharding is HF Trainer's
job (``DistributedSampler``), so there is no rank/dispatch logic to test here.
These tests cover the ``__getitem__`` contract: resample-on-miss, the
consecutive-failure circuit breaker, and the vLLM RDMA wire-format -> batch-dict
chain. Hidden states move over NIXL RDMA; the end-to-end tests inject a fake
``nixl`` agent (the library is not a test dependency) and route the sidecar HTTP
through ``httpx.MockTransport`` -- the RDMA transfer itself is a no-op, so they
exercise the orchestration + format chain, not real byte movement.
"""

import base64
import itertools
import json
import sys
import types
from pathlib import Path
from typing import Any, cast, get_type_hints
from unittest.mock import MagicMock

import httpx
import pytest
import torch

# hf_streaming_dataset imports LabelSmoother at module scope.
pytest.importorskip("transformers")

from modelopt.torch.speculative.plugins import hf_streaming_dataset
from modelopt.torch.speculative.plugins.hf_streaming_dataset import (
    EagleFormattedSample,
    EagleVllmStreamingConfig,
    EagleVllmStreamingDataset,
    StreamingConfig,
    StreamingDataset,
    normalize_streaming_entry,
    resolve_streaming_data_source,
)


def test_eagle_formatted_sample_has_precise_per_key_types() -> None:
    """Trainer tensor fields must not be widened by the boolean metadata field."""
    hints = get_type_hints(EagleFormattedSample)

    assert hints == {
        "input_ids": torch.Tensor,
        "base_model_hidden_states": torch.Tensor,
        "aux_hidden_states": torch.Tensor,
        "attention_mask": torch.Tensor,
        "loss_mask": torch.Tensor,
        "labels": torch.Tensor,
        "base_hidden_prenorm": bool,
    }


def _entries(n: int) -> list[dict]:
    """Minimal entry shape; ``id`` is the only field tests read back."""
    return [{"id": i} for i in range(n)]


def test_resolve_streaming_parquet_directory_without_conversion(tmp_path):
    (tmp_path / "part-01.parquet").touch()
    (tmp_path / "part-00.parquet").touch()

    dataset_format, data_files = resolve_streaming_data_source(tmp_path)

    assert dataset_format == "parquet"
    assert data_files == [
        str(tmp_path / "part-00.parquet"),
        str(tmp_path / "part-01.parquet"),
    ]


def test_normalize_openperfectblend_entry_has_stable_id_and_roles():
    entry = {
        "conversations": [
            {"from": "human", "value": "Question"},
            {"from": "gpt", "value": "Answer"},
        ]
    }

    first = normalize_streaming_entry(entry)
    second = normalize_streaming_entry(entry)

    assert first == second
    assert first is not None
    cid, conversations, tools = first
    assert len(cid) == 64
    assert tools is None
    assert conversations == [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
    ]


def test_tool_bearing_entry_preserves_exact_template_inputs_and_token_ids():
    """PTV3 tool declarations and structured tool calls reach the actual tokenizer unchanged."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read one repository file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ]
    messages = [
        {"role": "user", "content": "Inspect the failing test."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"tests/test_bug.py"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "name": "read_file",
            "content": "assert actual == expected",
        },
        {"role": "assistant", "content": "The implementation drops the expected value."},
    ]
    tokenizer = MagicMock()

    def apply_chat_template(actual_messages, **kwargs):
        assert actual_messages == messages
        assert kwargs["tools"] == tools
        return {"input_ids": torch.tensor([[101, 202, 303]])}

    tokenizer.apply_chat_template.side_effect = apply_chat_template
    dataset = StreamingDataset(
        [{"conversation_id": "swe-tool-1", "messages": messages, "tools": tools}],
        tokenizer=tokenizer,
        config=StreamingConfig(answer_only_loss=False),
    )

    sample = dataset._tokenize_entry(dataset.entries[0])

    assert sample is not None
    assert sample["token_ids"] == [101, 202, 303]
    tokenizer.apply_chat_template.assert_called_once()


def test_tool_declarations_are_part_of_derived_conversation_identity():
    """Changing only the tool schema changes the stable identity of an otherwise equal prompt."""
    base = {
        "messages": [{"role": "user", "content": "Use the repository tool."}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    }
    changed = {
        **base,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "open_file",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    }

    normalized_base = normalize_streaming_entry(base)
    normalized_changed = normalize_streaming_entry(changed)

    assert normalized_base is not None and normalized_changed is not None
    base_id, base_messages, base_tools = normalized_base
    changed_id, changed_messages, changed_tools = normalized_changed
    assert base_messages == changed_messages
    assert base_tools == base["tools"]
    assert changed_tools == changed["tools"]
    assert base_id != changed_id


def test_pretokenized_entry_preserves_exact_loss_mask_without_retokenizing():
    tokenizer = MagicMock()
    ds = StreamingDataset(
        [{"primary_id": "swe-7", "input_ids": [10, 11, 12, 13], "loss_mask": [0, 0, 1, 1]}],
        tokenizer=tokenizer,
        config=StreamingConfig(answer_only_loss=True, max_seq_len=3),
    )

    sample = ds._tokenize_entry(ds.entries[0])

    assert sample is not None
    assert sample["cid"] == "swe-7"
    assert sample["token_ids"] == [10, 11, 12]
    assert sample["loss_mask"].tolist() == [0, 0, 1]
    tokenizer.apply_chat_template.assert_not_called()


@pytest.mark.parametrize(
    "entry",
    [
        {"input_ids": [1, 2], "loss_mask": [1]},
        {"input_ids": [1, 2]},
        {"loss_mask": [0, 1]},
        {"input_ids": [1, 2], "loss_mask": [0, 2]},
    ],
)
def test_invalid_pretokenized_entry_fails_loudly(entry):
    ds = StreamingDataset([entry], tokenizer=MagicMock(), config=StreamingConfig())

    with pytest.raises(ValueError, match="pretokenized"):
        ds._tokenize_entry(entry)


def test_pretokenized_entry_with_truncated_zero_supervision_is_skipped():
    entry = {"input_ids": [1, 2, 3], "loss_mask": [0, 0, 1]}
    ds = StreamingDataset([entry], tokenizer=MagicMock(), config=StreamingConfig(max_seq_len=2))

    assert ds._tokenize_entry(entry) is None


def test_empty_corpus_raises():
    with pytest.raises(ValueError, match="entries is empty"):
        StreamingDataset([], tokenizer=MagicMock(), config=StreamingConfig())


def test_len_matches_corpus():
    ds = StreamingDataset(_entries(37), tokenizer=MagicMock(), config=StreamingConfig())
    assert len(ds) == 37


def test_map_corpus_is_not_materialized_per_rank():
    class _MapCorpus:
        def __len__(self):
            return 700_000

        def __getitem__(self, index):
            return {"id": index}

        def __iter__(self):
            raise AssertionError("authenticated map corpus must not be materialized")

    corpus = _MapCorpus()
    ds = StreamingDataset(corpus, tokenizer=MagicMock(), config=StreamingConfig())

    assert ds.entries is corpus
    assert len(ds) == 700_000


def test_getitem_resamples_past_unfit_entries():
    """An unfit entry (tokenize -> None) must not be returned; __getitem__ probes
    forward to the next fetchable index and returns that instead."""
    fetched_cids: list[int] = []

    class _Track(StreamingDataset):
        def _tokenize_entry(self, entry):
            # Even ids are "unfit" (e.g. truncated away / missing fields).
            if entry["id"] % 2 == 0:
                return None
            return {"cid": str(entry["id"]), "token_ids": [1], "loss_mask": None}

        def _fetch(self, sample):
            fetched_cids.append(int(sample["cid"]))
            return {"ok": True}

        def _format(self, fetched):
            return {"sentinel": fetched_cids[-1]}

    ds = _Track(_entries(10), tokenizer=MagicMock(), config=StreamingConfig())
    # idx 0 is unfit -> resamples forward to idx 1.
    out = ds[0]
    assert out == {"sentinel": 1}
    assert fetched_cids == [1]
    # An already-fit index is returned directly.
    assert ds[3] == {"sentinel": 3}


def test_q30_strict_exposure_retries_only_same_occurrence_and_records_uuid(tmp_path, monkeypatch):
    """Strict Q30 mode cannot probe/cycle/substitute a different dataset occurrence."""
    fetched_cids: list[str] = []

    class StrictDataset(StreamingDataset):
        def _tokenize_entry(self, entry):
            return {
                "cid": entry["prompt_uuid"],
                "prompt_uuid": entry["prompt_uuid"],
                "token_ids": [1],
                "loss_mask": None,
            }

        def _fetch(self, sample):
            fetched_cids.append(sample["cid"])
            if len(fetched_cids) == 1:
                raise httpx.ConnectError("retry same occurrence")
            return sample

        def _format(self, fetched):
            return {"sentinel": fetched["cid"]}

    evidence = tmp_path / "exposure"
    monkeypatch.setenv("Q30_STRICT_EXPOSURE", "1")
    monkeypatch.setenv("Q30_EXPOSURE_EVIDENCE_DIR", str(evidence))
    monkeypatch.setenv("RANK", "3")
    dataset = StrictDataset(
        [{"prompt_uuid": "a" * 64}, {"prompt_uuid": "b" * 64}],
        tokenizer=MagicMock(),
        config=StreamingConfig(fail_after_consecutive_skips=2),
    )

    sample = dataset[0]
    collator = hf_streaming_dataset.Q30StrictExposureCollator(lambda features: dict(features[0]))
    batch = collator([sample])
    hf_streaming_dataset.record_q30_consumed_exposure(batch)

    assert batch == {"sentinel": "a" * 64}
    assert fetched_cids == ["a" * 64, "a" * 64]
    records = [
        json.loads(line) for line in (evidence / "rank-00003.jsonl").read_text().splitlines()
    ]
    assert records == [{"dataset_index": 0, "prompt_uuid": "a" * 64}]


def test_q30_strict_exposure_records_consumed_not_prefetched_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Accelerate's one-batch lookahead may fetch, but must not evidence, batch 81."""
    accelerate = pytest.importorskip("accelerate")
    assert accelerate.__version__ == "1.14.0"
    from accelerate.data_loader import DataLoaderShard

    fetched_indices: list[int] = []

    class StrictDataset(StreamingDataset):
        def _tokenize_entry(self, entry):
            return {
                "cid": entry["prompt_uuid"],
                "prompt_uuid": entry["prompt_uuid"],
                "token_ids": [entry["dataset_index"]],
                "loss_mask": None,
            }

        def _fetch(self, sample):
            index = sample["token_ids"][0]
            fetched_indices.append(index)
            return {"index": index}

        def _format(self, fetched):
            return {"value": torch.tensor(fetched["index"])}

    evidence_root = tmp_path / "exposure"
    monkeypatch.setenv("Q30_STRICT_EXPOSURE", "1")
    monkeypatch.setenv("Q30_EXPOSURE_EVIDENCE_DIR", str(evidence_root))
    monkeypatch.setenv("RANK", "0")
    entries = [{"dataset_index": index, "prompt_uuid": f"{index:064x}"} for index in range(400)]
    dataset = StrictDataset(entries, tokenizer=MagicMock(), config=StreamingConfig())
    loader = DataLoaderShard(dataset, batch_size=4, shuffle=False)
    record_consumed = getattr(hf_streaming_dataset, "record_q30_consumed_exposure", None)

    for batch in itertools.islice(loader, 80):
        if callable(record_consumed):
            record_consumed(batch)

    assert fetched_indices == list(range(324))
    records = [
        json.loads(line) for line in (evidence_root / "rank-00000.jsonl").read_text().splitlines()
    ]
    assert records == [
        {"dataset_index": index, "prompt_uuid": f"{index:064x}"} for index in range(320)
    ]


def test_q30_exact_batch_sampler_gives_every_rank_three_final_examples_without_padding() -> None:
    """Pinned Accelerate must consume 700K once, ending with local batch three on all ranks."""
    accelerate = pytest.importorskip("accelerate")
    assert accelerate.__version__ == "1.14.0"
    from accelerate.data_loader import BatchSamplerShard, SeedableRandomSampler

    sampler_type = cast("Any", getattr(hf_streaming_dataset, "Q30ExactBatchSampler", None))
    assert callable(sampler_type), "exact final-global-batch sampler is missing"
    indices_by_rank: list[list[int]] = []
    final_batch_sizes: list[int] = []
    microsteps_by_rank: list[int] = []
    for rank in range(32):
        sampler = SeedableRandomSampler(range(700_000), data_seed=42)
        batches = sampler_type(
            sampler,
            local_batch_size=4,
            trainer_ranks=32,
            exact_exposure_count=700_000,
        )
        sharded = BatchSamplerShard(
            batches,
            num_processes=32,
            process_index=rank,
            split_batches=False,
            even_batches=True,
        )
        rank_batches = cast("list[list[int]]", list(sharded))
        indices_by_rank.append(list(itertools.chain.from_iterable(rank_batches)))
        final_batch_sizes.append(len(rank_batches[-1]))
        microsteps_by_rank.append(len(rank_batches))

    flattened = list(itertools.chain.from_iterable(indices_by_rank))
    assert microsteps_by_rank == [5_469] * 32
    assert final_batch_sizes == [3] * 32
    assert len(flattened) == 700_000
    assert len(set(flattened)) == 700_000
    assert set(flattened) == set(range(700_000))


def test_q30_strict_exposure_rejects_unfit_occurrence_without_forward_probe(tmp_path, monkeypatch):
    """An unusable selected occurrence fails instead of consuming the next UUID."""

    class StrictDataset(StreamingDataset):
        def _tokenize_entry(self, entry):
            return None if entry["id"] == 0 else {"cid": "next", "token_ids": [1]}

    monkeypatch.setenv("Q30_STRICT_EXPOSURE", "1")
    monkeypatch.setenv("Q30_EXPOSURE_EVIDENCE_DIR", str(tmp_path / "exposure"))
    monkeypatch.setenv("RANK", "0")
    dataset = StrictDataset([{"id": 0}, {"id": 1}], tokenizer=MagicMock())

    with pytest.raises(RuntimeError, match=r"strict exposure.*occurrence 0"):
        dataset[0]


def test_q30_strict_exposure_still_validates_fetch_payload(tmp_path, monkeypatch):
    class RequiredPayload:
        __required_keys__ = frozenset({"required"})

    class StrictDataset(StreamingDataset):
        fetch_payload_cls = RequiredPayload

        def _tokenize_entry(self, entry):
            return {"cid": entry["prompt_uuid"], "prompt_uuid": entry["prompt_uuid"]}

        def _fetch(self, sample):
            return {"substituted": True}

    monkeypatch.setenv("Q30_STRICT_EXPOSURE", "1")
    monkeypatch.setenv("Q30_EXPOSURE_EVIDENCE_DIR", str(tmp_path / "exposure"))
    monkeypatch.setenv("RANK", "0")
    dataset = StrictDataset([{"prompt_uuid": "a" * 64}], tokenizer=MagicMock())

    with pytest.raises(RuntimeError, match="missing required keys"):
        dataset[0]


def test_circuit_breaker_trips_on_consecutive_failures():
    """When _fetch keeps hitting transient errors (server down), __getitem__ raises
    after the threshold instead of silently resampling the whole corpus."""
    threshold = 3

    class _AlwaysFails(StreamingDataset):
        def _tokenize_entry(self, entry):
            return {"cid": str(entry["id"]), "token_ids": [1], "loss_mask": None}

        def _fetch(self, sample):
            # A down server surfaces as a transport error, which the breaker counts.
            raise httpx.ConnectError("simulated server down")

    ds = _AlwaysFails(
        _entries(20),
        tokenizer=MagicMock(),
        config=StreamingConfig(fail_after_consecutive_skips=threshold),
    )
    with pytest.raises(RuntimeError, match="consecutive _fetch failures"):
        ds[0]


def test_contract_violation_propagates_not_swallowed():
    """A non-transient error from _fetch (e.g. a contract violation / bug) must
    surface immediately, not be masked as a fetch miss and silently resampled."""

    class _BadContract(StreamingDataset):
        def _tokenize_entry(self, entry):
            return {"cid": str(entry["id"]), "token_ids": [1], "loss_mask": None}

        def _fetch(self, sample):
            raise RuntimeError("server token_ids drift")

    ds = _BadContract(
        _entries(20),
        tokenizer=MagicMock(),
        # High threshold: if the error were (wrongly) swallowed, the breaker wouldn't
        # fire, so a leaked breaker message would mask the regression.
        config=StreamingConfig(fail_after_consecutive_skips=100),
    )
    with pytest.raises(RuntimeError, match="server token_ids drift"):
        ds[0]


def test_fetch_returning_none_exhausts_then_raises():
    """If every entry's fetch yields None (e.g. all rejected), __getitem__ raises a
    clear 'no fetchable sample' error rather than hanging or returning junk."""

    class _AllNone(StreamingDataset):
        def _tokenize_entry(self, entry):
            return {"cid": str(entry["id"]), "token_ids": [1], "loss_mask": None}

        def _fetch(self, sample):
            return None

    ds = _AllNone(
        _entries(4),
        tokenizer=MagicMock(),
        config=StreamingConfig(fail_after_consecutive_skips=100),
    )
    with pytest.raises(RuntimeError, match="no fetchable sample"):
        ds[0]


def test_resume_skips_consumed_samples_without_refetching():
    """Map-style resume contract: HF Trainer skips consumed batches via
    accelerate.skip_first_batches, which drops their indices at the batch-sampler
    level so __getitem__ (and thus _fetch) is never called for them. This is why
    main.py leaves ignore_data_skip at its default (False) for streaming -- resume
    lands at the exact position with no re-fetch. Guards against a regression that
    would re-fetch (or re-stream) already-consumed samples on resume."""
    pytest.importorskip("accelerate")
    from accelerate import skip_first_batches
    from torch.utils.data import DataLoader, RandomSampler

    fetched: list[int] = []

    class _Recording(StreamingDataset):
        def _tokenize_entry(self, entry):
            return {"cid": str(entry["id"]), "token_ids": [1], "loss_mask": None}

        def _fetch(self, sample):
            cid = int(sample["cid"])
            fetched.append(cid)  # stands in for the RDMA fetch
            return {"cid": cid}

        def _format(self, fetched):
            return torch.tensor(fetched["cid"])

    n, batch_size, skip_batches = 20, 2, 3
    ds = _Recording(_entries(n), tokenizer=MagicMock(), config=StreamingConfig())

    def make_dl():
        # Fresh, identically-seeded sampler -> identical permutation across runs.
        return DataLoader(
            ds,
            batch_size=batch_size,
            sampler=RandomSampler(ds, generator=torch.Generator().manual_seed(0)),
        )

    # Full pass -> ground-truth consumption order (cid == requested index here).
    full_order = [int(x) for batch in make_dl() for x in batch]
    fetched.clear()

    # Resume: skip the first `skip_batches` batches.
    tail_order = [int(x) for batch in skip_first_batches(make_dl(), skip_batches) for x in batch]

    consumed = full_order[: skip_batches * batch_size]
    expected_tail = full_order[skip_batches * batch_size :]
    assert tail_order == expected_tail, "resume must continue at the exact data position"
    assert set(fetched).isdisjoint(consumed), "skipped (consumed) samples must not be re-fetched"
    assert fetched == expected_tail, "only the un-consumed tail is fetched after resume"


def test_server_urls_normalization():
    """server_urls accepts a single string, a comma-separated string, or a list, and
    strips trailing slashes."""

    def _urls(v):
        return EagleVllmStreamingConfig(server_urls=v, model="m", max_seq_len=128).server_urls

    assert _urls("http://a:8000/") == ["http://a:8000"]
    assert _urls("http://a:8000, http://b:8000/") == ["http://a:8000", "http://b:8000"]
    assert _urls(["http://a:8000", "http://b:8000"]) == ["http://a:8000", "http://b:8000"]
    with pytest.raises(ValueError, match="at least one non-empty URL"):
        EagleVllmStreamingConfig(server_urls="", model="m", max_seq_len=128)


def test_max_seq_len_is_required():
    """max_seq_len is optional on the base config but required for the RDMA backend (it
    pre-sizes the recv buffer), so a missing/non-positive value must fail at construction
    rather than crashing later in _fetch."""
    with pytest.raises(ValueError, match="max_seq_len"):
        EagleVllmStreamingConfig(server_urls="http://a:8000", model="m")
    with pytest.raises(ValueError, match=r"max_seq_len|greater than 0"):
        EagleVllmStreamingConfig(server_urls="http://a:8000", model="m", max_seq_len=0)


class _FakeNixlAgent:
    """Stand-in for a ``nixl`` agent: the RDMA transfer is a no-op (returns DONE
    immediately), so the recv buffer is left as-is. End-to-end tests assert shapes
    (driven by the sidecar's ``hs_shape``) and the format chain, not transferred bytes.
    """

    def register_memory(self, tensors):
        return MagicMock()

    def deregister_memory(self, reg):
        return None

    def get_agent_metadata(self):
        return b"agent-meta"

    def add_remote_agent(self, meta):
        return "remote-agent"

    def get_xfer_descs(self, views):
        return MagicMock()

    def deserialize_descs(self, blob):
        return MagicMock()

    def initialize_xfer(self, op, ldescs, rdescs, remote):
        return "xfer-handle"

    def transfer(self, handle):
        return None

    def check_xfer_state(self, handle):
        return "DONE"

    def release_xfer_handle(self, handle):
        return None


def _mock_rdma(monkeypatch, handler):
    """Inject a fake ``nixl._api`` (not installed in CI) and route the dataset's
    per-process ``httpx.Client`` (sidecar metadata/descriptor calls + the completions
    POST) through a ``MockTransport`` handler."""
    fake_api = types.ModuleType("nixl._api")
    fake_api.nixl_agent = lambda *a, **k: _FakeNixlAgent()
    fake_api.nixl_agent_config = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "nixl", types.ModuleType("nixl"))
    monkeypatch.setitem(sys.modules, "nixl._api", fake_api)

    real_client = httpx.Client

    def mock_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(hf_streaming_dataset.httpx, "Client", mock_client)


def _tokenizer_returning(seq: int) -> MagicMock:
    """Tokenizer mock whose apply_chat_template yields a fixed seq-len output."""
    tok = MagicMock()
    tok.apply_chat_template.return_value = {
        "input_ids": torch.arange(seq, dtype=torch.long).unsqueeze(0),
    }
    return tok


def _rdma_sidecar_handler(seq, n_layers, hidden, *, on_completion=None, done_valid=True):
    """Build an httpx handler emulating the RdmaHiddenStatesConnector sidecar:
    POST /v1/completions -> {hs_req_id}; GET /meta -> agent metadata;
    GET /desc -> ready descriptor (shape/dtype/token_ids); GET /done -> ack + ``valid``
    (False emulates the ring lapping the slot mid-read).
    ``token_ids`` mirrors the client prompt verbatim (no drift)."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/completions":
            if on_completion is not None:
                on_completion(request)
            return httpx.Response(200, json={"kv_transfer_params": {"hs_req_id": "req-1"}})
        if path == "/meta":
            return httpx.Response(200, json={"agent_metadata": base64.b64encode(b"m").decode()})
        if path == "/desc":
            return httpx.Response(
                200,
                json={
                    "ready": True,
                    "hs_descs": base64.b64encode(b"d").decode(),
                    "hs_shape": [seq, n_layers, hidden],
                    "hs_dtype": "float32",
                    "token_ids": list(range(seq)),
                    "slot": 0,
                },
            )
        if path == "/done":
            return httpx.Response(200, json={"freed": "req-1", "valid": done_valid})
        return httpx.Response(404, json={"error": "not found"})

    return handler


def test_eagle_vllm_dataset_end_to_end(monkeypatch):
    """Drive EagleVllmStreamingDataset against an in-process mocked RDMA server.

    Verifies the RDMA fetch -> tensor -> batch-dict chain produces dicts matching
    what EagleOfflineDataCollator expects.
    """
    seq, n_layers, hidden = 8, 3, 16  # n_layers = 1 final + 2 aux
    _mock_rdma(monkeypatch, _rdma_sidecar_handler(seq, n_layers, hidden))

    n_entries = 4
    entries = [
        {"conversation_id": f"c-{i}", "messages": [{"role": "user", "content": "x"}]}
        for i in range(n_entries)
    ]
    ds = EagleVllmStreamingDataset(
        entries=entries,
        tokenizer=_tokenizer_returning(seq),
        config=EagleVllmStreamingConfig(
            server_urls="http://mock:8000",
            model="mock-model",
            max_seq_len=seq,
        ),
    )

    batches = [ds[i] for i in range(n_entries)]

    expected_keys = {
        "input_ids",
        "base_model_hidden_states",
        "aux_hidden_states",
        "attention_mask",
        "loss_mask",
        "labels",
        "base_hidden_prenorm",
    }
    for b in batches:
        assert set(b) == expected_keys
        assert b["input_ids"].shape == (seq,)
        assert b["input_ids"].dtype == torch.int64
        assert b["base_model_hidden_states"].shape == (seq, hidden)
        # 2 aux layers * hidden, flattened
        assert b["aux_hidden_states"].shape == (seq, 2 * hidden)
        assert b["attention_mask"].shape == (seq,)
        assert b["loss_mask"].shape == (seq,)
        assert b["labels"].shape == (seq,)
        # labels are input_ids shifted by 1, last position is IGNORE
        assert torch.equal(b["labels"][:-1], b["input_ids"][1:])
        assert b["labels"][-1].item() == hf_streaming_dataset.IGNORE_TOKEN_ID


def test_fetch_round_robins_across_server_urls(monkeypatch):
    """With multiple server_urls, consecutive fetches alternate across endpoints so
    load is spread over replicas rather than pinned to the first one."""
    seq, n_layers, hidden = 8, 3, 16
    hosts: list[str] = []
    handler = _rdma_sidecar_handler(
        seq, n_layers, hidden, on_completion=lambda req: hosts.append(req.url.host)
    )
    _mock_rdma(monkeypatch, handler)

    n_entries = 4
    entries = [
        {"conversation_id": f"c-{i}", "messages": [{"role": "user", "content": "x"}]}
        for i in range(n_entries)
    ]
    ds = EagleVllmStreamingDataset(
        entries=entries,
        tokenizer=_tokenizer_returning(seq),
        config=EagleVllmStreamingConfig(
            server_urls=["http://a:8000", "http://b:8000"],
            model="mock-model",
            max_seq_len=seq,
        ),
    )

    for i in range(n_entries):
        ds[i]

    # Per-process round-robin cursor: a, b, a, b -- one completions POST each, alternating.
    assert hosts == ["a", "b", "a", "b"]


def test_lapped_slot_is_treated_as_miss(monkeypatch):
    """If /done reports valid=False (the ring overwrote the slot mid-read), the fetched
    bytes are stale and must be discarded as a miss -- not returned as training data.
    Here every read is reported lapped, so __getitem__ exhausts and raises rather than
    silently yielding corrupt hidden states."""
    seq, n_layers, hidden = 8, 3, 16
    _mock_rdma(monkeypatch, _rdma_sidecar_handler(seq, n_layers, hidden, done_valid=False))

    ds = EagleVllmStreamingDataset(
        entries=[{"conversation_id": "c-0", "messages": [{"role": "user", "content": "x"}]}],
        tokenizer=_tokenizer_returning(seq),
        config=EagleVllmStreamingConfig(
            server_urls="http://mock:8000",
            model="mock-model",
            max_seq_len=seq,
            fail_after_consecutive_skips=100,
        ),
    )
    with pytest.raises(RuntimeError, match="no fetchable sample"):
        ds[0]


def test_q30_strict_exposure_requires_explicit_true_done_ack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing ``valid`` bit cannot authorize exposure of the fetched occurrence."""
    seq, n_layers, hidden = 8, 3, 16
    completions: list[list[int]] = []
    done_calls = 0
    base_handler = _rdma_sidecar_handler(
        seq,
        n_layers,
        hidden,
        on_completion=lambda request: completions.append(json.loads(request.content)["prompt"]),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal done_calls
        if request.url.path == "/done":
            done_calls += 1
            if done_calls == 1:
                return httpx.Response(200, json={"freed": "req-1"})
        return base_handler(request)

    _mock_rdma(monkeypatch, handler)
    monkeypatch.setenv("Q30_STRICT_EXPOSURE", "1")
    monkeypatch.setenv("Q30_EXPOSURE_EVIDENCE_DIR", str(tmp_path / "exposure"))
    monkeypatch.setenv("RANK", "0")
    prompt_uuid = "a" * 64
    ds = EagleVllmStreamingDataset(
        entries=[
            {
                "prompt_uuid": prompt_uuid,
                "messages": [{"role": "user", "content": "x"}],
            }
        ],
        tokenizer=_tokenizer_returning(seq),
        config=EagleVllmStreamingConfig(
            server_urls="http://mock:8000",
            model="mock-model",
            max_seq_len=seq,
            fail_after_consecutive_skips=2,
        ),
    )

    sample = ds[0]
    collator = hf_streaming_dataset.Q30StrictExposureCollator(lambda features: dict(features[0]))
    batch = collator([sample])
    hf_streaming_dataset.record_q30_consumed_exposure(batch)

    assert completions == [list(range(seq)), list(range(seq))]
    assert done_calls == 2
    records = (tmp_path / "exposure/rank-00000.jsonl").read_text().splitlines()
    assert [json.loads(record) for record in records] == [
        {"dataset_index": 0, "prompt_uuid": prompt_uuid}
    ]


def test_oversize_server_response_raises(monkeypatch):
    """If the server captured more tokens than max_seq_len (its connector max_tokens >
    our recv buffer), reading would silently truncate the slice; fail loud instead so the
    size mismatch is configured away rather than trained on misaligned hidden states."""
    seq, n_layers, hidden = 8, 3, 16
    # Sidecar advertises a longer sequence than the recv buffer is sized for.
    _mock_rdma(monkeypatch, _rdma_sidecar_handler(seq + 4, n_layers, hidden))

    ds = EagleVllmStreamingDataset(
        entries=[{"conversation_id": "c-0", "messages": [{"role": "user", "content": "x"}]}],
        tokenizer=_tokenizer_returning(seq),
        config=EagleVllmStreamingConfig(
            server_urls="http://mock:8000",
            model="mock-model",
            max_seq_len=seq,
        ),
    )
    with pytest.raises(RuntimeError, match="max_seq_len"):
        ds[0]


def test_sidecar_port_taken_from_response(monkeypatch):
    """The connector advertises its sidecar port per completions response; the dataset
    must address the /meta, /desc and /done sidecar calls at that port (not a hardcoded
    default), so a non-default connector sidecar_port works without an env override."""
    seq, n_layers, hidden = 8, 3, 16
    advertised_port = 23456
    seen_ports: set[int] = set()
    base = _rdma_sidecar_handler(seq, n_layers, hidden)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/completions":
            return httpx.Response(
                200,
                json={
                    "kv_transfer_params": {"hs_req_id": "req-1", "hs_sidecar_port": advertised_port}
                },
            )
        seen_ports.add(request.url.port)
        return base(request)

    _mock_rdma(monkeypatch, handler)

    ds = EagleVllmStreamingDataset(
        entries=[{"conversation_id": "c-0", "messages": [{"role": "user", "content": "x"}]}],
        tokenizer=_tokenizer_returning(seq),
        config=EagleVllmStreamingConfig(
            server_urls="http://mock:8000",
            model="mock-model",
            max_seq_len=seq,
        ),
    )
    ds[0]
    assert seen_ports == {advertised_port}


def test_remote_agent_cache_distinguishes_sidecars_on_the_same_host(monkeypatch):
    """Replicas can share a hostname but always have distinct sidecar ports.

    Register each replica independently instead of reusing the first replica's
    NIXL remote-agent handle for every replica on that host.
    """
    seen_ports: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/meta"
        seen_ports.append(request.url.port)
        metadata = f"agent-{request.url.port}".encode()
        return httpx.Response(200, json={"agent_metadata": base64.b64encode(metadata).decode()})

    _mock_rdma(monkeypatch, handler)
    ds = EagleVllmStreamingDataset(
        entries=[{"conversation_id": "c-0", "messages": [{"role": "user", "content": "x"}]}],
        tokenizer=_tokenizer_returning(8),
        config=EagleVllmStreamingConfig(
            server_urls="http://shared:8000",
            model="mock-model",
            max_seq_len=8,
        ),
    )
    ds._rdma()

    ds._remote("shared", 19001)
    ds._remote("shared", 19002)
    ds._remote("shared", 19001)

    assert seen_ports == [19001, 19002]


# ---------------------------------------------------------------------------
# answer_only_loss template guard
# ---------------------------------------------------------------------------


def _fast_tokenizer_with_template(template: str, seq: int = 8) -> MagicMock:
    """Fast-tokenizer mock with a given chat template; returns ids + assistant_masks."""
    tok = MagicMock()
    tok.is_fast = True
    tok.chat_template = template
    tok.apply_chat_template.return_value = {
        "input_ids": torch.arange(seq, dtype=torch.long).unsqueeze(0),
        "assistant_masks": torch.ones(1, seq, dtype=torch.long),
    }
    return tok


_CONV = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]


def test_answer_only_loss_rejects_template_without_generation_tags():
    """A fast tokenizer whose template lacks {% generation %} tags fails loudly.

    Without the guard transformers only warns and returns an ALL-ZERO assistant
    mask -- every sample then trains at zero loss with no other symptom.
    """
    tok = _fast_tokenizer_with_template("{{ bos }}{% if add_generation_prompt %}x{% endif %}")
    with pytest.raises(RuntimeError, match="generation"):
        hf_streaming_dataset._tokenize_with_loss_mask(tok, _CONV, answer_only_loss=True)


def test_answer_only_loss_accepts_tagged_template():
    """Templates carrying {% generation %} (either whitespace-control form) pass."""
    for tag in ("{% generation %}", "{%- generation -%}"):
        tok = _fast_tokenizer_with_template("{{ bos }}" + tag + "{{ c }}")
        ids, mask = hf_streaming_dataset._tokenize_with_loss_mask(tok, _CONV, answer_only_loss=True)
        assert mask.sum() == ids.shape[-1]


def test_full_loss_skips_template_guard():
    """answer_only_loss=False never consults the template (mask is all ones)."""
    tok = _fast_tokenizer_with_template("{{ bos }}")
    ids, mask = hf_streaming_dataset._tokenize_with_loss_mask(tok, _CONV, answer_only_loss=False)
    assert mask.sum() == ids.shape[-1]


def test_answer_only_loss_rejects_slow_tokenizer_without_recovery():
    """A slow tokenizer with no registered recovery fails loudly even on a tagged template.

    Assistant-mask alignment needs the fast tokenizer's char_to_token; without the
    guard apply_chat_template fails downstream with an unrelated-looking error.
    """
    tok = _fast_tokenizer_with_template("{{ bos }}{% generation %}{{ c }}")
    tok.is_fast = False
    tok.convert_tokens_to_ids.return_value = None  # defeat recovery detect()s
    with pytest.raises(RuntimeError, match="fast tokenizer"):
        hf_streaming_dataset._tokenize_with_loss_mask(tok, _CONV, answer_only_loss=True)
