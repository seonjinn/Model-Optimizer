# Task 3 Report: Canonical Q30 Thinking Parent Receipts

## Result

Implemented a dedicated canonical receipt builder and loader for the two
design-approved Ptyche parents: DFlash completion job `2638009` and DSpark
completion job `2638015`, both Q30 Thinking B8 at step 25391 on 16 nodes. The
identity table binds each method to its exact run and checkpoint path.

Receipt generation and loading use no-follow descriptors, stable
device/inode/size/mtime/ctime comparisons, regular-file type checks, and
streamed hashing. Each receipt binds the complete checkpoint path/type/size/SHA
inventory, its canonical tree digest, `modelopt_state.pth`, the milestone
manifest, the exact completed-job lifecycle receipt, the target/source/runtime
identities, and the historical 1.3M receipt plus ordered UUID digest. Pair
reconciliation requires exactly one DFlash parent and one DSpark parent and
equality of both historical lineage digests.

`APPROVED_Q30T_PARENT_RECEIPT_FILE_SHA256S` intentionally remains empty.
Synthetic tests monkeypatch both path identities and approval roots only inside
the test process.

## RED Evidence

Before the production module existed, ran:

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tools/launcher/tests/test_q30t_parent_receipt.py
```

Collection failed with exit code 2 and:

```text
ImportError: cannot import name 'q30t_parent_receipt' from 'common.specdec'
```

Two additional RED cases were observed before their fixes:

- deleting `control/training-complete-s25391.json` was initially accepted;
- a nested `a/nested.json` next to `a.txt` exposed non-global descriptor order
  and caused the canonical receipt to reject its own bytes.

The implementation now requires the exact completion receipt and globally
sorts the descriptor inventory before hashing and publication.

## Local Verification Evidence

Focused parent plus existing continuation regressions:

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tools/launcher/tests/test_q30t_parent_receipt.py \
  tools/launcher/tests/test_qwen4b_ptv23_continuation.py
```

Result: `56 passed, 2 skipped`. Pytest also reported six pre-existing temporary
directory cleanup warnings from `test_stage_weight_only_parent_*`; no test
failed.

The focused parent-only result is `19 passed`.

## External Gate

No production parent receipt was generated and no receipt file SHA was added to
the approval root. The two remote checkpoint trees, milestone manifests,
completion receipts, exact target revision/tree, source/runtime identities,
and historical lineage must be regenerated from fresh reads and independently
reviewed together before either receipt SHA can be approved. This task did not
SSH, submit a job, or invent an external digest.

## Static Verification Evidence

Fresh checks on the two Python files completed with exit code 0:

```bash
../q4-ptv2-ab-integration/.venv/bin/ruff check --no-fix \
  tools/launcher/common/specdec/q30t_parent_receipt.py \
  tools/launcher/tests/test_q30t_parent_receipt.py
../q4-ptv2-ab-integration/.venv/bin/ruff format --check \
  tools/launcher/common/specdec/q30t_parent_receipt.py \
  tools/launcher/tests/test_q30t_parent_receipt.py
pyright tools/launcher/common/specdec/q30t_parent_receipt.py \
  tools/launcher/tests/test_q30t_parent_receipt.py
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m py_compile \
  tools/launcher/common/specdec/q30t_parent_receipt.py \
  tools/launcher/tests/test_q30t_parent_receipt.py
```

Ruff check reported `All checks passed`, Ruff format reported both files
already formatted, and Pyright reported `0 errors, 0 warnings, 0
informations`. Python compilation produced no output and exited successfully.
