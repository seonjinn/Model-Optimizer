#!/bin/bash
# Stage the four English PTV2 splits (18 shards) at a pinned revision.
# Token arrives on stdin and is never written to disk.
set -uo pipefail
DST="${1:?usage: stage_ptv2.sh <dest-dir>}"
REV=5c89e01dd720ae0f4058445ed49c5fb68a03c76e
BASE="https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2/resolve/${REV}/data"
read -r HFTOK

FILES="stem-00000-of-00002.parquet stem-00001-of-00002.parquet
math-00000-of-00002.parquet math-00001-of-00002.parquet
code-00000-of-00002.parquet code-00001-of-00002.parquet"
for i in $(seq 0 11); do FILES="$FILES chat-$(printf '%05d' "$i")-of-00012.parquet"; done

mkdir -p "$DST" || exit 1
ok=0; fail=0; failed=""
for f in $FILES; do
  if [ -s "$DST/$f" ]; then ok=$((ok+1)); continue; fi
  got=0
  for attempt in 1 2; do
    if curl -sSL --fail -m 2400 -H "Authorization: Bearer $HFTOK" \
         -o "$DST/$f.part" "$BASE/$f" 2>/dev/null; then
      mv "$DST/$f.part" "$DST/$f"; got=1; break
    fi
    rm -f "$DST/$f.part"
  done
  if [ "$got" = 1 ]; then ok=$((ok+1)); else fail=$((fail+1)); failed="$failed $f"; fi
done
{ echo "revision=$REV"; echo "ok=$ok fail=$fail"; [ -n "$failed" ] && echo "failed:$failed"; du -sh "$DST"; } > "$DST/../stage_status.txt"
