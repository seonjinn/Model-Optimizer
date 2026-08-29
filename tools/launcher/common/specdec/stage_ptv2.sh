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

# A non-empty file is not a staged file: a truncated shard is non-empty too.
# Take the last Content-Length so the value survives the CDN redirect.
remote_size() {
  curl -sIL --fail -m 120 -H "Authorization: Bearer $HFTOK" "$1" 2>/dev/null \
    | awk 'tolower($1) == "content-length:" { n = $2 + 0 } END { print n + 0 }'
}
# stat flags are not portable; wc -c is, and a missing file yields 0 either way.
local_size() { wc -c < "$1" 2>/dev/null | tr -d ' ' || echo 0; }

mkdir -p "$DST" || exit 1
ok=0; fail=0; failed=""; bytes=0
for f in $FILES; do
  want="$(remote_size "$BASE/$f")"
  [ "$want" -gt 0 ] 2>/dev/null || { fail=$((fail+1)); failed="$failed $f(no-size)"; continue; }
  if [ "$(local_size "$DST/$f")" = "$want" ]; then
    ok=$((ok+1)); bytes=$((bytes+want)); continue
  fi
  got=0
  for attempt in 1 2; do
    if curl -sSL --fail -m 2400 -H "Authorization: Bearer $HFTOK" \
         -o "$DST/$f.part" "$BASE/$f" 2>/dev/null \
       && [ "$(local_size "$DST/$f.part")" = "$want" ]; then
      mv "$DST/$f.part" "$DST/$f"; got=1; break
    fi
    rm -f "$DST/$f.part"
  done
  if [ "$got" = 1 ]; then ok=$((ok+1)); bytes=$((bytes+want)); else fail=$((fail+1)); failed="$failed $f"; fi
done
# du under-reports a Lustre directory that is still being written, so report
# the bytes actually accounted for instead.
{ echo "revision=$REV"; echo "ok=$ok fail=$fail"; echo "bytes=$bytes";
  [ -n "$failed" ] && echo "failed:$failed"; } > "$DST/../stage_status.txt"
[ "$fail" -eq 0 ]
