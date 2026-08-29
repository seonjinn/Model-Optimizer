#!/bin/bash
# Stage the PTv3 core candidate set at pinned revisions.
#
# The token arrives on stdin and is passed to curl through a `-K -` config on a
# pipe. It is never written to disk and never appears in argv: /proc on these
# login nodes is mounted without hidepid, so every user can read every other
# user's command line.
#
# Each file is verified against the size the tree API reports for the pinned
# revision, so a truncated transfer is refetched instead of being counted done.
#
# Large files are fetched as parallel byte ranges. A single stream measured
# 775 KB/s against this CDN, which is 5.4 hours for the 15 G shard -- longer
# than any wall clock a transfer should need and longer than curl would wait.
# Ranges are independent GETs, so they also make a retry cheap: a failed range
# costs its own chunk, not the whole file.
set -uo pipefail
DST="${1:?usage: stage_ptv3.sh <dest-dir>}"
read -r HFTOK
MANIFEST="
Nemotron-SFT-Math-v4:84d42ad0
Nemotron-RL-Math-v2:804418c1
Nemotron-SFT-SWE-v3:3f73de64
Nemotron-SFT-SWE-v3.5:ad641292
Nemotron-SFT-SWE-v2:bd151f3f
Nemotron-SWE-v1:0fe17a96
Open-SWE-Traces:c2114fc8
Nemotron-Agentic-v1:650d5909
Nemotron-SFT-Agentic-v2:7c804833
Nemotron-RL-Lightning-Training-Blend:262eb58c
"
mkdir -p "$DST" || exit 1
STATUS="$DST/stage_ptv3_status.txt"; : > "$STATUS"

# Token on a pipe, never in argv.
fetch() {
  printf 'header = "Authorization: Bearer %s"\n' "$HFTOK" \
    | curl -K - -sSL --fail "$@"
}

# Every "is this file the right length" check goes through here. Reading the
# size with a redirect means a missing file is a shell error rather than a
# value, and wc pads its output on some platforms, so a bare string compare is
# wrong twice over.
filesize() {
  [ -f "$1" ] || { echo -1; return; }
  wc -c < "$1" 2>/dev/null | tr -d '[:space:]'
}

CHUNK_BYTES=${STAGE_CHUNK_BYTES:-$((128 * 1024 * 1024))}
STREAMS=${STAGE_STREAMS:-16}
# Below this a single stream is already the whole file in a few chunks and the
# bookkeeping costs more than it saves.
RANGE_MIN_BYTES=${STAGE_RANGE_MIN_BYTES:-$((256 * 1024 * 1024))}

# A transfer that is moving slowly is not a transfer that is stuck. Abort on a
# stall (under 4 KB/s for two minutes), never on elapsed time: a wall-clock cap
# turns a slow link into a permanent failure, and without -C - every retry
# starts over.
SPEED_GUARD=(--speed-limit 4096 --speed-time 120)

# Fetch one byte range into its own file, then check it is exactly the length
# that range should hold. A server that ignores Range answers with the whole
# object, and that check is what catches it.
fetch_chunk() {
  local url="$1" cf="$2" start="$3" end="$4" len="$5" attempt
  for attempt in 1 2 3; do
    rm -f "$cf"
    if fetch "${SPEED_GUARD[@]}" -r "$start-$end" -o "$cf" "$url" 2>/dev/null &&
       [ "$(filesize "$cf")" -eq "$len" ] 2>/dev/null; then
      return 0
    fi
  done
  rm -f "$cf"
  return 1
}

# Returns 0 only when the reassembled file is exactly $want bytes. Chunks are
# kept on failure so the next run resumes instead of restarting.
fetch_ranged() {
  local url="$1" tgt="$2" want="$3"
  local dir="$tgt.parts" start end len cf running=0
  mkdir -p "$dir" || return 1
  start=0
  while [ "$start" -lt "$want" ]; do
    end=$((start + CHUNK_BYTES - 1))
    [ "$end" -ge "$want" ] && end=$((want - 1))
    len=$((end - start + 1))
    cf="$dir/$(printf '%020d' "$start")"
    if [ "$(filesize "$cf")" -ne "$len" ] 2>/dev/null; then
      fetch_chunk "$url" "$cf" "$start" "$end" "$len" &
      running=$((running + 1))
      # Drain the whole batch rather than refilling one slot at a time. The
      # barrier costs a little throughput and buys portability: `wait -n` needs
      # bash 4.3, and this script has to run wherever it is dropped.
      if [ "$running" -ge "$STREAMS" ]; then wait; running=0; fi
    fi
    start=$((end + 1))
  done
  wait
  # Re-derive the chunk list rather than trusting the jobs: the only thing that
  # matters is whether every range is on disk at its exact length.
  local -a chunks=()
  start=0
  while [ "$start" -lt "$want" ]; do
    end=$((start + CHUNK_BYTES - 1))
    [ "$end" -ge "$want" ] && end=$((want - 1))
    len=$((end - start + 1))
    cf="$dir/$(printf '%020d' "$start")"
    if [ "$(filesize "$cf")" -ne "$len" ] 2>/dev/null; then return 1; fi
    chunks+=("$cf")
    start=$((end + 1))
  done
  cat "${chunks[@]}" > "$tgt.part" || return 1
  if [ "$(filesize "$tgt.part")" -ne "$want" ] 2>/dev/null; then
    rm -f "$tgt.part"; return 1
  fi
  mv "$tgt.part" "$tgt" && rm -rf "$dir"
}

for entry in $MANIFEST; do
  REPO="${entry%%:*}"; SHA="${entry##*:}"
  OUT="$DST/$REPO"; mkdir -p "$OUT"
  # Resolve the file list and each file's size at the pinned revision.
  LISTING=$(fetch -m 300 \
    "https://huggingface.co/api/datasets/nvidia/$REPO/tree/$SHA?recursive=1" \
    | python3 -c 'import json,sys
try: t=json.load(sys.stdin)
except Exception: sys.exit(1)
for x in t:
    if x.get("type")=="file":
        print(x["size"], x["path"], sep="\t")' 2>/dev/null)
  if [ -z "$LISTING" ]; then echo "$REPO sha=$SHA TREE_FAILED" >> "$STATUS"; continue; fi
  ok=0; fail=0; refetched=0
  while IFS=$'\t' read -r want f; do
    [ -n "$f" ] || continue
    tgt="$OUT/$f"
    if [ -f "$tgt" ]; then
      have=$(filesize "$tgt")
      if [ "$have" -eq "$want" ] 2>/dev/null; then ok=$((ok+1)); continue; fi
      # Right name, wrong length: a truncated earlier transfer. Refetch.
      rm -f "$tgt"; refetched=$((refetched+1))
    fi
    mkdir -p "$(dirname "$tgt")"
    url="https://huggingface.co/datasets/nvidia/$REPO/resolve/$SHA/$f"
    got=0
    if [ "$want" -ge "$RANGE_MIN_BYTES" ] 2>/dev/null; then
      fetch_ranged "$url" "$tgt" "$want" && got=1
    fi
    if [ "$got" = 0 ]; then
      # Single stream, resumed rather than restarted, for small files and as
      # the fallback when the CDN will not serve ranges for this object.
      for attempt in 1 2 3; do
        if fetch "${SPEED_GUARD[@]}" -C - -o "$tgt.part" "$url" 2>/dev/null; then
          have=$(filesize "$tgt.part")
          if [ "$have" -eq "$want" ] 2>/dev/null; then mv "$tgt.part" "$tgt"; got=1; break; fi
        fi
        have=$(filesize "$tgt.part")
        # Keep a partial that is still growing toward the right length; discard
        # one that has overshot it, because -C - cannot undo that.
        if [ "$have" -gt "$want" ] 2>/dev/null; then rm -f "$tgt.part"; fi
      done
      rm -f "$tgt.part"
    fi
    [ "$got" = 1 ] && ok=$((ok+1)) || fail=$((fail+1))
  done <<< "$LISTING"
  bytes=$(find "$OUT" -type f ! -name '*.part' ! -path '*.parts/*' -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{print s+0}')
  echo "$REPO sha=$SHA ok=$ok fail=$fail refetched=$refetched bytes=$bytes" >> "$STATUS"
done
total=$(find "$DST" -type f ! -name '*.part' ! -path '*.parts/*' -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{print s+0}')
echo "TOTAL bytes=$total" >> "$STATUS"
