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

for entry in $MANIFEST; do
  REPO="${entry%%:*}"; SHA="${entry##*:}"
  OUT="$DST/$REPO"; mkdir -p "$OUT"
  find "$OUT" -name '*.part' -delete 2>/dev/null
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
      have=$(wc -c < "$tgt" 2>/dev/null || echo -1)
      if [ "$have" = "$want" ]; then ok=$((ok+1)); continue; fi
      # Right name, wrong length: a truncated earlier transfer. Refetch.
      rm -f "$tgt"; refetched=$((refetched+1))
    fi
    mkdir -p "$(dirname "$tgt")"
    got=0
    for attempt in 1 2 3; do
      if fetch -m 3600 -o "$tgt.part" \
           "https://huggingface.co/datasets/nvidia/$REPO/resolve/$SHA/$f" 2>/dev/null; then
        have=$(wc -c < "$tgt.part" 2>/dev/null || echo -1)
        if [ "$have" = "$want" ]; then mv "$tgt.part" "$tgt"; got=1; break; fi
      fi
      rm -f "$tgt.part"
    done
    [ "$got" = 1 ] && ok=$((ok+1)) || fail=$((fail+1))
  done <<< "$LISTING"
  bytes=$(find "$OUT" -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{print s+0}')
  echo "$REPO sha=$SHA ok=$ok fail=$fail refetched=$refetched bytes=$bytes" >> "$STATUS"
done
total=$(find "$DST" -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{print s+0}')
echo "TOTAL bytes=$total" >> "$STATUS"
