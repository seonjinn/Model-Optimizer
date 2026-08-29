"""Compare staged PTv3 shards against the file list at each pinned revision.

The stager writes a status file, but a restarted run truncates it and the sizes
it records disagree with disk, so it cannot be used as the completeness record.
The revision tree is the only authority: a repo is staged when every file the
pinned SHA lists is present at its published size.
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(sys.argv[1])
TOKEN = os.environ["HF_TOKEN"]
MANIFEST = {
    "Nemotron-SFT-Math-v4": "84d42ad0",
    "Nemotron-RL-Math-v2": "804418c1",
    "Nemotron-SFT-SWE-v3": "3f73de64",
    "Nemotron-SFT-SWE-v3.5": "ad641292",
    "Nemotron-SFT-SWE-v2": "bd151f3f",
    "Nemotron-SWE-v1": "0fe17a96",
    "Open-SWE-Traces": "c2114fc8",
    "Nemotron-Agentic-v1": "650d5909",
    "Nemotron-SFT-Agentic-v2": "7c804833",
    "Nemotron-RL-Lightning-Training-Blend": "262eb58c",
}

def tree(repo, sha):
    url = f"https://huggingface.co/api/datasets/nvidia/{repo}/tree/{sha}?recursive=1"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return [x for x in json.load(r) if x.get("type") == "file"]

for repo, sha in MANIFEST.items():
    out = ROOT / repo
    try:
        files = tree(repo, sha)
    except Exception as exc:
        print(f"{repo:38s} TREE_FAILED {exc}")
        continue
    missing = short = 0
    have_bytes = want_bytes = 0
    for entry in files:
        want = entry.get("size") or 0
        want_bytes += want
        target = out / entry["path"]
        if not target.exists():
            missing += 1
            continue
        actual = target.stat().st_size
        have_bytes += actual
        if want and actual != want:
            short += 1
    state = "COMPLETE" if (missing == 0 and short == 0 and files) else "INCOMPLETE"
    print(f"{repo:38s} {state:10s} files={len(files):4d} missing={missing:4d} "
          f"size_mismatch={short:3d} bytes={have_bytes}/{want_bytes}")
