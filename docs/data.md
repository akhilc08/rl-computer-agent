# Data

Everything under `data/` is git-ignored. The source copy is on akhilc08's Mac, so a fresh clone won't have it.

To get it, download the snapshot from the `data-20261025` release (376 MB, everything except `data/public/`) and unzip it at the repo root:

```bash
gh release download data-20261025 && unzip rlca-data-20261025.zip && rm rlca-data-20261025.zip
scripts/public/download_public.sh   # public datasets, 13 GB
```

The release is a point-in-time copy. Newer runs only exist locally until a new release is cut.

## What's there (as of 2026-10-25)

| Folder | Size | What it is | How to recreate |
|---|---|---|---|
| `data/explore/` | 585 MB | Explorer runs: screenshots, AX trees, and actions on real logged-in sites (Gmail, GitHub, Google, Amazon, Docs, Calendar, Canvas, NeetCode, claude.ai). The main run, `20261025-124135`, has 2,835 states and 712 unique screenshots. | `scripts/collect/explore.py` / `overnight.sh`. It needs the owner's accounts, so it can't really be recreated. |
| `data/source_eval/` | 8 MB | AX vs OCR vs detector comparison (`sources.jsonl`, `sources_summary.json`) | `scripts/optbench/replay_sources.py`, `compare_sources.py` |
| `data/public/` | 15 GB | ScreenSpot v1/v2/Pro, OmniACT, OS-Atlas macOS+Linux, OSWorld-G, MMBench-GUI (L2) | `scripts/public/download_public.sh` |
| `data/converted/` | | Public sets in Kev's element-list format: `screens.jsonl` (OCR + detector elements per screenshot) and `examples.jsonl` (instruction, action, target element ids, split). ScreenSpot-v2/Pro, OSWorld-G and MMBench-GUI are split `eval` only; OmniACT is held out completely (see [benchmarks.md](benchmarks.md)). | `scripts/public/convert_public.py --set <name>` (macOS, about 1 s per screenshot) |
| `data/labels/` | | 10,980 labelled voice commands on 508 Mac screens: 7,295 on 325 Arc pages and native-app snapshots, plus 3,685 on 183 native Slack, Notes, Notion and Spotify windows from the native explorer (near-duplicate windows dropped, `--native`). Each command has all correct targets, action, typed text and its source, irreversible / ambiguous / absent flags, casual and planner styles. Written by Claude agents under `scripts/labels/label_guide.txt`. | `scripts/labels/label_prep.py`, then labelling, then `scripts/labels/label_check.py` |
| `data/records/` | | Kev training/eval requests built from the labels and converted public sets (`train`, `val` = Mac labels only, `heldout_site` = NeetCode, `eval_ssv2`, `eval_sspro`, `eval_osworldg`, `eval_mmbench`, `eval_omniact`). Training screenshots that look like a benchmark screen are dropped. | `scripts/train/build_records.py` (run with `.vendor/kev/.venv/bin/python`) |
| `data/results/` | <1 MB | Kev 0.8B / 4B baseline outputs | `scripts/eval/run_kev_baseline.py` |
| `data/snapshots/` | <1 MB | AX snapshots of native apps | `scripts/collect/capture_snapshots.py` |
| `data/synthetic/` | <1 MB | `baseline_v0.jsonl` synthetic commands | not scripted yet |

Not downloaded yet: Multimodal-Mind2Web and AndroidControl (about 64 GB). The plan is to run `download_public.sh --big` on the AWS training instance.

## Privacy

`data/explore/`, `data/snapshots/` and `data/source_eval/` come from the owner's personal accounts and contain real emails, chats, calendars and repos. They're shared with the team through the private release only. Don't commit them, re-upload them anywhere else, or share screenshots outside the team. Anything going to AWS for training needs the owner's sign-off first.

The public datasets in `data/public/` are fine to download yourself with the script.
