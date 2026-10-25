# Data

Everything under `data/` is git-ignored. It exists only on akhilc08's Mac, so a fresh clone won't have it.

## What's there (as of 2026-10-25)

| Folder | Size | What it is | How to recreate |
|---|---|---|---|
| `data/explore/` | 585 MB | Explorer runs: screenshots, AX trees, and actions on real logged-in sites (Gmail, GitHub, Google, Amazon, Docs, Calendar, Canvas, NeetCode, claude.ai). The main run, `20261025-124135`, has 2,835 states and 712 unique screenshots. | `scripts/data/explore.py` / `overnight.sh`. It needs the owner's accounts, so it can't really be recreated. |
| `data/source_eval/` | 8 MB | AX vs OCR vs detector comparison (`sources.jsonl`, `sources_summary.json`) | `scripts/eval/replay_sources.py`, `compare_sources.py` |
| `data/public/` | 13 GB | ScreenSpot v1/v2/Pro, OmniACT, OS-Atlas macOS+Linux | `scripts/data/download_public.sh` |
| `data/results/` | <1 MB | Kev 0.8B / 4B baseline outputs | `scripts/baseline/run_kev_baseline.py` |
| `data/snapshots/` | <1 MB | AX snapshots of native apps | `scripts/data/capture_snapshots.py` |
| `data/synthetic/` | <1 MB | `baseline_v0.jsonl` synthetic commands | not scripted yet |

Not downloaded yet: Multimodal-Mind2Web and AndroidControl (about 64 GB). The plan is to run `download_public.sh --big` on the AWS training instance.

## Privacy

`data/explore/`, `data/snapshots/` and `data/source_eval/` come from the owner's personal accounts and contain real emails, chats, calendars and repos. Don't commit them, upload them, or share screenshots from them. If you need examples, ask the owner. Anything going to AWS for training needs the owner's sign-off first.

The public datasets in `data/public/` are fine to download yourself with the script.
