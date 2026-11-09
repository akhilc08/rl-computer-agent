# rl-computer-agent

RL to make generalized computer agents better. Kev is a sub-1B voice agent that picks one option from a list
built from the macOS accessibility (AX) tree, Apple Vision OCR and a pixel detector.

Data is git-ignored (`data/`, `runs/`); see [docs/data.md](docs/data.md). Results: [docs/results.md](docs/results.md).
Evaluation plan and baselines: [docs/benchmarks.md](docs/benchmarks.md). Related work: [docs/related_work.md](docs/related_work.md).

## Layout

| Path | What's there |
|---|---|
| `src/rlca/` | Library: AX walking, screen capture, OCR, detector, option-list merge, explorer safety policy, records |
| `scripts/sandbox/` | EC2 Mac sandbox: setup, virtual display, fake-content seeding, the collector, and the dataset pipeline (`sandbox_dataset.py` clean/split/sources/visible/pixel, `sandbox_label.py`) |
| `scripts/labels/` | Command-labeling guide and checker shared by every label set |
| `scripts/public/` | Download public GUI datasets and convert them to Kev's format |
| `scripts/collect/` | Earlier explorers that ran on the owner's Mac (that data is internal only) |
| `scripts/optbench/` | Option-source benchmark: which OCR/detector/merge builds the best option list |
| `scripts/eval/` | Kev and baseline evaluation and scoring (list models, VLM grounding, web agents) |
| `scripts/train/` | Training records and the AWS training pipeline |
| `scripts/probes/` | Small one-off measurements (AX coverage, prefill latency, tokenizer) |
| `scripts/setup/` | One-time model conversions |
| `docs/` | Results, data notes, benchmark plan, related work |
