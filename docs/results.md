# Results

## Where Kev's options come from: AX vs OCR vs detector (2026-10-25)

This is a replay of explorer run `data/explore/20261025-124135`: 712 unique Arc screenshots from 9 sites, run with `scripts/eval/replay_sources.py`. Raw output is in that run's `sources.jsonl` and `sources_summary.json` (see [data.md](data.md) for how to get the data). The Mac is an Apple Silicon laptop, and the detector and captioner run on MPS.

Each screen's options come from three sources:
- **Accessibility (AX):** the window's AX tree.
- **OCR:** Apple Vision text lines that don't overlap an AX element.
- **Detector icons:** OmniParser v2 YOLO boxes that overlap neither AX nor OCR.

Counts only include the page area, so the Arc sidebar is left out.

### Options found per screen

| Site | Screens | AX | + OCR | + detector icons |
|---|---|---|---|---|
| GitHub | 201 | 43 | +18 | +8 |
| Google | 153 | 33 | +13 | +8 |
| Amazon | 100 | 57 | +26 | +9 |
| Gmail | 100 | 68 | +10 | +18 |
| NeetCode | 89 | 39 | +11 | +15 |
| Docs | 43 | 16 | +12 | +6 |
| Canvas | 13 | 34 | +20 | +12 |
| Google Calendar | 8 | 72 | +54 | +33 |
| claude.ai | 5 | 1 | 0 | 0 |
| **All** | **712** | **44** | **+16** | **+10** |

The "+" columns are upper bounds. Some OCR lines are plain paragraphs and some detector boxes are decoration. A by-eye check of a sample is still to do.

### Latency per screen (median ms)

| Site | AX | OCR fast | OCR accurate | Detector |
|---|---|---|---|---|
| GitHub | 109 | 68 | 168 | 268 |
| Google | 108 | 66 | 170 | 268 |
| Amazon | 138 | 80 | 185 | 271 |
| Gmail | 111 | 66 | 171 | 266 |
| NeetCode | 124 | 54 | 153 | 271 |
| Docs | 102 | 43 | 125 | 266 |
| Canvas | 84 | 48 | 148 | 266 |
| Google Calendar | 87 | 56 | 160 | 274 |
| claude.ai | 75 | 35 | 107 | 262 |

Across all screens:

| Step | Median ms | p90 ms |
|---|---|---|
| Screenshot | 61 | 69 |
| AX | 111 | 190 |
| OCR fast | 61 | 111 |
| OCR accurate | 162 | 242 |
| Detector | 268 | 305 |
| Merge | 8 | 22 |
| Icon captions (Florence-2, 64×64 crops) | 203 | 312 |

Gmail's AX p90 is 463 ms. Kev 0.8B (4-bit MLX) takes about 170 ms per step.

### Cost before Kev runs (sources in parallel)

| Setup | Wait | Options per screen |
|---|---|---|
| AX only | ~110 ms | 44 |
| AX + fast OCR | ~110 ms (OCR finishes within the AX time) | up to 60 |
| AX + accurate OCR | ~170 ms | up to 60 |
| + detector | ~270 ms | up to 70 |

### By-eye check of the extras (2026-11-04)

These are hand judgments on 27 screenshots, 3 per site. `scripts/eval/overlay_sample.py draw` draws the AX, OCR-only and detector-only boxes; each extra was then marked as a real click target or not, and `score` tallies the marks. Judgments are in `overlays/judgments.jsonl` in the run folder.

| Site | AX | OCR extras | real | Detector extras | real |
|---|---|---|---|---|---|
| Amazon | 40 | 7.0 | 0.7 | 4.7 | 2.0 |
| Canvas | 33 | 12.0 | 4.7 | 10.3 | 1.7 |
| claude.ai | 1 | 0 | 0 | 0 | 0 |
| Docs | 15 | 8.0 | 2.0 | 4.7 | 3.7 |
| Google Calendar | 98 | 38.7 | 11.7 | 21.0 | 16.0 |
| GitHub | 26 | 16.0 | 0.3 | 4.0 | 1.0 |
| Gmail | 96 | 12.3 | 0 | 28.0 | 18.0 |
| Google | 17 | 2.3 | 0 | 0.3 | 0 |
| NeetCode | 35 | 10.3 | 3.7 | 22.0 | 4.0 |
| **All** | **40** | **11.9** | **2.6** | **10.6** | **5.1** |

- **Only about 1 in 5 OCR extras is a real target (2.6 of 11.9).** The rest are paragraphs, headings and labels. The real ones are mostly nav labels on pages where the AX page tree is missing.
- **About half the detector extras are real (5.1 of 10.6).** They're icon-only buttons AX doesn't expose: Gmail's star and importance markers, Docs/Sheets template tiles, theme toggles, carousel arrows, and mini-calendar dates.
- **No visible target was missed by all three sources** on these screens.
- **The extras matter most when Arc's AX tree has no web area.** That's 6 of 13 Canvas screens, 5 of 8 Calendar screens and 8 of 100 Amazon screens. There, AX only sees the browser chrome, and OCR plus the detector find the whole page. Next to try: setting `AXManualAccessibility` on Arc, which makes Chromium build the tree.
- **claude.ai isn't an AX problem.** All 12 visits captured `claude.ai/recents` still showing the loading spinner, so the page never finished loading in the background window.

So real targets per screen are roughly AX 40 + OCR 2.6 + detector 5.1, about 18% on top of AX, not the 60% upper bound. A short-line filter on OCR (nav labels, not sentences) would cut most of the OCR noise.

### Takeaways

- **AX + fast OCR in parallel is still the default.** It's free in wall time, but only about 6% more real targets, so OCR needs a short-line filter to keep the noise out of Kev's list.
- **The detector finds more real targets than OCR (5 vs 2.6 per screen) but costs 270 ms**, more than Kev itself. Run it when AX has no web area or the list looks thin, not on every screen.
- **Fixing the missing AX web area is the bigger win.** That's where most of the real extras come from.
- **Captions are weak on brand logos**, so icon text from the captioner can't be trusted alone.
- **claude.ai never finished loading in the background window.** The explorer needs to wait for it, or skip it.

## Kev 0.8B after the first fine-tune (2026-11-05)

`jaredpalmer/kev-0.8b@v1.0` fine-tuned on `train_mix` (28,155 examples) for 1 epoch, 3,508 steps on an AWS A10G, about 13 h. Scored with `scripts/eval/score_kev.py`. Run directories: `runs/eval-base-0.8b` and `runs/eval-sft-0.8b-v1`; the adapter is `runs/sft-0.8b-v1` under `.vendor/kev`.

**This run trained on OmniACT, and its test screens are the same screens as its training split, so the OmniACT row is not a clean result.** OmniACT is now held out completely (see [benchmarks.md](benchmarks.md)), and a clean retrain is planned.

### Element accuracy, before → after

| Set | Before | After | Act alone at 90%, before → after | Calibration error, before → after |
|---|---|---|---|---|
| Our Mac screens (val, Mac part) | 66% | **90%** | | |
| All of val (Mac + OmniACT) | 53% | 75% | 8% → 63% | 0.26 → 0.10 |
| NeetCode (site held out of training) | 75% | **86%** | 57% → **92%** | 0.42 → 0.09 |
| Synthetic set | 56% | 77% | 17% → 80% | 0.30 → 0.07 |
| ScreenSpot-v2 | 51% | 67% | 0% → 31% | 0.17 → 0.06 |
| ScreenSpot-Pro | 24% | 25% | 0% → 0% | 0.09 → 0.11 |
| OmniACT test (not clean, see above) | 45% | 58% | 0% → 19% | 0.21 → 0.09 |

### Other questions (val)

| Question | Before | After |
|---|---|---|
| Action (click / type / …) | 63% | 94% |
| Text source | 66% | 99% |
| Ambiguous | 70% | 92% (but it flags only 6 of 39 truly ambiguous; NeetCode: 32 of 102) |
| Irreversible | 95% | 99% (1% base rate, so not meaningful yet) |
| "Not on screen" on Mac screens | 21% | 82% |

### Notes

- **"Ambiguous" is too cautious.** When it says ambiguous it's usually right, but it misses most real cases. Fixing this is the RL step's job.
- **It never answers "none" on public sets** (0 of 356 on ScreenSpot-Pro): every public training example had a target present.
- **Synthetic-set action is low (32%)** because that old set labels menu items "open_menu_item" while the new labels say "click".
- **ScreenSpot-Pro didn't move.** Even when the target is in the list, it picks it only about 32% of the time; icon captions don't say enough on dense professional toolbars.

## Screenshot models on the Mac (2026-11-05)

UI-Venus-1.5-2B and Qwen3-VL-2B, 4-bit MLX (`mlx-vlm`) on the M5 MacBook (32 GB). 25 commands from each of ScreenSpot-v2, ScreenSpot-Pro, OSWorld-G and OmniACT, same prompts and image budgets as the GPU eval (`scripts/eval/vlm_mac_speed.py`). Time covers resize + prefill + decode. For comparison, Kev's path is about 0.3–0.45 s: the element list is 110–270 ms and Kev 0.8B about 170 ms.

| Model | Median per command | p90 | Hits (sample) | Peak memory |
|---|---|---|---|---|
| UI-Venus-1.5-2B | 5.6 s | 24.7 s | 70% | 5.0 GB |
| Qwen3-VL-2B | 7.2 s | 21.0 s | 37% | 3.5 GB |

UI-Venus by benchmark (time grows with screen resolution):

| Benchmark | Median | Hits |
|---|---|---|
| OSWorld-G (1080p) | 3.1 s | 76% |
| ScreenSpot-v2 | 5.6 s | 84% |
| OmniACT (Retina Mac screens) | 12.1 s | 68% |
| ScreenSpot-Pro (4K) | 24.5 s | 52% |

On the same Mac, the 2B screenshot models are 10–50× slower than Kev but far more accurate on public screenshots. ShowUI-2B (small image budget) and the fast list-based models weren't timed: the runs were stopped because they made the laptop unusable. The rest runs on the GPU box.

On the A10G: UI-Venus-1.5-2B about 0.86 s, ShowUI-2B about 0.82 s, Qwen3-VL-2B about 1.6 s per command on OSWorld-G (12-example smoke test). The full GPU runs are in progress.
