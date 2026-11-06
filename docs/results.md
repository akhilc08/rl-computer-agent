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
| OSWorld-G | 25% | 28% | 1% → 6% | 0.07 → 0.09 |
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
- **OSWorld-G barely moved (25% → 28%).** The target is in the parser's list for 79% of examples, and when it is, Kev picks it about 36% of the time. Published 2B screenshot models report about 59% (UI-Venus-1.5-2B). This checkpoint predates the lookalike filter, so it may have trained on some OS-Atlas Linux screens that resemble OSWorld-G; even so it scores low.
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

## Kev vs the open element-list models: MindAct and WebLINX (2026-11-06)

Every model picks from the same element list for the same commands. The lists come from the page's HTML on Mind2Web and from our pixel parser on the screenshot sets; our Mac set and NeetCode use AX trees. Each set is the same fixed sample of 500 decisions (seed 0, `data/records_s`). The exceptions are all 429 Mac val screens, plus 519 website and 521 domain steps on Mind2Web. On Mind2Web, MindAct's own DeBERTa ranker keeps the top 50 elements, and every model picks from those 50. The target makes the top 50 for 90% of task, 87% of website and 87% of domain steps; MindAct's paper reports about 86%. Steps whose target isn't in the top 50 are left out for every model. Scripts: `scripts/data/convert_mind2web.py`, `scripts/eval/webagent_baselines.py`, `scripts/eval/score_kev.py`. Outputs: `runs/webagents`, `runs/kev-box`.

### Element accuracy (%)

| Model | Params | M2W task | M2W website | M2W domain | ScreenSpot-v2 | ScreenSpot-Pro | OSWorld-G | MMBench-GUI L2 | OmniACT | Our Mac screens | NeetCode (held-out site) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Kev 0.8B, before training** | 0.8B | 2.4 | 3.3 | 4.0 | 53.0 | **27.4** | 24.6 | 30.8 | 42.4 | 66.0 | 75.2 |
| **Kev 0.8B, after training** | 0.8B | 7.6 | 10.6 | 10.8 | **68.8** | 26.8 | **28.2** | **37.8** | *60.2* | **90.4** | **86.0** |
| MindAct base | 0.25B | 49.8 | 37.6 | 38.0 | 33.4 | 18.8 | 16.4 | 13.8 | 20.2 | 34.5 | 26.4 |
| MindAct large | 0.8B | 56.2 | 43.5 | 44.7 | 37.8 | 17.2 | 16.2 | 18.2 | 23.6 | 49.4 | 35.4 |
| MindAct XL | 3B | **60.0** | **46.8** | **48.6** | 22.6 | 24.0 | 22.6 | 13.8 | 18.2 | 54.3 | 37.8 |
| WebLINX flan-T5 base | 0.25B | 5.6 | 5.8 | 6.9 | 26.6 | 6.4 | 10.8 | 10.2 | 16.6 | 19.8 | 23.4 |
| WebLINX flan-T5 large | 0.8B | 1.4 | 1.9 | 2.9 | 4.4 | 24.4 | 20.8 | 9.6 | 11.2 | 14.7 | 8.8 |
| WebLINX Sheared-LLaMA | 1.3B | 7.8 | 8.7 | 10.9 | 24.0 | 6.4 | 9.6 | 10.0 | 14.8 | 17.5 | 17.4 |
| WebLINX MindAct base | 0.25B | 5.6 | 6.0 | 8.1 | 25.2 | 6.6 | 10.2 | 10.8 | 15.2 | 18.9 | 24.0 |
| WebLINX MindAct large | 0.8B | 6.6 | 6.4 | 7.7 | 27.8 | 6.0 | 10.0 | 10.2 | 17.2 | 19.8 | 25.2 |

*Italic:* Kev's after-training OmniACT score isn't clean. That checkpoint trained on OmniACT's train split, which shares screens with its test split. The before-training score (42.4) is clean.

### Speed (A10G, each model alone on the GPU, 100 decisions)

| Model | Median per decision, M2W (50 elements) | Median, NeetCode (~100 elements) | Model calls per decision |
|---|---|---|---|
| Kev 0.8B (PyTorch bf16 eval path) | **0.23 s** | 0.68 s | 1 |
| MindAct base / large / XL | 1.28 / 2.09 / 2.56 s | 1.52 / 2.86 / 3.09 s | 11 on M2W, 20 on NeetCode (multiple choice in rounds of 5) |
| WebLINX flan-T5 base / large | 0.40 / 1.31 s | 0.40 / 0.74 s | 1 |
| WebLINX Sheared-LLaMA 1.3B | 5.09 s | 5.09 s | 1 |
| WebLINX MindAct base / large | 0.42 / 0.81 s | 0.43 / 0.81 s | 1 |

Kev's serving path on the Mac (4-bit MLX) takes about 0.17 s for the model. The A10G number above uses the unoptimized PyTorch path used for scoring. WebLINX models see only the 10 elements their ranker keeps, so their prompts are short. Neither Mind2Web's ranker nor WebLINX's is in the timings, and our parser isn't either.

### What this shows

- **Off Mind2Web, Kev wins every set, even before training.** It's ahead of the best MindAct/WebLINX model by 31 points on ScreenSpot-v2, 20 on MMBench-GUI, 36 on our Mac screens and 48 on NeetCode. It's also 5–10× faster than MindAct (one call instead of 11–20). The WebLINX models with similar speed (flan/MindAct base, ~0.4 s) score 10–25%.
- **On Mind2Web, MindAct wins by a lot (50–60% vs Kev's 8–11%).** There are two reasons:
  - MindAct was trained on Mind2Web's train split, which has the same task style and many of the same sites.
  - Mind2Web asks something different from what Kev does. It gives a high-level goal plus history ("Find all events in New York in September", done so far: …) and asks for the next step. Kev grounds a direct command ("click Change Location"). It usually picks the element most related to the goal, often the one already clicked.

  The WebLINX models, which weren't trained on Mind2Web, also score 2–11% there. Even the ranker's first pick alone gets 34%, so Kev isn't using the ranker's order.
- **To be competitive on Mind2Web, Kev needs Mind2Web training data, as MindAct and WebLINX had.** The options are to train on Mind2Web train (7,775 steps, no overlap with the test splits) or to report Mind2Web only as an out-of-domain number.
- **Training didn't hurt anything, and it helped most where our data matches** (Mac +24, NeetCode +11, ScreenSpot-v2 +16 points). The "none" problem from the earlier section shows up again. On MMBench-GUI, Kev says "none" correctly 1% of the time after training versus 39% before, so the retrain needs public "none" examples.

### Caveats

- **The screenshot sets have no HTML.** MindAct and WebLINX get a synthetic page built from our parser's list instead (role → tag, label → text), so those sets are new ground for them. Their Mind2Web numbers use the real page HTML and match their papers' setup.
- **WebLINX is approximated.** We don't run its own ranker on our candidates, and it's conversational: flan-T5 large usually answers with a chat reply ("Okay") instead of acting, which is why its scores are so low.
- **Mind2Web here is element accuracy for each step, with the ranker's top 50.** MindAct's paper reports element accuracy over all steps, including ones the ranker misses. To compare with the paper, multiply by the top-50 hit rate. For example, MindAct large on the domain split: 44.7% × 0.87 ≈ 39%. The paper reports about 40% for that split, so our setup reproduces theirs.

## Option sources: coverage vs speed (2026-11-06)

Every option source scored on the same screens: 712 Arc screenshots from explorer run `20261025-124135` plus public screenshots (OmniACT 98 Mac screens, ScreenSpot-Pro 300, OSWorld-G 250). Scripts: `scripts/eval/optbench_build.py` (manifest, AX, Vision OCR), `optbench_detect.py` (pixel detectors), `optbench_score.py` (coverage). Outputs are in `data/option_bench/`.

A target is covered when some option's center falls inside one of its correct boxes. That is, clicking the option would hit it. The ground truths:

- **label:** labelled command targets on our Arc screens.
- **press:** elements whose press changed the screen.
- **public sets:** human answer boxes.

Label and press both lean toward AX, because labels were written from AX-based lists and the explorer only presses AX elements. The public sets have no AX at all, so they stand in for screens where AX fails.

Times are Mac medians (M5) per screen, with each source running alone. The Core ML columns run on all screens; the MPS rows are a 60-screen sample. UI-DETR was only run on the A10G.

### Single sources

| Source | Mac ms | Options/screen | label | press | OmniACT | ScreenSpot-Pro | SS-Pro icons | OSWorld-G |
|---|---|---|---|---|---|---|---|---|
| Blind grid, 150 cells (chance) | 0 | 148 | 48.7 | 44.2 | 45.6 | 18.0 | 12.7 | 27.5 |
| AX (recorded live) | 111 | 78 | 95.4 | 99.8 | | | | |
| Vision OCR fast | 66 | 62 | 73.8 | 72.6 | 58.8 | 24.0 | 3.4 | 31.6 |
| Vision OCR accurate | 169 | 62 | 75.8 | 71.1 | 63.5 | 23.0 | 3.4 | 39.0 |
| OmniParser v2 @1280, MPS (today's detector) | 132 | 124 | 84.5 | 81.4 | 82.3 | 83.0 | 87.3 | 75.9 |
| OmniParser v2 @1280, Core ML | 93 | 131 | 85.4 | 82.3 | 85.4 | 85.0 | 89.8 | 76.9 |
| **GPA-GUI-Detector @1280, Core ML** | **87** | 151 | 89.0 | 89.1 | 90.9 | **90.0** | 85.6 | 77.5 |
| **GPA-GUI-Detector @640, Core ML** | **32** | 116 | 87.3 | 87.1 | 89.6 | 57.7 | 53.4 | 79.0 |
| Screen2AX YOLO11l @1280, MPS | 157 | 152 | 82.2 | 84.7 | 89.4 | 78.0 | 77.1 | 73.7 |
| OmniParser icon_detect_v3 @1280, MPS | 574 | 147 | 90.1 | 87.1 | 90.9 | 64.0 | 61.0 | 77.3 |
| UI-DETR-1 @1600 (A10G only) | 336 (GPU) | 261 | 93.6 | 94.3 | 95.5 | 80.7 | 74.6 | 84.3 |

Detectors are shown at confidence 0.05. With the same budget of 100 boxes each on ScreenSpot-Pro, the scores are GPA 54, OmniParser v2 49, Screen2AX 45, v3 43, UI-DETR 41 and the blind grid 10. So GPA's lead isn't just from returning more boxes.

### Combinations

Sources run in parallel. When run together on the Mac, fast OCR plus GPA (Core ML) takes 60 ms at 640px and 99 ms at 1280px, both under the 111 ms AX walk.

| Combination | Mac ms (parallel) | Options/screen | label | label not from AX | press | OmniACT | ScreenSpot-Pro | OSWorld-G |
|---|---|---|---|---|---|---|---|---|
| AX + OCR fast | ~111 | 137 | 99.8 | 95.3 | 99.8 | | | |
| AX + OCR fast (≤6 words) + GPA @640 | ~111 | 187 | 100.0 | 99.7 | 99.8 | | | |
| OCR fast (≤6 words) + GPA @640, no AX | ~60 | 171 | 91.0 | 99.7 | 90.9 | 90.7 | 66.3 | 81.4 |
| OCR fast (≤6 words) + GPA @1280, no AX | ~99 | 204 | 92.2 | 99.7 | 92.4 | 92.1 | 91.3 | 80.0 |
| OCR fast (≤6 words) + OmniParser v2 @1280, no AX | ~94 | 185 | 92.6 | 99.0 | 92.8 | 92.5 | 89.3 | 81.8 |

### What this shows

- **Swap the detector for GPA-GUI-Detector via Core ML.** At 640px it takes 32 ms, against 132 ms for today's OmniParser v2 on MPS (268 ms in the earlier replay). It covers as much or more on every set. It's also MIT-licensed, while OmniParser v2's weights are AGPL. fp16 Core ML stays within 1–3 points of the GPU runs.
- **640px is enough for normal windows. Dense 4K screens need 1280px.** ScreenSpot-Pro coverage goes from 58% at 640px to 90% at 1280px, while every other set is about the same. Switch to 1280px only when the screen is large or crowded.
- **The pixel sources add no wait.** Fast OCR plus GPA finishes inside the AX walk, so AX + OCR + detector costs about the same ~111 ms as AX alone, and the merged list reaches 100% of labelled targets.
- **Without AX, OCR + GPA reaches about 91% on ordinary screens.** The misses are on dense pro apps (ScreenSpot-Pro at 640px) and Linux desktops (OSWorld-G, about 80%).
- **The cost is a longer list:** about 187 options per screen instead of 137. That makes Kev's prompt longer and pushes toward its 255-option cap. Kev also hasn't been trained on detector-only answers yet (see the caption bug in label prep). Pixel options should go in when AX is thin, or be ranked, rather than appended in full.
- **Not worth it:**
  - OmniParser icon_detect_v3: 4× slower than v2 on the Mac and worse on dense screens.
  - UI-DETR-1: the best raw coverage, but it returns 260–300 boxes and takes 336 ms even on a GPU.
  - Screen2AX's macOS-trained YOLO: slower than GPA and worse on every set.
- **A small model listing the elements doesn't work yet:**
  - Florence-2-base region proposals return about one box per screen, usually the whole page, so they cover almost nothing (0–3%).
  - ScreenVLM-316M (IBM's screen parser) writes out the whole page as tagged boxes. It hit its 4,096-token cap on every screen and took about 340 s per screen on an A10G. On the 6 screens it finished, it found 6 of 19 pressed elements. Even a 100× faster runtime would be slower than the whole AX + OCR + detector stack.

### Where each step ran

Nothing here touched live apps. AX was not walked again: the scores use the AX trees the explorer recorded live on the owner's Mac (Arc on 2026-10-25, `data/explore/20261025-124135`). Everything else ran on saved screenshots.

| Step | Ran on | Input |
|---|---|---|
| AX | owner's MacBook (M5), live, during the 2026-10-25 explorer run | recorded `elements` and `ms.ax_window` in `states.jsonl` |
| Vision OCR, fast and accurate | owner's MacBook | saved screenshots (`optbench_build.py`) |
| Detector coverage on all screens | AWS g5.xlarge (A10G) | the same screenshots, uploaded with the owner's OK and deleted afterward |
| Detector timing | owner's MacBook: Core ML on all screens, MPS on a 60-screen sample | saved screenshots (`optbench_detect.py`) |
| Core ML export | AWS box (needs `numpy<2.3` and torch ≤ 2.7; coremltools 9 fails on the Mac's torch 2.14) | Ultralytics `export(format="coreml", half=True)` |
| Scoring | owner's MacBook | `optbench_score.py` |

Not tested yet:
- **Native apps.** The about 440 explorer screenshots of Slack, Notion, Notes, Spotify and Zed in `data/native/` already have recorded AX. Other apps would need fresh look-only captures.
- **Independent ground truth.** Targets that every source missed are invisible to these labels.

### Running it on an AWS Mac

Every step except AX runs on saved files, so on an EC2 Mac instance:

1. Copy `data/option_bench/manifest.jsonl` and the screenshots it lists. Don't copy the private explore screenshots unless the owner signs off; the public sets alone are fine.
2. Run `optbench_build.py`, then `optbench_detect.py --source gpa_coreml:640` (and the others), then `optbench_score.py`.
3. Copy the Core ML packages from `~/.cache/rlca/coreml/`, or export them again in an environment with `numpy<2.3`.

Live AX needs a bit more:

- **Permissions.** The Python or terminal process needs Accessibility and Screen Recording access. On an EC2 Mac these are granted once through a GUI session (VNC / Screen Sharing), or through an MDM profile.
- **Apps.** They have to be installed, signed in where it matters, and open, with a real window on the virtual display.
- **Scripts.**
  - `scripts/eval/compare_sources.py` walks AX on the current window of each running app and times it.
  - `scripts/data/explore_native.py` records full runs in the layout `optbench_build.py` reads (`states.jsonl` + `shots/`).
- **Timings.** EC2 Macs are M1 / M2 / M2 Pro / M4, not the owner's M5, so expect different numbers. Compare sources on the same machine, not against this table.
