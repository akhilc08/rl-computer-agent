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


## Every model on every benchmark (2026-11-07)

Every model was run on every benchmark it can run. Kev's numbers on the older sets are from the sections above; its WebLINX and AndroidControl runs were added on 2026-11-07 (A10G median 92 ms on AndroidControl, 139 ms on WebLINX). Each set is the same fixed sample for every model (`data/records_s`, seed 0): 500 per set, 519/521 on the Mind2Web website/domain splits, and Mac val = 809 (the old 429 plus 380 Slack, Notes and Notion commands). Screenshot models see the screenshot and must click inside the target's box; every other model picks from the element list. Table: `python3 scripts/eval/results_table.py`; outputs in `runs/webagents`, `runs/list-baselines`, `runs/vlm`.

New benchmarks:
- **AndroidControl, low-level split** (`scripts/data/convert_androidcontrol.py`): direct step instructions with real accessibility trees, 1,288 test steps across 237 apps.
- **WebLINX, all five test splits** (`scripts/data/convert_weblinx.py`): real pages with WebLINX's own released candidate ranking, top 50. The target is in the top 50 for 53–77% of turns, depending on the split, and turns where it isn't are left out for every model.

### Element accuracy (%)

| Group | Model | M2W task | M2W web | M2W domain | WebLINX | AndroidControl | SS-v2 | SS-Pro | OSWorld-G | MMBench | OmniACT | Mac val | NeetCode |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Kev | Kev 0.8B (after training) | 7.6 | 10.6 | 10.8 | 17.2 | 71.6 | 68.8 | 26.8 | 28.2 | 37.8 | 60.2 | 90.4 | 86.0 |
| Kev | Kev 0.8B (before training) | 2.4 | 3.3 | 4.0 | 9.4 | 57.6 | 53.0 | 27.4 | 24.6 | 30.8 | 42.4 | 66.0 | 75.2 |
| MindAct | mindact-base | 49.8 | 37.6 | 38.0 | 18.0 | 42.0 | 33.4 | 18.8 | 16.4 | 13.8 | 20.2 | 37.9 | 26.4 |
| MindAct | mindact-large | 56.2 | 43.5 | 44.7 | 18.4 | 42.6 | 37.8 | 17.2 | 16.2 | 18.2 | 23.6 | 49.9 | 35.4 |
| MindAct | mindact-xl | 60.0 | 46.8 | 48.6 | 19.6 | 34.6 | 22.6 | 24.0 | 22.6 | 13.8 | 18.2 | 50.1 | 37.8 |
| WebLINX | wl-flan-base | 5.6 | 5.8 | 6.9 | 30.6 | 37.8 | 26.6 | 6.4 | 10.8 | 10.2 | 16.6 | 23.4 | 23.4 |
| WebLINX | wl-flan-large | 1.4 | 1.9 | 2.9 | 29.8 | 0.6 | 4.4 | 24.4 | 20.8 | 9.6 | 11.2 | 17.1 | 8.8 |
| WebLINX | wl-llama-1.3b | 7.8 | 8.7 | n/r† | 0.0 | 41.6 | 24.0 | 6.4 | 9.6 | 10.0 | 14.8 | 19.4 | 17.4 |
| WebLINX | wl-mindact-base | 5.6 | 6.0 | 8.1 | 23.0 | 39.8 | 25.2 | 6.6 | 10.2 | 10.8 | 15.2 | 22.9 | 24.0 |
| WebLINX | wl-mindact-large | 6.6 | 6.4 | 7.7 | 23.2 | 42.4 | 27.8 | 6.0 | 10.0 | 10.2 | 17.2 | 23.5 | 25.2 |
| Small LLM, same list | HuggingFaceTB/SmolLM2-1.7B-Instruct | 8.8 | 10.8 | 7.3 | 9.0 | 31.0 | 19.0 | 0.8 | 3.4 | 7.6 | 8.6 | 7.2 | 4.0 |
| Small LLM, same list | HuggingFaceTB/SmolLM2-360M-Instruct | 30.4 | 28.1 | 26.9 | 31.2 | 1.4 | 1.8 | 0.8 | 0.8 | 1.2 | 3.2 | 1.0 | 0.2 |
| Small LLM, same list | Qwen/Qwen2.5-0.5B-Instruct | 21.4 | 19.3 | 17.5 | 30.2 | 5.6 | 3.6 | 0.4 | 0.2 | 1.2 | 3.6 | 2.0 | 0.0 |
| Small LLM, same list | Qwen/Qwen2.5-1.5B-Instruct | 4.8 | 4.8 | 4.2 | 9.8 | 50.0 | 35.4 | 9.6 | 13.2 | 17.8 | 21.0 | 35.0 | 28.6 |
| Small LLM, same list | Qwen/Qwen3-0.6B | 3.8 | 4.2 | 6.7 | 10.2 | 39.6 | 35.2 | 11.2 | 12.6 | 17.2 | 20.8 | 21.6 | 12.0 |
| Small LLM, same list | Qwen/Qwen3-1.7B | 7.8 | 7.7 | 10.9 | 15.4 | 47.0 | 48.8 | 17.4 | 15.4 | 22.4 | 26.8 | 35.1 | 22.2 |
| Small LLM, same list | model2vec | 6.8 | 6.6 | 7.7 | 14.4 | 49.8 | 56.6 | 17.8 | 18.2 | 19.0 | 38.2 | 62.5 | 73.0 |
| Screenshot model | gui-actor-2b | – | – | – | 31.2 | 84.6 | 89.0 | 35.6 | 38.2 | 65.2 | 61.8 | 68.4 | 71.0 |
| Screenshot model | qwen3vl-2b | – | – | – | 30.8 | 75.0 | 81.0 | 30.2 | 31.8 | 55.6 | 56.8 | 53.5 | 63.5 |
| Screenshot model | showui-2b | – | – | – | 29.0 | 80.0 | 77.0 | 7.6 | 20.2 | 44.0 | 51.8 | 30.2 | 34.9 |
| Screenshot model | tinyclick | – | – | – | 34.6 | 89.8 | 81.0 | 3.8 | 22.6 | 35.2 | 46.2 | 55.3 | 54.1 |
| Screenshot model | venus-2b | – | – | – | 32.2 | 88.6 | 93.6 | 59.2 | 64.4 | 79.2 | 69.0 | 74.6 | 80.6 |
| Screenshot model | venus-8b | – | – | – | 37.6 | 93.6 | 96.2 | 69.8 | 74.4 | 89.0 | 76.0 | 81.8 | 84.9 |

† WebLINX's Sheared-LLaMA almost never acts on its own test data: 26 of 500 outputs are element actions, and most of those name uids that aren't on the page. WebLINX's exact eval formatting gives the same result (0/20 on a check), while its flan-T5 models on the same prompts pick real candidates. We couldn't reproduce it, so that cell is left out.

### Median latency per decision (ms, A10G)

| Group | Model | M2W task | M2W web | M2W domain | WebLINX | AndroidControl | SS-v2 | SS-Pro | OSWorld-G | MMBench | OmniACT | Mac val | NeetCode |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MindAct | mindact-base | 1646 | 1751 | 1674 | 1112 | 405 | 721 | 4552 | 3162 | 1005 | 1283 | 1654 | 1624 |
| MindAct | mindact-large | 2710 | 2724 | 2591 | 1907 | 778 | 1324 | 6279 | 5446 | 2196 | 2857 | 1970 | 3513 |
| MindAct | mindact-xl | 3749 | 3677 | 2572 | 2287 | 816 | 1590 | 8995 | 5879 | 2504 | 3878 | 2243 | 3716 |
| WebLINX | wl-flan-base | 497 | 496 | 492 | 443 | 384 | 377 | 421 | 411 | 387 | 389 | 397 | 396 |
| WebLINX | wl-flan-large | 1081 | 1122 | 1148 | 878 | 740 | 714 | 755 | 761 | 1105 | 736 | 787 | 801 |
| WebLINX | wl-llama-1.3b | 5064 | 5061 | n/r† | 5284 | 5988 | 5001 | 5564 | 5164 | 5339 | 5349 | 5169 | 5086 |
| WebLINX | wl-mindact-base | 482 | 483 | 476 | 464 | 416 | 407 | 459 | 444 | 422 | 423 | 425 | 429 |
| WebLINX | wl-mindact-large | 907 | 912 | 897 | 957 | 796 | 771 | 822 | 814 | 795 | 788 | 808 | 801 |
| Small LLM, same list | HuggingFaceTB/SmolLM2-1.7B-Instruct | 158 | 155 | 153 | 130 | 118 | 148 | 335 | 244 | 159 | 207 | 173 | 179 |
| Small LLM, same list | HuggingFaceTB/SmolLM2-360M-Instruct | 217 | 216 | 216 | 214 | 212 | 210 | 258 | 229 | 216 | 216 | 210 | 226 |
| Small LLM, same list | Qwen/Qwen2.5-0.5B-Instruct | 79 | 81 | 80 | 66 | 65 | 65 | 150 | 112 | 70 | 82 | 85 | 78 |
| Small LLM, same list | Qwen/Qwen2.5-1.5B-Instruct | 148 | 147 | 148 | 123 | 106 | 113 | 250 | 173 | 131 | 146 | 156 | 161 |
| Small LLM, same list | Qwen/Qwen3-0.6B | 101 | 95 | 105 | 96 | 122 | 123 | 223 | 176 | 125 | 135 | 138 | 139 |
| Small LLM, same list | Qwen/Qwen3-1.7B | 175 | 173 | 160 | 137 | 127 | 134 | 290 | 226 | 154 | 176 | 185 | 189 |
| Small LLM, same list | model2vec | 1 | 1 | 1 | 1 | 1 | 1 | 3 | 2 | 1 | 1 | 2 | 2 |
| Screenshot model | gui-actor-2b | – | – | – | 1266 | 2374 | 2872 | 8694 | 1729 | 2874 | 7260 | 8802 | 8808 |
| Screenshot model | qwen3vl-2b | – | – | – | 1506 | 1799 | 2414 | 3743 | 2433 | 1937 | 2744 | 3392 | 3429 |
| Screenshot model | showui-2b | – | – | – | 703 | 730 | 752 | 768 | 749 | 752 | 765 | 783 | 800 |
| Screenshot model | tinyclick | – | – | – | 127 | 136 | 140 | 151 | 132 | 139 | 160 | 184 | 185 |
| Screenshot model | venus-2b | – | – | – | 629 | 1022 | 1182 | 2614 | 874 | 1134 | 2026 | 3671 | 3703 |
| Screenshot model | venus-8b | – | – | – | 1176 | 1872 | 2160 | 2976 | 1569 | 2108 | 3793 | 6889 | 6936 |

Kev's A10G latency (PyTorch eval path, alone on the GPU) is 230 ms on Mind2Web (50 elements) and 680 ms on NeetCode (~100 elements). On the Mac (4-bit MLX), Kev itself takes about 170 ms. List-model times leave out building the list (the ranker or our parser), as for Kev. Box 1, where the screenshot models ran, was shared with another job part of the time, so their times may be a little high.

### Same examples, apples to apples

Kev was scored on the original 429 Mac val commands, and screenshot models only where a target exists (absent commands have nothing to click). On exactly the same commands:

| Model | Mac val, target present (370) | NeetCode, target present (458) |
|---|---|---|
| **Kev 0.8B (after training)** | **91.9** | **87.6** |
| Kev 0.8B (before training) | 70.3 | 77.9 |
| UI-Venus-1.5-8B | 79.2 | 84.9 |
| UI-Venus-1.5-2B | 73.5 | 80.6 |
| GUI-Actor-2B | 68.4* | 71.0 |
| TinyClick (0.27B) | 60.0 | 54.1 |
| Best list model (model2vec / MindAct-XL, all 429) | 57.3 / 54.3 | 73.0 / 37.8 |

*GUI-Actor's Mac val number is over all 709 target-present commands.

### What this shows

- **On real Mac screens with accessibility, trained Kev beats every model tested, including 8B screenshot models**, and does it in a fraction of the time: about 0.2–0.7 s on the A10G vs 3.7 s (UI-Venus-2B) and 6.9 s (UI-Venus-8B) on the same screens.
- **On public screenshot benchmarks the screenshot models win clearly** (UI-Venus-2B: 94 vs 69 on ScreenSpot-v2, 59 vs 27 on ScreenSpot-Pro). Here Kev only sees what our pixel parser finds, and that list misses the target 20%+ of the time.
- **TinyClick is the fastest accurate screenshot model** (~0.14 s; 81 on ScreenSpot-v2, 90 on AndroidControl), but it collapses on dense professional screens (4% on ScreenSpot-Pro) and on our Mac screens (55–60%).
- **Among list models, nothing open comes close to Kev on our data.** MindAct only wins on Mind2Web, which it trained on. model2vec, a plain embedding match, is a surprisingly strong floor on our screens (57–73%): our commands often reuse the element's label.
- **Small instruct LLMs are poor at this out of the box.** Qwen3-1.7B is the best of them (49 on ScreenSpot-v2, 35 on Mac val). SmolLM2-360M and Qwen2.5-0.5B pick the first option 87–96% of the time, so their ~30% on Mind2Web and WebLINX is just the ranker's top-1 rate, not real decisions.
- **AndroidControl (real accessibility trees): trained Kev gets 71.6%, the best of every list model** (next: Qwen2.5-1.5B 50.0, model2vec 49.8, MindAct-large 42.6), at 92 ms. The screenshot models do better here (UI-Venus-8B 93.6, TinyClick 89.8 at about 140 ms); Kev has never trained on phone screens.
- **WebLINX is weak for Kev (17.2%)**, behind WebLINX's own flan-T5 (30.6) and the screenshot models (29–38). Like Mind2Web, its commands are chat turns that need dialogue context, not direct commands.

## Option sources v2: 3,158 screens, 153 apps (2026-11-07)

Which detector and merge rule to use to build Kev's option list, scored against dense element boxes
(`scripts/eval/optbench2_*.py`; detectors on an A10G, Apple Vision OCR on an EC2 Mac). Recall = share of
clickable elements with an option center inside them. Options = list length after merging near-duplicates.

- **GroundCUA** (2,131 screens, 87 desktop apps, people boxed every element). GPA-GUI-Detector was almost
  certainly trained on it (mean best IoU 0.80 vs ~0.48 for the others; it boxes 97% of the non-clickable
  "Information Display" text), so GPA's numbers there are not a fair test.
- **Screen2AX** (158 macOS screens, MacPaw). MacPaw's YOLO was trained on this data, so its numbers are not fair here.
- **Sandbox** (869 screens from our EC2 Mac collector, ~50 apps and sites). Targets are the live AX elements, so it
  measures how much of AX a pixel method recovers. It cannot show what AX misses; that needs hand labels.

### Clickable recall at conf 0.15 (union)

| method | Screen2AX (Mac, held out) | Sandbox (AX recovered) | GroundCUA | options (S2AX) | ms |
|---|---|---|---|---|---|
| OCR fast (≤6 words) | 51.2 | 44.1 | 30.5 | 32 | 48 |
| GPA @640 | 81.1 | 50.4 | *97.2 (trained on it)* | 41 | 28 |
| OmniParser v2 @640 | 66.2 | 45.2 | 68.1 | 25 | 27 |
| OmniParser v3 @1280 | 67.9 | 52.3 | 78.9 | 27 | 207 |
| UI-DETR-1 @1600 | 91.0 | 64.3 | 93.7 | 105 | 320 |
| OCR + GPA @640 | 84.9 | 57.8 | *97.6* | 72 | 50 |
| OCR + UI-DETR @1600 | 91.8 | 64.8 | 94.0 | 131 | 320 |
| OCR + UI-DETR @1600, conf 0.3 | 87.7 | 62.4 | 88.3 | 83 | 320 |
| OCR + GPA @640, conf 0.05 | 86.7 | 59.6 | *98.8* | 82 | 50 |

### What this shows

- **GPA stays the pick.** At the same list length (~82 options) OCR + GPA scores 86.7 on held-out Mac screens vs
  87.7 for OCR + UI-DETR, at 50 ms vs 320 ms. UI-DETR's lead at conf 0.15 comes from listing 105-130 options, most
  of them noise (precision 39% vs 49-54% for GPA).
- **OCR is worth adding**: +3-4 points over a detector alone at little cost. Fast OCR is enough; accurate OCR is 5x
  slower (250 ms) and adds about 1 point when combined.
- **Two detectors don't help.** Every two-detector "fill" combination scores below OCR + GPA alone.
- **Pixels recover only 55-65% of AX on real Mac UI**, so AX remains the backbone wherever it exists. *Correction
  (2026-11-08): this understates pixels. The sandbox screenshots are missing the window toolbar on ~75% of
  screens and sheets/popups on ~14% (see "Sandbox dataset" below), so the detector never saw those AX targets.* Gating the
  detector to AX-poor screens (gate10) gives the shortest lists (81 vs 100 options) at equal AX recall, but whether
  the detector's extra boxes are real targets AX missed can only be settled with hand labels.
- Decision: **AX + fast OCR (≤6 words) + GPA @640** (union, conf 0.05-0.15). About 120-130 ms on an M5 with the
  parts in parallel.

## Sandbox dataset (2026-11-08)

Three EC2 Macs ran the collector for 20 h each (`scripts/sandbox/collect.py`): 33,290 screenshots. After
`scripts/data/sandbox_dataset.py` (clean, split, sources, visible):

- **19,898 screens, 115 apps/sites** (40 websites); 13,258 near-duplicates and 134 blank frames dropped. Light and
  dark mode, four window sizes. 1,481 screens are AX-poor (< 8 labelled AX elements).
- **Split by app, frozen** (`data/sandbox_ds/splits.json`): train 80 apps / 14,291 screens, val 12 / 2,082, test 23 / 3,525.
- **Option list per screen** (AX + fast OCR + GPA@640): on average 61 AX, 13 OCR-only and 7 detector-only options;
  OCR 24 ms and GPA 23 ms median on the laptop.
- **Capture gap.** The collector's window capture left the toolbar strip blank on 14,936 screens (75%; Safari's
  tab bar and toolbar, app toolbars) and missed sheets, popovers, autocomplete lists and open menus, which are
  separate windows, on ~2,800 screens. The AX tree still lists those controls, so AX targets stay valid, but
  each AX option now carries `visible` and screens with more than 30% hidden carry `capture_mismatch`. Worth
  checking whether Kev's runtime capture has the same gap on macOS 26.
- **Commands (Claude drafts)** on 200 screens (100 train, 100 test, 103 apps), from `scripts/data/sandbox_label.py`:
  2,674 commands and 893 icon captions; 20% of commands target a detector-only icon, 10% are absent. The test
  drafts still need a human pass.
