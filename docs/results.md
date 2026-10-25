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

### Takeaways

- **AX + fast OCR in parallel is the default.** It gives up to 36% more options with no extra wait.
- **The detector costs more than Kev itself.** Only run it when the AX list looks thin, as on Calendar and Canvas.
- **Captions are weak on brand logos**, so icon text from the captioner can't be trusted alone.
- **claude.ai gives about 1 AX option per screen.** This needs investigating, possibly because Chromium didn't build the AX tree for the off-screen window.
