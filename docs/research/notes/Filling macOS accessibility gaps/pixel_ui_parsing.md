# Pixel-only UI element detection and screen parsing (2024-2026): replacing or improving OmniParser v2

Context: the agent merges macOS AX tree + Apple Vision OCR + OmniParser v2 (YOLO icon detector ~270 ms + Florence-2 captioner ~200 ms per screen on MPS), against ~170 ms for the decision model.

## Q1. What succeeded or competes with OmniParser v2?

### Takeaway
There is no "OmniParser v3". Microsoft's only post-v2 change is a new MIT-licensed YOLOv9-E detector (icon_detect_v3, July 2026). The real competitors split into three groups. (a) Drop-in class-agnostic or typed detectors: UI-DETR-1 (RF-DETR-M), uitag-yolo11s trained on GroundCUA desktop data, and Salesforce GPA-GUI-Detector. (b) Open-vocabulary or grounding models that answer "where is X" but do not list every element: GUIrilla-See (macOS, Florence-2 based), GroundNext, Ferret-UI Lite, Moondream. (c) Large screen-understanding VLMs (ScreenAI, Ferret-UI 2) that are too slow or unreleased for a per-step loop. For a list-of-options pipeline, group (a) is the realistic replacement, and the evidence says a desktop-trained detector (GroundCUA) is the best bet for macOS.

### Cited Findings

**OmniParser lineage (Microsoft)**
- Changelog: v1 (Oct 2024, interactive region detection plus icon description models); V1.5 (Nov 2024, "finer small-icon detection and interactability prediction"); V2 checkpoints (Feb 2025); OmniTool for a Windows 11 VM (Feb 2025); and, dated 2026/7, "a YOLOv9-E interactive region detector" whose weights live in [HF PR #37](https://huggingface.co/microsoft/OmniParser-v2.0/discussions/37). No version after V2 is named. — [microsoft/OmniParser README](https://github.com/microsoft/OmniParser)
- Licenses: icon_detect_v3 is built on the MIT YOLOv9 implementation. "Earlier Ultralytics-based icon detectors retain their original AGPL license", which matters if the agent ships the v2 `icon_detect` (YOLOv8) weights. Caption models are MIT. The repo sidebar says CC-BY-4.0 while the README badge says MIT, and the repo does not resolve this. — [microsoft/OmniParser](https://github.com/microsoft/OmniParser)
- V2 claims: trained on more interactive-element-detection and icon-caption data; 60% lower latency than V1 "by decreasing the image size of the icon caption model"; better accuracy on small interactable elements; 39.6 average on ScreenSpot-Pro when paired with GPT-4o (the README news item says 39.5). — [MSR blog](https://www.microsoft.com/en-us/research/articles/omniparser-v2-turning-any-llm-into-a-computer-use-agent/)
- Reported V2 latency: ~0.6 s/frame on A100 and ~0.8 s on a single RTX 4090 for the full pipeline (secondary source quoting the model card). — [OmniParser-v2.0 HF README](https://huggingface.co/microsoft/OmniParser-v2.0/blame/31cddfce855d124f1cff8ed8d4f810991db831c5/README.md)
- GUI-360° (Nov 2025) benchmarks OmniParser and OmniParser-v2 as screen-parsing baselines and calls the v1-to-v2 gain "modest but consistent". — [GUI-360° arXiv 2511.04307](https://arxiv.org/pdf/2511.04307)

**Core ML port of OmniParser v2 (directly relevant to Apple Silicon)**
- `1of2/omniparser-v2-coreml` ships icon_detect_v3 (YOLOv9-E) as a 117 MB mlpackage with `detect_640` and `detect_1280` functions. The Florence-2-base captioner is split into an encoder (270 MB) and decoder (192 MB) plus a 79 MB memory-mapped embedding table. The card claims "every executable op in all three graphs is planned for the Neural Engine (zero CPU or GPU fallbacks)". Requires macOS 15+. MIT. **No latency numbers are published.** Postprocessing: conf 0.05, NMS IoU 0.1, max 300 detections, merge boxes above IoU 0.7. Captions use 64×64 crops upscaled to 768×768 and greedy decoding over a 20-token prefix. — [HF: 1of2/omniparser-v2-coreml](https://huggingface.co/1of2/omniparser-v2-coreml)
- The card says detection "is strongest on desktop and web UIs and weaker on unusual custom-drawn interfaces", and that the models do not read on-screen text, so OCR is still needed. — [same](https://huggingface.co/1of2/omniparser-v2-coreml)

**UI-DETR-1 (racineai, ~2025)**
- RF-DETR-Medium fine-tuned as a class-agnostic "clickable element" detector on 3,550 screenshots merged from six Roboflow datasets. MIT. Outputs boxes and confidence only. — [HF: racineai/UI-DETR-1](https://huggingface.co/racineai/UI-DETR-1)
- WebClick (1,639 samples, Gemini 2.5 Pro as the decider): 70.8% vs 58.8% for OmniParser with tuned thresholds (40.7% at OmniParser defaults). Averages 82.3 elements per image vs 50.6 for OmniParser. Latency ~0.82 s vs 0.77 s for OmniParser, hardware not stated. Self-reported, web-only benchmark. — [same](https://huggingface.co/racineai/UI-DETR-1)

**uitag-yolo11s-ui-detect-v1 (laywens / ComputerUseAgents, Dec 2025), a desktop-trained YOLO**
- YOLO11s fine-tuned on ServiceNow GroundCUA (~55K desktop screenshots, 3.56M human-verified boxes) using 640×640 tiles with 20% overlap at both train and inference time. "Full-resolution inference yields very few detections." 18 MB weights, MIT, gated download. — [HF: laywens/uitag-yolo11s-ui-detect-v1](https://huggingface.co/laywens/uitag-yolo11s-ui-detect-v1)
- 9 types: Button, Menu, Input_Elements, Navigation, Information_Display, Sidebar, Visual_Elements, Others, Unknown. No captions, no OCR. — [same](https://huggingface.co/laywens/uitag-yolo11s-ui-detect-v1)
- Metrics: GroundCUA in-distribution (500 images, 30K GT boxes) R@0.5 94.0%, P@0.5 83.6%; UI-Vision (1,181 images) R@0.5 83.5%; val mAP50 0.792, mAP50-95 0.540. ScreenSpot-Pro center-hit coverage: 80.1% alone, 57.3% for Apple Vision alone, 90.8% combined (icons 42.5% → 87.6%). The card does not compare against OmniParser. — [same](https://huggingface.co/laywens/uitag-yolo11s-ui-detect-v1)
- The authors say it is not intended for mobile or web-only screenshots. — [search summary of model card](https://huggingface.co/laywens/uitag-yolo11s-ui-detect-v1)

**GPA-GUI-Detector (Salesforce, Apr 2026)**
- A YOLO-based interactive-element detector "finetuned from the OmniParser ecosystem", MIT, loaded with Ultralytics. It is the first stage of GPA (GUI Process Automation from one demo), which reports 100% success and roughly 10× faster runs than a Gemini computer-use baseline in a pilot study. Its model card compares it qualitatively against OmniParser V2 on a ScreenSpot-Pro screenshot. I found no quantitative detection metrics. — [GPA paper arXiv 2604.01676](https://arxiv.org/pdf/2604.01676); [alphaXiv summary](https://www.alphaxiv.org/abs/2604.01676)
- Community tooling uses it as a `detect_icons` tool that returns boxes without labels, paired with OCR. — [openskillindex gui-observe](https://openskillindex.com/skills/openclaw-skills-gui-observe)

**GUIrilla / GUIrilla-See (MacPaw, Oct 2025): macOS-specific**
- A macOS desktop exploration framework that crawls apps through the accessibility API (`macapptree`). It yields 27,171 tasks across 1,108 apps from ~4.2K unique screens, plus GUIrilla-Gold with ~1.28K human-verified tasks. — [arXiv 2510.16051](https://arxiv.org/html/2510.16051v2); [code](https://github.com/MacPaw/GUIrilla); [macapptree](https://github.com/MacPaw/macapptree/); [Gold dataset](https://huggingface.co/datasets/GUIrilla/GUIrilla-Gold/)
- GUIrilla-See 0.7B is Florence-2-Large with LoRA r=8, trained for **open-vocabulary grounding** (query in, box or polygon out). It is not an exhaustive detector, and no detection P/R or latency is reported. Grounding results:

  | Model | SS-v2 | SS-Pro | SS-Pro macOS |
  |---|---|---|---|
  | See-0.7B | 53.6 | 7.3 | 8.0 |
  | See-3B | 89.5 | 29.4 | 32.6 |
  | See-7B | 94.7 | 35.4 | 41.4 |
  | Florence-2 0.7B base | 1.8 | 0.1 | 0.2 |
  | UI-TARS-7B | 91.6 | 35.7 | 27.7 |

  — [arXiv 2510.16051](https://arxiv.org/html/2510.16051v2)
- The paper is CC BY-NC-SA 4.0. The license of the model weights is not stated in what I read. — [same](https://arxiv.org/html/2510.16051v2)

**GroundCUA / GroundNext (ServiceNow, ICLR 2026)**
- A desktop dataset from expert human demos: 87 apps in 12 categories, ~56K screenshots, 3.56M element annotations, public on HF. GroundNext-7B (Qwen2.5-VL-7B, SFT on 700K plus RL) reports 52.9 on ScreenSpot-Pro, 67.7 on OSWorld-G, and 60.3 on UI-Vision. It is a grounding model, not a parser, but the dataset is the best public source for training a desktop detector. — [arXiv 2511.07332](https://arxiv.org/abs/2511.07332); [project](https://groundcua.github.io); [HF model](https://huggingface.co/ServiceNow/GroundNext-7B-V0); [ServiceNow ICLR 2026](https://www.servicenow.com/research/publication/aarash-feizi-grou-iclr2026.html)

**Apple: Ferret-UI family**
- Ferret-UI 2 (ICLR 2025) is a multimodal LLM for universal UI understanding across iPhone, Android, iPad, web, and AppleTV. **macOS is not among them.** — [arXiv 2410.18967](https://arxiv.org/abs/2410.18967v2); [Apple ML](https://machinelearning.apple.com/research/ferret-ui-2)
- Ferret-UI Lite (Sep 2025) is a 3B end-to-end on-device GUI agent covering mobile, web, and desktop. It reports grounding of 91.6 on ScreenSpot-V2, 53.3 on ScreenSpot-Pro, and 61.2 on OSWorld-G, with navigation success of 28.0% on AndroidWorld and 19.8% on OSWorld. It is an agent or grounder, not a parser that lists every element. — [arXiv 2509.26539](https://www.arxiv.org/pdf/2509.26539); [InfoQ](https://www.infoq.com/news/2026/02/apple-ferret-ui-lite-on-device)

**Google ScreenAI (background, IJCAI 2024)**
- A 5B PaLI-plus-pix2struct VLM. Its core pretraining task is "screen annotation": the type, location, and description of UI elements. Pictograms go through a separate icon classifier with 77 icon types. Weights are not released as far as I found. Too large for a ~200 ms budget. — [Google Research blog](https://research.google/blog/screenai-a-visual-language-model-for-ui-and-visually-situated-language-understanding/); [IJCAI 2024](https://ijcai.org/proceedings/2024/0339.pdf)

**Apple Screen Recognition (background, CHI 2021)**
- A pixels-only iOS UI detector that generates VoiceOver metadata. Trained on 77,637 screens from 4,068 iPhone apps. Reports 71.3% mAP (82.7% frequency-weighted) over 13 classes on 5,002 test screens. It also predicts clickability and grouping, and shipped in iOS 14. — [Apple ML research](https://machinelearning.apple.com/research/mobile-applications-accessible); [arXiv 2101.04893](https://ar5iv.labs.arxiv.org/html/2101.04893); [CMU news](https://www.cmu.edu/news/stories/archives/2021/june/team-improves-app-accessibility.html)

**Small VLMs (Moondream)**
- Moondream 3 Preview (Dec 2025) claims fully MLX-native, quantized Mac inference, with point and detect "near instantaneous" because each coordinate is one token. This is a vendor claim with no numbers. — [Moondream blog](https://moondream.ai/blog/moondream-station-m3-preview)
- An independent Mac Studio test (May 2026) measured moondream2 (1.93B) at ~1.8-2.4 s per point or detect call on MPS. The same author said the newer local runtime was not usable from a plain pip install on Apple Silicon at the time. — [nipunbatra blog](https://nipunbatra.github.io/blog/posts/2026-05-02-moondream-photon/moondream.html)

### Inferences
- The likely lowest-risk upgrade is to keep the OmniParser structure but swap the YOLOv8 `icon_detect` for icon_detect_v3 run through the Core ML/ANE port. This also removes the AGPL exposure of the older Ultralytics weights. An alternative is uitag-yolo11s (18 MB, desktop-trained, typed classes), though tiling at 640 multiplies inference calls (a 2880×1800 Retina screen at 20% overlap is roughly 6×4 = 24 tiles), so it may be slower than one 1280 pass despite the small model. That would need measuring.
- GUIrilla-See, GroundNext, Ferret-UI Lite, and Moondream answer "find X" questions. They do not produce an exhaustive option list, so they fit as a fallback grounder for an option the agent decided on, not as a parser.
- ScreenAI and Ferret-UI 2 are background only: no released weights I could confirm, and too big for the latency budget.

### Gaps
- No Apple Silicon latency numbers were published for any of these detectors, including the Core ML OmniParser port. Benchmark them locally.
- I found no quantitative P/R for GPA-GUI-Detector, and no head-to-head of uitag-yolo11s vs OmniParser.
- Whether Ferret-UI 2 or Ferret-UI Lite weights are public, and under what license, is unconfirmed.
- I did not verify PaddleOCR layout models (PP-DocLayout and similar) or SAM-based UI segmentation in this pass. I found no 2024-2026 evidence that either is competitive for GUI element detection, but I did not search deeply.

## Q2. Outputs, metrics, size, latency, license, Apple Silicon support (comparison table)

### Takeaway
Numbers are not comparable across rows: they use different benchmarks (WebClick agent accuracy, GroundCUA box P/R, ScreenSpot center-hit coverage, ScreenSpot grounding accuracy). Only uitag-yolo11s reports box-level P/R on desktop data. No row has trustworthy Apple Silicon latency.

### Cited Findings

| System | Output | Reported quality | Size | Latency (reported) | License | Apple Silicon |
|---|---|---|---|---|---|---|
| OmniParser v2 (YOLOv8 icon_detect + Florence-2-base caption) | boxes, `interactivity` flag, OCR text or caption | +GPT-4o 39.6 SS-Pro ([MSR](https://www.microsoft.com/en-us/research/articles/omniparser-v2-turning-any-llm-into-a-computer-use-agent/)) | Florence-2-base ~0.23B | 0.6 s A100 / 0.8 s 4090 full pipeline ([HF](https://huggingface.co/microsoft/OmniParser-v2.0/blame/31cddfce855d124f1cff8ed8d4f810991db831c5/README.md)) | detector AGPL (Ultralytics), caption MIT ([repo](https://github.com/microsoft/OmniParser)) | MPS (user's setup) |
| OmniParser icon_detect_v3 (YOLOv9-E, 2026/7) | boxes | none published | 117 MB Core ML | none published | MIT ([repo](https://github.com/microsoft/OmniParser)) | Core ML, all-ANE claim ([HF](https://huggingface.co/1of2/omniparser-v2-coreml)) |
| UI-DETR-1 (RF-DETR-M) | class-agnostic boxes + conf | WebClick 70.8 vs OmniParser 58.8 | not stated | 0.82 s vs 0.77 s OmniParser, hardware unknown | MIT ([HF](https://huggingface.co/racineai/UI-DETR-1)) | not stated |
| uitag-yolo11s (GroundCUA) | boxes + 9 types | GroundCUA R 94.0 / P 83.6; UI-Vision R 83.5 | 18 MB | ~2.2 s tiled YOLO step inside uitag, Mac unspecified ([PyPI](https://pypi.org/project/uitag/)) | MIT, gated ([HF](https://huggingface.co/laywens/uitag-yolo11s-ui-detect-v1)) | has `coreml` extra ([PyPI](https://pypi.org/project/uitag/)) |
| GPA-GUI-Detector | boxes, no labels | qualitative only | YOLO | not stated | MIT ([paper](https://arxiv.org/pdf/2604.01676)) | Ultralytics (MPS possible) |
| GUIrilla-See 0.7B/3B/7B | box or polygon for a query (grounding) | SS-Pro macOS 8.0 / 32.6 / 41.4 | 0.7-7B | not reported | paper CC BY-NC-SA; weights unclear ([arXiv](https://arxiv.org/html/2510.16051v2)) | not stated |
| Ferret-UI Lite | grounding + agent actions | SS-v2 91.6, SS-Pro 53.3 | 3B | not reported | unclear ([arXiv](https://www.arxiv.org/pdf/2509.26539)) | built for on-device |
| ScreenAI (background) | element type + box + description | SOTA on WebSRC, MoTIF, Widget Captioning (2024) | 5B | n/a | weights not found ([Google](https://research.google/blog/screenai-a-visual-language-model-for-ui-and-visually-situated-language-understanding/)) | no |
| Screen Recognition (background, iOS) | 13 classes, clickability, grouping | 71.3 mAP | not stated | on-device iOS 14 | proprietary ([Apple](https://machinelearning.apple.com/research/mobile-applications-accessible)) | ships in iOS only |
| moondream2 | point / detect for a query | SS 80.4 F1@0.5 (unverified secondary) | 1.93B | ~1.8-2.4 s/call on MPS ([blog](https://nipunbatra.github.io/blog/posts/2026-05-02-moondream-photon/moondream.html)) | Apache-2.0 (from memory, unverified) | MPS; MLX in community bindings |

### Inferences
- In the user's own numbers, the detector (~270 ms on MPS) is already far faster than most of these reported figures. Those figures mostly include captioning or tiling, or come from unknown hardware. Treat any claimed "faster" alternative skeptically until it is benchmarked on the same Mac.
- Moving from MPS to Core ML/ANE is the most plausible way to get a large detector speedup, but nobody has published a number for it.

### Gaps
- Parameter counts for UI-DETR-1 and GPA-GUI-Detector, and a confirmed moondream license, are not verified here.

## Q3. Which approaches label interactability (clickable vs decoration)?

### Takeaway
Few models output an explicit interactable flag. OmniParser's "interactivity" field is effectively source-based: YOLO icon boxes are True and OCR-only text is False. Detectors trained on interactable-only labels (OmniParser, UI-DETR-1, GPA) encode interactability implicitly by what they detect at all. Screen Recognition (iOS, 2021) is the clearest work that predicts clickability explicitly. Typed detectors (uitag, 9 classes) give a proxy, since Information_Display and Visual_Elements are likely non-actionable.

### Cited Findings
- OmniParser V1.5 added "prediction of whether each screen element is interactable or not". — [OmniParser README](https://github.com/microsoft/OmniParser)
- In OmniParser's code, surviving YOLO boxes are emitted with `'interactivity': True` and OCR entries with `'interactivity': False`. Any OCR text inside an icon box is absorbed into that icon's `content`. — [util/utils.py](https://raw.githubusercontent.com/microsoft/OmniParser/master/util/utils.py)
- UI-DETR-1 is described as finding "any clickable element" (class-agnostic). — [HF](https://huggingface.co/racineai/UI-DETR-1)
- GPA-GUI-Detector targets "interactive UI elements (icons, buttons, etc.)". — [GPA paper](https://arxiv.org/pdf/2604.01676)
- uitag-yolo11s outputs types (Button, Menu, Input_Elements, Navigation, Information_Display, Sidebar, Visual_Elements, …), not a binary flag. — [HF](https://huggingface.co/laywens/uitag-yolo11s-ui-detect-v1)
- Screen Recognition predicts whether elements are clickable, alongside 13 element classes and grouping. — [Apple ML](https://machinelearning.apple.com/research/mobile-applications-accessible)
- ScreenAI annotates element type (button, icon type from 77 classes, etc.), from which interactability can be inferred. — [Google blog](https://research.google/blog/screenai-a-visual-language-model-for-ui-and-visually-situated-language-understanding/)

### Inferences
- For this agent, the AX tree already provides roles and actions (AXPress). Training or calibrating a small pixel detector's interactable head on AX-labeled macOS screens (as GUIrilla does with macapptree) is a natural way to get a macOS-specific clickable flag for apps where AX is thin.
- OmniParser's interactivity flag should not be trusted as a real clickability classifier. It reflects which source a box came from.

### Gaps
- No 2024-2026 paper I found reports precision or recall specifically for interactable-vs-decorative classification on desktop.

## Q4. Tricks that reduce captioning cost

### Takeaway
OmniParser already does two of the main tricks: batching crops (128 per batch, 64×64) and captioning only boxes that have no OCR text. The other levers are a smaller caption input (the source of V2's 60% speedup), caching captions by crop hash across frames, skipping captions for boxes the AX tree already labels, running Florence-2 on the ANE through Core ML, or replacing captions with a type label from a typed detector.

### Cited Findings
- `get_parsed_content_icon` defaults to `batch_size=128` ("128 roughly takes 4 GB of GPU memory for florence v2") and resizes each crop to 64×64. — [utils.py](https://raw.githubusercontent.com/microsoft/OmniParser/master/util/utils.py)
- Only boxes without OCR content are captioned: OCR-labeled boxes are sorted first and captioning starts at `starting_idx`. The source has an edge-case bug: if `starting_idx` is 0, every box is captioned, and if it is -1 (no box lacks content), the slice still captions the last box. — [utils.py](https://raw.githubusercontent.com/microsoft/OmniParser/master/util/utils.py)
- V2's 60% latency reduction came from shrinking the caption model's input image size. — [MSR blog](https://www.microsoft.com/en-us/research/articles/omniparser-v2-turning-any-llm-into-a-computer-use-agent/)
- The Core ML port uses greedy decoding over a fixed 20-token prefix with no repeated 3-grams, and puts the encoder and decoder on the ANE. — [HF Core ML port](https://huggingface.co/1of2/omniparser-v2-coreml)
- uitag skips captioning by default. Its optional VLM step (MAI-UI-2B in MLX format) only classifies non-text detections by element type and raises total time from ~3 s to ~15 s. Vision text is given priority over other sources on overlap. — [uitag PyPI](https://pypi.org/project/uitag/)
- Moondream point and detect use one token per coordinate (vendor claim of speed). — [Moondream](https://moondream.ai/blog/moondream-station-m3-preview)

### Inferences
- Skip captions for boxes with high IoU against an AX element that has a title, label, or description. On AX-rich apps this likely removes most caption work. This is the biggest win, and it is specific to this pipeline.
- Cache by perceptual hash of the crop, plus size. Icons such as toolbar buttons repeat across frames and apps, so after warm-up most captions become lookups.
- Cap new captions per frame, prioritizing boxes near the cursor or focus, and return "unlabeled icon" for the rest.
- Check the `starting_idx` edge cases in the local fork. When every box has OCR text, the upstream code still captions one box, which wastes a call.

### Gaps
- I found no published measurements of caption caching hit rates or AX-guided caption skipping.

## Q5. Desktop/macOS-specific work and transfer from web/mobile

### Takeaway
Desktop-specific data arrived in 2025: GroundCUA (87 desktop apps, 3.56M boxes), UI-Vision (83 desktop apps), and GUIrilla (macOS-only, 1,108 apps, AX-derived). Evidence that web- and mobile-trained models transfer poorly to desktop or macOS is indirect but consistent. Grounding models score much lower on ScreenSpot-Pro (professional desktop apps), and UI-TARS-7B drops from 35.7 overall to 27.7 on the macOS subset. Apple Vision OCR alone misses most icons on pro apps. A detector trained on desktop data (uitag-yolo11s) reaches 83.5% recall on the unseen UI-Vision set.

### Cited Findings
- UI-Vision is a "desktop-centric" offline benchmark covering 83 applications with dense element annotations, permissively licensed. — [arXiv 2503.15661](https://arxiv.org/html/2503.15661v2)
- GroundCUA covers 87 desktop apps and ~56K screenshots. GroundNext is trained only on desktop data yet reportedly generalizes to mobile and web. — [arXiv 2511.07332](https://arxiv.org/abs/2511.07332); [HF](https://huggingface.co/ServiceNow/GroundNext-7B-V0)
- GUIrilla is macOS-only and collects data automatically through the accessibility API. Its macOS-specific SS-Pro numbers show models trained on general data lag on macOS: UI-TARS-7B scores 27.7 on SS-Pro macOS vs 35.7 overall, UI-TARS-2B 15.4 vs 27.7, and UGround-7B 12.3 vs 16.5. GUIrilla-See-7B scores 41.4 on macOS vs 35.4 overall. — [arXiv 2510.16051](https://arxiv.org/html/2510.16051v2)
- On ScreenSpot-Pro, Apple Vision alone covers 57.3% of targets (42.5% of icons). Adding the GroundCUA-trained YOLO raises this to 90.8% (87.6% of icons). Vision-only text coverage on the 604 macOS screenshots is 71.1%. The weakest apps for the combined pipeline are AutoCAD (68%) and FL Studio (79%). — [HF uitag model](https://huggingface.co/laywens/uitag-yolo11s-ui-detect-v1); [uitag PyPI](https://pypi.org/project/uitag/)
- The Core ML OmniParser card notes weaker detection on "unusual custom-drawn interfaces". — [HF](https://huggingface.co/1of2/omniparser-v2-coreml)
- Ferret-UI 2 covers iPhone, iPad, Android, web, and AppleTV, but not macOS. Ferret-UI Lite adds desktop. — [Apple ML Ferret-UI 2](https://machinelearning.apple.com/research/ferret-ui-2); [arXiv 2509.26539](https://www.arxiv.org/pdf/2509.26539)
- WinSpot (ACL 2025) is a Windows-specific grounding benchmark. WorldGUI (arXiv 2502.08047) is an interactive desktop benchmark. — [search summary listing](https://github.com/Khang-9966/Computer-Browser-Phone-Use-Agent-Datasets); [WorldGUI](https://arxiv.org/pdf/2502.08047)

### Inferences
- The agent can generate its own macOS detector training data for free: AX-tree boxes with roles and actions on AX-rich apps serve as labels (the GUIrilla and macapptree approach). Fine-tuning OmniParser's YOLO or YOLO11s on that data, plus GroundCUA, would target exactly the macOS gap. The AX-poor apps (Electron, games, canvas UIs) are where the pixel detector matters, so evaluation should weight those.
- The ScreenSpot-Pro "coverage" numbers from uitag are center-hit rates for target elements, not box recall over all elements. They overstate how complete an option list would be.

### Gaps
- No paper reports OmniParser v2 box recall specifically on macOS screenshots.
- No public macOS-only box-level detection benchmark exists. GUIrilla reports grounding only, and UI-Vision has no per-OS split that I found.
- The Mac model behind uitag's timings is not stated.
