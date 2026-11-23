# Option-list quality for GUI agents: benchmarks, metrics, and the macOS accessibility-gap research landscape

Scope: how to measure whether a perception pipeline's actionable-element list contains the target, how that affects downstream success, and whether filling macOS accessibility (AX) gaps is an open problem worth a project. Focus is 2024-2026. Unless marked otherwise, numbers come from primary arXiv pages; secondary-source numbers are flagged.

## Q1. Are there benchmarks or metrics for candidate recall (target coverage), interactability detection, or AX-tree completeness?

### Takeaway
No standard desktop benchmark scores "does the element list contain the target" (candidate recall@K) directly. The closest precedent is web-only: Mind2Web's candidate-ranking Recall@K. On desktop, the field measures either end-to-end grounding (ScreenSpot, ScreenSpot-Pro, OSWorld-G, UI-Vision), detector mAP or F1 against developer AX labels (Screen2AX), or task success under different observation pipelines (Windows Agent Arena, Screen2AX-Task, OSWorld). A desktop "target coverage" metric tied to agent success is still missing, and that is an opening.

### Cited Findings
- **Mind2Web (web, 2023) defines candidate generation as ranking.** A small LM ranks DOM elements and keeps the top-k (k=50) for a larger LM, and the ranker is scored with Recall@K. Third-party summaries put MindAct DeBERTa ranker Recall@50 at roughly 85-89% across test splits. These numbers are unverified against the paper. — [Mind2Web paper](https://arxiv.org/pdf/2306.06070); [emergentmind summary](https://www.emergentmind.com/topics/mind2web-benchmark)
- **Leaderboard figures from Dual-VCR on Mind2Web element ranking:** Recall@50 90.1, Recall@10 79.7, Recall@1 38.4. — [Dual-View Visual Contextualization](https://arxiv.org/pdf/2402.04476); [leaderboard](https://www.wizwand.com/sota/element-ranking-on-mind2web-test)
- **Prune4Web (2025) on HTML pruning:** existing methods "either rely on overly simplistic heuristic filtering or require separate language models for element-ranking". Candidate-set construction is still treated as an open subproblem on the web. — [Prune4Web](https://arxiv.org/html/2511.21398v1)
- **OSWorld-G (2025):** 564 annotated grounding samples across 32 UI element types, taken from OSWorld rollouts. Categories are text matching, element recognition, layout understanding, fine-grained manipulation, and refusal (target not on screen, e.g. "open Firefox" when Firefox is not visible). It ships with the 4M-example JEDI grounding dataset. — [OSWorld-G on alphaxiv](https://www.alphaxiv.org/benchmarks/the-university-of-hong-kong/osworld-g); [search summary of JEDI/OSWorld-G](https://arxiv.org/pdf/2505.13227)
- **GTA1-7B scores 67.7% on OSWorld-G**, which gives a reference grounding level for 2025 models. — [GTA1](https://web3.arxiv.org/pdf/2507.05791)
- **UI-Vision (ICML 2025, ServiceNow/Mila):** a license-permissive desktop benchmark covering 83 open-source apps in 6 domains. Tasks are Element Grounding (basic, functional, spatial), Layout Grounding, and Action Prediction. A secondary source says the best model (UI-TARS-72B) reached only 25.5% on element grounding (unverified). Data is at HF `ServiceNow/ui-vision`. — [paper list entry](https://osu-nlp-group.github.io/GUI-Agents-Paper-List/papers/arxiv-2503-15661); [HF dataset](https://huggingface.co/datasets/ServiceNow/ui-vision)
- **Screen2AX (2025) scores AX-tree reconstruction against developer AX labels on macOS.** All-YOLOv11 element detection reaches P 49.1 / R 43.4 / F1 43.8, mAP50 46.6, at 0.204 s on an M1 Pro. Hierarchy reaches edge-F1 77% (heuristics 67%), group IoU 55%, leaves-F1 31%, and graph edit distance 200. Per-class AP50 runs from AXLink 60.6 and AXButton 57.2 down to AXTextArea 27.5. — [Screen2AX HTML](https://arxiv.org/html/2507.16704)
- **OmniParser (Microsoft, 2024) trained an "interactable region" detector on 67k web screenshots**, with boxes taken from DOM interactable elements on 100k popular URLs. The dataset has no desktop-native or mobile data. The fine-tuned detector added 4.3% accuracy on ScreenSpot over raw Grounding DINO. — [OmniParser paper](https://arxiv.org/pdf/2408.00203); [project page](https://microsoft.github.io/OmniParser); [secondary note on missing desktop data](https://www.emergentmind.com/topics/interactable-region-detection-dataset)
- **Interactive GUI Grounding (IGG, ICML 2026)** covers targets that cannot be localized or acted on from the first screenshot, so the agent must interact to expose the target. This is next to abstention but is a different setting. — [IGG ICML 2026](https://icml.cc/virtual/2026/69259)

### Inferences
- The Mind2Web Recall@K framing (does the top-K candidate list contain the gold element?) carries over directly to desktop option lists. No published desktop benchmark appears to report it. Gold targets from OSWorld-G, ScreenSpot-Pro (macOS subset), UI-Vision or GUIrilla-Gold could be matched (by IoU or center-in-box) against a pipeline's element list to compute desktop target-coverage@K. This is feasible with existing data.
- Detector mAP against developer AX labels (Screen2AX) mismeasures completeness. The labels are themselves incomplete (see Q3), so missing-in-AX elements count as false positives, not as wins. A completeness metric needs human-verified "all interactable elements" annotations.
- OSWorld-G's refusal category is the only standard abstention test found. Option-list pipelines also need a "target absent from list" signal, which is a different question from "target absent from screen".

### Gaps
- I did not check the exact metric definitions in ScreenSpot-Pro or OS-Atlas, or their macOS sample counts.
- I found no benchmark that labels interactability (clickable vs. decorative) for desktop-native apps. OmniParser's labels are web-only.
- I did not confirm whether VisualWebArena reports SoM-candidate coverage. Not researched.

## Q2. What macOS-specific datasets exist with screenshots plus AX trees?

### Takeaway
MacPaw (Kyiv) plus Ukrainian Catholic University is the main source of macOS AX-tree data: Screen2AX datasets, GUIrilla-Task and Gold, MacApp Trees, and the MacArena benchmark. OpenCUA AgentNet has about 5K macOS trajectories with AX trees recorded. macOSWorld is screenshot-only. Interactive macOS benchmarks are now well established, but none ships human-completed element lists.

### Cited Findings
- **Screen2AX datasets (112 macOS apps):** Screen2AX-Tree (1,127 screenshots, 52 AX classes), Screen2AX-Element (986 images, 44,150 annotations, 5 classes; the Fig. 4 class counts sum to about 40.6K, an internal inconsistency), Screen2AX-Group (808 images, 30,774 annotations), and Screen2AX-Task (435 images, 5,934 command rows, 166 held-out eval images). Labels come from developer AX data. — [Screen2AX HTML](https://arxiv.org/html/2507.16704)
- **GUIrilla (MacPaw, Oct 2025):** the crawler ran on 12,298 macOS apps, and only 1,108 supported AX well enough to produce valid trees. GUIrilla-Task has 27,171 tasks over 1,108 apps and about 4.2K screens (train 881 apps, test 227 apps). MacApp Trees is 561 GB compressed. GUIrilla-Gold has about 1.3K human-verified tasks (1,283 vs. 1,319, inconsistent within the paper), with a human baseline of 90.26%. Code is at github.com/MacPaw/GUIrilla and github.com/MacPaw/macapptree. — [GUIrilla](https://arxiv.org/html/2510.16051v2); [HF GUIrilla-Trees](https://huggingface.co/datasets/macpaw-research/GUIrilla-Trees)
- **OpenCUA AgentNet (XLANG, 2025):** about 22.6K human demonstration trajectories (12K Windows, 5K macOS, 5K Ubuntu) across 100+ apps and 200+ sites. AgentNetTool records screen video, input events and accessibility trees. — [OpenCUA site](https://opencua.xlang.ai/); [OpenCUA paper](https://arxiv.org/pdf/2508.09123v2); [VentureBeat](https://venturebeat.com/ai/opencuas-open-source-computer-use-agents-rival-proprietary-models-from-openai-and-anthropic)
- **macOSWorld (NeurIPS 2025):** 202 interactive tasks over 30 apps in 5 languages. Observations are screenshots only, and agents may add their own SoM. AX trees are neither provided nor evaluated. Proprietary CUAs score above 30%, and open lightweight models below 2%. v2 was withdrawn for an experimental-results error, and v4 is current. — [macOSWorld v4](https://arxiv.org/html/2506.04135v4); [abs](https://arxiv.org/abs/2506.04135)
- **MacArena (ICML 2026 workshop, MacPaw):** 421 tasks over 50 apps on real Apple Silicon. It has 221 OSWorld ports, 151 macOSWorld tasks and 49 new native tasks. The AX tree is an optional observation, but there is no screenshot-vs-AX ablation. Overall success: OpenAI CUA 31.83, Qwen3-VL-4B 24.23, UI-TARS-1.5-7B 21.14, Qwen3-VL-2B 11.40. On the native subset, CUA scores 36.73 and UI-TARS 10.20. Every model scored 3-10 points lower on macOS ports than on Ubuntu OSWorld, and multi-app tasks were near 0%. — [MacArena notes](https://en.papernotes.org/ICML2026/llm_agent/macarena_benchmarking_computer_use_agents_on_an_online_macos_environment/)
- **ScreenSpot** includes macOS and Windows desktop screenshots (600+ screens across platforms). Screen2AX used 172 macOS instructions from it. — [OmniParser summary](https://learnopencv.com/omniparser-vision-based-gui-agent/); [Screen2AX HTML](https://arxiv.org/html/2507.16704)

### Inferences
- The small Kev models (0.8B/4B) are near the "open lightweight model" regime. MacArena's Qwen3-VL-2B/4B numbers (11-24%) are the most comparable public macOS baselines.
- GUIrilla's 1,108-of-12,298 yield is a strong argument that AX-only crawling misses most of the macOS app long tail. That long tail is where a vision-completed element list would matter.

### Gaps
- I could not confirm the size or availability of OS-Atlas macOS data. I did not search for it directly.
- I did not verify whether UI-Vision includes macOS screens. It is described as open-source desktop software, likely mostly Linux and Windows, but this is unverified.
- I did not verify the exact AX schema in AgentNet macOS (full tree vs. focused subtree).

## Q3. How often are AX trees missing or incomplete on desktop, and how much failure is perception/grounding vs. planning?

### Takeaway
On macOS, about 30-36% of apps have "full" AX coverage, and about 18-33% have none. Even in apps that crawl successfully, only 40% of elements carry a correct role plus description. Across desktop agent studies, grounding/execution errors dominate failures (OSWorld >75% of failures involve click inaccuracy; Agent S grounding 53% vs. planning 35%). Adding structured element lists (UIA/AX) on top of vision gives large relative gains on Windows and macOS. Native macOS AX alone does worse than vision parsers.

### Cited Findings
- **Screen2AX survey (human-checked bounding boxes):** in a random 452-app sample (App Store plus Brew Cask), 32.7% had no AX, 37.8% partial and 29.4% full. In the top-99 popular apps, 17.7% had none, 45.9% partial and 36.4% full. "Full" means every on-screen element appears with a correct box. Descriptions were not graded, so these figures are optimistic. The abstract's "33% full" does not match the table's 29.4%. — [Screen2AX HTML](https://arxiv.org/html/2507.16704)
- **GUIrilla AX quality audit:** 64% of screens GOOD, 24% MEDIUM, 12% BAD. Only 40% of elements had a correct role plus description, 49% had role only and 11% were mislabeled. Boxes were accurate for 80% of elements, and 10% extended outside the visible window. Coverage was weak for icon-dense pro and creative tools (Vivado, Illustrator, DaVinci Resolve). — [GUIrilla](https://arxiv.org/html/2510.16051v2)
- **Screen2AX-Task success rate (GPT-4 agent, macOS):** built-in AX 16.9%, OmniParser V2 28.0%, Screen2AX without hierarchy 30.7%, Screen2AX 33.7%. ScreenSpot-macOS (172 instructions): image-only 6.9%, OmniParser V2 31.9%, Screen2AX 36.6%. The abstract's "2.2x" claim works out to about 2.0x from the table. — [Screen2AX HTML](https://arxiv.org/html/2507.16704)
- **OSWorld (2024) input-modality table (success rate):** GPT-4V screenshot 5.26%, screenshot plus a11y 12.17%, SoM 11.77%. GPT-4o: a11y 11.36%, screenshot 5.03%, screenshot plus a11y 11.21%, SoM 4.59%. Gemini-ProV dropped with a11y (5.80% to 3.48%) and SoM (1.06%). Raw a11y XML runs to "millions of tokens", and filtered trees still need about 6K tokens for 90% of observations. Of 550 failures, >75% involved mouse-click inaccuracy. The authors note the a11y tree "may not be well-supported across all software". — [OSWorld HTML](https://arxiv.org/html/2404.07972)
- **Agent S (2024) error analysis on OSWorld failures:** grounding errors 53.06%, planning errors 34.69%, execution errors 79.59% (categories overlap). — [Agent S](https://arxiv.org/pdf/2410.08164)
- **Windows Agent Arena (2024):** the best config (UIA plus OmniParser SoM with GPT-4V) reached 19.5% vs. a human 74.5%. Secondary summaries say adding UIA markers raised OmniParser-based SoM by about 57% relative (52% for open-source detectors, 15% for proprietary pixel models). UIA parsing can slow queries "from seconds to minutes". The relative-gain figures were not checked against Table 4. — [WAA paper](https://arxiv.org/pdf/2409.08264); [chatpaper summary](https://chatpaper.com/paper/58597)
- **UFO2 (Microsoft, 2025) uses hybrid control detection in production.** It runs UIA and OmniParser, merges their outputs, and drops visual boxes with IoU >0.1 against UIA controls. The paper reports hybrid detection raising completion "in cases where UIA alone fails". Exact numbers not verified. — [UFO2 hybrid detection docs](https://microsoft.github.io/UFO/ufo2/core_features/control_detection/hybrid_detection); [UFO2 paper](https://arxiv.org/html/2504.14603v2)
- **macOSWorld:** GPT-4o in English went from 8.8% to 19.9% with OmniParser-v2 SoM. Qualitatively, open research models plan poorly and ground well, while general VLMs plan well and ground poorly. No percentage breakdown is given. — [macOSWorld v4](https://arxiv.org/html/2506.04135v4)
- **A 2026 analysis (arXiv 2607.28367)** proposes assigning each failed trajectory to its earliest decisive failure (Tier 1 planning/spec, Tier 2 grounding) rather than the most visible symptom. This bears on how perception-vs-planning attribution should be measured. — [How Benchmarks Mis-Score CUAs](https://arxiv.org/pdf/2607.28367)
- **GUI-Perturbed (2026)** reports systematic brittleness of grounding models under domain randomization and argues grounding errors compound across steps. — [GUI-Perturbed](https://huggingface.co/papers/2604.14262.md)

### Inferences
- The evidence consistently shows that structured element lists help most when fused with vision, and that native macOS AX alone is weak (16.9% vs. 28-34% for vision parsers on Screen2AX-Task). A hybrid AX-plus-vision list is the empirically supported design. Pure-AX and pure-vision pipelines are both dominated.
- Published failure attributions use different, overlapping taxonomies. None isolates "target absent from candidate list" as its own failure class. That class is exactly what an option-list recall metric would measure.

### Gaps
- I found no measurement of per-element AX miss rate (the fraction of visible interactable elements missing from the AX tree) on macOS. Both papers grade at screen or app level.
- I did not verify the Electron-, canvas- or game-specific AX miss rates. No source quantified them.

## Q4. Who is working on this now, and what is unsolved?

### Takeaway
Active groups include MacPaw/UCU (Screen2AX, GUIrilla, MacArena; the only macOS-AX-focused line), Microsoft (OmniParser, UFO2 hybrid, Windows Agent Arena), XLANG/HKU (OSWorld, OSWorld-G/JEDI, OpenCUA), ServiceNow/Mila (UI-Vision, GroundNext), and Show Lab NUS (macOSWorld). Several problems remain open: fast on-device hybrid parsing (Screen2AX reports about 0.2 s detection plus 0.76 s hierarchy and says it may not be real-time), rare and custom controls in pro or creative apps, interactability labels for desktop-native apps, and abstention when the target is not in the list.

### Cited Findings
- **Screen2AX's stated limitations:** labels inherit inconsistent developer AX, rare and custom elements (Photoshop) are missed, grouping conventions vary, a mobile-trained captioner mis-captions desktop icons, OCR confuses icons with symbols, and speed "may not meet real-time" needs. Future work listed: semantic group classes, desktop icon data, other OSes, and real-time accessibility validation. — [Screen2AX HTML](https://arxiv.org/html/2507.16704)
- **GUIrilla's limitations:** it depends on developer AX quality, supports macOS only, and is weak on icon-dense pro and creative tools. — [GUIrilla](https://arxiv.org/html/2510.16051v2)
- **GUIrilla-See 7B** scores 35.36% on ScreenSpot-Pro (41.39% on its macOS subset), and the 0.7B version scores 53.55% on ScreenSpot-v2. Small models still lag on pro desktop UIs. — [GUIrilla](https://arxiv.org/html/2510.16051v2)
- **Microsoft OmniParser's interactable detector is web-trained (67k DOM-labelled screenshots).** Desktop-native interactability data is absent. — [OmniParser](https://arxiv.org/pdf/2408.00203)
- **UFO2 makes hybrid UIA plus vision the Windows default option.** Windows has a production reference design, and no macOS equivalent was found. — [UFO2 docs](https://microsoft.github.io/UFO/ufo2/core_features/control_detection/overview)
- **The searches found no 2026 paper on repairing (patching) an existing macOS AX tree with vision.** Screen2AX regenerates a tree from pixels and does not merge it with the live AX tree. — [Screen2AX abs](https://arxiv.org/abs/2507.16704); search result, no primary source for the absence

### Inferences
- The specific combination is unclaimed in what I found: a live AX tree plus a vision detector, merged and deduplicated, with a learned "missing-element" repair. It would be on-device and sub-200 ms on Apple Silicon, and evaluated by target-coverage recall plus downstream success on macOS. UFO2 does the merge on Windows without a recall metric. Screen2AX does vision-only on macOS.
- Abstention ("none of the options is the target, so scroll, open a menu or re-perceive") is under-studied. OSWorld-G refusal and IGG are the nearest benchmarks, and neither is framed around a candidate list.

### Gaps
- I did not survey Apple's own work (Ferret-UI, screen recognition/VoiceOver Screen Recognition) in this pass. It is relevant prior art for on-device macOS/iOS element detection.
- I did not check 2025-2026 GUI-agent survey papers for an authoritative open-problems list.

## Q5. What would be a credible contribution?

### Takeaway
The strongest evidence-backed contribution is a measurement-first package for macOS: (1) human-completed element lists for a few hundred Mac screens across native, Electron, canvas and pro apps, (2) a target-coverage recall@K metric with an abstention-aware variant, and (3) a fast hybrid AX-plus-vision parser evaluated against both. A learned AX-repair model is a natural extension. The novelty claim must be positioned against Screen2AX, GUIrilla and UFO2.

### Cited Findings
- **Baselines exist to compare against:** native AX (16.9%), OmniParser V2 (28.0%) and Screen2AX (33.7%) on Screen2AX-Task. Screen2AX reports detector F1 43.8 at 0.2 s on an M1 Pro. — [Screen2AX HTML](https://arxiv.org/html/2507.16704)
- **Recall@K precedent and reference numbers come from Mind2Web element ranking** (Recall@50 about 85-90%). — [Mind2Web](https://arxiv.org/pdf/2306.06070); [Dual-VCR](https://arxiv.org/pdf/2402.04476)
- **Reusable gold targets:** GUIrilla-Gold (about 1.3K human-verified macOS tasks), OSWorld-G (564, with refusal), the ScreenSpot-Pro macOS subset, and MacArena for downstream success. — [GUIrilla](https://arxiv.org/html/2510.16051v2); [OSWorld-G](https://www.alphaxiv.org/benchmarks/the-university-of-hong-kong/osworld-g); [MacArena](https://en.papernotes.org/ICML2026/llm_agent/macarena_benchmarking_computer_use_agents_on_an_online_macos_environment/)

### Inferences (speculative and clearly marked)
- **Dataset:** existing macOS sets label only what developer AX exposes (Screen2AX, GUIrilla). A set where humans add the missing interactable elements would be new. It enables a true completeness metric (element recall vs. human ground truth) rather than agreement with AX. Even 300-1,000 screens weighted toward partial or no-AX apps would be credible, given Screen2AX's eval used 166 images.
- **Metric:** report target-coverage@K, list precision (non-interactable noise), the abstention AUROC for "target not in list", and the correlation of these with MacArena or Screen2AX-Task success. Showing that list recall predicts task success would itself be a publishable finding.
- **Parser:** AX first, with vision filling only regions where AX is empty or suspicious, deduplicated by IoU as in UFO2. The target is under about 100-200 ms on Apple Silicon to beat Screen2AX's reported latency.
- **Risk:** the space is moving fast. MacPaw has a three-paper macOS line, and Screen2AX or GUIrilla follow-ups could claim the hybrid angle. Positioning on measurement (recall metric plus completed lists) is more defensible than claiming "first hybrid parser".

### Gaps
- No source quantifies how much list-recall improvement translates into task-success improvement. This is the core open empirical question, and nothing found answers it.
