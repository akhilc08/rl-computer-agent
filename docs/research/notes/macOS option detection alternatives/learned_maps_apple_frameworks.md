# Learned, cached and temporal option detection + new Apple frameworks (macOS agent)

Context: the options list is built from an AX walk + Apple Vision OCR + OmniParser YOLO + Florence-2 captions (~470 ms for detector+captions, vs ~170 ms decision model). The unattended explorer has saved 2,835 states / 712 screenshots with AX trees and press transitions. Pixel detectors, grounding VLMs and CDP/DOM are covered elsewhere. Items marked "[prior knowledge]" were not re-verified in this session. Items marked "[speculation]" are my inferences.

## 1. Per-app / per-site UI maps and memories built from exploration

### Takeaway
Exploration-built memories mostly store (a) page/screen nodes linked by action transitions and (b) LLM-written descriptions of what each element does. They match a new screen by element identity (resource ID or text) or by screen similarity, and they mainly improve task success and step count. None of them is a pixel-level option detector. For this project the useful pattern is AutoDroid/AppAgentX style: a per-app transition graph plus element-function tables, keyed by a stable element signature, injected as hints. The 2,835-state explorer log already has the right shape for this.

### Cited Findings
- **AutoDroid** (Wen et al.): randomly explores target apps, extracts a **UI transition graph (UTG)**, then asks an LLM to summarise element functions from it. Output is a "simulated task table" (NL task → UI navigation sequence) and a "UI function table". GUI states are turned into simplified HTML-style text. Benchmark: 158 hand-labelled tasks across 13 open-source apps. v1 reports 90.9% action accuracy and 71.3% task success. The memory-injection ablation shows large gains for every LLM tested (exact deltas not captured here) — [arXiv 2308.15272](https://arxiv.org/abs/2308.15272), [v4 HTML](https://arxiv.org/html/2308.15272v4). Venue is reported elsewhere as MobiCom'24, which I did not confirm this session.
- **AppAgentX** (Jiang et al., 2025): keeps history as a chain of **page nodes** joined by transitions, and evolves "shortcut nodes" that replace repeated low-level action sequences. If an identified element links to a shortcut, the agent runs the high-level action; otherwise it falls back to low-level actions. Memory raises success from 16.9% (no memory) to 70.8%, and memory+evolution gives 71.4%. A secondary summary says prompt tokens drop from about 9.26k to 4.94k — [arXiv 2503.02268](https://arxiv.org/abs/2503.02268v2), [alphaXiv overview](https://www.alphaxiv.org/overview/2503.02268). The AndroidWorld 62.5 / DroidTask 88.6 numbers come from an aggregator and are unverified — [wizwand](https://www.wizwand.com/paper/68fa78121e209131cea76a2d).
- **Mobile-Agent-E** (NeurIPS 2025): keeps a persistent long-term memory of **Tips** (general lessons) and **Shortcuts** (executable atomic-op sequences), updated after each task by two "Experience Reflectors". Self-evolution adds +6.5% absolute, and the paper reports +22% absolute over prior SOTA on Mobile-Eval-E across three backbones — [arXiv 2501.11733](https://arxiv.org/html/2501.11733v2). It stores knowledge about procedure, not element positions.
- **Mobile-Agent-v2**: a three-agent design (planning, decision, reflection). My search did not confirm what its memory unit stores — [aitopics summary](https://aitopics.org/doc/arxivorg:FC53F49F).
- **Synapse** (ICLR 2024): stores whole successful trajectories as exemplars and retrieves them by embedding similarity. It also does "state abstraction", which compresses the raw observation to the relevant elements. On Mind2Web the paper reports a 56% relative gain in avg step SR over the previous SOTA prompting method. The GPT-3.5 ablation increments are 32% / 50% / 56%. On CodeLlama-7B it gets about 2.5x MindAct's step SR — [arXiv 2306.07863](https://arxiv.org/pdf/2306.07863), [code](https://github.com/ltzheng/synapse).
- **Agent Workflow Memory** (ICML 2025): induces reusable sub-task workflows from past trajectories, offline or online. It reports +24.6% relative success on Mind2Web and +51.1% on WebArena. A third-party code review says the released code just loads a workflow text file into the prompt — [PMLR](https://proceedings.mlr.press/v267/wang25bx.html), [review](https://zby.github.io/commonplace/agent-memory-systems/reviews/agent-workflow-memory/).
- **AppAgent** (Zhang et al., 2023): an exploration or demo phase writes a per-element "documentation" entry (what tapping it does), keyed by the element's Android resource ID, and reuses it in later tasks [prior knowledge] — [arXiv 2312.13771](https://arxiv.org/abs/2312.13771).
- **PastForward** (UNIST/Meta, 2026) keeps an "interaction memory" of completed transitions keyed by a **perceptual screen hash + action signature**, with a latent action-embedding fallback (details in section 2) — [arXiv 2609.32166](https://arxiv.org/html/2609.32166v1).

### Inferences
- [speculation] The most direct gap-filler: for each (app or site, screen signature), store the union of elements seen across past visits. Use AX elements plus detector/caption output that was confirmed by a press transition, with boxes stored relative to the window or a stable AX anchor. On a revisit, match by screen hash or AX-tree fingerprint and merge the stored elements into the live AX list. That would let you skip Florence-2 for already-known crops. AutoDroid's "UI function table" and AppAgent's per-element docs are the precedent for the description field.
- Reported gains are task success, not element recall. No paper here measures how much a memory fills in missing interactive elements. You would need to measure that on your 712 screenshots.

### Gaps
- No paper found that reports per-site element-map hit rates under real UI drift (A/B layouts, dynamic content) on desktop or web.
- Exact AutoDroid memory-ablation numbers and the final MobiCom figures were not retrieved.

## 2. Frame-to-frame tracking / incremental perception

### Takeaway
macOS already gives you dirty rectangles per frame through ScreenCaptureKit. Two 2026 papers show 2-3.5x per-step speedups from reusing computation across steps, but both reuse VLM computation (KV cache, tokens), not detector output. I found no paper that measures the speedup from re-running a detector or captioner only on changed regions. A crop-hash caption cache is a straightforward engineering win, but nobody has published a measurement of it.

### Cited Findings
- **ScreenCaptureKit `SCStreamFrameInfo.dirtyRects`**: per-frame array of rectangles that "contain changes" (redrawn plus moved regions), as CGRect in pixels, available since macOS 12.3 — [Apple docs](https://developer.apple.com/documentation/screencapturekit/scstreamframeinfo/dirtyrects.md), [objc2 binding](https://docs.rs/objc2-screen-capture-kit/latest/objc2_screen_capture_kit/static.SCStreamFrameInfoDirtyRects.html). I found no documentation on how reliably it is populated.
- **GUIFlow** (Tsinghua/CUHK/JD, 2026): uses **patch-level pixel differences against a rolling reference** to choose which vision-encoder patches to recompute. Unchanged patches keep their cached KV. It also re-injects cached KV for recurring screenshots (with M-RoPE correction) and uses previous responses as speculative drafts. Reported FLOPs reduction is 3.37x (Qwen3-VL-4B) and 3.39x (UI-TARS-1.5-7B) offline, 3.49x / 3.32x online on OSWorld. Per-step latency drops 1.94-2.55x, e.g. GUI-Owl 4363 ms → 1713 ms. OSWorld score is 27.9% vs 24.3% dense. Hardware: RTX 4090, no on-device numbers — [arXiv 2610.34959](https://arxiv.org/html/2610.34959).
- **PastForward** (2026): (1) reuses past token sequences as verified speculative drafts, (2) predicts the next screen from stored transitions and starts inference during the action→observation wait, keeping the result only if the observed screen hash matches, (3) carries KV over for unchanged prefixes and retained screenshots. Edge speedups with Qwen3-VL-2B are 1.63-2.36x (Jetson Orin Nano 2.36x, Galaxy S26 Ultra 1.63x, Xiaomi 15 1.65x) at roughly equal success (21.70% vs 21.84%). Server speedup is 1.40-1.68x. The Agent-X speculative-decoding baseline gets 1.18-1.56x — [arXiv 2609.32166](https://arxiv.org/html/2609.32166v1).
- **ShowUI**: removes about 33% redundant visual tokens through UI-guided token selection, for a 1.4x training speedup. This is within-frame redundancy, not temporal — [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Lin_ShowUI_One_Vision-Language-Action_Model_for_GUI_Visual_Agent_CVPR_2025_paper.pdf), as summarised in the [Efficient GUI Agents survey, arXiv 2609.02309](https://arxiv.org/pdf/2609.02309).
- A CHART-paper component keeps a persistent cache of screenshots and their predicted region boxes so it does not annotate them twice — [arXiv 2610.35237](https://arxiv.org/html/2610.35237).

### Inferences
- [speculation] Applied to the existing pipeline:
  - Run YOLO and Florence-2 only on dirty rects (or on tiles that intersect them).
  - Keep previous-frame detections whose boxes don't intersect any dirty rect.
  - Key Florence-2 captions by a perceptual hash of the crop, so repeated icons across screens and sites hit the cache.
  - Since captions take most of the ~470 ms, a high cache-hit rate on revisited sites could bring the perception step down to the AX-walk plus OCR cost.
  - Measure all of this on the explorer's 712 screenshots, which include revisits.
- PastForward's "predict next screen from transition memory, then pre-compute" also fits: the explorer's press-transition log could precompute options for the likely next screen while the click animates.
- dirtyRects only exists in a live SCStream, not in one-shot screenshot APIs. Scrolling shows up as large moved rects, which invalidate most of the cache.

### Gaps
- No published measurement of detector/caption caching or dirty-rect-gated detection for GUI agents.
- No data on dirtyRects behaviour with browser compositing (for example, whether Chrome/Safari report full-window damage).

## 3. Learning interactability from interaction outcomes

### Takeaway
There is direct precedent: Apple's own "Never-ending UI Learner" (UIST 2023) labels tappability by actually tapping elements and watching whether the screen changes, then trains vision models on those labels. The explorer's press transitions are exactly this signal, so self-supervised fine-tuning of the YOLO detector or a tappability classifier on 2,835 states is a well-precedented next step.

### Cited Findings
- **Never-ending UI Learner** (Wu, Krosnick, Schoop, Swearngin, Bigham, Nichols, UIST '23; Apple/CMU): rather than inferring tappability from appearance or unreliable metadata, the crawler programmatically taps the element and observes the effect. It installed apps from the app store and crawled them for >5,000 device-hours, >500,000 actions, 6,000 apps. It trains three CV models: tappability, draggability and screen similarity. Accuracy and per-iteration gains were not visible in the abstract — [arXiv 2308.08726](https://arxiv.org/abs/2308.08726).
- **WebUI** (Wu et al., CHI 2023): 400k crawled web pages with auto-extracted metadata (accessibility tree, boxes). An FCOS detector pretrained on WebUI predicts element boxes plus multi-hot labels from the AX tree. Web pages average about 20 clickable elements vs fewer on mobile. My search did not confirm a dedicated outcome-based "interactability" label; clickability appears to come from DOM/AX metadata — [arXiv 2301.13280](https://arxiv.org/abs/2301.13280).
- **PixelWeb** (2025) criticises WebUI for pervasive annotation errors, including boxes with no corresponding GUI element. This is a warning about metadata-derived labels, and an argument for outcome-verified ones — [arXiv 2504.16419](https://arxiv.org/html/2504.16419v2).
- The screen-similarity model in the Never-ending learner is also what you need to decide whether "pressing changed the screen" — [arXiv 2308.08726](https://arxiv.org/abs/2308.08726).

### Inferences
- [speculation] Proposed labelling scheme:
  - Positive: a press that causes an AX-tree diff or pixel diff (outside the cursor or focus ring).
  - Negative: a press with no change.
  - Unknown: hover-only effects.
  - Train a small tappability head on crops, or fine-tune OmniParser's YOLO, on these labels.
  - Caveats: delayed effects (network loads), toggles that change only a pixel, and destructive actions. Your overnight-scope safety rules already limit what can be pressed.
- AX-tree diffs give a cleaner change signal than pixel diffs on macOS, because animations and caret blinks add pixel noise.

### Gaps
- The Never-ending learner's quantitative results (precision/recall vs static-label baselines) were not retrieved. Read the full PDF.
- No desktop/macOS replication found.

## 4. Apple frameworks in macOS 15 / macOS 26

### Takeaway
Apple's frameworks have no public "UI element detection" request. The useful pieces are:
- Vision's `RecognizeDocumentsRequest` (WWDC25): structured text, tables, lists, barcodes and data detectors.
- VisionKit `ImageAnalyzer` (Live Text, data detectors, subjects).
- The Foundation Models framework, which in the WWDC26 cycle gained image input and Vision-backed tools (OCR, barcode, saliency), so it could serve as a cheap on-device describer.
- App Intents / `NSUserActivity.appEntityIdentifier`, which exposes onscreen content only for apps that adopt it, and only to Siri / Apple Intelligence.

None of these give latency numbers on Apple Silicon for screenshot workloads. You have to measure them.

### Cited Findings
- **Vision `RecognizeDocumentsRequest`** (WWDC25 session 272): new request that returns a hierarchical document observation (text in 26 languages, tables grouped into rows and cells, lists, barcodes). It uses the new **DataDetection** framework for phone numbers, emails and addresses. Available on iOS, macOS, iPadOS, tvOS and visionOS. It is aimed at documents, not GUIs — [WWDC25 #272](https://developer.apple.com/videos/play/wwdc2025/272/). A third-party wiki says macOS 26+ (unofficial) — [project-aries](https://project-aries-cdl.pages.dev/entities/recognizedocumentsrequest).
- **VisionKit `ImageAnalyzer` / `ImageAnalysis`**: macOS 13+. Subject identification and lifting came in macOS 14 via `ImageAnalysisOverlayView` (AppKit) with `subject(at:)`. Accepts URL, NSImage, CGImage, CIImage and CVPixelBuffer on macOS, so it can run on screenshots. I found no latency data — [Apple VisionKit docs](https://developer.apple.com/documentation/visionkit.md), [overlay view](https://developer.apple.com/documentation/visionkit/imageanalysisoverlayview.md), [visionkit-rs](https://docs.rs/visionkit-rs/latest/index.html).
- **Foundation Models framework / on-device AFM**:
  - The model is about 3B parameters, multimodal (vision encoder), 2-bit quantisation-aware trained, with KV-cache sharing. The 2025 report says the framework gives developers direct access to it — [Apple tech report 2025](https://machinelearning.apple.com/research/apple-foundation-models-tech-report-2025), [arXiv 2507.13575](https://arxiv.org/abs/2507.13575).
  - WWDC26 adds image inputs and "image-based tool calling", with built-in Vision tools (OCR, barcode, saliency) according to session chapters and write-ups — [WWDC26 videos](https://developer.apple.com/wwdc/videos/), [MacRumors SOTU](https://www.macrumors.com/2026/06/09/apple-outlines-major-ai-and-developer-tool-updates/), [dev.to summary (secondary)](https://dev.to/hariharanjagan/whats-new-in-apples-foundation-models-framework-at-wwdc-2026-5227).
  - A secondary blog gives an 8,192-token context and says larger images cost more tokens. Its throughput claims (~50-100 tok/s) are unsourced — [dev.to](https://dev.to/owen_fox/apples-third-generation-foundation-models-a-developers-read-on-wwdc-2026-2ej1).
  - No official TTFT or image-latency numbers found.
- **Onscreen awareness APIs**: `NSUserActivity.appEntityIdentifier` (iOS 18.2 / macOS 15.2) associates an AppEntity with the current activity so Siri and Apple Intelligence can use what is onscreen. The flow is: describe the content as an AppEntity, adopt an assistant schema, conform to Transferable — [Apple docs](https://developer.apple.com/documentation/foundation/nsuseractivity/appentityidentifier.md), [App Intents updates](https://developer.apple.com/documentation/updates/appintents.md). A Flutter plugin note says iOS 26+, which conflicts with Apple's listing — [pub.dev](https://pub.dev/documentation/app_intents/latest/app_intents_platform_interface/AppIntentsPlatform/setOnscreenEntity.html).
- **Context for small VLMs in Q5**: OmniParser-style detectors and the ~3B on-device model are positioned as complementary. No Apple model emits UI-element boxes publicly (absence noted, not sourced).

### Inferences
- [speculation] `appEntityIdentifier` is app opt-in, and the data flows to Siri/Apple Intelligence, not to third-party processes. A third-party agent probably cannot read another app's onscreen entities. Treat it as unusable for gap-filling.
- [speculation] Foundation Models with image input could replace Florence-2 captioning for icon crops, if its per-call latency on M-series is under about 50 ms per crop or it can batch many crops in one prompt. Benchmark it against Florence-2 on cached crops. The 2-bit 3B model is memory-cheap, but generation adds token latency, so constrain output to short labels with guided generation.
- [speculation] `RecognizeDocumentsRequest` and DataDetection could tag links, phone numbers, emails and table cells in web content as extra actionable options that the AX tree misses (for example, canvas-rendered tables).

### Gaps
- No measured latency for ImageAnalyzer, RecognizeDocumentsRequest or Foundation Models image prompts on Apple Silicon.
- Exact WWDC26 Foundation Models image API signatures not verified from primary docs.

## 5. Small on-device VLMs as element "listers" vs detectors

### Takeaway
On the evidence, general VLMs asked to "list all interactive elements" have poor recall (best F1 about 0.13 on GUI-360°). A purpose-trained small parser (ScreenVLM, 316M) can beat larger VLMs on dense parsing, but dedicated detectors still win on out-of-distribution transfer and tight-box recall. A detector plus selective captioning stays the better choice for full-screen enumeration. VLMs fit better as describers of crops.

### Cited Findings
- **GUI-360°** (arXiv 2511.04307): its screen-parsing task asks for all interactable elements. The best general VLM F1 is about 0.128, with low recall. OmniParser-v2 improves modestly over v1 — [arXiv 2511.04307](https://arxiv.org/pdf/2511.04307).
- **ScreenParse / ScreenVLM** (ICML 2026, arXiv 2602.14276): 316M-param model decoding a "ScreenTag" markup. Page-IoU is 0.592 vs 0.294 for a larger foundation VLM, and OmniParser V2 fine-tuned on ScreenParse gets 0.503. On GroundCUA OOD, plain OmniParser V2 gets 0.361 vs ScreenVLM 0.251. On ScreenSpot, detectors have higher Recall@50 (OmniParser V2 about 0.48-0.54) — [arXiv 2602.14276](https://arxiv.org/html/2602.14276), [project page](https://saidgurbuz.github.io/screenparse).

### Inferences
- [speculation] Generating a full element list autoregressively costs about (#elements × tokens per element) decode steps, likely hundreds of ms to seconds even for sub-1B models on M-series. That is worse than YOLO's single pass. VLM listers fit better as an offline labeller for the explorer data than at runtime.

### Gaps
- No same-screenshot head-to-head of small VLM lister vs OmniParser latency on Apple Silicon.

## 6. OS-level accessibility repair: Screen Recognition on macOS

### Takeaway
I found no evidence that macOS (including 26) VoiceOver has iOS-style Screen Recognition. Mac users on AppleVis were still asking for it in October 2026. Even on iOS, Screen Recognition output is not exposed through public AX APIs to third-party processes (not confirmed for macOS). Reproducing it yourself, as a detector trained on AX-labelled data, is the practical route.

### Cited Findings
- AppleVis forum, Oct 2026: Mac users wish macOS VoiceOver had Screen Recognition "like iOS". They list hard apps: MS Word, Google Docs, Office web, Firefox/Thunderbird, Qt apps (KeePassXC), Parallels — [applevis comment 218666](https://www.applevis.com/comment/218666). Earlier hope for a Mac version — [applevis 172375](https://applevis.com/comment/172375).
- On iOS, Screen Recognition makes some unreachable controls usable (for example, a checkbox), but it can freeze VoiceOver and degrade Safari use — [applevis thread](https://applevis.com/forum/ios-ipados/screen-recognition-can-save-day), [comment 193438](https://applevis.com/comment/193438).
- Apple's research basis for Screen Recognition (Zhang et al., CHI 2021, "Screen Recognition: Creating Accessibility Metadata for Mobile Applications from Pixels") trained an on-device detector on about 77k screens [prior knowledge; outside the 2023-26 window] — [arXiv 2101.04893](https://arxiv.org/abs/2101.04893).

### Inferences
- [speculation] Even if Apple shipped it on Mac, iOS synthesises the recognised elements inside VoiceOver. There is no indication they would enter the target app's AX tree for other AX clients. Don't plan around it.

### Gaps
- No official Apple statement found about macOS 26 VoiceOver Screen Recognition, or about whether "Screen Recognition"-derived elements are visible to AX clients on any platform.
