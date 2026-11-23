# Direct coordinate grounding vs option-list / set-of-marks selection

Context: a macOS agent currently picks an element id from an option list built from AX + OCR + detector, using a small Qwen-based model with a pointer head. Question: keep it, replace it with direct coordinate grounding, or add grounding as a fallback when the target isn't in the list.

Research cutoff: 2026-11-04. Most benchmark numbers are self-reported by model authors. The official ScreenSpot-Pro leaderboard page (gui-agent.github.io/grounding-leaderboard, last updated Sep 10 2026) did not render model rows through my fetch tool, so the per-model numbers below come from model cards and papers.

## Q1. State-of-the-art grounding models (2024-2026): scores and sizes

### Takeaway
Pure-vision grounding has improved a lot since 2024. On ScreenSpot-Pro (high-res professional desktop apps), open 7-8B models went from about 29% (Qwen2.5-VL-7B) to about 58-68% (Holo2-8B, UI-Venus-1.5-8B). The best open models report 66-78%. ScreenSpot-v2 is saturated, with most current models at 91-95%. OSWorld-G, the most desktop-like benchmark, sits at about 56-76% for current models.

### Cited findings

**Holo2 model card comparison table (H Company; ScreenSpot-Pro / OSWorld-G / ScreenSpot-v2, %)** — [Holo2-4B card](https://huggingface.co/Hcompany/Holo2-4B)
| Model | Size | SS-Pro | OSWorld-G | SS-v2 | License (per card) |
|---|---|---|---|---|---|
| Holo2-30B-A3B | 30B MoE, 3B active | 66.1 | 76.1 | 94.9 | research-only, non-commercial |
| Holo2-8B | 8B | 58.9 | 70.1 | 93.2 | Apache 2.0 |
| Holo2-4B | 4B (from Qwen3-VL-4B-Thinking) | 57.2 | 69.4 | 93.2 | Apache 2.0 |
| Holo1.5-72B | 72B | 63.3 | 71.8 | 94.4 | – |
| Holo1.5-7B | 7B | 57.9 | 66.2 | 93.3 | Apache 2.0 ([Holo1.5-3B card](https://huggingface.co/Hcompany/Holo1.5-3B)) |
| Holo1.5-3B | 3B | 51.4 | 61.5 | 91.6 | Qwen Research License ([card](https://huggingface.co/Hcompany/Holo1.5-3B)) |
| Qwen3-VL-30B-A3B-Thinking | 30B-A3B | 49.9 | 65.8 | 91.8 | – |
| Qwen3-VL-8B-Thinking | 8B | 38.5 | 56.0 | 91.5 | – |
| Qwen3-VL-4B-Thinking | 4B | 41.4 | 56.4 | 90.0 | – |
| Qwen2.5-VL-72B | 72B | 55.6 | 62.0 | 93.3 | – |
| Qwen2.5-VL-7B | 7B | 29.0 | 40.6 | 85.6 | – |
| Qwen2.5-VL-3B | 3B | 29.3 | 34.3 | 80.7 | – |
| UI-TARS-1.5-7B | 7B | 39.0 | 61.0 | 94.0 | – |
| UI-Venus-72B | 72B | 61.9 | 70.4 | 95.3 | – |
| UI-Venus-7B | 7B | 50.8 | 58.8 | 94.1 | – |

- The Holo2 card also reports end-to-end navigation: Holo2-4B gets 37.7% on OSWorld, about the same as Holo2-30B-A3B (37.4%). Holo1.5-3B gets 5.8%, so strong grounding alone does not make a good agent. — [Holo2-4B card](https://huggingface.co/Hcompany/Holo2-4B)
- UI-Venus (v1) reports 94.1 / 50.8 (7B) and 95.3 / 61.9 (72B) on SS-v2 / SS-Pro and claims to beat GTA1 and UI-TARS-1.5. — [UI-Venus-Ground-7B card](https://huggingface.co/inclusionAI/UI-Venus-Ground-7B), [72B card](https://huggingface.co/inclusionAI/UI-Venus-Ground-72B), [GitHub](https://github.com/inclusionAI/UI-Venus)
- UI-Venus-1.5 (Feb 2026, arXiv 2602.09082) reports ScreenSpot-Pro of 57.7% (2B), 68.4% (8B) and 69.6% (30B-A3B). The 30B-A3B also reports 70.6% on OSWorld-G and 76.4% on OSWorld-G-R. — [UI-Venus-1.5-30B-A3B card](https://huggingface.co/inclusionAI/UI-Venus-1.5-30B-A3B), [UI-Venus-1.5-8B card](https://huggingface.co/inclusionai/ui-venus-1.5-8b), [paper](https://huggingface.co/papers/2602.09082)
- GTA1 (Salesforce, RL-trained grounding plus test-time scaling) reports 50.1 / 92.4 / 67.7 on SS-Pro / SS-v2 / OSWorld-G for GTA1-7B, and 63.6% on SS-Pro for GTA1-32B. — [GTA1 GitHub](https://github.com/Yan98/GTA1), [arXiv 2507.05791](https://web3.arxiv.org/pdf/2507.05791)
- GUI-Actor (Microsoft, NeurIPS 2025) reports 44.6 on SS-Pro for GUI-Actor-7B (Qwen2.5-VL) and 40.7 with Qwen2-VL, above UI-TARS-72B's 38.1. — [arXiv 2506.03143](https://arxiv.org/abs/2506.03143)
- Jedi / OSWorld-G (XLANG/HKU, NeurIPS 2025 D&B): OSWorld-G has 564 samples across text matching, element recognition, layout understanding, fine-grained manipulation and refusal. The JEDI training set has about 4M examples. On OSWorld-G, Jedi-3B scores 50.9 and Jedi-7B 54.1, against UI-TARS-7B 47.5, Qwen2.5-VL-3B 27.3 and Qwen2.5-VL-7B 31.4 (authors' table). — [OSWorld-G project page](https://osworld-grounding.github.io), [arXiv 2505.13227](https://arxiv.org/pdf/2505.13227)
- The Step-GUI technical report gives OSWorld-G scores of 58.2 for Qwen3-VL-8B, 58.0 for GUI-Owl-32B and 47.5 for UI-TARS-1.5-7B. — [Step-GUI report arXiv 2512.15431](https://arxiv.org/pdf/2512.15431)
- Top of the ScreenSpot-Pro board (aggregators, unverified): Holo2-235B-A22B is listed at 78.5 in an agentic/zoom setting, MAI-UI-32B at 77.5 (MVP) or 73.5 (zoom-in), UI-Venus-1.5-30B-A3B at 69.6 and UI-Venus-1.5-8B at 68.4. — [Papers with Code SS-Pro](https://paperswithcode.co/benchmark/screenspot-pro?eval=5370), [HF blog: Holo2-235B](https://huggingface.co/blog/Hcompany/introducing-holo2-235b-a22b). One aggregator claims "GPT-6 Astra 92.7%" ([benchlm.ai](https://benchlm.ai/benchmarks/screenspot-pro)). It is low-credibility and should not be relied on.
- Scores conflict across tables. Holo2's table lists UI-TARS-1.5-7B at 39.0 on SS-Pro and Qwen2.5-VL-3B at 29.3, which is higher than Qwen2.5-VL-7B (29.0). Each paper uses its own prompts and image resolution, so treat ±5 points as noise across sources.

### Inferences
- If you want an open grounder to sit beside the option-list policy, the strongest candidates at a size that runs on a Mac are Holo2-4B (Apache 2.0, Qwen3-VL-4B base, SS-Pro 57.2 / OSWorld-G 69.4) and UI-Venus-1.5-2B (SS-Pro 57.7). Both are far ahead of the vanilla Qwen2.5-VL-3B grounding that a Qwen-based policy inherits (~29 SS-Pro).
- OSWorld-G is the closest proxy for macOS desktop apps among these benchmarks. ScreenSpot-v2 is saturated and doesn't separate the models.

### Gaps
- I did not retrieve primary-source numbers for OS-Atlas, UGround, Aguvis, ShowUI, SeeClick, Phi-Ground, UI-TARS-2, or Claude/OpenAI computer-use on ScreenSpot-Pro or OSWorld-G within the tool budget. These models are mostly 2024-era and below the current open SOTA. The one exception is OS-Atlas, which [its paper](https://arxiv.org/pdf/2410.23218) reports outperformed SeeClick and Set-of-Mark as the grounding module for a GPT-4o agent on OSWorld.
- The official leaderboard page's model table could not be extracted.
- Licenses are confirmed only for Holo models and browserground. Check every other model card before using it commercially. Several of these models are research-only or under the Qwen Research License.

## Q2. Smallest models (<=4B) with good grounding, and Apple Silicon / MLX latency

### Takeaway
Grounding at 2-4B is now competitive. UI-Venus-1.5-2B (57.7 SS-Pro), Holo2-4B (57.2 SS-Pro, 69.4 OSWorld-G), Holo1.5-3B (51.4 / 61.5), Jedi-3B (50.9 OSWorld-G) and GUI-Actor-3B (91.0-92.4 SS-v2) are all strong. I found almost no rigorous measured latency on Apple Silicon. The anecdotal data suggests seconds per call, not tens of milliseconds, for 3B 4-bit models in mlx-vlm on full screenshots.

### Cited findings
- UI-Venus-1.5-2B: SS-Pro 57.7%. — [UI-Venus-1.5 card](https://huggingface.co/inclusionAI/UI-Venus-1.5-30B-A3B), [paper](https://huggingface.co/papers/2602.09082)
- Holo2-4B: SS-Pro 57.2, OSWorld-G 69.4, SS-v2 93.2, Apache 2.0, fine-tuned from Qwen3-VL-4B-Thinking. — [card](https://huggingface.co/Hcompany/Holo2-4B)
- Holo1.5-3B: SS-Pro 51.4, OSWorld-G 61.5, under the Qwen Research License (inherited from Qwen2.5-VL-3B). — [card](https://huggingface.co/Hcompany/Holo1.5-3B)
- Jedi-3B: OSWorld-G 50.9, compared with base Qwen2.5-VL-3B at 27.3. — [OSWorld-G page](https://osworld-grounding.github.io)
- GUI-Actor small variants: the 2B model (Qwen2-VL) scores 88.6 on SS-v2, or 89.3 with the verifier. The 3B model (Qwen2.5-VL) scores 91.0, or 92.4 with the verifier. A separate GUI-Actor-Verifier-2B is released. — [GUI-Actor-7B-Qwen2.5-VL card](https://huggingface.co/microsoft/GUI-Actor-7B-Qwen2.5-VL/blob/bf9f35c390e5fe48dcf1bf3a959ddc9d47a7f12a/README.md), [GUI-Actor-2B-Qwen2-VL card](https://huggingface.co/microsoft/GUI-Actor-2B-Qwen2-VL/blob/refs%2Fpr%2F2/README.md), [Verifier-2B](https://huggingface.co/microsoft/GUI-Actor-Verifier-2B)
- SE-GUI reports 47.2% (7B) and 35.9% (3B), trained on about 3k samples. Per the ScreenSpot-Pro repo's news, these figures are presumably ScreenSpot-Pro. — [ScreenSpot-Pro repo](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding)
- MLX availability: `browserground-mlx` is a Qwen3-VL-2B-Instruct UI-grounding fine-tune in 4-bit MLX format, 1.78 GB, Apache 2.0. Its card claims "~10x faster on Apple Silicon" than transformers but gives no measured numbers. — [HF card](https://huggingface.co/renezander030/browserground-mlx)
- `mlx-community/GUI-Owl-7B-6bit` is an mlx-vlm conversion. Its card says grounding was essentially unchanged after quantization but gives no latency figures. — [card](https://huggingface.co/mlx-community/GUI-Owl-7B-6bit/blob/main/README.md)
- Anecdotal macOS latency: Qwen2.5-VL-3B at 4-bit via mlx-vlm was 2.9 GB, loaded in 1.9 s and ran inference in 3-7 s per call. The author judged it too weak and switched to 7B 4-bit. — [DEV Community post](https://dev.to/nivdvir/building-a-real-time-screen-reader-on-macos-that-actually-works-471)
- Ferret-UI reportedly requires CUDA flash-attention, which rules it out on a Mac. — same [DEV post](https://dev.to/nivdvir/building-a-real-time-screen-reader-on-macos-that-actually-works-471)

### Inferences
- Most of the latency in a generative grounder is the vision encoder on a high-res screenshot (thousands of visual tokens for a Retina display) plus decoding about 10 coordinate tokens. GUI-Actor's head emits no coordinate tokens; it needs only one forward pass and the attention map. That makes it the better fit for a latency-sensitive Mac pipeline, and the closest in spirit to the existing pointer head.
- A Qwen-based pointer-head policy already pays for the vision encoder. A grounding head on the same backbone (GUI-Actor style, ~100M params on 7B) could reuse those visual tokens at almost no extra cost, instead of running a second 2-4B model for 3-7 s.

### Gaps
- I found no published, controlled latency benchmarks (tokens/s, time-to-first-token, per-click latency) for any GUI grounding model on M-series chips with mlx-vlm, and none for UI-TARS or ShowUI on MLX. These need to be measured locally.
- Accuracy loss from 4-bit quantization on grounding benchmarks has not been measured, beyond the claim on the GUI-Owl 6-bit card.

## Q3. Set-of-marks / option-list vs pure-vision grounding: when does each win?

### Takeaway
Option lists (SoM, AX/UIA lists) win when the element source is complete. Selecting an id is far easier than regressing pixels, which is especially true for general LLMs: OmniParser took GPT-4V from 16% to ~73% on ScreenSpot. They fail exactly when the element source misses the target, and in UFO2 more than 62% of failures were detection failures. Specialized pure-vision grounders now beat SoM as a grounding module. The best measured results combine an accessibility API with vision detection, and the combination mainly helps when the planner is strong.

### Cited findings
- The OmniParser method (Microsoft, 2024) combines a detector with OCR and icon captions, overlays numbered marks, and has GPT-4V pick an id instead of coordinates. — [arXiv 2408.00203](https://arxiv.org/html/2408.00203)
- On ScreenSpot, raw GPT-4V averaged 16.2%. With OmniParser's fine-tuned detection plus local semantics it reached ~73.0%. Without the grounding detector it averaged 58.38%, and adding local semantics added 10.32 points. Local semantics also raised id-assignment accuracy on SeeAssign from 0.705 to 0.938. — [OmniParser review](https://liner.com/review/omniparser-for-pure-vision-based-gui-agent), [paper](https://arxiv.org/html/2408.00203). The 73.0% comes from a secondary review; check it against Table 2 of the paper.
- Set-of-Mark prompting (Yang et al., 2023) introduced numbered marks, masks and boxes to unlock visual grounding in GPT-4V. — [arXiv 2310.11441](https://arxiv.org/pdf/2310.11441)
- The OSWorld paper says the accessibility tree and SoM "can be helpful" but "can also lead to potential misguidance and varies across models." — [OSWorld NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d413e48f84dc61244b6be550f1cd8f5-Paper-Datasets_and_Benchmarks_Track.pdf)
- OS-Atlas reports that, as a drop-in grounding module for a GPT-4o agent on OSWorld, it significantly outperforms SeeClick and Set-of-Mark. — [arXiv 2410.23218](https://arxiv.org/pdf/2410.23218)
- **UFO2 (Microsoft, Windows) ablation, Table 3.** UFO2 compares UIA-only, OmniParser-v2-only and hybrid control detection, then applies SoM over the merged list. — [arXiv 2504.14603v2](https://arxiv.org/html/2504.14603v2)

  | Detector | LLM | WAA SR | OSWorld-W SR |
  |---|---|---|---|
  | UIA only | GPT-4o | 23.4% | 22.4% |
  | OmniParser-v2 only | GPT-4o | 26.6% | 14.3% |
  | Hybrid | GPT-4o | 26.6% | 22.4% |
  | UIA only | o1 | 25.3% | 24.5% |
  | OmniParser-v2 only | o1 | 20.8% | 14.3% |
  | Hybrid | o1 | 27.9% | 28.6% |

  - Hybrid recovers 9.9% (WAA) and 12.5-25% (OSWorld-W) of the tasks that UIA-only fails (Control Recovery Ratio).
  - Over 62% of WAA failures for UFO2-base (GPT-4o) were control-detection failures.
  - UIA misses controls in apps that use non-standard toolkits and in apps like LibreOffice that don't fully follow accessibility standards.
  - Merge rule: a vision detection is discarded if its IoU with any UIA control exceeds 10%. Survivors are wrapped as pseudo-UIA objects, and 27.9% / 56.7% of OmniParser detections were discarded as duplicates.
- Agent S2 (Simular, 2025) uses "Mixture of Grounding," routing to visual (a UI-TARS grounder), textual (OCR) and structural experts. Ablations show it improves subtask completion and overall success. Agent S2 reports 27.0% (15-step) and 34.5% (50-step) on OSWorld from screenshots only. — [arXiv 2504.00906](https://arxiv.org/pdf/2504.00906), [Agent-S GitHub](https://github.com/dowithless/Agent-S). One aggregator gives a 3-5 point drop without MoG ([Simular technical review](https://www.simular.ai/articles/agent-s2-technical-review)); I did not verify this against the paper.

### Inferences
- When the AX tree is complete (native Cocoa apps), the option list is the better primary path. It is cheaper, its click targets are exact, and its labels are semantic. Pure vision wins on canvases, Electron and custom-drawn UIs, games, and anywhere AX or the detector misses elements. UFO2's numbers show vision-only falls behind the structured list on OSWorld-W (14.3% vs 22.4%), while the hybrid is never worse and is sometimes better.
- With a weak policy, hybrid gains vanished (GPT-4o on OSWorld-W: 22.4% → 22.4%). Adding candidates helps only if the selector can use them, so a small selector model needs training data that includes vision-only candidates.
- Your pipeline is already a hybrid of AX, OCR and detector, the same design as UFO2. Its remaining failure mode is targets that no source produced, and only a coordinate grounder can recover those.

### Gaps
- I found no controlled, apples-to-apples ablation of SoM id-selection against direct coordinate prediction using the same small, fine-tuned model. Most comparisons use GPT-4V or GPT-4o as the selector.
- I found no macOS-specific study. All the hybrid evidence comes from Windows (UFO2) or Linux (OSWorld).

## Q4. Hybrid designs: grounding as a candidate proposer or as a "none of the above" fallback

### Takeaway
Several published systems already use grounding as a separate stage. UFO2 merges vision detections into the accessibility list. Agent S2 routes to a grounder per action. GUI-Actor proposes several candidate regions in one pass and lets a verifier pick. The natural design for this agent is to keep the option list, add a "none of the above" option, and on that choice call a grounding head or model. Alternatively, grounding proposals can be injected into the list as extra candidates with deduplication.

### Cited findings
- GUI-Actor adds an `<ACTOR>` token whose attention over visual patch tokens yields "one or more action regions in a single forward pass," with no numeric coordinate generation. A separate grounding verifier picks the most plausible candidate and consistently improves results. — [arXiv 2506.03143](https://arxiv.org/abs/2506.03143), [MSR page](https://www.microsoft.com/en-us/research/publication/gui-actor-coordinate-free-visual-grounding-for-gui-agents/), [NeurIPS 2025](https://neurips.cc/virtual/2025/poster/119841)
- GUI-Actor also reports that fine-tuning only the action head (~100M params for 7B) with a frozen backbone, plus the verifier, matches prior SOTA. This suggests a grounding head can be bolted onto an existing frozen VLM. — [arXiv 2506.03143](https://arxiv.org/abs/2506.03143)
- The verifier gains on SS-v2 are 88.6 → 89.3 (2B), 91.0 → 92.4 (3B) and 92.1 → 92.5 (7B). — [GUI-Actor model cards](https://huggingface.co/microsoft/GUI-Actor-7B-Qwen2.5-VL/blob/bf9f35c390e5fe48dcf1bf3a959ddc9d47a7f12a/README.md)
- Later attention-based grounders, GUI-AIMA and Trifuse, build on the same "attention map as grounding" idea. — [GUI-AIMA arXiv 2511.00810](https://arxiv.org/pdf/2511.00810), [Trifuse arXiv 2602.06351](https://arxiv.org/pdf/2602.06351)
- GTA1 uses test-time scaling: it samples several candidate actions and has a judge select one. — [GTA1 GitHub](https://github.com/Yan98/GTA1)
- UFO2 adds vision-detected elements to the accessibility list with an IoU>10% dedupe rule and treats them as pseudo-native controls, so the selector sees one unified list. — [arXiv 2504.14603v2](https://arxiv.org/html/2504.14603v2)
- Agent S2 lets the generalist planner describe the target in language, and a specialist grounder (UI-TARS, or Claude via API) returns the coordinates. — [Agent-S GitHub](https://github.com/dowithless/Agent-S)

### Inferences
Three designs fit this architecture, from cheapest to most capable:
1. **Fallback head.** Add a `NONE` option to the pointer head. When it wins or confidence is low, run a GUI-Actor-style attention head on the same backbone's patch tokens, or an external grounder such as Holo2-4B or UI-Venus-1.5-2B, to get a point. Snap the point to the nearest AX, OCR or detector box when one contains it.
2. **Candidate injection.** Run the grounder's top-k attention peaks as extra options, deduplicated against existing options at a low IoU like UFO2's 10%, so the pointer head still selects from one list. This keeps a single decision mechanism and makes failures auditable.
3. **Dual-head with agreement check.** Predict both an id and a point. If the point falls inside the chosen element's box, act. If they disagree, defer, re-observe or zoom. This is a cheap verifier.

Design 1 or 2 should be trainable from existing labels. Each labeled option already has a box, so the attention-head target is the patch mask under that box, which is GUI-Actor's training signal.

### Gaps
- I found no paper that evaluates a "none of the above → grounding fallback" design specifically, or that reports how often option lists miss the target on macOS.

## Q5. Confidence, calibration, and abstention when the target is absent

### Takeaway
Grounding models are poorly calibrated for abstention. Models trained for computer use almost never refuse, even when trained with refusal data. If the agent needs a reliable "target not here" signal, the option-list side (an explicit NONE option, or the probability mass over options) is the better place for it. Any coordinate fallback needs its own verifier or threshold.

### Cited findings
- OSWorld-G includes a refusal category of 54 infeasible instructions, credited only when the model abstains, for example by outputting (-1,-1). — [OSWorld-G arXiv 2505.13227v3](https://arxiv.org/html/2505.13227v3), [Trifuse](https://arxiv.org/pdf/2602.06351)
- The Jedi authors report that despite including refusal data in training, their model "rarely produces refusal responses." Refusal was consistently absent in every model except Gemini-2.5-Pro, "especially those specifically trained for computer-use tasks." — [arXiv 2505.13227v3](https://arxiv.org/html/2505.13227v3)
- LookAgain reports that many strong GUI models reach competitive grounding accuracy but nearly fail on refusal cases. — [arXiv 2608.09723](https://arxiv.org/pdf/2608.09723)
- AdaGround's joint abstention rule, tuned for few false refusals (0.8% and 0.4%), catches only 15 and 8 of the 54 absent-target instructions in OSWorld-G. — [arXiv 2610.36874](https://arxiv.org/html/2610.36874)
- MEGA-GUI explicitly trades refusal accuracy against false-positive rate. — [arXiv 2511.13087](https://arxiv.org/pdf/2511.13087)
- UFO2 validates actions against UIA preconditions (is_enabled, is_visible) and stops if a check fails. On macOS, the equivalent would be a post-hoc AX check. — [UFO2](https://arxiv.org/html/2504.14603v2)

### Inferences
- A pointer head over options gives a natural confidence score through its softmax over options plus NONE, which can be calibrated by temperature scaling on held-out data. A generative coordinate grounder gives no comparable score. An attention-map head (GUI-Actor) does give a peak value or entropy that could be thresholded, but no paper I found calibrates it.
- The fallback should be gated by a verifier. Possible checks include whether the point lands inside a detected box or OCR span that matches the instruction, a zoomed re-check, or an AX hit-test at the point (`AXUIElementCopyElementAtPosition`). Without one, the grounder will confidently click somewhere when the target is absent.

### Gaps
- I found no study reporting calibration metrics (ECE, reliability curves) for GUI grounding models, and none for pointer-over-options GUI selectors.
- AdaGround and LookAgain are recent arXiv preprints (2026) and have not been peer-reviewed.
