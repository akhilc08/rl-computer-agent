# Structural (non-pixel / hybrid) UI element sources and AX-tree repair on macOS

Context: the observed gaps are Spotify (3 AX buttons, ignores force flags), claude.ai in Arc (~1 element) and Gmail (AX walk p90 463 ms). The agent already sets AXManualAccessibility and AXEnhancedUserInterface.

Research budget note: about 18 tool calls. Several sub-questions are only partly sourced, and the Gaps sections say which.

## Q1. Chromium/Electron/CEF force flags: what they do, why apps ignore them, workarounds

### Takeaway
In Chrome, `AXEnhancedUserInterface` turns on `kAXModeComplete | kScreenReader`, but only after a **2-second debounce**. Electron listens for `AXManualAccessibility` instead. CEF apps such as Spotify have neither hook unless the embedder wires one in, so their documented lever is the `--force-renderer-accessibility` launch flag. Before giving up on Arc, check that the agent is not reading the tree inside the 2 s debounce window.

### Cited Findings
- Chrome's `chrome_browser_application_mac.mm` handles only `AXEnhancedUserInterface` in `accessibilitySetValue:forAttribute:` and calls `enableScreenReaderCompleteModeAfterDelay:`. `AXManualAccessibility` does not appear in that file. — [Chromium source](https://chromium.googlesource.com/chromium/src/+/HEAD/chrome/browser/chrome_browser_application_mac.mm)
- The source comment calls the attribute "undocumented", says it is "set when VoiceOver is turned on/off", and says Chrome also monitors it "in case other ATs use it to request accessibility activation." — [Chromium source](https://chromium.googlesource.com/chromium/src/+/HEAD/chrome/browser/chrome_browser_application_mac.mm)
- The enable path applies `ui::kAXModeComplete` + `AXMode::kFromPlatform` + `AXMode::kScreenReader`. Because the attribute "can change spuriously on macOS Sonoma", Chrome debounces with a **2-second delay**, and each new request cancels the pending one. If VoiceOver is already explicitly enabled, requests from other ATs are ignored. A disable with no pending enable takes effect immediately. — [Chromium source](https://chromium.googlesource.com/chromium/src/+/HEAD/chrome/browser/chrome_browser_application_mac.mm)
- AXMode flags in Chromium:
  - `kNativeAPIs` means a third-party client is using platform APIs, but web contents stay inaccessible without another mode.
  - `kWebContents` is the minimum for web content to be exposed.
  - `kInlineTextBoxes`, `kScreenReader` and `kHTML` add detail. `kHTML` is memory-heavy.
  - `kAXModeBasic` = NativeAPIs + WebContents. `kAXModeComplete` adds inline text boxes, screen reader and HTML. `kAXModeCompleteNoHTML` is the same without HTML.
  - Filter flags suppress or permit modes that AT interactions set.
  - Source: [ax_mode.h](https://chromium.googlesource.com/chromium/src/+/lkgr/ui/accessibility/ax_mode.h)
- Electron added `AXManualAccessibility` because `AXEnhancedUserInterface` is "reserved by VoiceOver" and setting it breaks window managers such as Magnet. Third-party ATs set it on the app-level AXUIElement for the PID. Apps can also call `app.setAccessibilitySupportEnabled(true)`. System assistive utilities override the setting. — [Electron accessibility docs](https://www.electronjs.org/docs/tutorial/accessibility); [electron/electron PR #7206](https://github.com/electron/electron/pull/7206)
- Spotify desktop (CEF): AppleVis users say the only known way to get a usable tree is to launch with `--force-renderer-accessibility` (for example via `open -a Spotify --args --force-renderer-accessibility`). No permanent setting exists. Even then, Artists, Albums and Playlists are not announced as buttons or links. — [AppleVis thread](https://mail.applevis.com/comment/78234). Caveat: this is an older community thread, and I found no 2025–2026 confirmation that the flag still works.
- Per-attribute AX reads in Chromium/Electron cost about 10–50 ms+ each, versus under 1 ms for native apps, because each read is a cross-process IPC to the renderer's accessibility bridge. — [cotabby issue #209](https://github.com/FuJacob/cotabby/issues/209)

### Inferences
- Arc is Chromium-based. If Arc keeps Chrome's `BrowserCrApplication` handling, setting `AXEnhancedUserInterface` and reading right away will hit the 2 s debounce and see only the browser chrome. That would match the ~1 element seen on claude.ai. Test this by setting the flag once per process, waiting more than 2 s, and re-walking. Also avoid re-setting the flag on every step, since each set restarts the timer.
- `AXManualAccessibility` is Electron-only (it lives in Electron's `atom_application.mm` / `electron_application.mm`). Chrome-family browsers and CEF hosts like Spotify will reject or ignore it. That fits the "rejects force flags" observation.
- For Spotify, the realistic options are: (a) relaunch with `--force-renderer-accessibility`, which is intrusive and was not verified on current builds; (b) app scripting or the Web API (Q5); (c) a vision fallback (Q4).

### Gaps
- I could not verify whether Arc ships Chrome's `AXEnhancedUserInterface` handler unchanged, or whether Arc and current Spotify CEF honour `--force-renderer-accessibility`. Both need a local empirical test.
- I did not confirm the current Electron source file and behaviour of `AXManualAccessibility` in 2025–2026 Electron versions.

## Q2. Browsers: CDP, Playwright, extensions, and how browser agents build element lists

### Takeaway
For Chromium browsers, CDP gives a much richer and cheaper structural source than the macOS AX bridge. `DOMSnapshot.captureSnapshot` returns `isClickable` (which includes JS click listeners), paint order and bounds in one call. Browser Use merges DOM + DOMSnapshot + the CDP AX tree and applies a 16-rule interactivity heuristic that the agent can copy.

### Cited Findings
- CDP `DOMSnapshot`:
  - Layout nodes carry `isClickable`, which covers nodes with JS click listeners and anchors.
  - `paintOrders` is available when `includePaintOrder=true`, and nodes painted together share an index.
  - `captureSnapshot` takes a computed-style whitelist (for example `cursor`, `visibility`, `opacity`). `getSnapshot` is deprecated.
  - Source: [DOMSnapshot.pdl](https://chromium.googlesource.com/devtools/devtools-frontend/+/HEAD/node_modules/devtools-protocol/pdl/domains/DOMSnapshot.pdl)
- A browser-use module (`enhanced_snapshot.py`) reads `isClickable`, `cursor`, bounds, `paintOrders` and `stackingContexts` from DOMSnapshot. — [mirror of browser_use/dom/enhanced_snapshot.py](https://glama.ai/mcp/servers/@dsouza-anush/browser-use-heroku/blob/430d35aa1e00a656b3bcf1f2292d05e41a40f657/browser_use/dom/enhanced_snapshot.py). Caveat: this is a third-party mirror.
- Browser Use `clickable_elements.py` `is_interactive` rules, in order (first match wins):
  - **Reject:** html/body tags, `disabled` or `hidden` AX properties, and `label[for]`.
  - **Accept:**
    - A CDP-detected JS click listener (React, Vue, Angular).
    - An iframe or frame larger than 100×100.
    - A label or span that wraps an input within 2 levels.
    - Search-like keywords in class, id or `data-*`.
    - AX properties focusable, editable, settable, required, autocomplete or keyshortcuts.
    - AX state checked, expanded, pressed or selected.
    - Native interactive tags: button, input, select, textarea, a, details, summary and option.
    - `on*` handler attributes or `tabindex`.
    - Interactive ARIA or AX roles.
    - Icon-sized elements (10–50 px) with class, role, onclick, data-action or aria-label.
    - Computed `cursor: pointer`.
  - Source: [browser-use clickable_elements.py](https://raw.githubusercontent.com/browser-use/browser-use/main/browser_use/dom/serializer/clickable_elements.py)
- CDP `Accessibility.getFullAXTree` can be formatted into compact text with uids. Puppeteer's AX snapshot prunes "uninteresting" nodes by default. Some tools fall back to the unfiltered tree when the pruned snapshot is empty, and run up to 50 `DOM.describeNode` calls concurrently to cut latency. — [chrome-agent snapshot.rs](https://docs.rs/crate/chrome-agent/0.15.0/source/src/snapshot.rs); [puppeteer Accessibility.ts](https://vps14.tadam.co.il/PDF-SRV/node_modules/puppeteer-core/src/cdp/Accessibility.ts); [viewpoint-core aria_snapshot](https://docs.rs/crate/viewpoint-core/latest/source/src/page/aria_snapshot/mod.rs)

### Inferences
- For Arc and Chrome, attaching CDP (a `--remote-debugging-port` launch or an extension using `chrome.debugger`) avoids per-attribute macOS AX IPC entirely. One `DOMSnapshot.captureSnapshot` plus one `Accessibility.getFullAXTree` should be far cheaper than a 463 ms AX walk on Gmail. I have no measured numbers for this, so the agent should benchmark it.
- The Browser Use heuristic, especially `cursor:pointer` plus the JS-listener flag, recovers the div-based buttons that the AX tree mislabels as generic groups.
- Safari has no CDP. The options there are WebDriver (`safaridriver`) or a Safari Web Extension content script that runs the same DOM heuristic. Safari's AX tree (WebKit) is generally complete, so it matters less.

### Gaps
- No primary sources found on Stagehand's observe/hybrid snapshot internals, Agent-E's DOM distillation, WebVoyager's set-of-marks JS, OpenAI Operator/Atlas, or Claude for Chrome element-list construction. The searches returned nothing usable.
- No measured CDP latency numbers (`getFullAXTree` or `captureSnapshot` on Gmail-scale pages) were found.
- Not verified: whether Arc permits `--remote-debugging-port` or extensions with the `debugger` permission.

## Q3. Faster AX traversal on macOS

### Takeaway
The big wins come from batching attributes per element (`AXUIElementCopyMultipleAttributeValues`), cutting bridge overhead, pruning the traversal and caching by focus or app. Parallelizing across threads within one app does not help, because each app's AX server appears to serialize requests.

### Cited Findings
- MacOS-MCP v0.3.15 sped up state capture about 10× with identical output (0 nodes lost or gained by role, name and bbox):
  - Browser + system UI capture went from 695.7 ms to about 69 ms.
  - A heavy DOM page (7,174 elements walked) went from 18.0 s to 0.83 s.
  - Elements walked dropped from 738 to 409 with no node loss, because the walk only descends into collapsed containers when the parent has a usable bbox.
  - About two-thirds of the gain came from unboxing AXValue with `AXValueGetValue` (~0.82 µs) instead of a failing `hasattr` probe (~178 µs).
  - It uses `GetMultipleAttributeValues`, memoises type classification, and materialises children arrays once.
  - Source: [MacOS-MCP release v0.3.15](https://github.com/CursorTouch/MacOS-MCP/releases/tag/v0.3.15)
- MacOS-MCP rejected prefetching attribute batches on a worker pool because it was "measurably slower", since one application's accessibility server appears to serialise requests. It also rejected two-tier attribute fetches, because per-call base cost dominated. — [MacOS-MCP release v0.3.15](https://github.com/CursorTouch/MacOS-MCP/releases/tag/v0.3.15)
- `AXUIElementCopyAttributeValues` with an index and max count can fetch a slice of a large children array (such as a table) instead of the full list. — [talon issue #496](https://github.com/talonvoice/talon/issues/496)
- Cotabby's proposed fixes for Chromium/Electron slowness:
  - Cache snapshots while the frontmost app and focused element are unchanged.
  - Track per-bundle-ID AX response time.
  - Cap depth or nodes (they used 200 nodes, depth 10).
  - Wrap calls in about 50 ms timeouts with a last-snapshot fallback.
  - The project polls synchronously because "AXObserver delivery is inconsistent across apps."
  - Source: [cotabby issue #209](https://github.com/FuJacob/cotabby/issues/209)
- Apple accessibility-dev (2009): batched or ranged attribute methods help most where AppKit would otherwise build the full array. — [accessibility-dev 2009](https://lists.apple.com/archives/accessibility-dev/2009/Sep/msg00001.html)

### Inferences
- Gmail at 463 ms p90 most likely reflects per-attribute IPC to the Chromium renderer. Batching (one Multiple call per node), visibility and bbox pruning, and CDP for browser content (Q2) are the best levers.
- Parallelizing across *different* apps (for example menu bar vs. front app) may still help, since serialization seems to be per app. This is untested.
- Use `AXObserver` (`kAXUIElementDestroyed`, `kAXValueChanged`, `kAXFocusedUIElementChanged`, `kAXLayoutChanged`) only as a cache-invalidation hint, not as ground truth, given the reported inconsistency.

### Gaps
- No published benchmark found that isolates the speedup of `AXUIElementCopyMultipleAttributeValues` itself or of `AXUIElementSetMessagingTimeout`.
- No primary source found on the per-app serialization of AX servers beyond MacOS-MCP's empirical note.

## Q4. Research on AX-tree repair and augmentation (hybrid trees)

### Takeaway
Two well-supported patterns exist. The UFO2 pattern runs the accessibility API first, then adds vision detections that do not overlap it (IoU > 10% → discard), and wraps the survivors as pseudo-controls. The Screen Recognition / Screen2AX pattern rebuilds the tree from pixels. On macOS specifically, Screen2AX reports that only 33% of apps have full accessibility support.

### Cited Findings
- **UFO2** (Microsoft, arXiv 2504.14603). Source for this bullet: [UFO2 paper](https://arxiv.org/html/2504.14603v2).
  - The pipeline queries UIA for controls passing runtime predicates (`is_visible`, `is_enabled`).
  - OmniParser-v2 (YOLO-v8 + Florence-2 0.23B) adds detections. Visual detections with **IoU > 10%** against any UIA control are discarded, and the rest become pseudo-UIA objects through a UIAWrapper.
  - Legacy or custom apps "render critical controls using non-standard toolkits that bypass UIA entirely."
  - Over 62% of UFO2-base failures on WAA were control-detection failures.
  - 27.9% and 56.7% of OmniParser detections were discarded as overlapping UIA (the split by benchmark is unclear).
- **UFO2 ablation** (success rate). Source: [UFO2 paper](https://arxiv.org/html/2504.14603v2).

  | Model | Benchmark | UIA only | OmniParser only | Hybrid |
  |---|---|---|---|---|
  | GPT-4o | WAA | 23.4% | 26.6% | 26.6% |
  | GPT-4o | OSWorld-W | 22.4% | 14.3% | 22.4% |
  | o1 | WAA | 25.3% | 20.8% | 27.9% |
  | o1 | OSWorld-W | 24.5% | 14.3% | 28.6% |

  Vision alone is worse than UIA on OSWorld-W. The hybrid never loses to either source alone.
- **UFO2 GUI+API action layer** (12 APIs for Word, Excel and PowerPoint, on 27 OSWorld office tasks). Source: [UFO2 paper](https://arxiv.org/html/2504.14603v2).
  - Success rate went from 16.3% to 22.4% with GPT-4o and from 16.3% to 24.5% with o1.
  - Average steps went from 16.0 to 6.6 with o1.
  - Speculative multi-action cut LLM calls by about 12% (WAA) to about 44–51% (OSWorld-W) with roughly flat success rate.
- **Screen2AX** (Muryn et al., arXiv 2507.16704, July 2025, macOS). Source: [Screen2AX abstract](https://arxiv.org/abs/2507.16704).
  - Only 33% of macOS apps offer full accessibility support.
  - Three datasets across 112 macOS apps cover element detection, grouping and hierarchy.
  - It reaches 77% F1 at reconstructing a complete AX tree.
  - It gives a 2.2× task improvement over native AX representations on Screen2AX-Task and beats OmniParser V2 on ScreenSpot.
  - Name, Description, Role Description and Value are frequently missing on macOS. — [Screen2AX HTML](https://arxiv.org/html/2507.16704v1)
- **Screen Recognition** (Zhang et al., CHI 2021, Apple). Source: [arXiv 2101.04893](https://arxiv.org/abs/2101.04893).
  - It builds VoiceOver metadata from pixels only, using 77,637 screens from 4,068 iPhone apps.
  - An on-device detector is combined with grouping and ordering heuristics plus content, state and interactivity models.
  - A study with 9 screen-reader users made some previously unusable apps usable.
  - The follow-up is Screen Parsing (Wu et al., UIST 2021), which infers hierarchy from screenshots.
- **Agent S** uses an "image-augmented accessibility tree" with screenshots. — [Agent S](https://arxiv.org/html/2410.08164v1)
- **Agent S2** adds a textual grounding expert that uses OCR. It reaches 27.0% (15-step) and 34.5% (50-step) on OSWorld. — [Agent S2](https://arxiv.org/html/2504.00906v1)
- MacArena / online macOS benchmark observations are screenshot + AX tree. — [HF paper 2606.06560](https://huggingface.co/papers/2606.06560.md)

### Inferences
- For Spotify and other CEF apps, a UFO2-style merge fits directly. Keep the 3 AX buttons as authoritative, then add OCR or detector boxes with IoU > 0.1 dedup as pseudo-elements, labelled by OCR text. Mark these pseudo-elements so the model knows they cannot be AXPress'd and must be clicked by coordinate.
- Screen2AX is the most directly relevant macOS model. Check whether code and weights are released before relying on it.

### Gaps
- I did not verify a Screen2AX code or weights release, its latency, or its license.
- I did not retrieve the OSWorld, macOSWorld or UFO (v1) statements on a11y-tree incompleteness, or OSWorld's a11y-tree vs screenshot ablation numbers.
- I did not cover Windows UIA "cache request" (`IUIAutomationCacheRequest`) batching as an analogue. It is known to exist but no source was pulled.

## Q5. App-specific APIs as alternatives to clicking

### Takeaway
Scriptable apps sidestep missing AX entirely, and UFO2 shows that APIs cut steps sharply. Spotify on macOS exposes an AppleScript dictionary for playback control, which covers much of what the 3 visible AX buttons fail to.

### Cited Findings
- Spotify AppleScript works through `osascript`:
  - `playpause`, `next track`, `previous track` and `play track "<spotify uri>"`.
  - The app must be running.
  - The first use triggers the macOS Automation permission prompt.
  - Sources: [spotify-applescript skill doc](https://playbooks.com/skills/openclaw/skills/spotify-applescript); [skills.sh mirror](https://skills.sh/moltbot/skills/spotify-applescript). Caveat: these are community docs, so check them against the dictionary in Script Editor (File → Open Dictionary → Spotify).
- UFO2 evidence that API actions beat GUI actions where available: +6.1 to +8.2 points of success rate, and average steps 16.0 → 6.6 (o1). — [UFO2 paper](https://arxiv.org/html/2504.14603v2)

### Inferences
- Rough order of sources by app type:
  1. Scripting dictionary or JXA (Spotify, Music, Mail, Finder, Safari `do JavaScript`, Chrome/Arc `execute javascript` if enabled).
  2. URL schemes (`spotify:` URIs).
  3. CDP for Chromium web content.
  4. Batched AX.
  5. Vision fill-in.
- Chrome-family AppleScript `execute javascript` can run the Browser Use heuristic in-page with no CDP port. It needs the "Allow JavaScript from Apple Events" developer setting, and I did not verify whether Arc supports it.

### Gaps
- No source pulled for: the Spotify Web API (playback control needs Premium plus an OAuth `user-modify-playback-state` scope; not verified here), Arc's AppleScript dictionary, Shortcuts / App Intents as an agent action surface, or usage numbers for these in published agents.
