# Interaction-based probing for actionable UI elements on macOS (when the AX tree is incomplete)

Scope: techniques that find clickable things by querying or poking the live UI (hit-tests, cursor shape, hover diffs, focus traversal, AX notifications, learning from user clicks). This excludes AX force flags, CDP/DOM, pixel detectors and grounding VLMs, which other notes cover. Agent context: AX walk takes ~110 ms median (Gmail p90 463 ms). Known gaps are Spotify (3 AX buttons), Arc pages with no AX web area, and icon-only buttons.

Research budget note: about 17 search/fetch calls. Several key questions have thin public sources. Gaps are flagged, and claims from unsourced prior knowledge are labelled "(unsourced)" or put in Inferences.

## 1. Hit-testing with AXUIElementCopyElementAtPosition (grid / point scans)

### Takeaway
A per-point hit-test is a cheap and practical way to resolve OCR or detector box centers to AX elements. It costs about 2 to 70 ms per call depending on the app, so a dense grid scan is too slow for a voice loop, but a few dozen targeted probes are affordable. In Chromium it can first return a coarse AXGroup and resolve to the precise element only after retries. Public evidence that it surfaces elements unreachable through AXChildren is anecdotal, and no source confirms it in general.

### Cited Findings
- API: returns the accessibility object at a screen position (top-left-origin coordinates) for an application element. It can return `kAXErrorNoValue` when nothing is there. — [Apple AXUIElement.h (archived)](https://leopard-adc.pepas.com/documentation/Accessibility/Reference/AccessibilityLowlevel/AXUIElement_h/CompositePage.html); [objc2 docs.rs binding](https://docs.rs/objc2-application-services/latest/objc2_application_services/fn.AXUIElementCopyElementAtPosition.html)
- Measured cost, one developer's PR (locant), Device Hub / Xcode simulated iPhone screen: "The hit test itself (`AXUIElementCopyElementAtPosition` on the app) costs 64–70 ms and is now about 70% of a lookup." Grid probe over 1,950 points took p50 90–92 ms, p95 148 ms and max 221–304 ms per lookup (hit-test plus attribute reads, ~19 AX calls per lookup). On the older reader it was p50 141 ms, p95 270 ms, ~100 AX calls. — [locant PR #78](https://github.com/Malik1942/locant/pull/78)
- Same PR, Device Hub sidebar (native AppKit) over 1,418 points: p50 10.2 ms, p95 16.3 ms, max 29 ms per lookup. Hit-test cost varies by more than 5x across surfaces in the same app. — [locant PR #78](https://github.com/Malik1942/locant/pull/78)
- Same PR: a hit-test on a window or screen group "answers nothing", and the application element is never returned by a hit-test. In Claude (Electron), 58 batched attribute reads failed and fell back to single reads. In Grok Bot (Chromium), batched and single reads agreed across 239 web nodes. — [locant PR #78](https://github.com/Malik1942/locant/pull/78)
- Chromium quirk, reported in a search summary of developer projects: the first `AXUIElementCopyElementAtPosition` answer inside Chromium is a coarse AXGroup, and the precise element arrives "a few frames later". The workaround was up to 5 retries 50 ms apart. — reported in search results around [locant PR #78](https://github.com/Malik1942/locant/pull/78) / related PRs. The exact originating page was not verified, so treat this as anecdotal.
- A scroll optimization replaced a full AX tree walk (124 ms for 593 nodes) with a single hit-test at a cached position, which took about 2 ms. It kept the slow scan when identifiers were empty, to avoid scrolling the wrong container. — [remote-mic-app PR #851](https://github.com/HD838A/remote-mic-app/pull/851)
- Chromium implements `accessibilityHitTest:` on its Cocoa node wrappers. Unit tests check that a hit at (50, 50) returns the child labeled "Child1" and that non-accessible nodes return nil. — [browser_accessibility_mac_unittest.mm](https://chromium.googlesource.com/chromium/src/+/c2ebc26fe8e23f94d295bf1cb1523402616f569a/content/browser/accessibility/browser_accessibility_mac_unittest.mm); [ax_native_widget_mac_unittest.mm](https://chromium.googlesource.com/chromium/src/+/d4afc97b7/ui/views/widget/ax_native_widget_mac_unittest.mm)
- Chromium added a hit-test redirect for remote app shims (PWAs) so web content objects can be found by hit-test from the shim process (issue 1400682, tested on macOS 12.6.1 and 13.3.1). Hit-test paths in Chromium have special-case plumbing that the child walk does not share. — [Chromium code search results, "Implement accessibilityHitTest for remote app shims"](https://chromium.googlesource.com/chromium/src/+/913eaefa55beab765aa1d99da170db474d8c63ca/ui/accessibility/platform/ax_platform_node_mac.mm)
- A 2026 macOS computer-use tool uses a "hit-test → find clickable ancestor → AXPress" action path. It notes that OpenAI's SkyComputerUseService imports `AXUIElementCopyElementAtPosition` (a symbol observation only, not behavior). End-to-end AX snapshots: Finder p50 120 ms, Chrome p50 100 ms. — [stable-computer-use](https://github.com/zhangyinglong3550/stable-computer-use)
- Element detection by hit-test "can take seconds on grid-like screens", per an issue in a snip tool. — [snip-pin issue #9](https://github.com/felsenuboot/snip-pin/issues/9)

### Inferences
- Budget math: at 10–70 ms per probe, a 20×12 grid (240 points) would cost 2.4–17 s, which is unusable for voice. Probing only the centers of OCR/OmniParser boxes that have no matching AX node (typically 5–40 per screen) costs about 0.05–2.8 s. Probes can run in parallel per process only if the target app's AX server handles concurrent requests; that is untested and unsourced.
- The best use is to attach a semantic AX element (role, title, AXPress) to a pixel detection, or to reject a detection, rather than to discover new elements. For Spotify (3 AX buttons), a hit-test will most likely return the same AXGroup/AXWebArea container. This is speculation and needs testing on this machine.
- Use `AXUIElementSetMessagingTimeout` on the app element to cap hung probes. This API exists, but no source was fetched here (unsourced).
- Whether a hit-test returns nodes missing from AXChildren: plausible in Chromium because hit-testing goes through a different code path (deepest node, redirected for remote shims) and the child list prunes ignored nodes. No source was found that confirms the two disagree in practice. It is cheap to test empirically by hit-testing detector centers and checking `AXParent` chains against the walked set.

### Gaps
- No Apple-published latency figure. No public measurement on Spotify, Arc or Gmail specifically.
- No primary confirmation of the Chromium "coarse AXGroup first" behavior or of hit-test-only nodes in WebKit/Electron.

## 2. Cursor-shape probing (pointing hand / I-beam as an interactability signal)

### Takeaway
macOS exposes the current system cursor through `NSCursor.currentSystem` (reportedly deprecated in macOS 15 and later). Identifying "pointing hand" needs image comparison because `isEqual:` is unreliable. Each probe needs a real cursor move and a hover dwell, so it is slow (tens of ms per point) and has side effects. No desktop tool or paper was found that uses the cursor as an interactability signal. The web analogue (`cursor: pointer` in computed style) is a known heuristic in browser agents.

### Cited Findings
- `NSCursor.currentSystem` "returns the current system cursor regardless of which application set the cursor, and whether Cocoa or Carbon APIs were used". — [Apple NSCursor.currentSystem](https://developer.apple.com/documentation/appkit/nscursor/currentsystem)
- Apple Developer Forums thread: "What can I use in place of NSCursor.currentSystemCursor?" says it is deprecated in macOS 15. Apple availability metadata still lists it, so the sources conflict. — [Apple forums 760305](https://developer.apple.com/forums/thread/760305); [Microsoft .NET binding page](https://learn.microsoft.com/en-us/dotnet/api/appkit.nscursor.currentsystemcursor?view=net-macos-26.4-10.0)
- Stack Overflow: comparing with `[NSCursor pointingHandCursor]` via `isEqual:` is unreliable (two `currentSystemCursor` results are not equal). The workaround is to compare the cursor image or hotspot against known cursor images. — [SO 41241736](https://kiwix.lema.org/stackoverflow.com_en_all_2023-11/questions/41241736/how-to-detect-current-system-cursor-on-osx); [SO 8198380](https://kiwix.lema.org/stackoverflow.com_en_all_2023-11/questions/8198380/how-to-get-current-type-of-mouse-cursor-in-mac-os-x)
- Web analogue: elements with a pointer cursor are either default-clickable (links, buttons) or have CSS `cursor: pointer`. Scanning every element for this is costly. Computed-style inheritance has edge cases (e.g., `<area>`). — [SO getcomputedstyle tag page](https://dev.library.kiwix.org/content/stackoverflow_en_nopic_2021-08/questions/tagged/getcomputedstyle_page=5); [Mozilla bug 1055933](https://bugzilla.mozilla.org/show_bug.cgi?id=1055933)
- No result was found for `CGSCurrentCursorSeed` or other private CGS cursor APIs (searched).

### Inferences
- Mechanics (unsourced, from general AppKit knowledge): the cursor is set by the app under the mouse, via tracking areas or `cursorRect`s in AppKit, or via Chromium's cursor setting from CSS. So the signal only exists for the app under the real mouse pointer, after it has processed a mouse-moved event. That requires warping the real cursor (`CGWarpMouseCursorPosition` / posted mouse-moved), waiting one or more frames (~16–50 ms), then reading the cursor. A 40-point probe costs ~1–2 s and visibly moves the user's mouse.
- In Chromium/Electron/Arc web content, `cursor: pointer` is the dominant clickability cue, so cursor probing could plausibly recover icon-only web buttons where the AX web area is missing (Arc). This is speculation; verify on Arc.
- A comparison via the cursor image's hash (`NSCursor.image` TIFF bytes) plus hotspot is probably more robust than `isEqual:` (speculation, consistent with the SO advice).
- Not viable on occluded/background windows, because the cursor reflects only what is under the pointer.

### Gaps
- No paper or tool found that uses desktop cursor-shape changes for widget discovery. No timing measurements. Private `CGSCurrentCursorSeed` usage was not confirmed by any source found.

## 3. Hover diffs (mouse-move + screenshot diff, tooltips) and GUI-testing literature

### Takeaway
Hover probing is standard in web testing (Selenium `moveToElement`, Cypress mouseover triggers). Desktop GUI rippers (GUITAR lineage) discover widgets by opening windows and reading toolkit properties, not by hover diffs. No hover/tooltip-diff desktop crawler was found. Synthetic mouse events to background windows (`CGEventPostToPid`) are unreliable for mouse events, so hover probing effectively requires the real cursor over a frontmost, unoccluded window.

### Cited Findings
- GUI Ripping (Memon, Banerjee, Nagarajan, WCRE 2003) traverses the GUI by opening all windows and extracting widgets, properties and values to build a GUI forest and event-flow graphs. The 2013 retrospective notes extensions to web, iOS, Android, JFC, SWT and UNO. No hover probing is described. — [Memon WCRE 2003](https://www.cs.umd.edu/~atif/pubs/MemonWCRE2003-abstract.html); [The First Decade of GUI Ripping (WCRE 2013)](https://www.cs.umd.edu/~atif/papers/WCRE2013.pdf)
- Crawljax fires events on candidate elements and infers a state-flow graph from DOM state changes. This is the "poke and diff" paradigm, applied to DOM rather than pixels. — [Mesbah et al., TWEB](https://people.ece.ubc.ca/amesbah/resources/papers/tweb-final.pdf)
- Web hover testing: dispatching `mouseover` does not trigger CSS-only `:hover`, so native events are needed. Listeners may use `mouseenter`/`pointerover`. Hover fails if an element is overlaid. — [BrowserAct hover node docs](https://docs.browseract.com/workflow/learn/node-types/hover-element-node); [birdeatsbug Selenium hover](https://birdeatsbug.com/blog/mouse-hover-in-selenium)
- Background delivery: `postToPid` can deliver events to background windows, but many apps check that their window is key/main before processing clicks. — [bridge.surf blog](https://bridge.surf/blog/macos-can-support-two-cursors-at-the-same-time)
- A Stack Overflow user found that `postToPid` worked only for keyboard events, while the global `post` worked for keyboard and mouse (2023, anecdotal). — [SO 75511477](https://kiwix.lema.org/stackoverflow.com_en_all_2023-11/questions/75511477/how-to-post-to-event-to-specific-window-on-macos)
- `silentmouse` CLI posts mouse events to the process owning a target window without moving the global cursor. It warns that the target may still see normal mouse side effects, and it needs Accessibility permission. — [ph0ryn/silentmouse](https://github.com/ph0ryn/silentmouse)
- Posting events to other apps is not allowed from a sandboxed app. — [Apple Community JP](https://discussionsjapan.apple.com/thread/110194517)

### Inferences
- Hover diff cost (speculation): move, then wait for the hover transition (CSS transitions are often 100–200 ms; macOS tooltips appear after ~1 s), then screenshot (ScreenCaptureKit single frame, typically tens of ms) and diff. Highlight-only probing is about 150–250 ms per point, and tooltip probing is over 1 s per point. That fits a lazy, single-target disambiguation step ("is this icon a button? what's its tooltip label?"), not enumeration.
- Tooltips are a strong naming source for icon-only buttons. Many apps also expose the tooltip as `AXHelp`, which is cheaper to read than a hover (unsourced; worth checking before hovering).
- Tracking-area-driven hover in AppKit generally requires the window to be active or the tracking area to use `activeAlways` (unsourced), so background hover will often show nothing.

### Gaps
- No desktop GUI crawler paper (ICSE/ISSTA/ASE 2022–2026) found that uses hover/tooltip probes. Humanoid and newer desktop crawlers were not reached within budget.

## 4. Keyboard focus traversal and Voice Control "Show numbers"

### Takeaway
Tab traversal with `AXFocusedUIElement` reads only enumerates focusable controls. It depends on the "Keyboard navigation" setting (off by default, so Tab only reaches text fields and lists). It changes focus, which is a user-visible side effect, and in browsers it often skips links. It is a weak fit for Spotify and icon-only buttons. Voice Control's "Show numbers" targets "all clickable or tappable onscreen items". Apple documents it as built on accessibility tagging (explicitly on iOS), with a numbered grid as the fallback for unlabeled items. No public API exposes its overlay data.

### Cited Findings
- The macOS setting "Use keyboard navigation to move focus between controls" (Catalina+, formerly "All controls") is off by default. When off, Tab moves only between text fields and lists. Ctrl+F7 toggles it. Turning on VoiceOver switches Full Keyboard Access to "All controls". — [a11yproject](https://a11yproject.com/posts/macos-browser-keyboard-navigation); [tempertemper](https://www.tempertemper.net/blog/using-the-keyboard-to-navigate-on-macos); [Apple VoiceOver guide](https://support.apple.com/ur-in/guide/voiceover/with-the-tab-key-vo2753/9/mac/10.14)
- Firefox has its own preference that overrides the macOS setting, so even with the system setting on, Tab may not focus links. — [Mozilla bug 1628476](https://bugzilla.mozilla.org/show_bug.cgi?id=1628476); [Mozilla support](https://support.mozilla.org/bm/questions/1232153)
- Apple Voice Control tech brief (Sept 2019): "Saying 'Show numbers' assigns numbers to all clickable or tappable onscreen items… especially useful for selecting unlabeled buttons." Item Names on iOS/iPadOS: "Developers can tag UI elements with Item Names and Item Numbers using the standard UIKit framework." Numbered Grid: "For unlabeled elements that are unreachable through Item Names and Item Numbers… 'Show grid'… interact more deeply with apps that haven't fully incorporated accessibility labels." — [Apple Voice Control Tech Brief](https://apple.com/accessibility/pdf/Voice_Control_Tech_Brief_Sept_2019.pdf)
- 2019 commentary argued numbers and grids work without labels because "all the OS needs to know are the locations of the UI items" (commentary, not Apple docs). — [TechCrunch](https://techcrunch.com/?p=1836381)
- Recent user reports (2025–2026): the numbers overlay still shows when set to None, and "Show Numbers Once" is misheard in macOS 26.5. — [Apple dev forums 814857](https://developer.apple.com/forums/thread/814857); [Apple Discussions 256317435](https://discussions.apple.com/thread/256317435)

### Inferences
- The brief's wording (numbers for clickable items, grid for items numbers cannot reach) strongly implies that macOS Voice Control numbers come from the AX tree (elements with press actions). It would then miss exactly what this agent's AX walk misses (Spotify, Arc), so it offers no new coverage. This is an inference: Apple does not document the macOS mechanism. A cheap empirical check is to say "show numbers" in Spotify and Arc and count the overlays against the AX walk.
- No public API for reading the Voice Control overlay was found. Reading it would mean screen-capturing the overlay window (it is a system overlay, likely capturable with ScreenCaptureKit) and OCR-ing the numbers. That is fragile and speculative.
- Tab traversal cost (speculation): each Tab plus a `AXFocusedUIElement` read is about 10–30 ms with a render wait. 50 controls would take ~1 s, and it moves focus (it can scroll pages, trigger focus handlers, or commit text fields). Only use it as an opt-in fallback on native forms or dialogs.

### Gaps
- No source on how Voice Control on macOS chooses targets internally (AX vs. on-device screen recognition). iOS has VoiceOver "Screen Recognition"; no confirmation was found that macOS Voice Control uses a vision model.

## 5. AX notifications (AXObserver) for incremental discovery and caching

### Takeaway
AXObserver can register per-app, per-element notifications (`AXCreated`, `AXLayoutChanged`, `AXValueChanged`, `AXFocusedUIElementChanged`, `AXUIElementDestroyed`). Window managers rely on the window-level ones. No source was found on how reliable element-level notifications are in Chromium/Electron. Use them as cache-invalidation hints, not as a complete discovery source.

### Cited Findings
- Notification constants `AXCreated`, `AXLayoutChanged`, `AXUIElementDestroyed` and others sit alongside window notifications in the Rust accessibility bindings. — [accessibility-sys notification constants](https://docs.rs/crate/accessibility-sys/latest/source/src/notification_constants.rs); [axuielement ax_notification.rs](https://docs.rs/crate/axuielement/latest/source/src/ax_notification.rs)
- Tiling WM paneru creates one AXObserver per app PID and registers notifications per element. It combines this with CGEventTap and NSWorkspace observers. — [paneru platform.rs](https://docs.rs/crate/paneru/latest/source/src/platform.rs)
- Amethyst saw rare crashes in the `AXUIElementDestroyed` callback, suspected to be a race during window close. — [Amethyst #671](https://github.com/ianyh/Amethyst/issues/671)
- The element delivered with a destroyed notification may have lost its attributes, but it should CFEqual cached references, so match by cached reference. — [accessibility-dev 2010](https://lists.apple.com/archives/accessibility-dev/2010/Jun/msg00022.html)
- A 2026 agent tool uses a cheap "AX signature" check of about 46 ms after app switches to decide whether to re-snapshot, a polling alternative to notifications. — [stable-computer-use](https://github.com/zhangyinglong3550/stable-computer-use)

### Inferences
- Practical design: subscribe to `AXFocusedWindowChanged`, `AXMainWindowChanged`, `AXFocusedUIElementChanged`, `AXLayoutChanged` and `AXValueChanged` on the app element. Mark the cached option list dirty on any event, and re-walk lazily, at most once per N ms, when the user speaks. This hides most of the ~110 ms walk latency behind idle time. It cannot discover elements that are not in the tree, so it does not help Spotify or Arc.
- Chromium only emits AX notifications once its accessibility mode is enabled (consistent with the Electron docs on AXManualAccessibility; see [Electron accessibility docs](https://electronjs.org/docs/latest/tutorial/accessibility)). Notification volume in Chromium can be high during scrolling or animation, so debounce.

### Gaps
- No source quantifies missed or delayed `AXCreated`/`AXLayoutChanged` in Chrome, Electron, Spotify or Arc. Mature WMs (yabai, Rectangle) were not reviewed within budget.

## 6. Learning interactability from the user's own clicks (CGEventTap logging)

### Takeaway
Logging the user's clicks (CGEventTap, listen-only), together with a hit-test and screenshot crop at the click point and a post-click change signal, can build per-app maps of clickable regions. That gives a way to cover AX-blind apps like Spotify over time. The mobile literature shows click logs at scale are a rich learning signal (20M+ clicks). No macOS-specific system was found.

### Cited Findings
- Zhou & Li (2021): over 20 million clicks from more than 4,000 Android users, each paired with the view hierarchy. A Transformer predicts the next tapped element at 48% top-1 and 71% top-3 on held-out users. — [arXiv 2108.05342](https://arxiv.org/abs/2108.05342)
- Automatic Macro Mining from Interaction Traces at Scale (2023) mines reusable macros from interaction traces and reviews element-representation work (Screen2Vec, UIBert, ActionBert). — [arXiv 2310.07023](https://arxiv.org/pdf/2310.07023)
- Illinois thesis (2016): unsupervised extraction of frequently used interactive elements from Android traces (autoencoders + k-means). — [IDEALS 93109](https://www.ideals.illinois.edu/items/93109)
- CGEventTap is used routinely by macOS WMs alongside AXObserver. — [paneru platform.rs](https://docs.rs/crate/paneru/latest/source/src/platform.rs)

### Inferences
- Per-app map (speculation, design proposal): for each click, store the app bundle ID, window-relative point, normalized crop hash, the hit-test result (role, path), and whether the screen or AX changed within ~300 ms (a click "did something"). Cluster per app and layout. At query time, inject remembered targets that match a crop-hash or template match on the current screen as options. This directly targets Spotify's static chrome (play, shuffle and like icons are at stable positions).
- Privacy: a click log plus crops is sensitive and should stay on device (fits the project's data-on-owner-Mac policy). It needs Input Monitoring / Accessibility permission for the event tap (unsourced detail).

### Gaps
- No desktop/macOS paper found that learns interactability from personal click logs. No data on how many clicks are needed for per-app coverage.

## 7. Side effects and safety; background or occluded windows

### Takeaway
Read-only AX calls (hit-test, attribute reads, observers) have no visible side effects and work on background windows, but they can be slow or hang on busy apps. Cursor probing, hover diffs and Tab traversal all mutate visible state (cursor position, hover menus, tooltips, focus, scroll) and generally need the target frontmost and unoccluded. Synthetic background mouse events are unreliable.

### Cited Findings
- Background `postToPid` delivery exists, but apps may ignore mouse events unless their window is key/main. Mouse events via `postToPid` were reported not to work at all in one case. — [bridge.surf](https://bridge.surf/blog/macos-can-support-two-cursors-at-the-same-time); [SO 75511477](https://kiwix.lema.org/stackoverflow.com_en_all_2023-11/questions/75511477/how-to-post-to-event-to-specific-window-on-macos)
- Background mouse posting still causes "normal mouse side effects" in the target. — [silentmouse](https://github.com/ph0ryn/silentmouse)
- Homerow (Vimac's successor): Electron apps (Slack, Discord, Signal) need opt-in "Electron Support", which can cause slower performance and high CPU. Homerow 1.5.3 fixed issues where Chrome a11y caused Homerow to be auto-disabled. Enabling accessibility in Chromium-based apps has real performance costs. — [Homerow user guide](https://dexterleng.notion.site/Homerow-User-Guide-b51b701ba23f4fbbbb75692faef3d994); [Softpedia Homerow](https://mac.softpedia.com/get/Utilities/Homerow.shtml)
- Keyboard-click tools (Vimac, Homerow, Shortcat) all enumerate actionable elements via the AX API. Vimac reduced hints "to what is just clickable" and had "a major performance buff in webkit/electron". Shortcat enumerates the active window's elements, app menus, and open windows. Users report missed elements in apps with poor AX (e.g., Obsidian), and grid modes are the fallback. — [Vimac GitHub](https://github.com/dexterleng/vimac/); [Shortcat docs](https://shortcat.app/docs/getting-started/); [braindetox Shortcat review](https://braindetox.kr/en/posts/shortcat_macos_keyboard_control_2026.html)
- Open-source Neru labels targets using either AX or "Native Vision OCR". It is the same AX+OCR hybrid this agent already runs, and evidence that no hidden third discovery channel is in common use. — [Neru via awesome.ecosyste.ms](https://awesome.ecosyste.ms/projects/github.com%2Fy3owk1n%2Fneru)

### Inferences
- Risk ranking (speculation, low to high): AX hit-test and attribute reads (none visible; risk of hangs, so use a messaging timeout) < AXObserver (none) < passive click logging (none, but privacy) < cursor probe (moves the user's pointer and can open hover menus) < hover diff (opens hover menus, mega-menus and previews, and can trigger autoplay on hover, e.g., video thumbnails) < Tab traversal (changes focus and scroll, and can fire onblur validation or commit text).
- During voice use the user may be hands-off, so a brief cursor warp is tolerable if the cursor is restored. Never probe on hover in apps with hover-triggered actions, such as Gmail row hover actions and Spotify hover play buttons. Hover there reveals targets, which is useful but changes the screenshot used by OCR/YOLO.
- Recommended ordering for this agent: (1) hit-test detector/OCR centers to enrich or verify, (2) AXObserver-driven cache to hide walk latency, (3) learned per-app click maps for AX-blind apps, (4) single-target cursor/hover probe only to disambiguate a specific candidate when the user's command is ambiguous.

### Gaps
- No measurements of side-effect rates or of cursor/hover probe latency on macOS. No source on whether ScreenCaptureKit captures hover state of occluded windows (it captures window content, but the hover state will not be triggered without a pointer).
