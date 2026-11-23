# App-level semantic action channels on macOS (complement to the AX-tree option list)

Method note: web sources plus **local measurements on the owner's Mac** (macOS 27.0, build 26A428, 116 apps in /Applications + /System/Applications, measured 2026-11-06). The local commands are listed inline so someone else can rerun them. "Local measurement" findings cite the command, not a URL.

## 1. App Intents / Shortcuts / Spotlight / Siri (App Entities)

### Takeaway
App Intents give the richest typed action catalog on macOS, and the OS already keeps a machine-readable index of it: a per-user SQLite "ToolKit" database listing about 2,163 tools across 104 bundles. But coverage is mostly Apple's own apps. Spotify, Slack, Chrome, Arc, Notion, Discord and VS Code ship **zero** intents. Mail has `ArchiveMessageIntent`, which its AppleScript dictionary lacks. No public API lets a third-party agent invoke an arbitrary intent directly. The documented route is wrapping it in a user shortcut and calling `shortcuts run`.

### Cited Findings
- App Intents are "the foundation for integrating your Apple Siri and Apple Intelligence, and provides a structured way to describe what your app can do". Intents describe "the actions your app supports, the parameters they require, and when they're safe to run". They surface in Shortcuts, Spotlight and Widgets. — [WWDC26 session 240](https://developer.apple.com/videos/play/wwdc2026/240/)
- App Schemas are "a specialization of App Intents... shaped in a way that Siri knows how to process". Domains (for example Messages, with `sendMessage`/`draftMessage`) are "categories of contracts between your app and Siri". Siri can run them "all without opening the app". — [WWDC26 session 240](https://developer.apple.com/videos/play/wwdc2026/240/)
- Onscreen awareness: apps tie visible content to App Entities through `NSUserActivity` (one primary item) or the `.appEntityIdentifier(...)` view annotation (lists), so Siri can resolve "this"/"that". Entities can be `IndexedEntity` (in the system semantic index) or resolved through `EntityStringQuery`. — [WWDC26 session 240](https://developer.apple.com/videos/play/wwdc2026/240/); a third-party wrapper says `NSUserActivity.appEntityIdentifier` is iOS 26+ — [pub.dev app_intents](https://pub.dev/documentation/app_intents/latest/app_intents_platform_interface/AppIntentsPlatform/setOnscreenEntity.html)
- The session says nothing on MCP, macOS-specific behavior, or whether non-Apple agents can invoke intents. — [WWDC26 session 240](https://developer.apple.com/videos/play/wwdc2026/240/)
- macOS 26 Tahoe Spotlight can run "hundreds of system and app actions" plus user shortcuts, with auto-generated "Quick Keys" (for example "sm" for send message). Third-party apps contribute through App Intents. — [MacRumors, Jun 2025](https://www.macrumors.com/2025/06/09/apple-supercharges-spotlight-in-macos-tahoe-with-quick-keys-and-more/); [iThinkDiff](https://www.ithinkdiff.com/spotlight-macos-tahoe-quick-keys-clipboard/)
- MCP in App Intents: the macOS/iOS 26.1 betas contained code "laying the groundwork to bring MCP support to App Intents". It was not user-facing at the time. — [9to5Mac](https://9to5mac.com/?p=1020830); [Computerworld](https://www.computerworld.com/article/4061614/apple-to-deploy-mcp-support-for-powerful-ai-experiences.html); [AlternativeTo](https://alternativeto.net/news/2025/9/apple-begins-testing-early-model-context-protocol-mcp-support-in-ios-ipados-and-macos). I found no confirmation that it shipped.
- **Local measurement: the system action catalog.** `~/Library/Shortcuts/ToolKit/Tools-prod.v79-*.sqlite` (readable with `sqlite3 -readonly`, no TCC prompt from a terminal that already has file access) has tables `Tools`, `Parameters`, `ParameterLocalizations`, `ToolLocalizations`, `Types`, `EntityProperties`, `EnumerationCases`, `SearchKeywords`, `SampleInvocationPhrase`, `ContainerMetadata` and others.
  - Row counts: 2,163 tools (appIntent 1,579, action 387, siriIntent 144, flowTool 53) from 104 source containers.
  - There is also a `~/Library/Shortcuts/ToolEmbeddingDatabase/index.sql` (42 MB), which looks like a precomputed embedding index over these tools. That reading is an inference; I did not inspect its schema.
  - Command: `sqlite3 -readonly Tools-*.sqlite "select m.id,count(*) from Tools t join ContainerMetadata m on m.rowId=t.sourceContainerId group by 1 order by 2 desc"`.
- **Local measurement: coverage per app (ToolKit tool counts).**
  - Apple apps: Shortcuts 434, System Settings 431 (+ Settings 211), Photos 80, News 60, Notes 57, Reminders 51, Safari 42, Messages 37, Clock 37, Mail 32, Home 31, Calendar 24, Finder 16, Music 12.
  - Third-party apps with intents: Ghostty 13, AlDente 12, Tailscale 12, Outlook 6, Teams 6, Word/Excel/PowerPoint 2 each, Raycast 3, Cisco AnyConnect 3, Lightroom 1, Hammerspoon 1.
  - **No entries:** Spotify, Slack, Google Chrome (only an iOS `com.google.chrome.ios.openurl` interchange action), Arc, Notion, Discord, Obsidian, VS Code.
- **Local measurement: bundle metadata.** Apps embed `Contents/Resources/Metadata.appintents/extract.actionsdata` (JSON). 33 apps on this Mac contain one, and 24 more sit inside PlugIns/Extensions (for example `Mail.app` lists `ArchiveMessageIntent, DeleteMessageIntent, ForwardMessageIntent, ReplyMessageIntent, SetMailMessageIsRead, UnsubscribeMessageIntent, SendDraftIntent, SearchMailIntent…`, 23 in total). Notes has 51, including `PinNotesLinkAction, AppendToNoteLinkAction, MoveNotesToFolderLinkAction`. Music's bundle file has 8 (the ToolKit DB shows 12, so the catalog also merges extension intents).
- **Local measurement: Shortcuts CLI.** `shortcuts` has only `run`, `list`, `view` and `sign`. It runs **user-created shortcuts by name** (with `-i` input and `-o` output) and cannot list or invoke raw App Intents. `shortcuts list` took 36 ms on this machine (4 user shortcuts).

### Inferences
- The ToolKit SQLite is the best "options" source for intents. One query yields names, localized titles, parameter types, enum cases, search keywords and sample phrases, which makes it close to a ready-made retrieval index for voice commands ("archive this email" maps to `com.apple.mail.ArchiveMessageIntent`). It is a private, versioned (`v79`) format, so the schema could change in any OS update. Treat it as best-effort and read-only.
- Invocation path (speculative but practical): pre-generate one signed `.shortcut` per intent the agent cares about, using `shortcuts sign` on a generated plist. Have the user import it once, then call `shortcuts run "<name>"`. Calling intents directly through private frameworks (LinkServices/WorkflowKit) may work, but it is undocumented and could break with entitlements or OS updates. I did not test it.
- "Archive this email" resolves to a real entity only if the target message is known. Without an onscreen App Entity handle, the agent has to fall back to the selected message, for example through Mail's AppleScript `selection` (see section 2). So intents and AppleScript complement each other.
- Third-party adoption is low among the apps this project targets. Intents won't fix the Spotify or Arc AX gaps.

### Gaps
- No public statistic for how many App Store or popular Mac apps ship App Intents. The local count (about 10 third-party apps of about 70 non-Apple apps on this Mac) is one sample.
- I found no source on whether non-Apple processes can read the onscreen-awareness entity annotations (`appEntityIdentifier`). Probably not.
- Latency of `shortcuts run` for a single intent was not measured. I didn't want to run actions on the owner's Mac.
- MCP-in-App-Intents shipping status in macOS 26.x or 27 is unconfirmed.

## 2. AppleScript / JXA scripting dictionaries (sdef)

### Takeaway
AppleScript is the most practical semantic channel for the media and Apple apps. `sdef /path/App.app` dumps the dictionary in milliseconds, and an Apple Event round trip costs about 11 ms in-process (about 130 ms per `osascript` spawn). Spotify's dictionary has exactly the actions its AX tree hides (`next track`, `play track`, …). Electron apps (Slack, VS Code, Notion, Obsidian, Discord) have no dictionary at all. Every target app needs a one-time Automation (TCC) grant per sending app. Until it is granted, events fail with error -1743.

### Cited Findings
- **Local measurement: dictionary coverage (`sdef <app> | grep -c '<command '`).**

  | App | sdef commands |
  |---|---|
  | Spotify | 6 (`next track, previous track, playpause, pause, play, play track`) |
  | Music | 31 (incl. `next track, back track, search, add, download, fast forward, reveal`) |
  | Mail | 19 |
  | Safari | 10 |
  | Arc | 10 |
  | Chrome | 23 |
  | Notes | 2 |
  | Calendar | 8 |
  | Messages | 3 |
  | Reminders | 1 |
  | Finder | 25 |
  | Outlook | 70 |
  | Slack, VS Code, Notion, Obsidian, Discord, Raycast | 0 (no `NSAppleScriptEnabled`, no .sdef) |

  Command counts undercount capability, because much of the power lives in object properties (for example Spotify `player state`, `current track`, `sound volume`, `shuffling`; Mail `selection`, setting `mailbox` of a message). Mail's dictionary has no "archive" verb. The usual archive is `move msg to mailbox "Archive"`, while the App Intent path has `ArchiveMessageIntent`.
- Chrome and Arc dictionaries expose windows, tabs, URLs and `execute javascript` (Chrome; Arc has a similar verb). That gives a DOM path without CDP, but Chrome requires the user to enable "Allow JavaScript from Apple Events" in the View > Developer menu. This comes from general knowledge of Chrome's dictionary and was not re-verified for this note. Check with `sdef "/Applications/Google Chrome.app" | grep -i javascript`.
- **Local measurement: latency.** `osascript -e 'tell application "Finder" to get name of front window'` took 0.13–0.17 s wall time, mostly process spawn. A JXA loop sending 20 Apple Events to Finder from one process averaged **11.45 ms per event**.
- **Local measurement: TCC.** `tell application "Spotify" to get player state` from the terminal returned `Not authorized to send Apple events to Spotify. (-1743)` (0.15–0.30 s). Automation consent is per (sender, target) pair and is granted through a one-time prompt, or denied and then reset with `tccutil reset AppleEvents`. An app that sends Apple Events needs `NSAppleEventsUsageDescription`, and a hardened-runtime binary needs the `com.apple.security.automation.apple-events` entitlement. That last point is general Apple platform knowledge, not re-sourced here.
- An existing MCP server, steipete/macos-automator-mcp (MIT, about 870 stars at the snapshot), runs AppleScript/JXA from a knowledge base of scripts, raw code, or files. It says the host app needs Automation access, and "first-time attempts to control a new application... may still trigger a macOS confirmation prompt". — [GitHub](https://github.com/steipete/macos-automator-mcp)

### Inferences
- An agent can build "options" from sdef by parsing the XML once per app version (`<command>` names, `<parameter>` types, `<class>`/`<property>` with `access="rw"`). Map each verb or settable property to a templated JXA snippet, then cache it keyed on `CFBundleVersion`.
- For Spotify, sdef plus state properties covers next, previous, pause, play, play a URI, volume, shuffle, repeat and seek (`player position`). That is more than the 3 AX buttons. It can't like a song, add to a playlist or search; those need the Web API.
- Keep one long-lived JXA/ScriptingBridge process (or `NSAppleScript` in-process) to stay near 10 ms per event, instead of spawning `osascript` (about 130 ms) every time.
- Apple Events to an app that isn't running will launch it. Check `NSRunningApplication` first.

### Gaps
- No public coverage statistic for sdef among popular Mac apps. The table above is one machine.
- The latency of a Spotify event specifically (after granting TCC) was not measured.

## 3. Electron / Chromium-embedded apps: CDP or `--inspect` access

### Takeaway
The Chromium switch `--remote-debugging-port` is **not controlled by any Electron fuse**. Fuses only gate Node's `--inspect`, `ELECTRON_RUN_AS_NODE` and `NODE_OPTIONS`. So relaunching Slack, Discord, Notion, Obsidian or VS Code with `--remote-debugging-port=922x` usually gives renderer DOM access over CDP, unless the app strips the switch in its own code (unverified per app). The Node-side routes are locked down in Slack and Notion and open in VS Code. Spotify is CEF, not Electron.

### Cited Findings
- Electron fuses and their defaults:
  - `runAsNode` (ELECTRON_RUN_AS_NODE), default Enabled
  - `nodeOptions` (NODE_OPTIONS / NODE_EXTRA_CA_CERTS), default Enabled
  - `nodeCliInspect` ("whether the `--inspect`, `--inspect-brk`, etc. flags are respected"; when disabled it also blocks SIGUSR1), default Enabled
  - `embeddedAsarIntegrityValidation`, default Disabled
  - `onlyLoadAppFromAsar`, default Disabled
  - `cookieEncryption`, `grantFileProtocolExtraPrivileges` and others

  The fuses page never mentions `--remote-debugging-port`. — [Electron docs: Fuses](https://www.electronjs.org/docs/latest/tutorial/fuses)
- **Local measurement: fuse states (`npx @electron/fuses read --app …`).**

  | App | RunAsNode | NodeOptions env | NodeCliInspect | ASAR integrity | OnlyLoadFromAsar |
  |---|---|---|---|---|---|
  | Slack | Disabled | Disabled | Disabled | Enabled | Enabled |
  | Notion | Disabled | Disabled | Disabled | Enabled | Enabled |
  | Discord | Disabled | **Enabled** | Disabled | Disabled | Disabled |
  | Obsidian | Disabled | Disabled | Disabled | Disabled | Disabled |
  | VS Code | **Enabled** | **Enabled** | **Enabled** | Disabled | Disabled |

  Spotify ships `Chromium Embedded Framework.framework` (CEF), not Electron.
- Community agent "skills" document relaunching Slack and Discord with `--remote-debugging-port` (for example `slack --remote-debugging-port=9222`) and note that the flag must be present at launch. "Connection refused" means the app was started without it. These are secondary sources. — [vercel-labs agent-browser electron skill (mirror)](https://openskillindex.com/skills/vercel-labs-agent-browser-electron)
- Chrome itself (136+) ignores `--remote-debugging-port`/`--remote-debugging-pipe` on the **default** user-data-dir, to stop infostealers from pulling cookies. It requires a non-default `--user-data-dir`, which means a fresh, logged-out profile. Chrome for Testing and Chromium are unaffected. — [Chrome for Developers blog](https://developer.chrome.com/blog/remote-debugging-port); a report says the behavior persists in Chrome 151 — [trycua/cua issue #2916](https://github.com/trycua/cua/issues/2916)

### Inferences
- Electron apps: relaunching with `--remote-debugging-port` most likely works. This is unverified on these builds, because I didn't relaunch the user's apps, and it needs a quick test per app (`curl localhost:9222/json`).
- Slack and Notion block Node-level injection (no inspect, no RUN_AS_NODE, ASAR integrity on), so CDP renderer access is the only DOM route.
- VS Code is wide open (it also has a CLI and an extension API, which make a better channel).
- Spotify CEF: CEF apps honor `--remote-debugging-port` only if the embedder enables it. Spicetify-style tooling suggests Spotify does in some builds, but I did not verify this. Label it speculative.
- Security: an open CDP port on localhost gives any local process full control of the logged-in session (cookies, tokens). Bind it to 127.0.0.1, use a random port, close it after the session, and never enable it on the owner's daily profile without consent. Relaunching also interrupts the user, since the app has to quit.
- Chrome's 136 rule means the CDP route for the user's real logged-in Chrome is effectively closed. Use an extension (section 5) or Apple Events `execute javascript` instead. The Arc/Dia default-profile behavior wasn't checked; Arc tracks Chromium, so it probably inherited the same rule.

### Gaps
- No vendor documentation found on whether Slack, Discord or Notion deliberately strip `--remote-debugging-port`.
- Spotify CEF debugging-port behavior on current macOS builds is unverified.

## 4. Per-app APIs and MCP servers as action channels; how existing agents mix API and GUI

### Takeaway
Web APIs and MCP servers give the deepest actions (like a song, archive in Gmail, GitHub operations), but they need OAuth, network calls and per-service quotas. Spotify's Web API became much harder to use in 2026 (Development Mode limited to 5 users and Premium-only). Leading agents now pair GUI control with an optional code or tool path: UFO2's API-first-then-GUI "Puppeteer", and Agent S3's `call_code_agent`.

### Cited Findings
- Spotify Web API (Feb 6, 2026): new Development Mode client IDs (from Feb 11) require Premium, one client ID per developer, up to **5 authorized users**, and a smaller set of endpoints. Changes to endpoint access for existing integrations were later postponed. Quotas are now counted per developer account. — [Spotify developer blog](https://developer.spotify.com/blog/2026-02-06-update-on-developer-access-and-platform-security); [Spotify community thread](https://community.spotify.com/t5/Spotify-for-Developers/Web-API-quota-updates-for-Development-Mode/td-p/7508852). Extended Quota reportedly requires a registered business and 250k MAU (secondary source) — [Activepieces](https://www.activepieces.com/blog/spotify-web-api-free-access-modes-scopes-limits)
- UFO2 (Windows) uses a "Puppeteer" executor that "harmonizes GUI actions and native API calls", prefers an API when a semantically equivalent one exists, and falls back to GUI. APIs are registered with a Python decorator (name, arg schema, app binding), and Office APIs are built on COM. MCP is not used. Only 10–12 hand-written APIs. — [UFO2 arXiv 2504.14603](https://arxiv.org/html/2504.14603v2)
- Simular Agent S3 adds an optional local coding environment (`call_code_agent`) for tasks "that can be completed through programming rather than GUI interaction". It reaches 66% on OSWorld alone at 100 steps and 72.6% with Behavior Best-of-N. Simular warns that the environment runs arbitrary Python/Bash locally. — [simular-ai/Agent-S](https://github.com/simular-ai/agent-s); [PyPI gui-agents](https://pypi.org/project/gui-agents)
- macOS community MCP: steipete/macos-automator-mcp wraps AppleScript/JXA as MCP tools (see section 2). — [GitHub](https://github.com/steipete/macos-automator-mcp)

### Inferences
- For a voice agent running locally, the order of preference is probably: App Intent/Shortcut or AppleScript (local, about 10–100 ms, no network) > Web API/MCP (100s of ms, OAuth, quotas) > GUI click. Use Web APIs only for actions with no local verb (Spotify "save to library", Gmail archive in a browser tab).
- Gmail in a browser has no local semantic channel. Its keyboard shortcut (`e` = archive, when enabled) or the Gmail API (`users.messages.modify` removing `INBOX`; general knowledge, not re-sourced) are the options.

### Gaps
- I did not get primary sources on how Claude computer use, OpenAI Operator/Codex or Raycast AI combine API tools with GUI. Raycast exposes 3 intents locally, and Raycast AI Extensions/MCP are known from general knowledge only. Search budget ran out.
- No sourced latency numbers for Spotify Web API or Gmail API calls.

## 5. Browser extensions and content scripts (Arc, Chrome, Safari)

### Takeaway
An extension content script is the one way to export a page's interactive elements continuously from the user's real, logged-in Chrome or Arc profile, because Chrome 136+ blocks CDP on the default profile. Arc still runs Chrome Web Store extensions but has been in maintenance mode since May 2025.

### Cited Findings
- Arc entered maintenance mode around May 2025 (security and Chromium updates only) as The Browser Company moved to Dia. Atlassian acquired The Browser Company (reported $610M, closing Oct 2025). Arc and Dia both still run Chrome extensions. Mostly secondary or SEO sources. — [Wikipedia: Arc](https://en.wikipedia.org/wiki/Arc_(web_browser)); [Neowin](https://www.neowin.net/news/atlassian-acquires-the-browser-company-creators-of-arc-browser/); [SupaSidebar](https://supasidebar.com/blog/is-arc-browser-dead)
- CDP on the default Chrome profile is blocked since Chrome 136 (see section 3). — [Chrome blog](https://developer.chrome.com/blog/remote-debugging-port)

### Inferences
- Design: an MV3 extension whose content script runs on `document_idle`, with a `MutationObserver` to keep the list fresh. It walks the DOM with the same interactivity heuristics as Browser Use (covered elsewhere). It sends a compact element list through `chrome.runtime` native messaging to a local host process, and the host merges it with the AX options. This also fixes Arc pages with no AX web area, since the content script doesn't depend on Arc's AX bridge.
- Compared with CDP:
  - Pros: works on the real profile, needs no relaunch, can't be blocked by the 136 rule, and is a lower-privilege surface.
  - Cons: blocked on `chrome://` pages and the Web Store, needs installation per browser, and Safari needs a separate Safari Web Extension built and signed in Xcode.
  - Cons: no cross-origin iframe access without `all_frames`, and closed shadow roots stay opaque.
- Safari has no CDP. Its remote automation is WebDriver (`safaridriver`) in a separate automation session, so for Safari an extension or AppleScript `do JavaScript` is the realistic route. This comes from general knowledge.

### Gaps
- No primary Browser Company documentation found confirming the current state of Arc's extension API, such as MV3 parity or native messaging support.

## 6. Menu-bar commands, keyboard shortcuts and the Services menu

### Takeaway
The menu bar is the most universal semantic channel on macOS. Almost every app, Electron included, exposes its commands as `AXMenuBarItem` → `AXMenu` → `AXMenuItem` with titles and `AXMenuItemCmdChar`/`AXMenuItemCmdModifiers`. An agent can list these and press them through AX without clicking, even when the app's content has no AX buttons. I didn't find external sources for numbers in the budget available.

### Cited Findings
- The AppKit App Intents container has 18 tools in the local ToolKit DB (`com.apple.AppKit`), and `com.apple.WindowManager` has 7. That suggests generic window and menu operations are partly exposed as system intents. — local measurement (ToolKit query above)
- UFO2 relies on exactly this kind of always-available fallback: GUI actions when no API exists. — [UFO2 arXiv](https://arxiv.org/html/2504.14603v2)

### Inferences
- These come from macOS AX knowledge, not re-sourced:
  - Menu items support `AXPress` even while the menu is closed, so no visual change is needed.
  - Shortcut attributes let the agent send a keystroke instead.
  - Spotify's menu bar has a Playback menu (Play/Pause, Next, Previous, Shuffle, Repeat, Volume), which covers "skip this song" even though the window exposes only 3 AX buttons.
  - Slack, Notion and VS Code have rich menus too.
  - Enumerating them requires Accessibility permission, which the project already has.
- The Services menu (`NSServices` in each app's Info.plist, enumerable with `/System/Library/CoreServices/pbs -dump_pboard`, or from each bundle's Info.plist) gives text- and file-transform actions on the current selection. It is narrow but useful for "make a new note from this".
- Latency: walking a full app menu bar through AX is probably tens to hundreds of ms for large apps. This wasn't measured; cache it per app and invalidate when the window changes.

### Gaps
- No measured counts of menu items per app or of AX menu-walk latency. That would be a quick local benchmark worth running.
- No external source found on agent papers that use macOS menu bars specifically as an action space.

## 7. Evidence that API/tool channels improve agent success over GUI-only

### Takeaway
Results are consistent across benchmarks: adding API/MCP tools to a GUI agent raises success by about 4–10 points absolute and often cuts steps sharply. Tool-only agents do worse than hybrid ones, and models still under-use available tools (about 30–36% tool invocation rate).

### Cited Findings
- UFO2 ablation (OSWorld-W office tasks; the text says 27 tasks):

  | Model | Mode | Success rate | Steps (ACS) |
  |---|---|---|---|
  | GPT-4o | GUI-only | 16.3% | 13.8 |
  | GPT-4o | GUI+API | 22.4% | 12.9 |
  | o1 | GUI-only | 16.3% | 16.0 |
  | o1 | GUI+API | 24.5% | 6.6 (−58.5%) |

  Speculative multi-action cut steps by up to 51.5%. — [UFO2 arXiv 2504.14603 Tables 5, 7](https://arxiv.org/html/2504.14603v2)
- OSWorld-MCP (ICLR 2026): MCP tools raise Claude 4 Sonnet success from 38.9% to 45.0% at 50 steps per the abstract. Table 1 reports 40.1 → 49.5, so the sources disagree. The best tool invocation rate is only about 33–36%, and accuracy is 0 GUI-only versus 16.7% with MCP on tasks needing 4 tool calls. — [arXiv 2510.24563](https://arxiv.org/abs/2510.24563v2); [GitHub x-plug/osworld-mcp](https://github.com/x-plug/osworld-mcp); [ICLR poster](https://iclr.cc/virtual/2026/poster/10007133)
- MCPWorld (2025, Claude 3.7 Sonnet): Hybrid 75.1% versus GUI-only 70.7% versus MCP-only 53.2% success. Aggregator figure; primary paper not checked. — [Emergent Mind: hybrid GUI-API action space](https://api.emergentmind.com/topics/hybrid-gui-api-action-space)
- UltraCUA (Oct 2025): OSWorld 27.0% versus 23.4% (7B) and 41.0% versus 33.3% (32B) compared with GUI-only bases, with about 11% fewer steps. Aggregator figure. — [arXiv 2510.17790](https://arxiv.org/html/2510.17790v2)
- MAS-Bench (mobile, Sep 2025): shortcut-augmented hybrid agents reach up to 68.3% success with 39% better efficiency than GUI-only. — [arXiv 2509.06477](https://www.arxiv.org/abs/2509.06477)
- WebArena hybrid browsing+API agent (Song et al. 2024): API-only 29.2%. The hybrid figure is reported as 35.8% or 38.9% depending on the paper version. — [arXiv 2410.16464](https://arxiv.org/pdf/2410.16464v3)

### Inferences
- MCP-only falls below GUI-only (MCPWorld), so API channels should add options to the AX list rather than replace it.
- Low tool-invocation rates suggest that a small model like Kev will need tool options placed in the same candidate list as AX options, ranked by a retriever, instead of being offered as a separate mode.

### Gaps
- No macOS-specific hybrid benchmark found; all the evidence is from Windows, Ubuntu (OSWorld), web or mobile.
- MCPWorld and UltraCUA numbers come from aggregators and need checking against the primary PDFs.
