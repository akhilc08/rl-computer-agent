# MacBench {version} (draft, not released)

A grounding benchmark of native macOS apps and websites in Safari: given a screenshot of the frontmost window and a spoken-style command, pick the element to act on, or say the target isn't on screen.

Status: **draft**. Labels are Claude drafts, not yet human-verified. The privacy scan has open flags (see `privacy/summary.json`). Don't publish until both are cleared.

## Contents

| | {version} |
|---|---|
| Screens | {screens:,} ({screens_native:,} native, {screens_web:,} web) |
| Apps / sites | {apps} ({n_native_apps} native apps, {n_web_apps} websites) |
| Commands | {commands:,} |
| Target is an AX element | {kind_ax:,} |
| Target is OCR text only | {kind_ocr:,} |
| Target is a detector-only icon (no AX, no text) | {kind_icon:,} |
| Target is a menu-bar item (menus closed, no box) | {kind_menu:,} |
| Target not on screen ("none") | {kind_none:,} |
| Ambiguous (should ask the user) | {ambiguous:,} |
| Irreversible (send, delete, buy, ...) | {irreversible:,} |

The apps are the held-out ({split}) split of an app-level split: {native_apps}, plus {web_apps}. No screen or app here is in Kev's training data.

All screens come from throwaway EC2 Mac sandboxes seeded with made-up content (fake contacts, notes and mail, all `@example.com` with 555 phone numbers). Websites were browsed logged out.

## Files

- `images/<screen>.jpg`: the screenshot as captured (the window, or window plus menu bar).
- `screens.jsonl`: one row per screen.
  - App, kind (native/web), window frame, scale, light/dark mode.
  - `ax`: the accessibility elements of the window and menu bar, each with role, label, context and enabled state.
  - `ocr`: reference OCR lines (Apple Vision).
  - `detector`: reference detector boxes (GPA-GUI-Detector @640).
  - `options`: the merged reference list (AX + OCR + detector).
- `commands.jsonl`: one row per command.
  - `targets`: acceptable option ids.
  - `target_boxes`: in screen points and image pixels.
  - `action`, `text`, `text_source`, `absent`, `ambiguous`, `irreversible`.
  - `tracks`: which tracks the command counts in.
  - `verified`: false until a human checks it.
- `list_ax.jsonl`, `list_pixels.jsonl`: ready-made decision requests (state plus choice questions, Kev/Jev format) for list-picking models on the two tracks.
  - Detector-only icons in these lists carry the captions the labelling pass wrote, which are better than what a deployed captioner produces. Systems should be scored with their own captions too, and the paper must say which.

Boxes are `[x, y, w, h]` in screen points. Image pixels are `((x - frame_x) * scale, (y - frame_y) * scale, w * scale, h * scale)`; each box also carries `box_px`.

## Tracks

1. **Pixels only.** Input: the screenshot and the command. The reference OCR and detector output may be used, but AX may not. {track_pixels:,} commands; menu-bar targets are left out because a closed menu has no pixels.
2. **AX + pixels.** Input: the screenshot, the command and the AX tree (window plus menu bar). All {track_ax:,} commands.

A system answers with either a point or an option id, or with "none".
- **Point answers** (any system, including screenshot models) are right if the point falls inside any target box.
- **Option-id answers** are right if the id is in `targets`.
- **On the ready-made lists**, a pixels-track target counts if a list item overlaps a target box (IoU ≥ 0.5, or its centre is inside the box and it is at most 4× the box's area).
  - In {list_pixels_target_missing:,} pixels-track commands, no OCR or detector item covers the target, so `list_pixels.jsonl` leaves them out. They still count for point answers.
  - In {list_ax_target_missing:,} AX-track commands, the target sits past the 255-option cap, so `list_ax.jsonl` leaves them out too.

## Metrics

- **Element accuracy**, overall and by target kind: AX, OCR, icon, menu.
- **"None" accuracy:** on commands with `absent=true`, the system must abstain. Also report false-abstain on present targets.
- **Calibration**, for systems that give a confidence: ECE, plus the error rate among answers at ≥0.9 confidence, and coverage at 1% and 5% error.
- **Latency per decision**, with the hardware named. End to end on a Mac is the headline.
{small_note}

## License (suggested)

CC BY-NC 4.0, for research use only. Screenshots show Apple's UI and third-party websites; those rights stay with their owners, as in ScreenSpot and OmniACT.

## Known gaps

- **Unverified labels.** Every label is a Claude draft, so a human pass is needed before release.
- **Size.** {version} covers {screens:,} labelled screens out of {pool:,} {split}-app screens captured.
- **Real people on websites.** News and Wikipedia pages show real public figures' names and photos, and an anonymous editor's IP on a Wikipedia history page.
- **Device ids.** Disk Utility shows the sandbox Mac's volume UUID. See `privacy/flags.jsonl`.
