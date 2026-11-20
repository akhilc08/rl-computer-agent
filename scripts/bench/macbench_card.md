---
license: cc-by-nc-4.0
language:
- en
pretty_name: MacBench
size_categories:
- 10K<n<100K
task_categories:
- image-text-to-text
tags:
- gui-grounding
- computer-use
- gui-agents
- accessibility
- macos
- benchmark
configs:
- config_name: commands
  data_files:
  - split: test
    path: commands.jsonl
  default: true
- config_name: screens
  data_files:
  - split: test
    path: screens.jsonl
---

# MacBench

A grounding benchmark for macOS. Each item gives a screenshot and a command, such as "hit the little plus at the top". The task is to pick the element to act on, or to say the target isn't on screen.

- **Scale:** {commands} commands on {screens} screens. {screens_native} screens come from 15 native apps and {screens_web} from 8 websites in Safari.
- **Inputs:** every screen ships with its screenshot, the window's and menu bar's accessibility (AX) tree, reference OCR and reference detector boxes.
- **Tracks:** the same commands are scored *pixels only* and *AX + pixels*.

## What's in it

| | |
|---|---|
| Screens (native / web) | {screens} ({screens_native} / {screens_web}) |
| Commands | {commands} |
| Target is an AX element | {kind_ax} |
| Target is OCR text with no AX element | {kind_ocr} |
| Target is an icon only a detector finds | {kind_icon} |
| Target is a menu-bar item | {kind_menu} |
| Target not on screen ("none") | {kind_none} |
| Ambiguous (should ask the user) | {ambiguous} |
| Irreversible (send, delete, buy, ...) | {irreversible} |
| Human-checked sample (stratified, 60 per target kind) | {verified}, all judged correct |

**Apps:**
- Native: Audio MIDI Setup, Blender, Disk Utility, Font Book, GnuCash, Godot, IINA, Journal, Keka, Krita, MuseScore 4, Music, Notes, Photo Booth, Weather.
- Sites: bbc.com, irs.gov, khanacademy.org, npr.org, reddit.com, wikipedia.org, ycombinator.com, youtube.com.

**Collection:**
- Screens were captured on throwaway cloud Macs seeded with made-up people, notes and mail (`@example.com`, 555 numbers).
- Websites were browsed logged out.
- These apps are the held-out split of an app-level split, so a model trained on the other apps has seen none of them.

## Files

- `images/<screen>.jpg`: the frontmost window, composited with any sheet, popover and the menu bar.
- `screens.jsonl`, one row per screen:
  - app, kind (native/web), title, URL, window frame, scale, light/dark mode;
  - `ax`: role, label, context, enabled, box;
  - `ocr`: reference Apple Vision lines;
  - `detector`: reference GUI-detector boxes;
  - `options`: the merged reference list.
- `commands.jsonl`, one row per command:
  - `targets`: every acceptable option id;
  - `target_boxes`: in screen points and `box_px` image pixels;
  - `target_kind`, `action`, `text`, `text_source`;
  - `absent`, `ambiguous`, `irreversible`;
  - `tracks`, `verified`.
- `list_ax.jsonl`, `list_pixels.jsonl`: ready-made option lists for list-picking models. Each item is a state plus choice questions.
- `captions_deployed.jsonl`: the deployed captioner's name for every detector box, used in the lists.
- `annotations_icon_captions.jsonl`: the labelling pass's icon captions. These are annotations only, not inputs.
- `tools/score.py`: the scorer.
- `tools/label_guide.txt`: the labelling guide.

**Boxes** are `[x, y, w, h]`. Screen points map to image pixels as `((x - frame_x) * scale, (y - frame_y) * scale, w * scale, h * scale)`, and `box_px` is precomputed.

## Tracks and scoring

1. **Pixels only:** screenshot plus command. The reference OCR and detector output may be used, but not AX. This covers {track_pixels} commands; menu-bar targets are left out because a closed menu has no pixels.
2. **AX + pixels:** screenshot plus command plus the AX tree. All {track_ax} commands.

A system answers with an image point, an option id, or "none".

- A **point** is right if it falls inside any target box.
- An **option id** is right if it is in `targets`.
- On an absent command, only "none" is right.

Every command in a track counts. Commands a system gives no answer for count as wrong, and so do pixels-track targets that the ready-made list doesn't contain ({list_pixels_target_missing} commands). This way, list and point systems share one denominator.

```
python tools/score.py --bench . --track ax --pred predictions.jsonl
```

Predictions can take any of these shapes, one per line:
- `{"id", "option"}`
- `{"id", "point": [x, y] | null}`
- Kev/Jev's `{"id", "prediction": {"probabilities": {"element": {...}}}}`

Report:
- element accuracy, overall and by target kind;
- abstention accuracy on absent commands, and the false-abstain rate;
- accuracy on ambiguous and irreversible commands;
- calibration, if your system gives a confidence.

Icon items in the ready-made lists are named by a deployed captioner (`captions_deployed.jsonl`), not by the labelling pass. The labelling pass's icon captions are in `annotations_icon_captions.jsonl`; they echo the commands, so never feed them to a system. If you use your own captioner, say so.

## Baselines

Element accuracy (%). List models use the ready-made lists; Venus answers with points.

| Track | Model | All | AX | OCR | Icon | None |
|---|---|---|---|---|---|---|
| AX + pixels | Kev-0.8B | 56.6 | 63.2 | 73.3 | 18.3 | 44.9 |
| AX + pixels | Kev-0.8B fine-tuned on our 80 training apps | 82.5 | 87.7 | 77.6 | 49.9 | 87.7 |
| Pixels only | Kev-0.8B | 43.8 | 48.0 | 54.3 | 21.3 | 34.0 |
| Pixels only | Kev-0.8B fine-tuned | 62.2 | 60.5 | 58.8 | 50.4 | 88.3 |
| Pixels only | UI-Venus-1.5-2B | 78.1 | 84.6 | 61.8 | 86.8 | 35.3 |

## Privacy

Before release every string and screenshot was scanned for emails, phone numbers, IP addresses, cloud host names, UUIDs, serial numbers, names and faces.

**Redacted** (text replaced, boxes blurred):
- Safari tab, sidebar and suggestion entries that pointed at a private person or place: street addresses, real-estate listings, user profiles, @handles, name searches;
- volume and snapshot UUIDs;
- cloud host names and the sandbox account name;
- public IP addresses.

**Removed:** screens showing personal profiles with emails, a system dialog naming the collection machine, and an adult forum page.

**Left as on the live site:** public figures in news content, public business phone numbers, and public usernames on public pages. If you find something that should not be here, open an issue and it will be removed in a point release.

## License

- Labels, metadata and lists: CC BY-NC 4.0.
- Scorer: MIT.
- Screenshots show Apple's interface and third-party websites. Those rights stay with their owners, and the screenshots are included for non-commercial research only.

## Citation

```
@inproceedings{macbench2027,
  title     = {MacBench: Grounding Commands on macOS Apps and the Web, with Accessibility and Pixel Tracks},
  author    = {Chilaka, Saiakhil},
  booktitle = {Companion Proceedings of the ACM Web Conference 2027},
  year      = {2027}
}
```
