# Datasheet for MacBench

Following *Datasheets for Datasets* (Gebru et al., 2021).

## Motivation

**Why was it created?** Existing GUI grounding benchmarks have three gaps for desktop agents. They are mostly mobile, web and Windows. They give only pixels. They assume the target is always on screen. MacBench measures grounding on macOS, with and without the accessibility tree, and includes commands an agent should not act on directly.

**Who created it and who funded it?** Saiakhil Chilaka (Cornell University). The cloud machines were paid for by the author.

## Composition

**What do the instances represent?** Each instance is one command on one screen: a screenshot of a macOS window or a web page in Safari, and a natural-language command. The labels are the acceptable target elements, the action, and three flags:
- absent: the target is not on screen;
- ambiguous: the command fits two or more elements with different outcomes;
- irreversible: the action can't be undone, or other people see it.

**How many?** {commands} commands on {screens} screens:
- 33 apps and sites: 22 native, 11 web;
- {kind_ax} AX targets, {kind_ocr} OCR-only targets, {kind_icon} icon-only targets, {kind_menu} menu-bar targets, {kind_none} absent;
- {ambiguous} ambiguous and {irreversible} irreversible.

**Is it a sample?** Yes.
- The screens are {screens} of 9,338 captured screens from the 33 held-out apps (the 23 test-split apps plus 10 never-trained validation apps).
- They are spread evenly over apps.
- Hard screens (accessibility-poor, or with 3+ icons only a detector finds) alternate with ordinary ones, and light mode alternates with dark.
- Near-duplicates were removed.

**What does each instance contain?**
- The screenshot (JPEG), window geometry and scale.
- The accessibility elements of the window and menu bar.
- Apple Vision OCR lines and GUI-detector boxes.
- A merged option list.
- The command and its labels, with target boxes in points and pixels.

**Is there a label?** Yes, as above. Labels are LLM-drafted; {verified} of {commands} commands are human-verified (see Collection).

**Recommended splits.** MacBench is a test set only. Do not train on it. A model trained on the 92 other apps of the same collection has seen none of these apps.

**Errors, noise, redundancy.**
- OCR lines contain recognition errors.
- Detector boxes include non-controls.
- Some screenshots miss toolbar strips, which affects the pixels track only. Targets that are not visible were excluded at labelling time.
- {list_pixels_target_missing} pixels-track targets are covered by no reference OCR line or detector box.

**Is it self-contained?** Yes.

**Does it contain confidential or personal data?**
- The machines held only made-up personal data.
- Public web pages show real public figures (news, Wikipedia) and public usernames. These are left as on the live sites.
- Entries pointing at a private person or place were redacted, along with machine identifiers (see Preprocessing).

**Offensive content?** Screens from an adult forum section reached by the crawler were removed. News pages can contain distressing headlines.

## Collection process

**How was it collected?**
- A collector app ran for about 20 hours on each of four disposable EC2 Mac instances, at 1440×900 HiDPI.
- Each Mac was seeded with invented contacts, notes, reminders, calendar events and mail (`@example.com`, 555 numbers).
- The collector opened apps and logged-out websites and clicked through controls.
- At each step it saved the composited frontmost window, its accessibility tree and its geometry.
- All instances were terminated afterwards.

**Who labelled it?**
- Commands were drafted by LLM agents (Claude). The agents saw the screenshot, with detector-only boxes drawn and numbered, and the option list, and followed `tools/label_guide.txt`.
- A human then reviewed a stratified sample of 300 commands, all judged correct ({verified} remain after later filters).
- A model from another family (OpenAI, via the Codex CLI) judged every command against the screenshot; 316 confirmed errors were removed.

**Over what timeframe?** November 2026.

**Ethical review.** No human subjects were involved. No personal accounts or devices were used, and websites were visited logged out without posting or purchasing.

## Preprocessing

Every string and screenshot was scanned for emails, phone numbers, IP addresses, cloud host names, UUIDs, serial numbers, person names and faces. The fixes, all done by `scripts/bench/macbench_redact.py`:
- Browser (Safari, Firefox) tab, sidebar, history and address-bar entries that could point at a private person or place (every search-results tab, user-profile tab and address-bar suggestion) were replaced with "tab" or "suggestion" and blurred.
- Volume and snapshot UUIDs, cloud host names, the sandbox account name and public IPs were replaced and blurred.
- Screens with personal profiles, a system dialog naming the collection machine, or IP-range block notices were dropped.

## Uses

**Intended:** evaluating GUI grounding and abstention for computer-use agents on macOS, with or without accessibility data.

**Not suitable for:**
- training;
- commercial use;
- inferring anything about real people;
- measuring real users' phrasing, since commands reflect an LLM's phrasing guided by the labelling guide.

## Distribution

- **Where:** Hugging Face, versioned (v1).
- **License:** CC BY-NC 4.0 for labels and metadata, MIT for the scorer. Screenshot rights stay with Apple and the website owners; the screenshots are for non-commercial research only.

## Maintenance

- The author maintains it.
- Errors and privacy issues go to the repository's issue tracker.
- Fixes ship in point releases with a changelog. A privacy report is acted on by removing the item.
