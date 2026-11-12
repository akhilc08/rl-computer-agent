"""Privacy redaction for MacBench, applied by macbench_build.py to every screen before release.

What leaks on the sandbox screens and how it is removed (text everywhere it appears, pixels by blurring):
  - Safari's other tabs, the tab sidebar, History menu items and address-bar suggestions show pages the sandbox
    browsed. That history is automated, not a person's, and most of it is public pages ("Search : NPR"), so those
    stay. Entries that point at a real private person or place are redacted: street addresses and real-estate
    listings, user profile pages, @handles, and searches for a person's name. Labels become "tab" /
    "suggestion" / "history item" and the boxes are blurred; commands that target one are dropped.
  - Machine identifiers in any text: volume and snapshot UUIDs (OCR-tolerant), long hex ids on Disk Utility,
    EC2 host names and the "ec2-user" account, public IPv4 addresses (e.g. a Wikipedia editor's IP).
    Replaced with [UUID] / [ID] / user / host / [IP]; the element or OCR line is blurred.
OCR lines whose centre falls in a redacted box are dropped. No MacBench command targets any redacted element; the
build checks this and drops a command if one ever does.
"""
import ipaddress
import re

from PIL import Image, ImageFilter

UUID = re.compile(r"\b[0-9A-Z]{8}-[0-9A-Z]{4}-[0-9A-Z]{4}-[0-9A-Z]{4}-[0-9A-Z]{12}\b", re.I)
HEX_ID = re.compile(r"\b(?=[0-9A-F]*\d)[0-9A-F]{10,}\b")
EC2 = [(re.compile(r"\bec2-user\b", re.I), "user"),
       (re.compile(r"\bip-\d+-\d+-\d+-\d+(\.ec2\.internal)?\b", re.I), "host"),
       (re.compile(r"\bec2-\d+-\d+-\d+-\d+[\w.-]*", re.I), "host"),
       (re.compile(r"\bi-[0-9a-f]{8,17}\b"), "[ID]")]
IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
# Invented by a labelling agent but not a reserved domain: swap for the seeded fake user.
EMAIL_SWAP = {"kev.tester@icloud.com": "sam.rivera@example.com"}


def _public_ip(m):
    try:
        ip = ipaddress.ip_address(m.group(0))
    except ValueError:
        return False
    return ip.is_global


def scrub(text, app):
    """Text with machine identifiers replaced."""
    if not text:
        return text
    out = UUID.sub("[UUID]", text)
    if app == "Disk Utility":
        out = HEX_ID.sub("[ID]", out)
    for rx, rep in EC2:
        out = rx.sub(rep, out)
    out = IPV4.sub(lambda m: "[IP]" if _public_ip(m) else m.group(0), out)
    for a, b in EMAIL_SWAP.items():
        out = out.replace(a, b)
    return out


def _centre_in(a, boxes):
    cx, cy = a[0] + a[2] / 2, a[1] + a[3] / 2
    return any(b[0] <= cx <= b[0] + b[2] and b[1] <= cy <= b[1] + b[3] for b in boxes)


RISKY = re.compile(r"zillow|redfin|realtor\.com|\bMLS\b|\b\d{2,6} [A-Z][a-z]+(?: [A-Z][a-z]+)* (?:St|Street|Ave|Avenue|Rd|Road|"
                   r"Blvd|Dr|Drive|Ln|Lane|Ct|Court|Way|Pl|Place|Ter|Terrace|Cir|Circle)\b|\bUser:|user\?id=|\bprofile\b|"
                   r"linkedin|whitepages|spokeo|@[A-Za-z0-9_]{3,}|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}", re.I)
NAME_SEARCH = re.compile(r"^\s*([A-Z][a-z]+) ([A-Z][a-z]+(?:-[A-Z][a-z]+)?)\s+-\s+(?:Google Search|Bing|DuckDuckGo)")


def _given_names():
    import ast
    from pathlib import Path
    src = (Path(__file__).parent / "macbench_privacy.py").read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "GIVEN":
            return {w for c in ast.walk(node.value) if isinstance(c, ast.Constant) and isinstance(c.value, str)
                    for w in c.value.split()}
    return set()


GIVEN = _given_names()


def risky(label):
    if not label:
        return False
    m = NAME_SEARCH.match(label)
    return bool(RISKY.search(label) or (m and m.group(1) in GIVEN))


def history_label(screen, el):
    """The replacement label if this AX element shows a browsed page that points at a private person or place."""
    if screen["kind"] != "web" or not risky(el.get("label")):
        return None
    ctx = " ".join(map(str, el.get("context") or [])).lower()
    box = el.get("box")
    if el["role"] == "AXRadioButton" and box and box[1] - screen["frame"][1] < 100 and el.get("label") != screen["title"]:
        return "tab"
    if "sidebar" in ctx and el["role"] in ("AXRow", "AXCell", "AXStaticText", "AXLink", "AXButton") and el.get("label"):
        return "tab"
    if "auto-complete" in ctx or "autocomplete" in ctx:
        return "suggestion"
    if el.get("source") == "ax_menubar" and "history" in ctx:
        return "history item"
    return None


def redact(screen, opts):
    """Redacted copies of a screen's AX elements ("targets"), option list, OCR lines and detector boxes.

    Returns (targets, options, ocr, gpa, blur_boxes, redacted_option_ids); boxes in screen points."""
    app = screen["app"]
    blur, targets = [], []
    for el in screen["targets"]:
        el = dict(el)
        new = history_label(screen, el)
        if new is None:
            s = scrub(el.get("label"), app)
            new = s if s != el.get("label") else None
        if new is not None:
            el["label"] = new
            if el.get("box"):
                blur.append(el["box"])
        targets.append(el)
    ocr = []
    for line in opts["ocr"]:
        if _centre_in(line["box"], blur):
            continue
        s = scrub(line["text"], app)
        if s != line["text"]:
            blur.append(line["box"])
            line = dict(line, text=s)
        ocr.append(line)
    # The option list repeats AX labels (a<i> = targets[i]) and OCR lines (o*) and carries detector boxes (d*).
    options, dropped = [], set()
    for x in opts["options"]:
        x = dict(x)
        if x["id"].startswith("a") and x["id"][1:].isdigit() and int(x["id"][1:]) < len(targets):
            t = targets[int(x["id"][1:])]
            if t.get("box") == x.get("box"):
                x["label"] = t["label"]
        elif x["id"].startswith("o"):
            if _centre_in(x["box"], blur) and not any(l["box"] == x["box"] for l in ocr):
                dropped.add(x["id"])
                continue
            x["label"] = scrub(x["label"], app)
        options.append(x)
    return targets, options, ocr, opts["gpa"], blur, dropped


def blur_image(src, dst, boxes, frame, scale, radius=14, pad=3):
    """Copy the screenshot with each box (screen points) Gaussian-blurred."""
    img = Image.open(src).convert("RGB")
    for x, y, w, h in boxes:
        l = max(0, int((x - frame[0]) * scale) - pad)
        t = max(0, int((y - frame[1]) * scale) - pad)
        r = min(img.width, int((x + w - frame[0]) * scale) + pad)
        b = min(img.height, int((y + h - frame[1]) * scale) + pad)
        if r > l and b > t:
            img.paste(img.crop((l, t, r, b)).filter(ImageFilter.GaussianBlur(radius)), (l, t))
    img.save(dst, quality=92)
