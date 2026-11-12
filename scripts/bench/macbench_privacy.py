"""Privacy scan of a MacBench build before release: flags, never deletes.

Text: every string a screen or command carries (window title, URL, AX labels, OCR lines, option labels, commands,
typed text) is checked for emails, phone numbers, IPv4 addresses, AWS/EC2 traces (ip-*.ec2.internal hostnames,
instance ids, 12-digit account-like numbers, ec2-user), UUIDs, serial numbers, and person names. The seeded fake
content (scripts/sandbox/seed_content.py: FIRST x LAST names, @example.com mail, (555) phones) is known fake and
not flagged. Images: Apple Vision face detection on every screenshot.

Writes <bench>/privacy/flags.jsonl (one row per hit) and <bench>/privacy/summary.json (counts by type, screens hit).
Runs in the project venv (needs pyobjc Vision):
  .venv/bin/python scripts/bench/macbench_privacy.py [--bench data/macbench/v0] [--no_faces]
"""
import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def seeded_names():
    """FIRST and LAST from seed_content.py, read without importing it (it drives the sandbox when run)."""
    tree = ast.parse((ROOT / "scripts/sandbox/seed_content.py").read_text())
    names = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("FIRST", "LAST"):
            names[node.targets[0].id] = set(ast.literal_eval(node.value))
    return names.get("FIRST", set()), names.get("LAST", set())


# Common given names (US/UK/global top lists) for the "First Last" check; seeded fake first names are removed later.
GIVEN = set("""James John Robert Michael William David Richard Joseph Thomas Charles Christopher Daniel Matthew Anthony
Mark Donald Steven Paul Andrew Joshua Kenneth Kevin Brian George Timothy Ronald Edward Jason Jeffrey Ryan Jacob Gary
Nicholas Eric Jonathan Stephen Larry Justin Scott Brandon Benjamin Samuel Gregory Alexander Frank Patrick Raymond Jack
Dennis Jerry Tyler Aaron Jose Adam Nathan Henry Douglas Zachary Peter Kyle Ethan Walter Noah Jeremy Christian Keith
Roger Terry Gerald Harold Sean Austin Carl Arthur Lawrence Dylan Jesse Jordan Bryan Billy Joe Bruce Gabriel Logan
Albert Willie Alan Juan Wayne Elijah Randy Roy Vincent Ralph Eugene Russell Bobby Mason Philip Louis Mary Patricia
Jennifer Linda Elizabeth Barbara Susan Jessica Sarah Karen Lisa Nancy Betty Margaret Sandra Ashley Kimberly Emily Donna
Michelle Carol Amanda Dorothy Melissa Deborah Stephanie Rebecca Sharon Laura Cynthia Kathleen Amy Angela Shirley Anna
Brenda Pamela Emma Nicole Helen Samantha Katherine Christine Debra Rachel Carolyn Janet Catherine Maria Heather Diane
Ruth Julie Olivia Joyce Virginia Victoria Kelly Lauren Christina Joan Evelyn Judith Megan Andrea Cheryl Hannah Jacqueline
Martha Gloria Teresa Ann Sara Madison Frances Kathryn Janice Jean Abigail Alice Judy Sophia Grace Denise Amber Doris
Marilyn Danielle Beverly Isabella Theresa Diana Natalie Brittany Charlotte Marie Kayla Alexis Lori Donald Joe Kamala
Barack Hillary Elon Taylor Kim Vladimir Xi Narendra Rishi Emmanuel Olaf Volodymyr Benjamin Keir Justin""".split())

PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(r"(?<!\d)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\d)"),
    "ipv4": re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])"),
    "ec2_host": re.compile(r"\bip-\d+-\d+-\d+-\d+\b|\.ec2\.internal\b|\bec2-\d+-\d+-\d+-\d+\b|\bec2-user\b", re.I),
    "ec2_instance": re.compile(r"\bi-[0-9a-f]{8,17}\b"),
    "aws_account_like": re.compile(r"(?<![\d-])\d{12}(?![\d-])"),
    "uuid": re.compile(r"\b[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\b"),
    "serial": re.compile(r"serial(?: number)?[:\s]+[A-Z0-9]{8,14}|\b(?=[A-Z0-9]{10,12}\b)(?=\w*\d)(?=\w*[A-Z])(?:[A-Z0-9]){10,12}\b", re.I),
    "name": re.compile(r"\b([A-Z][a-z]+) ([A-Z][a-z]+(?:-[A-Z][a-z]+)?)\b"),
}


def fake(kind, m, first, last):
    s = m.group(0)
    if kind == "email":
        return s.lower().endswith("@example.com")
    if kind == "phone":
        return bool(re.search(r"\(?555\)?[ .-]", s))
    if kind == "ipv4":
        parts = [int(p) for p in s.split(".")]
        return any(p > 255 for p in parts) or s.startswith(("0.", "127.")) or parts[0] < 10 and parts[1] < 10   # version numbers
    if kind == "name":
        f, l = m.group(1), m.group(2)
        return f not in GIVEN or f in first or l in last
    if kind == "serial":
        return s.isdigit()
    return False


def texts(screen, cmds):
    yield "title", screen.get("title") or ""
    yield "url", screen.get("url") or ""
    for e in screen["ax"]:
        yield f"ax:{e['id']}", e.get("label") or ""
    for i, l in enumerate(screen["ocr"]):
        yield f"ocr:{i}", l["text"]
    for o in screen["options"]:
        yield f"option:{o['id']}", (o.get("label") or "") + " " + (o.get("caption") or "")
    for c in cmds:
        yield f"command:{c['id']}", c["command"] + " " + (c.get("text") or "")


def faces(path):
    import Quartz
    import Vision
    url = Quartz.CFURLCreateWithFileSystemPath(None, str(path), 0, False)
    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, None)
    req = Vision.VNDetectFaceRectanglesRequest.alloc().init()
    ok, _ = handler.performRequests_error_([req], None)
    if not ok:
        return []
    return [(float(r.confidence()), [round(v, 3) for v in (r.boundingBox().origin.x, r.boundingBox().origin.y,
                                                            r.boundingBox().size.width, r.boundingBox().size.height)])
            for r in (req.results() or [])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=str(ROOT / "data/macbench/v0"))
    ap.add_argument("--no_faces", action="store_true")
    args = ap.parse_args()
    bench = Path(args.bench)
    first, last = seeded_names()
    screens = [json.loads(l) for l in open(bench / "screens.jsonl")]
    cmds = defaultdict(list)
    for c in map(json.loads, open(bench / "commands.jsonl")):
        cmds[c["screen"]].append(c)
    (bench / "privacy").mkdir(exist_ok=True)
    counts, hit_screens, seen = Counter(), defaultdict(set), set()
    with open(bench / "privacy/flags.jsonl", "w") as f:
        for s in screens:
            for field, text in texts(s, cmds[s["id"]]):
                for kind, rx in PATTERNS.items():
                    for m in rx.finditer(text):
                        if fake(kind, m, first, last):
                            continue
                        key = (s["id"], kind, m.group(0))
                        if key in seen:
                            continue
                        seen.add(key)
                        counts[kind] += 1
                        hit_screens[kind].add(s["id"])
                        f.write(json.dumps({"screen": s["id"], "app": s["app"], "field": field, "type": kind,
                                            "match": m.group(0), "text": text[:160]}, ensure_ascii=False) + "\n")
            if not args.no_faces:
                found = [x for x in faces(bench / s["image"]) if x[0] >= 0.5]
                if found:
                    counts["face"] += len(found)
                    hit_screens["face"].add(s["id"])
                    f.write(json.dumps({"screen": s["id"], "app": s["app"], "field": "image", "type": "face",
                                        "faces": found}) + "\n")
    summary = {"screens": len(screens), "hits": dict(counts),
               "screens_hit": {k: len(v) for k, v in hit_screens.items()},
               "screens_any": len(set().union(*hit_screens.values())) if hit_screens else 0}
    json.dump(summary, open(bench / "privacy/summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
