"""Check labelled commands against their screens and print the mix (styles, verbs, flags).

  .venv/bin/python scripts/data/label_check.py [data/labels/commands.jsonl]
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERBS = {"click", "select", "toggle", "type", "open_menu_item"}


def problems(r, ids):
    out = []
    if r["verb"] not in VERBS:
        out.append(f"verb {r['verb']!r}")
    if r["style"] not in ("casual", "planner"):
        out.append(f"style {r['style']!r}")
    unknown = [t for t in r["targets"] if t not in ids]
    if unknown:
        out.append(f"unknown ids {unknown}")
    if r["absent"] != (not r["targets"]):
        out.append("absent must mean no targets")
    if r["ambiguous"] and len(r["targets"]) < 2:
        out.append("ambiguous needs 2+ targets")
    if r["verb"] == "type" and not (r.get("text") and r.get("text_source") in ("user", "compose")):
        out.append("type needs text and text_source")
    return out


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/labels/commands.jsonl"
    screens = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data/labels/screens.jsonl")}
    rows = [json.loads(l) for l in open(path)]
    bad = 0
    for r in rows:
        s = screens.get(r["screen"])
        p = ["unknown screen"] if s is None else problems(r, {o["id"] for o in s["options"]})
        if p:
            bad += 1
            print(r["id"], "|", r["command"], "|", "; ".join(p))
    n = len(rows)
    print(f"{n} commands on {len({r['screen'] for r in rows})} screens, {bad} with problems")
    print("style", dict(Counter(r["style"] for r in rows)), "verb", dict(Counter(r["verb"] for r in rows)))
    for k in ("ambiguous", "absent", "irreversible"):
        print(f"{k}: {sum(r[k] for r in rows) / max(n, 1):.0%}")


if __name__ == "__main__":
    main()
