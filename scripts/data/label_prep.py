"""Build the screens to write labelled commands for: one per distinct page from an explorer run, plus the
native-app AX snapshots, each with the option list Kev would see.

Options are the AX window elements (w*), OCR lines no AX element covers that are short enough to be a
label rather than a paragraph (o*), captioned detector icons (d*), and enabled menu-bar items (m*).
Screens where the page never loaded (claude.ai stuck on its spinner) are skipped.

Writes data/labels/screens.jsonl (one row per screen, with options) and data/labels/screens/<id>.txt
(the same options as text, for whoever writes the commands).

  .venv/bin/python scripts/data/label_prep.py --run data/explore/<stamp>
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/labels"
MAX_OCR_WORDS = 6
SKIP_TARGETS = {"claude"}


def page_key(state):
    return state["target"], state["url"].split("#")[0].split("?")[0]


def option(eid, e):
    return {"id": eid, "role": e["role"], "label": e["label"], "context": e.get("context", []),
            "enabled": e.get("enabled", True), "box": e.get("frame") or e.get("box")}


def explore_screens(run):
    sources = {}
    if (run / "sources.jsonl").exists():
        for line in open(run / "sources.jsonl"):
            r = json.loads(line)
            sources[r["shot"]] = r
    seen = {}
    for line in open(run / "states.jsonl"):
        s = json.loads(line)
        if s["target"] in SKIP_TARGETS or not s["url"]:
            continue
        # One screen per page; prefer a state whose screenshot went through the source replay.
        k = page_key(s)
        if k not in seen or (s["shot"] in sources and seen[k]["shot"] not in sources):
            seen[k] = s
    for s in seen.values():
        opts = [option(e["id"], e) for e in s["elements"]]
        extras = sources.get(s["shot"], {}).get("items", [])
        no, nd = 0, 0
        for it in extras:
            if "ax" in it["sources"] or not it["label"].strip():
                continue
            if "ocr" in it["sources"]:
                if len(it["label"].split()) > MAX_OCR_WORDS:
                    continue
                opts.append(option(f"o{no}", {**it, "role": "text"}))
                no += 1
            else:
                opts.append(option(f"d{nd}", {**it, "role": "icon"}))
                nd += 1
        # Explorer states number menu items per menu (m0 in Arc and m0 in File), so renumber them.
        opts += [option(f"m{i}", e) for i, e in enumerate(e for e in s.get("menubar", []) if e.get("enabled", True))]
        yield {"id": f"{run.name}-{s['id']}", "kind": "explore", "app": s["app"], "site": s["target"],
               "url": s["url"], "title": s.get("title"), "shot": str((run / s["shot"]).relative_to(ROOT)), "options": opts}


def snapshot_screens():
    for f in sorted((ROOT / "data/snapshots").glob("*.json")):
        snap = json.load(open(f))
        opts = [option(e["id"], e) for e in snap["elements"] if e["source"] == "window" or e.get("enabled", True)]
        yield {"id": f"snap-{f.stem}", "kind": "snapshot", "app": snap["app"], "site": None, "url": None,
               "title": snap.get("window_title"), "shot": None, "options": opts}


def render(screen):
    """Options as compact text: window/page items one per line, menu items grouped by menu."""
    lines = [f"App: {screen['app']}", f"Title: {screen['title']}"]
    if screen["url"]:
        lines.append(f"URL: {screen['url']}")
    menus = {}
    for o in screen["options"]:
        if o["id"].startswith("m"):
            menus.setdefault(" › ".join(o["context"]) or "?", []).append(f"{o['id']} {o['label']}")
            continue
        role = o["role"].removeprefix("AX").lower()
        ctx = f"  · in: {' › '.join(o['context'][-2:])}" if o["context"] else ""
        off = " · disabled" if not o["enabled"] else ""
        lines.append(f'{o["id"]} {role} "{o["label"]}"{ctx}{off}')
    if menus:
        lines.append("Menu bar:")
        lines += [f"  [{m}] " + " | ".join(items) for m, items in menus.items()]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    (OUT / "screens").mkdir(parents=True, exist_ok=True)
    n = 0
    with open(OUT / "screens.jsonl", "w") as f:
        for screen in [*explore_screens(ROOT / args.run), *snapshot_screens()]:
            f.write(json.dumps(screen) + "\n")
            (OUT / "screens" / f"{screen['id']}.txt").write_text(render(screen))
            n += 1
    print(f"{n} screens -> {OUT / 'screens.jsonl'}")


if __name__ == "__main__":
    main()
