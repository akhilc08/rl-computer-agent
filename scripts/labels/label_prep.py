"""Build the screens to write labelled commands for: one per distinct page from an explorer run, plus the
native-app AX snapshots, each with the option list Kev would see.

Options are the AX window elements (w*), OCR lines no AX element covers that are short enough to be a
label rather than a paragraph (o*), captioned detector icons (d*), and enabled menu-bar items (m*).
Screens where the page never loaded (claude.ai stuck on its spinner) are skipped.

Writes data/labels/screens.jsonl (one row per screen, with options) and data/labels/screens/<id>.txt
(the same options as text, for whoever writes the commands).

  .venv/bin/python scripts/labels/label_prep.py --run data/explore/<stamp>
  .venv/bin/python scripts/labels/label_prep.py --native data/native/<stamp> [...]   (adds native-app screens)

Native runs have no URL, so one screen per distinct window (its element signature), and screens whose
(role, label) set is at least --near (Jaccard) like one already kept in the same app are dropped: the
explorer revisits the same Notes / Slack views with different text. --native appends to screens.jsonl.
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


def option(eid, e, source):
    return {"id": eid, "source": source, "role": e["role"], "label": e["label"], "context": e.get("context", []),
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
        opts = [option(e["id"], e, "window") for e in s["elements"]]
        extras = sources.get(s["shot"], {}).get("items", [])
        no, nd = 0, 0
        for it in extras:
            if "ax" in it["sources"] or not it["label"].strip():
                continue
            if "ocr" in it["sources"]:
                if len(it["label"].split()) > MAX_OCR_WORDS:
                    continue
                opts.append(option(f"o{no}", {**it, "role": "text"}, "ocr"))
                no += 1
            else:
                opts.append(option(f"d{nd}", {**it, "role": "icon"}, "det"))
                nd += 1
        # Explorer states number menu items per menu (m0 in Arc and m0 in File), so renumber them.
        opts += [option(f"m{i}", e, "menubar") for i, e in enumerate(e for e in s.get("menubar", []) if e.get("enabled", True))]
        yield {"id": f"{run.name}-{s['id']}", "kind": "explore", "app": s["app"], "site": s["target"],
               "url": s["url"], "title": s.get("title"), "shot": str((run / s["shot"]).relative_to(ROOT)), "options": opts}


def native_screens(runs, near):
    kept = []
    for run in runs:
        best = {}
        for line in open(run / "states.jsonl"):
            st = json.loads(line)
            k = (st["app"], st["sig"])
            if k not in best or (st.get("shot") and not best[k].get("shot")):
                best[k] = st
        for st in best.values():
            toks = {(e["role"], e["label"]) for e in st["elements"]}
            if not toks or any(a == st["app"] and len(toks & t) / len(toks | t) >= near for a, t in kept):
                continue
            kept.append((st["app"], toks))
            opts = [option(e["id"], e, "window") for e in st["elements"]]
            opts += [option(f"m{i}", e, "menubar") for i, e in enumerate(e for e in st.get("menubar", []) if e.get("enabled", True))]
            yield {"id": f"{run.name}-{st['id']}", "kind": "native", "app": st["app"], "site": None, "url": None,
                   "title": st.get("title"), "shot": str((run / st["shot"]).relative_to(ROOT)) if st.get("shot") else None,
                   "options": opts}


def snapshot_screens():
    for f in sorted((ROOT / "data/snapshots").glob("*.json")):
        snap = json.load(open(f))
        opts = [option(e["id"], e, e["source"]) for e in snap["elements"] if e["source"] == "window" or e.get("enabled", True)]
        yield {"id": f"snap-{f.stem}", "kind": "snapshot", "app": snap["app"], "site": None, "url": None,
               "title": snap.get("window_title"), "shot": None, "options": opts}


def render(screen):
    """Options as compact text: window/page items one per line, menu items grouped by menu."""
    lines = [f"App: {screen['app']}", f"Title: {screen['title']}"]
    if screen["url"]:
        lines.append(f"URL: {screen['url']}")
    menus = {}
    for o in screen["options"]:
        if o["source"] == "menubar":
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
    ap.add_argument("--run")
    ap.add_argument("--native", nargs="+", help="native explorer runs to add to the existing screens")
    ap.add_argument("--near", type=float, default=0.95)
    args = ap.parse_args()
    assert args.run or args.native, "--run or --native"
    (OUT / "screens").mkdir(parents=True, exist_ok=True)
    n = 0
    if args.native:
        have = {json.loads(line)["id"] for line in open(OUT / "screens.jsonl")} if (OUT / "screens.jsonl").exists() else set()
        new = [x for x in native_screens([ROOT / r for r in args.native], args.near) if x["id"] not in have]
    else:
        new = [*explore_screens(ROOT / args.run), *snapshot_screens()]
    with open(OUT / "screens.jsonl", "a" if args.native else "w") as f:
        for screen in new:
            f.write(json.dumps(screen) + "\n")
            (OUT / "screens" / f"{screen['id']}.txt").write_text(render(screen))
            n += 1
    print(f"{n} screens -> {OUT / 'screens.jsonl'}")


if __name__ == "__main__":
    main()
