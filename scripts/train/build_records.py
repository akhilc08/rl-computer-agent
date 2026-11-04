"""Build Kev training and eval records from our labelled Mac commands and the converted public sets.

Mac labels (data/labels/commands.jsonl + parts/part_*.jsonl) get every question: element, action, text
source, irreversible, ambiguous. Option lists follow serving: all window/page items, then the menu-bar
items the prefilter ranks highest for that command, then "none". Public sets get element and action only.

Splits:
  train / val      Mac labels (val = ~10% of screens, by screen) + OmniACT train / val + OS-Atlas
  heldout_site     every Mac command on the --heldout_sites sites (never trained on; the generalisation test)
  eval_ssv2 / eval_sspro / eval_omniact   public benchmarks and OmniACT's test split

Examples whose target didn't make it into the option list (menu prefilter or the 254-option cap) are
dropped from training and counted, since Kev can't get them right however it's trained.

  .venv/bin/python scripts/train/build_records.py [--heldout_sites neetcode]
"""
import argparse
import glob
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rlca.candidates import MAX_OPTIONS, MenuPrefilter  # noqa: E402
from rlca.elements import Element  # noqa: E402
from rlca.records import record  # noqa: E402

OUT = ROOT / "data/records"
MENU_K = 40
PUBLIC_ACTIONS = {"click": "click", "double_click": "click", "right_click": "click", "type": "type"}


def frac(key, salt="val"):
    return int(hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def mac_options(screen, command, prefilter):
    """Serving's option list for one command: page/window items, then the top MENU_K menu items."""
    page = [o for o in screen["options"] if not o["id"].startswith("m")]
    menu = [o for o in screen["options"] if o["id"].startswith("m")]
    if menu:
        els = [Element(id=o["id"], source="menubar", role=o["role"], label=o["label"], context=o["context"]) for o in menu]
        scores = prefilter.scores(command, els)
        menu = [menu[i] for i in sorted(range(len(menu)), key=lambda i: -scores[i])[:MENU_K]]
    return (page + menu)[: MAX_OPTIONS - 1]


def mac_records(heldout_sites, prefilter):
    screens = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data/labels/screens.jsonl")}
    files = [ROOT / "data/labels/commands.jsonl", *sorted(glob.glob(str(ROOT / "data/labels/parts/part_*.jsonl")))]
    for f in files:
        for line in open(f):
            c = json.loads(line)
            s = screens[c["screen"]]
            opts = mac_options(s, c["command"], prefilter)
            ids = {o["id"] for o in opts}
            targets = [t for t in c["targets"] if t in ids]
            if c["targets"] and not targets:
                yield "dropped_prefilter", None
                continue
            site = s["site"] or s["app"]
            split = "heldout_site" if site in heldout_sites else ("val" if frac(s["id"]) < 0.1 else "train")
            text_source = c.get("text_source") or "none"
            yield split, record(s["app"], c["command"], opts, targets, action=c["verb"], text_source=text_source,
                                irreversible=c["irreversible"], ambiguous=c["ambiguous"],
                                meta={"source": "mac_labels", "id": c["id"], "screen": s["id"], "site": site, "style": c["style"]})


def public_records():
    split_names = {"screenspot_v2": {"eval": "eval_ssv2"}, "screenspot_pro": {"eval": "eval_sspro"},
                   "omniact": {"train": "train", "val": "val", "test": "eval_omniact"}}
    for name in ("screenspot_v2", "screenspot_pro", "omniact", "os_atlas_macos", "os_atlas_linux"):
        d = ROOT / "data/converted" / name
        if not (d / "examples.jsonl").exists():
            continue
        screens = {json.loads(l)["image"]: json.loads(l) for l in open(d / "screens.jsonl")}
        for line in open(d / "examples.jsonl"):
            ex = json.loads(line)
            if not ex["target_ids"]:
                # Kept for eval (the pixel sources missed it, so the right answer is "none"); never trained on.
                if not ex["split"].startswith("eval"):
                    yield "dropped_unmatched", None
                    continue
            opts = screens[ex["image"]]["elements"][: MAX_OPTIONS - 1]
            ids = {o["id"] for o in opts}
            targets = [t for t in ex["target_ids"] if t in ids]
            if ex["target_ids"] and not targets:
                yield "dropped_cap", None
                continue
            action = PUBLIC_ACTIONS.get(ex.get("action"))
            if action is None:
                yield "dropped_action", None
                continue
            split = split_names.get(name, {}).get(ex["split"], "train")
            app = ex.get("app") or ex.get("platform") or name
            yield split, record(app, ex["instruction"], opts, targets, action=action,
                                text_source=("user" if action == "type" else None),
                                meta={"source": name, "id": ex["id"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout_sites", default="neetcode", help="comma-separated Mac sites/apps kept out of training")
    args = ap.parse_args()
    held = set(filter(None, args.heldout_sites.split(",")))
    OUT.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    by_split = defaultdict(list)
    for split, rec in [*mac_records(held, MenuPrefilter()), *public_records()]:
        counts[split] += 1
        if rec is not None:
            by_split[split].append(rec)
    for split, recs in by_split.items():
        with open(OUT / f"{split}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    src = Counter((s, r["_meta"]["source"]) for s, recs in by_split.items() for r in recs)
    for (s, name), n in sorted(src.items()):
        print(f"{s:14s} {name:16s} {n:6d}")
    print({k: v for k, v in counts.items() if k.startswith("dropped")})


if __name__ == "__main__":
    main()
