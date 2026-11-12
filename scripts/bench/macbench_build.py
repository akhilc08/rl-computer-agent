"""Build MacBench v0: a Mac-only grounding benchmark from the sandbox screens of the held-out (test-split) apps.

Source: data/sandbox_ds (screens.jsonl, options.jsonl, splits.json, label*/commands.jsonl, label*/captions.jsonl).
Only commands whose screen belongs to a test-split app are kept. Kev's training data (data/records_pixel) uses
train-split apps only, so nothing here was trained on. Screens on the drop list (drop_screens.tsv) are left out.

Output, data/macbench/v0/:
  images/<screen>.jpg     the screenshot as captured (window, or window + menu bar for "composite")
  screens.jsonl           per screen: app, kind, size, frame, scale, the AX elements (window + menu bar), the
                          reference OCR lines and detector boxes, and the reference option list (AX + OCR + detector)
  commands.jsonl          per command: targets (option ids) and target boxes (points and image pixels), action,
                          text + text_source, ambiguous, irreversible, absent ("none"), and which tracks it counts in
  list_ax.jsonl           the AX + pixels track as Kev/Jev decision requests (reference list, labels included)
  list_pixels.jsonl       the pixels-only track the same way (OCR + detector items only, no AX)
  stats.json

Boxes are [x, y, w, h] in screen points; image pixels = ((x - frame_x) * scale, (y - frame_y) * scale, w * scale,
h * scale). A point answer is right if it falls inside any target box.

  python scripts/bench/macbench_build.py [--out data/macbench/v0] [--no_images]
"""
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "data/sandbox_ds"
sys.path.insert(0, str(ROOT / "src"))


def to_px(box, frame, scale):
    if box is None:
        return None
    x, y, w, h = box
    return [round((x - frame[0]) * scale), round((y - frame[1]) * scale), round(w * scale), round(h * scale)]


def label_dirs():
    return sorted(p for p in DS.glob("label*") if (p / "commands.jsonl").exists())


def pixel_options(o, caps):
    """The pixels-only reference list, from the raw OCR lines and detector boxes (never AX's roles or labels).
    Detector boxes keep a caption only where the labelling pass captioned that box (detector-only icons)."""
    cap_by_box = {tuple(x["box"]): caps[x["id"]] for x in o["options"] if x["id"] in caps}
    items = [{"id": f"t{i}", "role": "text", "label": l["text"], "box": l["box"]} for i, l in enumerate(o["ocr"])]
    items += [{"id": f"i{j}", "role": "icon", "label": cap_by_box.get(tuple(b["box"]), ""), "box": b["box"]}
              for j, b in enumerate(o["gpa"])]
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/macbench/v0"))
    ap.add_argument("--split", default="test", help="which app split to build from (test; val for a dev set)")
    ap.add_argument("--no_images", action="store_true")
    args = ap.parse_args()
    from rlca.candidates import MAX_OPTIONS
    from rlca.records import record

    apps = json.load(open(DS / "splits.json"))["apps"]
    drop = {l.split("\t")[0] for l in open(DS / "drop_screens.tsv") if l.strip()}
    screens = {s["id"]: s for s in map(json.loads, open(DS / "screens.jsonl"))}
    opts = {o["id"]: o for o in map(json.loads, open(DS / "options.jsonl"))}
    captions = {}
    for d in label_dirs():
        if (d / "captions.jsonl").exists():
            for c in map(json.loads, open(d / "captions.jsonl")):
                captions[(c["screen"], c["option"])] = c["caption"]
    cmds = [c for d in label_dirs() for c in map(json.loads, open(d / "commands.jsonl"))
            if apps.get(screens.get(c["screen"], {}).get("app")) == args.split and c["screen"] not in drop]

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    stats = Counter()
    used = sorted({c["screen"] for c in cmds})
    with open(out / "screens.jsonl", "w") as f:
        for sid in used:
            s, o = screens[sid], opts[sid]
            frame, scale = s["frame"], s["scale"]
            options = []
            for x in o["options"]:
                x = dict(x)
                if x["id"].startswith("d") and (sid, x["id"]) in captions:
                    x["caption"] = captions[(sid, x["id"])]
                x["box_px"] = to_px(x["box"], frame, scale)
                options.append(x)
            ax = [dict(t, box_px=to_px(t.get("box"), frame, scale)) for t in s["targets"]]
            row = {"id": sid, "app": s["app"], "kind": s["kind"], "split": args.split, "title": s["title"],
                   "url": s["url"] if s["kind"] == "web" else None, "mode": s["mode"], "screen_size": s["size"],
                   "frame": frame, "scale": scale, "img_size": s["img_size"], "capture": s["capture"],
                   "ax_poor": s["ax_poor"], "image": f"images/{sid}.jpg", "ax": ax,
                   "ocr": [dict(l, box_px=to_px(l["box"], frame, scale)) for l in o["ocr"]],
                   "detector": [dict(b, box_px=to_px(b["box"], frame, scale)) for b in o["gpa"]],
                   "options": options}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if not args.no_images:
                shutil.copyfile(ROOT / s["shot"], out / "images" / f"{sid}.jpg")
            stats["screens"] += 1
            stats[f"screens_{s['kind']}"] += 1

    menubar = {sid: {t["id"]: t for t in screens[sid]["targets"] if t["source"] == "ax_menubar"} for sid in used}
    lists = {"ax": open(out / "list_ax.jsonl", "w"), "pixels": open(out / "list_pixels.jsonl", "w")}
    with open(out / "commands.jsonl", "w") as f:
        for c in cmds:
            s = screens[c["screen"]]
            frame, scale = s["frame"], s["scale"]
            by_id = {x["id"]: x for x in opts[c["screen"]]["options"]}
            boxes = [dict(tb, box_px=to_px(tb.get("box"), frame, scale)) for tb in c["target_boxes"]]
            kinds = {t[0] for t in c["targets"]}
            # Pixels track: needs a box to click. Menu items (closed menus, no box) can't be reached from pixels alone.
            pix_ok = c["absent"] or any(b.get("box") for b in boxes)
            tracks = ["ax"] + (["pixels"] if pix_ok else [])
            kind = ("none" if c["absent"] else "menu" if kinds == {"m"} else "icon" if kinds <= {"d"} and kinds
                    else "ocr" if kinds <= {"o", "d"} else "ax")
            row = {"id": c["id"], "screen": c["screen"], "app": s["app"], "command": c["command"], "style": c["style"],
                   "action": c["verb"], "targets": c["targets"], "target_boxes": boxes, "target_kind": kind,
                   "text": c["text"], "text_source": c["text_source"], "absent": c["absent"],
                   "ambiguous": c["ambiguous"], "irreversible": c["irreversible"], "tracks": tracks,
                   "labeler": c["labeler"], "verified": False}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["commands"] += 1
            stats[f"kind_{kind}"] += 1
            stats["ambiguous"] += c["ambiguous"]
            stats["irreversible"] += c["irreversible"]
            for t in tracks:
                stats[f"track_{t}"] += 1
            meta = {"source": "macbench", "id": c["id"], "screen": c["screen"]}
            # Reference lists for list-picking models: AX + OCR + detector (+ menu bar), and OCR + detector only.
            ax_opts = [{**x, "label": x["label"] or captions.get((c["screen"], x["id"]), "")} for x in by_id.values()]
            ax_opts += list(menubar[c["screen"]].values())
            caps = {i: t for (sid, i), t in captions.items() if sid == c["screen"]}
            pix_opts = pixel_options(opts[c["screen"]], caps)
            for t, options in (("ax", ax_opts), ("pixels", pix_opts)):
                if t not in tracks:
                    continue
                options = options[:MAX_OPTIONS]
                ids = {x["id"] for x in options}
                targets = [i for i in c["targets"] if i in ids]
                if t == "pixels":      # an AX-only target counts if a pixel item covers the same spot
                    targets = sorted({x["id"] for x in options for b in boxes if b.get("box") and inside_same(x["box"], b["box"])})
                if c["targets"] and not targets:
                    stats[f"list_{t}_target_missing"] += 1
                    continue
                rec = record(s["app"], c["command"], options, targets, action=c["verb"],
                             text_source=c.get("text_source") or "none", irreversible=c["irreversible"],
                             ambiguous=c["ambiguous"], meta=meta)
                lists[t].write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats[f"list_{t}"] += 1
    for fh in lists.values():
        fh.close()
    stats["apps"] = len({screens[s]["app"] for s in used})
    json.dump(dict(sorted(stats.items())), open(out / "stats.json", "w"), indent=1)
    print(json.dumps(dict(sorted(stats.items())), indent=1))
    write_readme(out, args.split, stats, [screens[s] for s in used],
                 sum(apps.get(s["app"]) == args.split for s in screens.values()))


def write_readme(out, split, stats, used, pool):
    """README from scripts/bench/macbench_README.md, with every count filled in from this build."""
    names = lambda kind: sorted({s["app"].removeprefix("web:") for s in used if s["kind"] == kind}, key=str.lower)
    native, web = names("native"), names("web")
    join = lambda xs: ", ".join(xs[:-1]) + (" and " if len(xs) > 1 else "") + xs[-1] if xs else "none"
    small = [k for k in ("ambiguous", "irreversible") if stats[k] < 50]
    fields = {k: 0 for k in ("kind_ax", "kind_ocr", "kind_icon", "kind_menu", "kind_none", "ambiguous", "irreversible",
                             "track_ax", "track_pixels", "list_ax_target_missing", "list_pixels_target_missing",
                             "screens_native", "screens_web")}
    fields.update(stats)
    fields.update(version=out.name, split=split, pool=pool, n_native_apps=len(native), n_web_apps=len(web),
                  native_apps=", ".join(native), web_apps=join(web),
                  small_note=f"- {join(small).capitalize()} counts are too small in {out.name} to report separately." if small else "- Report ambiguous and irreversible accuracy separately.")
    tpl = (Path(__file__).parent / "macbench_README.md").read_text()
    (out / "README.md").write_text(tpl.format(**fields))


def inside_same(a, b):
    """Two [x, y, w, h] boxes name the same element: IoU >= 0.5, or a's centre inside b with a at most 4x b's area."""
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    if union and inter / union >= 0.5:
        return True
    cx, cy = a[0] + a[2] / 2, a[1] + a[3] / 2
    return b[0] <= cx <= b[0] + b[2] and b[1] <= cy <= b[1] + b[3] and a[2] * a[3] <= 4 * b[2] * b[3]


if __name__ == "__main__":
    main()
