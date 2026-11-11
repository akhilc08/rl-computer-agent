"""Kev option lists for the labelled sandbox commands: AX + OCR + ranked detector icons (screen-options pipeline).

Each list keeps every AX element and every OCR line AX doesn't cover, as before. The detector-only icons are
replaced by screen-options' pipeline: the AX-trained tiled detector (axgpa_1280), a SigLIP ranker that scores
each icon against the command (context crops), and the AX-style labeller that captions only the icons kept.

  prep   (laptop)  data/pixel_lists/bundle.jsonl + images/<id>.jpg (longest side 1280), to rsync to a GPU box
  run    (GPU box) -> data/pixel_lists/extras.jsonl: per screen the new icons, each command's ranker scores,
                    captions for the union of every command's top --cap_k icons
  build  (laptop)  -> data/records_pixel/{train,val}.jsonl Kev records, keeping the top --k icons per command

On the box (screen-options checked out at ~/screen-options, its venv):
  PYTHONPATH=~/screen-options/src:~/screen-options/scripts/eval python scripts/train/pixel_lists.py run \
      --det ~/screen-options/runs/det/axgpa_1280/weights/best.pt --ranker ~/screen-options/runs/siglip/ctx/best
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "data/sandbox_ds"
OUT = ROOT / "data/pixel_lists"
LONG = 1280


def label_dirs():
    return sorted(p for p in DS.glob("label_train*") if (p / "commands.jsonl").exists())


def iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0


# ---------------------------------------------------------------- prep (laptop)

def prep(args):
    from PIL import Image
    cmds = {}
    for d in label_dirs():
        for c in map(json.loads, open(d / "commands.jsonl")):
            cmds.setdefault(c["screen"], []).append(c)
    screens = {}
    for line in open(DS / "screens.jsonl"):
        if line[8:20] in cmds:
            s = json.loads(line)
            screens[s["id"]] = s
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    n = 0
    with open(OUT / "bundle.jsonl", "w") as f:
        for line in open(DS / "options.jsonl"):
            sid = line[8:20]
            if sid not in screens:
                continue
            o, s = json.loads(line), screens[sid]
            img_out = OUT / "images" / f"{sid}.jpg"
            if not img_out.exists():
                im = Image.open(ROOT / s["shot"]).convert("RGB")
                r = LONG / max(im.size)
                if r < 1:
                    im = im.resize((round(im.width * r), round(im.height * r)), Image.LANCZOS)
                im.save(img_out, quality=90)
            w = Image.open(img_out).width
            keep = [x for x in o["options"] if x["id"][0] != "d"]
            f.write(json.dumps({"id": sid, "app": s["app"], "frame": s["frame"], "scale": w / s["frame"][2],
                                "options": keep,
                                "commands": [{"id": c["id"], "command": c["command"]} for c in cmds[sid]]}) + "\n")
            n += 1
    print(f"{n} screens, {sum(map(len, cmds.values()))} commands -> {OUT}")


# ---------------------------------------------------------------- run (GPU box)

def run(args):
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    from inst_feats import context_crop, vec
    from screen_options.detect import Captioner, Detector
    from screen_options.sources import merge

    det = Detector(model=args.det, imgsz=1280, tile=1, conf=args.conf)
    proc = AutoProcessor.from_pretrained(args.ranker)
    sig = AutoModel.from_pretrained(args.ranker, dtype=torch.float16).cuda().eval()
    cap = Captioner(model_dir=Path(args.labeler).expanduser())
    out = OUT / "extras.jsonl"
    done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    bundle = [b for b in map(json.loads, open(OUT / "bundle.jsonl")) if b["id"] not in done]
    print(f"{len(bundle)} screens to do, {len(done)} done", flush=True)
    with open(out, "a") as f, torch.inference_mode():
        for i, b in enumerate(bundle):
            img = Image.open(OUT / "images" / f"{b['id']}.jpg").convert("RGB")
            fx, fy, fw, fh = b["frame"]
            sc = b["scale"]
            ax = [SimpleNamespace(frame=tuple(o["box"]), label=o["label"], role=o["role"], context=o.get("context") or [])
                  for o in b["options"] if "ax" in o["sources"] and o.get("box")]
            ocr = [{"box": tuple(o["box"]), "text": o["label"]} for o in b["options"] if "ax" not in o["sources"]]
            items, need = merge(ax, ocr, det.boxes(img, b["frame"], sc), tuple(b["frame"]))
            icons = [items[k] for k in need]
            px = [[(it.box[0] - fx) * sc, (it.box[1] - fy) * sc, it.box[2] * sc, it.box[3] * sc] for it in icons]
            sims = {}
            if icons:
                embs = []
                crops = [context_crop(img, p) for p in px]
                for j in range(0, len(crops), 64):
                    pv = proc(images=crops[j:j + 64], return_tensors="pt")["pixel_values"].cuda().half()
                    embs.append(torch.nn.functional.normalize(vec(sig.get_image_features(pixel_values=pv)), dim=-1))
                img_e = torch.cat(embs)
                texts = [c["command"] for c in b["commands"]]
                tok = proc(text=texts, padding="max_length", truncation=True, return_tensors="pt",
                           max_length=min(64, sig.config.text_config.max_position_embeddings)).to("cuda")
                txt_e = torch.nn.functional.normalize(vec(sig.get_text_features(**tok)), dim=-1)
                m = (txt_e @ img_e.T).float().cpu()
                sims = {c["id"]: [round(v, 4) for v in row] for c, row in zip(b["commands"], m.tolist())}
                top = set()
                for row in m:
                    top.update(row.argsort(descending=True)[:args.cap_k].tolist())
            else:
                top = set()
            top = sorted(top)
            caps = cap.captions(img, batch=32, pixel_boxes=[(p[0], p[1], p[0] + p[2], p[1] + p[3]) for p in (px[k] for k in top)]) if top else []
            f.write(json.dumps({"id": b["id"],
                                "icons": [{"box": list(it.box), "conf": round(it.conf, 4)} for it in icons],
                                "captions": dict(zip(map(str, top), caps)), "sims": sims}) + "\n")
            if i % 200 == 0:
                print(f"{i}: {len(items)} items, {len(icons)} icons, {len(top)} captioned", flush=True)


# ---------------------------------------------------------------- build (laptop)

def parse_caption(text):
    """The labeller answers "<role>: <label>"; anything else is a plain label on an icon."""
    role, sep, label = text.partition(":")
    if sep and role.strip() and len(role.split()) <= 3 and not any(ch.isdigit() for ch in role):
        return role.strip().lower(), label.strip()
    return "icon", text.strip()


def remap(tb, new_icons):
    """Old detector-only target box -> index of the new icon covering it, or None."""
    best, best_v = None, 0.0
    for k, it in enumerate(new_icons):
        v = iou(tb, it["box"])
        cx, cy = tb[0] + tb[2] / 2, tb[1] + tb[3] / 2
        x, y, w, h = it["box"]
        if v < 0.5 and x <= cx <= x + w and y <= cy <= y + h and w * h <= 4 * tb[2] * tb[3]:
            v = 0.5
        if v > best_v:
            best, best_v = k, v
    return best if best_v >= 0.5 else None


def frac(key):
    return int(hashlib.sha256(f"val:{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def build(args):
    sys.path.insert(0, str(ROOT / "src"))
    from rlca.candidates import MAX_OPTIONS
    from rlca.records import record
    bundle = {b["id"]: b for b in map(json.loads, open(OUT / "bundle.jsonl"))}
    extras = {e["id"]: e for e in map(json.loads, open(OUT / "extras.jsonl"))}
    cmds = [c for d in label_dirs() for c in map(json.loads, open(d / "commands.jsonl"))]
    out_dir = ROOT / "data/records_pixel"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats, ranks = Counter(), []
    files = {s: open(out_dir / f"{s}.jsonl", "w") for s in ("train", "val")}
    for c in cmds:
        b, e = bundle.get(c["screen"]), extras.get(c["screen"])
        if b is None or e is None:
            stats["no_screen"] += 1
            continue
        sims = e["sims"].get(c["id"], [])
        order = sorted(range(len(e["icons"])), key=lambda k: -sims[k])[:args.k]
        kept = [k for k in order if str(k) in e["captions"]]
        icons = []
        for n, k in enumerate(kept):
            role, label = parse_caption(e["captions"][str(k)])
            icons.append({"id": f"n{n}", "source": "det", "role": role, "label": label, "box": e["icons"][k]["box"]})
        opts = (b["options"] + icons)[: MAX_OPTIONS - 1]
        ids = {o["id"] for o in opts}
        targets, lost = [], False
        for tb in c.get("target_boxes") or [{"id": t} for t in c["targets"]]:
            if tb["id"][0] == "d":
                k = remap(tb["box"], e["icons"])
                if k is None:
                    stats["det_target_not_detected"] += 1
                    lost = True
                    continue
                r = sorted(range(len(e["icons"])), key=lambda j: -sims[j]).index(k)
                ranks.append(r)
                if k not in kept:
                    stats["det_target_ranked_out"] += 1
                    lost = True
                    continue
                targets.append(f"n{kept.index(k)}")
            elif tb["id"] in ids:
                targets.append(tb["id"])
            else:
                lost = None                      # cut by Kev's option cap (very long AX trees): not learnable, dropped
                break
        if lost is None or any(t not in ids for t in targets):
            stats["dropped_over_cap"] += 1
            continue
        if c["targets"] and not targets:
            stats["became_none"] += 1           # the pipeline lost the target: on this list the right answer is "none"
        stats["absent" if not c["targets"] else ("icon" if any(t[0] == "n" for t in targets) else "ax_ocr")] += 1
        split = "val" if frac(b["app"] + c["screen"]) < args.val else "train"
        rec = record(b["app"], c["command"], opts, targets, action=c["verb"], text_source=c.get("text_source") or "none",
                     irreversible=c["irreversible"], ambiguous=c["ambiguous"],
                     meta={"source": "sandbox", "id": c["id"], "screen": c["screen"], "style": c["style"],
                           "lost_target": lost})
        files[split].write(json.dumps(rec, ensure_ascii=False) + "\n")
        stats[split] += 1
        stats["options_sum"] += len(opts)
    for f in files.values():
        f.close()
    n = stats["train"] + stats["val"]
    print(json.dumps({k: v for k, v in stats.items() if k != "options_sum"}, indent=1))
    print(f"mean options {stats['options_sum'] / max(n, 1):.1f}")
    if ranks:
        for K in (5, 10, 20, 30, 50):
            print(f"detected icon targets in top {K}: {sum(r < K for r in ranks) / len(ranks):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("prep", "run", "build"))
    ap.add_argument("--det", default="~/screen-options/runs/det/axgpa_1280/weights/best.pt")
    ap.add_argument("--ranker", default="~/screen-options/runs/siglip/ctx/best")
    ap.add_argument("--labeler", default="~/.cache/rlca/ax-labeler")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--cap_k", type=int, default=50, help="run: caption each command's top cap_k icons")
    ap.add_argument("--k", type=int, default=50, help="build: detector icons kept per command")
    ap.add_argument("--val", type=float, default=0.05, help="build: share of screens held out for val")
    args = ap.parse_args()
    for a in ("det", "ranker"):
        setattr(args, a, str(Path(getattr(args, a)).expanduser()))
    {"prep": prep, "run": run, "build": build}[args.step](args)


if __name__ == "__main__":
    main()
