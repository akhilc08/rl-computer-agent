"""Public grounding sets with the screen-options option list, so Kev is scored on the list it was trained on.

The old eval records (convert_public.py) use OmniParser boxes + OmniParser captions for every icon. Kev's
training records (scripts/train/pixel_lists.py) use the screen-options pipeline instead: the AX-trained GPA
detector, a SigLIP ranker that keeps each command's top --k detector-only icons, and the AX-style labeller
("role: label"). This rebuilds the public sets the same way. OCR is reused from convert_public.py's cache
(Apple Vision, run on the Mac), so only the GPU stages run on the box.

  run   (GPU box)  python scripts/public/pixel_public.py run  --sets screenspot_v2 ...   -> data/pixel_public/<set>.jsonl
  build (laptop)   python scripts/public/pixel_public.py build                          -> data/records_pixpub/eval_*.jsonl
                                                                                           and data/records_pixpub_s/ (same ids
                                                                                           as data/records_s)
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/pixel_public"
SETS = {"screenspot_v2": "eval_ssv2", "screenspot_pro": "eval_sspro", "osworld_g": "eval_osworldg", "omniact": "eval_omniact"}


def run(args):
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    so = Path(args.so).expanduser()
    sys.path[:0] = [str(so / "src"), str(so / "scripts/eval")]
    from inst_feats import context_crop, vec
    from screen_options.detect import Captioner, Detector
    from screen_options.sources import merge

    det = Detector(model=str(Path(args.det).expanduser()), imgsz=1280, tile=1, conf=0.05)
    proc = AutoProcessor.from_pretrained(Path(args.ranker).expanduser())
    sig = AutoModel.from_pretrained(Path(args.ranker).expanduser(), dtype=torch.float16).cuda().eval()
    maxlen = min(64, sig.config.text_config.max_position_embeddings)
    cap = Captioner(model_dir=Path(args.labeler).expanduser())
    OUT.mkdir(parents=True, exist_ok=True)
    for s in args.sets:
        d = ROOT / "data/converted" / s
        screens = {r["image"]: r for r in map(json.loads, open(d / "screens.jsonl"))}
        exs = {}
        for e in map(json.loads, open(d / "examples.jsonl")):
            exs.setdefault(e["image"], []).append(e)
        out = OUT / f"{s}.jsonl"
        done = {json.loads(l)["image"] for l in open(out)} if out.exists() else set()
        todo = [im for im in exs if im not in done and screens.get(im, {}).get("size")]
        print(s, len(todo), "screens to do", flush=True)
        with open(out, "a") as f, torch.inference_mode():
            for i, im in enumerate(todo):
                img = Image.open(ROOT / "data/public" / im).convert("RGB")
                frame = (0, 0, img.width, img.height)
                ocr = [{"box": tuple(e["box"]), "text": e["label"]} for e in screens[im]["elements"]
                       if "ocr" in e["sources"] and len(e["label"].split()) <= 6]
                items, need = merge([], ocr, det.boxes(img, frame, 1.0), frame)
                need = set(need)
                texts = [{"box": list(it.box), "label": it.label} for k, it in enumerate(items) if k not in need]
                icons = [items[k] for k in sorted(need)]
                ranks, top = {}, set()
                if icons:
                    embs = []
                    crops = [context_crop(img, it.box) for it in icons]
                    for j in range(0, len(crops), 64):
                        pv = proc(images=crops[j:j + 64], return_tensors="pt")["pixel_values"].cuda().half()
                        embs.append(torch.nn.functional.normalize(vec(sig.get_image_features(pixel_values=pv)), dim=-1))
                    ie = torch.cat(embs)
                    cmds = exs[im]
                    tok = proc(text=[e["instruction"] for e in cmds], padding="max_length", truncation=True,
                               max_length=maxlen, return_tensors="pt").to("cuda")
                    te = torch.nn.functional.normalize(vec(sig.get_text_features(**tok)), dim=-1)
                    m = (te @ ie.T).float().cpu()
                    for e, row in zip(cmds, m):
                        order = row.argsort(descending=True)[:args.k].tolist()
                        ranks[e["id"]] = order
                        top.update(order)
                top = sorted(top)
                px = [(icons[k].box[0], icons[k].box[1], icons[k].box[0] + icons[k].box[2], icons[k].box[1] + icons[k].box[3])
                      for k in top]
                caps = cap.captions(img, px, batch=32) if px else []
                f.write(json.dumps({"image": im, "size": [img.width, img.height], "texts": texts,
                                    "icons": [{"box": list(it.box), "conf": round(it.conf, 4)} for it in icons],
                                    "captions": dict(zip(map(str, top), caps)), "ranks": ranks}) + "\n")
                f.flush()
                if i % 100 == 0:
                    print(f"{s} {i}/{len(todo)}: {len(texts)} text, {len(icons)} icons, {len(top)} captioned", flush=True)


def build(args):
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/public"), str(ROOT / "scripts/train")]
    from convert_public import match
    from pixel_lists import parse_caption
    from rlca.records import MAX_OPTIONS, record
    from build_records import PUBLIC_ACTIONS
    out_full, out_s = ROOT / "data/records_pixpub", ROOT / "data/records_pixpub_s"
    out_full.mkdir(parents=True, exist_ok=True)
    out_s.mkdir(parents=True, exist_ok=True)
    for s, name in SETS.items():
        lists = {r["image"]: r for r in map(json.loads, open(OUT / f"{s}.jsonl"))}
        sample = {json.loads(l)["_meta"]["id"] for l in open(ROOT / "data/records_s" / f"{name}.jsonl")}
        stats = {"n": 0, "hit": 0, "dropped_cap": 0, "dropped_action": 0, "no_list": 0, "opts": 0}
        with open(out_full / f"{name}.jsonl", "w") as ff, open(out_s / f"{name}.jsonl", "w") as fs:
            for ex in map(json.loads, open(ROOT / "data/converted" / s / "examples.jsonl")):
                L = lists.get(ex["image"])
                if L is None:
                    stats["no_list"] += 1
                    continue
                opts = [{"id": f"o{i}", "role": "text", "label": t["label"], "box": t["box"]} for i, t in enumerate(L["texts"])]
                kept = [k for k in L["ranks"].get(ex["id"], []) if str(k) in L["captions"]]
                for n, k in enumerate(kept):
                    role, label = parse_caption(L["captions"][str(k)])
                    opts.append({"id": f"n{n}", "role": role, "label": label, "box": L["icons"][k]["box"]})
                opts = opts[:MAX_OPTIONS - 1]
                ids = match(opts, gt_box=ex.get("gt_box"), gt_point=None if ex.get("gt_box") else ex.get("gt_point"))
                action = PUBLIC_ACTIONS.get(ex.get("action"))
                if action is None:
                    stats["dropped_action"] += 1
                    continue
                rec = record(ex.get("app") or ex.get("platform") or s, ex["instruction"], opts, ids, action=action,
                             text_source=("user" if action == "type" else None),
                             meta={"source": s, "id": ex["id"], "split": ex["split"], "pipeline": "screen-options",
                                   **({"platform": ex["platform"]} if ex.get("platform") else {})})
                line = json.dumps(rec) + "\n"
                ff.write(line)
                if ex["id"] in sample:
                    fs.write(line)
                stats["n"] += 1
                stats["hit"] += bool(ids)
                stats["opts"] += len(opts)
        print(name, {**stats, "opts": round(stats["opts"] / max(1, stats["n"]), 1),
                     "target_in_list": round(stats["hit"] / max(1, stats["n"]), 3)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "build"])
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--so", default="~/screen-options")
    ap.add_argument("--det", default="~/screen-options/runs/det/axgpa_1280/weights/best.pt")
    ap.add_argument("--ranker", default="~/screen-options/runs/siglip/ctx/best")
    ap.add_argument("--labeler", default="~/.cache/rlca/ax-labeler")
    args = ap.parse_args()
    {"run": run, "build": build}[args.stage](args)


if __name__ == "__main__":
    main()
