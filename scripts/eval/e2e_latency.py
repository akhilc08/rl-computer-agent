"""End-to-end latency of Kev's pixels-only path on this Mac, screenshot in to answer out, per stage.

Same commands as the screenshot-model Mac timing (vlm_mac_speed.py: random.Random(0).sample of each set's
examples, first 3 are warm-up). Two stages, since the list pipeline and Kev live in different venvs:

  lists  (.venv): Vision OCR -> GPA detector (AX-trained, Core ML) -> merge -> context crops of the
                  detector-only icons -> SigLIP image embeddings -> command text embedding -> rank ->
                  caption the top --cap_k icons with the AX-style labeller. Writes each option list + times.
  kev    (.vendor/kev/.venv): Kev 0.8B 4-bit MLX picks from each list; adds its time and a hit flag.

  .venv/bin/python scripts/eval/e2e_latency.py lists
  .vendor/kev/.venv/bin/python scripts/eval/e2e_latency.py kev --run data/kev-runs/sft-0.8b-pixel
  python3 scripts/eval/e2e_latency.py report

Stages before the command is known (OCR, detect, merge, crops, image embeddings) can run while the user is
still speaking; "after_speech" counts only the rest (text embedding, rank, captions, Kev).
"""
import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SO = Path("~/Coding/screen-options").expanduser()
OUT = ROOT / "runs/e2e-latency"
SETS = ["screenspot_v2", "screenspot_pro", "osworld_g", "omniact"]
PRE = ("ocr", "detect", "merge", "crop", "embed_img")
POST = ("embed_text", "rank", "caption", "kev")


def examples(n):
    for s in SETS:
        exs = [json.loads(l) for l in open(ROOT / "data/converted" / s / "examples.jsonl")]
        for k, ex in enumerate(random.Random(0).sample(exs, min(len(exs), n + 3))):
            yield s, k < 3, ex


def parse_caption(text):
    role, sep, label = text.partition(":")
    if sep and role.strip() and len(role.split()) <= 3 and not any(ch.isdigit() for ch in role):
        return role.strip().lower(), label.strip()
    return "icon", text.strip()


def lists(args):
    sys.path[:0] = [str(SO / "src"), str(SO / "scripts/eval")]
    import Quartz
    import objc
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    from inst_feats import context_crop, vec
    from screen_options.detect import Captioner, Detector
    from screen_options.ocr import ocr_lines
    from screen_options.sources import merge

    def sync():
        torch.mps.synchronize()
        return time.perf_counter()

    det = Detector(model=str(SO / "runs/det/axgpa_1280/weights/best.pt"), imgsz=1280, tile=1, conf=0.05, coreml=args.coreml)
    proc = AutoProcessor.from_pretrained(SO / "runs/siglip/ctx/best")
    sig = AutoModel.from_pretrained(SO / "runs/siglip/ctx/best", dtype=torch.float16).to("mps").eval()
    maxlen = min(64, sig.config.text_config.max_position_embeddings)
    cap = Captioner(model_dir=Path("~/.cache/rlca/ax-labeler").expanduser())
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "lists.jsonl", "w") as f, torch.inference_mode():
        for s, warm, ex in examples(args.n):
            path = ROOT / "data/public" / ex["image"]
            img = Image.open(path).convert("RGB")
            frame = (0, 0, img.width, img.height)
            with objc.autorelease_pool():
                src = Quartz.CGImageSourceCreateWithURL(Quartz.CFURLCreateWithFileSystemPath(None, str(path), 0, False), None)
                cg = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
                t = {}
                t0 = time.perf_counter()
                ocr = [o for o in ocr_lines(cg, frame, fast=args.fast_ocr) if len(o["text"].split()) <= 6]
                t["ocr"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            boxes = det.boxes(img, frame, 1.0)
            t["detect"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            items, need = merge([], ocr, boxes, frame)
            t["merge"] = time.perf_counter() - t0
            icons = [items[k] for k in need]
            t0 = time.perf_counter()
            crops = [context_crop(img, it.box) for it in icons]
            t["crop"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            embs = []
            for j in range(0, len(crops), 64):
                pv = proc(images=crops[j:j + 64], return_tensors="pt")["pixel_values"].to("mps", torch.float16)
                embs.append(torch.nn.functional.normalize(vec(sig.get_image_features(pixel_values=pv)), dim=-1))
            t["embed_img"] = sync() - t0
            t0 = time.perf_counter()
            tok = proc(text=[ex["instruction"]], padding="max_length", truncation=True, max_length=maxlen,
                       return_tensors="pt").to("mps")
            te = torch.nn.functional.normalize(vec(sig.get_text_features(**tok)), dim=-1)
            t["embed_text"] = sync() - t0
            t0 = time.perf_counter()
            top = (te @ torch.cat(embs).T).squeeze(0).argsort(descending=True)[:args.cap_k].tolist() if embs else []
            t["rank"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            px = [(icons[k].box[0], icons[k].box[1], icons[k].box[0] + icons[k].box[2], icons[k].box[1] + icons[k].box[3])
                  for k in top]
            caps = cap.captions(img, px, batch=32) if px else []
            t["caption"] = sync() - t0
            opts = [{"text": f'text "{it.label}"', "box": list(it.box)} for k, it in enumerate(items) if k not in set(need)]
            for k, c in zip(top, caps):
                role, label = parse_caption(c)
                opts.append({"text": f'{role} "{label}"' if label else role, "box": list(icons[k].box)})
            opts = opts[:args.max_options - 1]
            row = {"set": s, "warmup": warm, "id": ex["id"], "instruction": ex["instruction"], "gt_box": ex.get("gt_box"),
                   "app": ex.get("platform") or s, "size": [img.width, img.height], "n_ocr": len(ocr),
                   "n_icons": len(icons), "n_captioned": len(top), "options": opts,
                   "ms": {k: round(v * 1000, 1) for k, v in t.items()}}
            f.write(json.dumps(row) + "\n")
            f.flush()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            print(s, ex["id"], row["ms"], flush=True)


def kev(args):
    sys.path.insert(0, str(ROOT / "src"))
    from rlca.kev_backend import load_kev, score_options
    tok, model = load_kev(args.run, bits=4)
    rows = [json.loads(l) for l in open(OUT / "lists.jsonl")]
    with open(OUT / "e2e.jsonl", "w") as f:
        for r in rows:
            state = f"App: {r['app']}\nUser command: {r['instruction']}"
            options = [o["text"] for o in r["options"]] + ["None of these elements"]
            probs, ms, ntok = score_options(tok, model, state, options)
            k = max(range(len(probs)), key=probs.__getitem__)
            hit = None
            if r["gt_box"] and k < len(r["options"]):
                x, y, w, h = r["options"][k]["box"]
                gx, gy, gw, gh = r["gt_box"]
                hit = x <= gx + gw / 2 <= x + w and y <= gy + gh / 2 <= y + h
            r["ms"]["kev"] = round(ms, 1)
            f.write(json.dumps({**{k2: v for k2, v in r.items() if k2 != "options"}, "n_options": len(options),
                                "prompt_tokens": ntok, "hit": hit}) + "\n")
            print(r["set"], r["id"], round(ms), "ms", "hit" if hit else "", flush=True)


def report(args):
    rows = [json.loads(l) for l in open(OUT / "e2e.jsonl") if not json.loads(l)["warmup"]]

    def st(xs):
        xs = sorted(xs)
        return f"{statistics.median(xs):7.0f} {xs[int(0.9 * (len(xs) - 1))]:7.0f}"
    print(f"{'set':15s}{'n':>4s}  " + "  ".join(f"{k:>15s}" for k in PRE + POST + ("total", "after_speech")) + "   hit")
    for s in SETS + ["all"]:
        rs = [r for r in rows if s in ("all", r["set"])]
        if not rs:
            continue
        cols = [st([r["ms"].get(k, 0) for r in rs]) for k in PRE + POST]
        cols.append(st([sum(r["ms"].values()) for r in rs]))
        cols.append(st([sum(r["ms"].get(k, 0) for k in POST) for r in rs]))
        hits = [r["hit"] for r in rs if r["hit"] is not None]
        print(f"{s:15s}{len(rs):4d}  " + "  ".join(f"{c:>15s}" for c in cols) + f"  {100 * sum(hits) / max(1, len(hits)):4.0f}%")
    print("(each cell: median p90, ms)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["lists", "kev", "report"])
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--cap_k", type=int, default=50)
    ap.add_argument("--max_options", type=int, default=255)
    ap.add_argument("--coreml", type=int, default=1)
    ap.add_argument("--fast_ocr", type=int, default=1)
    ap.add_argument("--run", default="data/kev-runs/sft-0.8b-pixel")
    args = ap.parse_args()
    {"lists": lists, "kev": kev, "report": report}[args.stage](args)


if __name__ == "__main__":
    main()
