"""Caption every reference detector box on MacBench with the deployed AX-style captioner (Florence-2 trained on AX
labels), so the ready-made option lists name icons the way a deployed system would.

The labelling pass also wrote captions, but only for icons it looked at while writing commands, in words that
echo those commands (98% of icon targets share a content word with their command). Lists built from them leak
the answer, so they are kept as annotations only and the lists use these captions instead.

GPU box, screen-options checkout on the path:
  python scripts/bench/macbench_captions.py --bench data/macbench/v1 --so ~/screen-options --labeler ~/.cache/rlca/ax-labeler
  -> <bench>/captions_deployed.jsonl  {"screen": id, "captions": ["role: label", ...]}  (one per screens.jsonl detector box)
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data/macbench/v1")
    ap.add_argument("--so", default="~/screen-options")
    ap.add_argument("--labeler", default="~/.cache/rlca/ax-labeler")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.so).expanduser() / "src"))
    import torch
    from PIL import Image
    from screen_options.detect import Captioner
    cap = Captioner(model_dir=Path(args.labeler).expanduser())
    bench = Path(args.bench)
    out = bench / "captions_deployed.jsonl"
    done = {json.loads(l)["screen"] for l in open(out)} if out.exists() else set()
    screens = [json.loads(l) for l in open(bench / "screens.jsonl")]
    with open(out, "a") as f, torch.inference_mode():
        for i, s in enumerate(screens):
            if s["id"] in done:
                continue
            img = Image.open(bench / s["image"]).convert("RGB")
            px = [(b[0], b[1], b[0] + b[2], b[1] + b[3]) for b in (d["box_px"] for d in s["detector"])]
            caps = cap.captions(img, px, batch=args.batch) if px else []
            f.write(json.dumps({"screen": s["id"], "captions": caps}) + "\n")
            f.flush()
            if i % 100 == 0:
                print(f"{i}/{len(screens)} {len(px)} boxes", flush=True)


if __name__ == "__main__":
    main()
