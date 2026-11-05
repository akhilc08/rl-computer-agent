"""Time 2B screenshot-grounding VLMs on this Mac (4-bit MLX), on the same prompts and image budgets as the GPU eval.

The deciding number for "Kev vs a vision model" is time per command on the Mac, so this runs a small sample per
benchmark one model at a time and reports wall time per call (image resize + prefill + decode), prompt/image
tokens, peak memory, and a sanity hit rate (4-bit can lose accuracy vs the bf16 GPU run). Prompts and parsing
come from vlm_grounding.py, so both runs see the same text.

Runs in .vendor/vlm-venv (mlx-vlm; qwen3vl also needs qwen-agent + soundfile and the Qwen3-VL cookbooks):
  .vendor/vlm-venv/bin/python scripts/eval/vlm_mac_speed.py --model venus-2b --path .vendor/mlx-models/venus-2b-4bit
  models: venus-2b showui-2b qwen3vl-2b
"""
import argparse
import json
import math
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_grounding import SHOWUI_SYSTEM, VENUS_PROMPT, gt_box, inside, parse_venus  # noqa: E402

# (resize factor, min pixels, max pixels): Qwen3-VL = 16 px patches merged 2x2, Qwen2-VL = 14 px merged 2x2
BUDGET = {"venus-2b": (32, 4 * 32 * 32, 16384 * 28 * 28), "qwen3vl-2b": (32, 16 * 32 * 32, 6400 * 32 * 32),
          "showui-2b": (28, 256 * 28 * 28, 1344 * 28 * 28)}


def smart_resize(h, w, factor, min_px, max_px):
    """qwen_vl_utils' rule: round to the patch grid, then scale into [min_px, max_px]."""
    hb, wb = max(factor, round(h / factor) * factor), max(factor, round(w / factor) * factor)
    if hb * wb > max_px:
        beta = math.sqrt(h * w / max_px)
        hb, wb = math.floor(h / beta / factor) * factor, math.floor(w / beta / factor) * factor
    elif hb * wb < min_px:
        beta = math.sqrt(min_px / (h * w))
        hb, wb = math.ceil(h * beta / factor) * factor, math.ceil(w * beta / factor) * factor
    return hb, wb


def messages(model, instruction, cookbook):
    if model == "venus-2b":
        return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": VENUS_PROMPT.format(instruction.removesuffix("."))}]}]
    if model == "showui-2b":
        return [{"role": "user", "content": [{"type": "text", "text": SHOWUI_SYSTEM}, {"type": "image"}, {"type": "text", "text": instruction}]}]
    from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import ContentItem, Message, NousFnCallPrompt
    sys.path.insert(0, str(Path(cookbook).expanduser()))
    from utils.agent_function_call import ComputerUse
    tool = ComputerUse(cfg={"display_width_px": 1000, "display_height_px": 1000}).function
    msgs = NousFnCallPrompt().preprocess_fncall_messages(
        messages=[Message(role="system", content=[ContentItem(text="You are a helpful assistant.")]),
                  Message(role="user", content=[ContentItem(text=instruction), ContentItem(image="file://screen.png")])],
        functions=[tool], lang=None)
    out = []
    for m in msgs:
        m = m.model_dump()
        content = m["content"] if isinstance(m["content"], list) else [{"text": m["content"]}]
        out.append({"role": m["role"], "content": [{"type": "image"} if c.get("image") else {"type": "text", "text": c.get("text") or ""}
                                                   for c in content]})
    return out


def point(model, raw, size):
    """Model text -> point in the resized image's relative frame, as (fx, fy) in 0-1, or None."""
    if model == "qwen3vl-2b":
        import re
        m = re.search(r'"coordinate"\s*:\s*\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)', raw)
        return (float(m.group(1)) / 1000, float(m.group(2)) / 1000) if m else None
    p = parse_venus(raw)
    if not p or p == (-1, -1):
        return None
    return (p[0] / 1000, p[1] / 1000) if model == "venus-2b" else p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=BUDGET)
    ap.add_argument("--path", required=True, help="local 4-bit MLX model folder or mlx-community repo")
    ap.add_argument("--sets", nargs="+", default=["screenspot_v2", "screenspot_pro", "osworld_g", "omniact"])
    ap.add_argument("--n", type=int, default=25, help="examples per set (after 3 warm-up calls)")
    ap.add_argument("--qwen_cookbook", default="~/Qwen3-VL/cookbooks")
    ap.add_argument("--out", default=str(ROOT / "runs/vlm-mac"))
    a = ap.parse_args()
    import mlx.core as mx
    from mlx_vlm import generate, load
    from PIL import Image
    model, proc = load(a.path)
    factor, min_px, max_px = BUDGET[a.model]
    rows = []
    for s in a.sets:
        f = ROOT / "data/converted" / s / "examples.jsonl"
        if not f.exists():
            continue
        exs = [json.loads(l) for l in open(f)]
        exs = random.Random(0).sample(exs, min(len(exs), a.n + 3))
        for k, ex in enumerate(exs):
            img = Image.open(ROOT / "data/public" / ex["image"]).convert("RGB")
            box = gt_box(ex, img.size)
            if box is None:
                continue
            t0 = time.perf_counter()
            h, w = smart_resize(img.height, img.width, factor, min_px, max_px)
            with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
                img.resize((w, h)).save(tmp.name)
                prompt = proc.apply_chat_template(messages(a.model, ex["instruction"], a.qwen_cookbook), tokenize=False,
                                                  add_generation_prompt=True)
                res = generate(model, proc, prompt, image=[tmp.name], max_tokens=64, temperature=0.0)
            ms = (time.perf_counter() - t0) * 1000
            p = point(a.model, res.text, (w, h))
            pt = (p[0] * img.width, p[1] * img.height) if p else None
            rows.append({"set": s, "id": ex["id"], "warmup": k < 3, "ms": round(ms, 1), "hit": bool(pt and inside(box, pt)),
                         "prompt_tokens": res.prompt_tokens, "gen_tokens": res.generation_tokens,
                         "resized": [w, h], "raw": res.text})
        print(s, "done", flush=True)
    timed = [r for r in rows if not r["warmup"]]
    summary = {"model": a.model, "path": a.path, "n": len(timed),
               "median_ms": statistics.median(r["ms"] for r in timed),
               "p90_ms": sorted(r["ms"] for r in timed)[int(0.9 * (len(timed) - 1))],
               "median_prompt_tokens": statistics.median(r["prompt_tokens"] for r in timed),
               "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2),
               "hit_rate": round(sum(r["hit"] for r in timed) / len(timed), 3),
               "per_set": {s: {"n": len(v), "median_ms": statistics.median(r["ms"] for r in v),
                               "hit_rate": round(sum(r["hit"] for r in v) / len(v), 3)}
                           for s in a.sets if (v := [r for r in timed if r["set"] == s])}}
    out = Path(a.out) / a.model
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
