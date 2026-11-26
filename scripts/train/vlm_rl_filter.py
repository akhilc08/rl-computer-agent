"""Pick the GRPO training commands the SFT model is unsure about: sample --k answers per training command with vLLM
and keep the commands where some answers are rewarded and some are not (0 < hits < k). Commands the model always gets
right or always gets wrong give GRPO no signal (every advantage is zero), and on the full split that was 76% of them.

Same prompt, image resize (qwen_vl_utils, --max_pixels) and reward as scripts/train/vlm_grpo.py. Writes the kept
commands in the train.jsonl screen format, each with "p_hit" (share of its k answers rewarded), plus <out>.stats.json.

  python scripts/train/vlm_rl_filter.py --model runs/vlm-rl/qwen-sft/merged --out data/vlm_rl/train_hard.jsonl
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
sys.path.insert(0, str(ROOT / "scripts/train"))
from vlm_grounding import VENUS_PROMPT  # noqa: E402
from vlm_grpo import reward  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/vlm_rl/train.jsonl")
    ap.add_argument("--data_root", default=str(ROOT))
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max_pixels", type=int, default=3200 * 32 * 32)
    ap.add_argument("--chunk", type=int, default=48, help="screens per vLLM call")
    ap.add_argument("--limit", type=int, default=0, help="first N screens only (smoke test)")
    args = ap.parse_args()
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    root = Path(args.data_root)
    screens = [json.loads(l) for l in open(ROOT / args.data)]
    if args.limit:
        screens = screens[: args.limit]
    proc = AutoProcessor.from_pretrained(args.model)
    llm = LLM(model=args.model, limit_mm_per_prompt={"image": 1}, max_model_len=8192, gpu_memory_utilization=0.85,
              enable_prefix_caching=True, seed=0)
    sp = SamplingParams(n=args.k, temperature=args.temp, top_p=1.0, max_tokens=24, seed=0)
    hist = collections.Counter()
    kept_cmds = n_cmds = 0
    t0 = time.time()
    with open(ROOT / args.out, "w") as f:
        for i in range(0, len(screens), args.chunk):
            batch, owners = [], []
            for r in screens[i: i + args.chunk]:
                img = None
                for c in r["cmds"]:
                    msgs = [{"role": "user", "content": [
                        {"type": "image", "image": str(root / r["shot"]), "max_pixels": args.max_pixels},
                        {"type": "text", "text": VENUS_PROMPT.format(c["command"].removesuffix("."))}]}]
                    if img is None:             # one resize per screen; every command on it shares the image
                        img = process_vision_info(msgs)[0][0]
                    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    batch.append({"prompt": text, "multi_modal_data": {"image": img}})
                    owners.append((r, c))
            outs = llm.generate(batch, sp, use_tqdm=False)
            keep = collections.defaultdict(list)
            for (r, c), o in zip(owners, outs):
                hits = sum(reward(x.text, c, 0.0)[0] == 1.0 for x in o.outputs)
                hist[hits] += 1
                n_cmds += 1
                if 0 < hits < args.k:
                    keep[id(r)].append({**c, "p_hit": hits / args.k})
            for r in screens[i: i + args.chunk]:
                if keep[id(r)]:
                    f.write(json.dumps({**r, "cmds": keep[id(r)]}) + "\n")
                    kept_cmds += len(keep[id(r)])
            el = time.time() - t0
            done = min(i + args.chunk, len(screens))
            print(f"screens {done}/{len(screens)} cmds {n_cmds} kept {kept_cmds} ({100 * kept_cmds / n_cmds:.1f}%) "
                  f"{el / 60:.1f}m eta {el / done * (len(screens) - done) / 60:.0f}m", flush=True)
    stats = {"commands": n_cmds, "kept": kept_cmds, "k": args.k, "temp": args.temp,
             "hits_histogram": {str(h): hist[h] for h in range(args.k + 1)}, "minutes": round((time.time() - t0) / 60, 1)}
    Path(ROOT / (args.out + ".stats.json")).write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
