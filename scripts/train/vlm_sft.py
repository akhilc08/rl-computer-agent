"""LoRA fine-tune of a Qwen3-VL-architecture screenshot model (UI-Venus-1.5-2B, Qwen3-VL-2B-Instruct) to answer
grounding commands with a point, on the sandbox train split (scripts/train/vlm_points_prep.py).

Prompt and answer follow UI-Venus's grounding format (scripts/eval/vlm_grounding.py VENUS_PROMPT): "[x,y]" in 0-1000
coordinates, "[-1,-1]" for an absent target. Each screen is one multi-turn conversation (image once, then up to
--per_seq commands), so the image is encoded once per chunk; loss is on the answers only. Images go through
qwen_vl_utils with the same max_pixels the evaluation uses. LoRA on the language model's linear layers, vision
tower frozen. Writes the merged model to <out>/merged (load it with vlm_grounding.py --repo <out>/merged).

  python scripts/train/vlm_sft.py --base inclusionAI/UI-Venus-1.5-2B --data data/vlm_points/train.jsonl --out runs/vlm-ft/venus-2b
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
from vlm_grounding import VENUS_PROMPT  # noqa: E402


def chunks(rows, per_seq, rng):
    out = []
    for r in rows:
        cmds = r["cmds"][:]
        rng.shuffle(cmds)
        for i in range(0, len(cmds), per_seq):
            out.append({"shot": r["shot"], "cmds": cmds[i:i + per_seq]})
    rng.shuffle(out)
    return out


def build(proc, process_vision_info, item, max_px, data_root):
    from PIL import Image
    img = Image.open(data_root / item["shot"]).convert("RGB")
    msgs = []
    for j, c in enumerate(item["cmds"]):
        text = {"type": "text", "text": VENUS_PROMPT.format(c["command"].removesuffix("."))}
        content = [{"type": "image", "image": img, "max_pixels": max_px}, text] if j == 0 else [text]
        msgs.append({"role": "user", "content": content})
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": c["answer"]}]})
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    images, _ = process_vision_info(msgs)
    enc = proc(text=[text], images=images, return_tensors="pt")
    ids = enc["input_ids"][0]
    labels = ids.clone().fill_(-100)
    tok = proc.tokenizer
    start = tok.convert_tokens_to_ids("<|im_start|>")
    end = tok.convert_tokens_to_ids("<|im_end|>")
    asst = tok.encode("assistant\n", add_special_tokens=False)
    i, n = 0, len(ids)
    while i < n:
        if ids[i] == start and ids[i + 1:i + 1 + len(asst)].tolist() == asst:
            j = i + 1 + len(asst)
            k = j
            while k < n and ids[k] != end:
                k += 1
            labels[j:k + 1] = ids[j:k + 1]          # answer and its <|im_end|>
            i = k
        i += 1
    enc["labels"] = labels[None]
    return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", default="data/vlm_points/train.jsonl")
    ap.add_argument("--data_root", default=str(ROOT))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_pixels", type=int, default=3200 * 32 * 32)
    ap.add_argument("--per_seq", type=int, default=12)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="first N sequences (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import torch
    from peft import LoraConfig, get_peft_model
    from qwen_vl_utils import process_vision_info
    from transformers import AutoModelForImageTextToText, AutoProcessor
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(ROOT / args.data)]
    seqs = chunks(rows, args.per_seq, rng)
    total = int(len(seqs) * args.epochs)
    if args.limit:
        total = min(total, args.limit)
    proc = AutoProcessor.from_pretrained(args.base)
    model = AutoModelForImageTextToText.from_pretrained(args.base, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                        device_map="cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                     target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps = math.ceil(total / args.accum)
    warm = max(1, steps // 30)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    print(f"{len(rows)} screens -> {len(seqs)} sequences, training on {total} ({steps} optimizer steps)", flush=True)
    model.train()
    t0, run_loss, run_n = time.time(), 0.0, 0
    for i in range(total):
        item = seqs[i % len(seqs)]
        enc = build(proc, process_vision_info, item, args.max_pixels, Path(args.data_root)).to("cuda")
        loss = model(**enc).loss
        (loss / args.accum).backward()
        run_loss += loss.item()
        run_n += 1
        if (i + 1) % args.accum == 0 or i + 1 == total:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
        if (i + 1) % 50 == 0 or i + 1 == total:
            el = time.time() - t0
            print(f"seq {i + 1}/{total} loss {run_loss / run_n:.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{el / (i + 1):.2f}s/seq eta {el / (i + 1) * (total - i - 1) / 60:.0f}m", flush=True)
            run_loss, run_n = 0.0, 0
    model.save_pretrained(out / "adapter")
    merged = model.merge_and_unload()
    merged.save_pretrained(out / "merged")
    proc.save_pretrained(out / "merged")
    (out / "train_args.json").write_text(json.dumps(vars(args), indent=1))
    print("saved", out / "merged", flush=True)


if __name__ == "__main__":
    main()
