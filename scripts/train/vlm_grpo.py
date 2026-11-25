"""GRPO (group-relative policy gradient) on single-step grounding for a Qwen3-VL-architecture screenshot model, on
top of its SFT merge (scripts/train/vlm_sft.py), with LoRA on the language model and the vision tower frozen.

Each command is a contextual bandit: one screenshot + command in Venus's prompt format, --group sampled answers,
reward per answer, advantage = (reward - group mean) / group std. Groups where every answer got the same reward
carry no signal and are skipped. One gradient step per batch of fresh samples (on-policy, so the PPO ratio is 1 and
the loss is -advantage * mean token log-prob of the answer), optional KL to the starting model (adapter disabled).

Reward (matches MacBench's scoring, which counts a hit and a correct abstention the same):
  target present: 1 if the point falls inside any target box, else 0 ([-1,-1] on a present target is 0)
  target absent:  1 for [-1,-1], else 0
  unparseable:    --format_penalty (default -0.5)

Checkpoints are chosen on the held-out dev apps (data/vlm_rl/dev.jsonl, none of them in SFT or RL training), never
on MacBench. Writes <out>/log.jsonl, <out>/ckpt-<step>/ adapters, <out>/dev.jsonl and <out>/merged (best dev ckpt).

  python scripts/train/vlm_grpo.py --init runs/vlm-rl/qwen-sft/merged --out runs/vlm-rl/qwen-grpo
  python scripts/train/vlm_grpo.py --init runs/vlm-rl/qwen-sft/merged --out runs/vlm-rl/qwen-sft-dev --eval_only
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
from vlm_grounding import VENUS_PROMPT, parse_venus  # noqa: E402


def reward(text, cmd, format_penalty):
    p = parse_venus(text)
    if p is None:
        return format_penalty, "unparsed"
    if tuple(p) == (-1, -1):
        return (1.0 if not cmd["boxes"] else 0.0), "abstain"
    if not cmd["boxes"]:
        return 0.0, "point"
    x, y = p
    return (1.0 if any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in cmd["boxes"]) else 0.0), "point"


def load(path):
    """-> flat list of commands, each with its screen's shot."""
    out = []
    for r in map(json.loads, open(ROOT / path)):
        for c in r["cmds"]:
            out.append({**c, "shot": r["shot"], "app": r["app"], "screen": r["screen"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="SFT merge to start from (local dir or hub id)")
    ap.add_argument("--data", default="data/vlm_rl/train.jsonl")
    ap.add_argument("--dev", default="data/vlm_rl/dev.jsonl")
    ap.add_argument("--data_root", default=str(ROOT))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_pixels", type=int, default=3200 * 32 * 32, help="same budget as SFT and the MacBench eval")
    ap.add_argument("--group", type=int, default=8, help="samples per command")
    ap.add_argument("--batch", type=int, default=8, help="commands per optimizer step")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--kl", type=float, default=0.0, help="KL weight to the starting model (0 = none)")
    ap.add_argument("--format_penalty", type=float, default=-0.5)
    ap.add_argument("--dev_n", type=int, default=800, help="dev commands scored at each eval (fixed sample)")
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--eval_only", action="store_true", help="score --init on the dev sample and exit")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from transformers import AutoModelForImageTextToText, AutoProcessor
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    root = Path(args.data_root)
    proc = AutoProcessor.from_pretrained(args.init)
    tok = proc.tokenizer
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    model = AutoModelForImageTextToText.from_pretrained(args.init, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                        device_map="cuda")

    def encode(cmd):
        img = Image.open(root / cmd["shot"]).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image", "image": img, "max_pixels": args.max_pixels},
                                             {"type": "text", "text": VENUS_PROMPT.format(cmd["command"].removesuffix("."))}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        images, _ = process_vision_info(msgs)
        return proc(text=[text], images=images, return_tensors="pt").to("cuda")

    dev = load(args.dev)
    dev = random.Random(1234).sample(dev, min(args.dev_n, len(dev)))

    @torch.no_grad()
    def evaluate(step):
        model.eval()
        tally = {"all": [0, 0], "present": [0, 0], "absent": [0, 0], "false_abstain": [0, 0], "unparsed": 0}
        rows = []
        for c in dev:
            enc = encode(c)
            o = model.generate(**enc, max_new_tokens=24, do_sample=False, temperature=None, top_p=None, top_k=None)
            text = tok.decode(o[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
            r, kind = reward(text, c, 0.0)
            key = "present" if c["boxes"] else "absent"
            for k in ("all", key):
                tally[k][0] += r
                tally[k][1] += 1
            if c["boxes"]:
                tally["false_abstain"][0] += kind == "abstain"
                tally["false_abstain"][1] += 1
            tally["unparsed"] += kind == "unparsed"
            rows.append({"id": c["id"], "raw": text, "reward": r})
        res = {"step": step, **{k: round(100 * a / b, 2) if b else None for k, (a, b) in
                                ((k, v) for k, v in tally.items() if k != "unparsed")},
               "unparsed": tally["unparsed"], "n": len(dev)}
        with open(out / "dev.jsonl", "a") as f:
            f.write(json.dumps(res) + "\n")
        print("dev", json.dumps(res), flush=True)
        return res, rows

    if args.eval_only:
        res, rows = evaluate(0)
        with open(out / "dev_rows.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                     target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    warm = max(1, args.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    train = load(args.data)
    rng.shuffle(train)
    print(f"{len(train)} train commands, {len(dev)} dev commands, {args.steps} steps x {args.batch} commands x "
          f"{args.group} samples", flush=True)

    def answer_logprob(enc, seq, n_prompt):
        """Mean log-prob of one sampled answer (tokens after the prompt, through the first <|im_end|>). Only the
        answer positions' logits are computed (logits_to_keep), not the ~3k image+prompt positions."""
        ans = seq[n_prompt:]
        stop = (ans == eos).nonzero()
        ans = ans[: int(stop[0]) + 1] if len(stop) else ans
        ids = torch.cat([enc.input_ids[0], ans])[None]
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=enc.pixel_values,
                       image_grid_thw=enc.image_grid_thw, logits_to_keep=len(ans) + 1).logits[0, :-1].float()
        return torch.log_softmax(logits, -1).gather(1, ans[:, None]).squeeze(1).mean()

    best, pos = None, 0
    if args.eval_every:
        best = evaluate(0)[0]
        best["ckpt"] = "init"
    t0 = time.time()
    for step in range(1, args.steps + 1):
        stats = {"reward": 0.0, "groups": 0, "skipped": 0, "unparsed": 0, "abstain": 0, "loss": 0.0}
        opt.zero_grad(set_to_none=True)
        for _ in range(args.batch):
            c = train[pos % len(train)]
            pos += 1
            enc = encode(c)
            n_prompt = enc.input_ids.shape[1]
            model.eval()
            with torch.no_grad():
                seqs = model.generate(**enc, max_new_tokens=24, do_sample=True, temperature=args.temp, top_p=1.0,
                                      top_k=0, num_return_sequences=args.group)
            texts = tok.batch_decode(seqs[:, n_prompt:], skip_special_tokens=True)
            scored = [reward(t, c, args.format_penalty) for t in texts]
            rs = torch.tensor([r for r, _ in scored])
            stats["reward"] += rs.mean().item() / args.batch
            stats["unparsed"] += sum(k == "unparsed" for _, k in scored)
            stats["abstain"] += sum(k == "abstain" for _, k in scored)
            stats["groups"] += 1
            if rs.std() < 1e-6:
                stats["skipped"] += 1
                continue
            adv = (rs - rs.mean()) / (rs.std() + 1e-4)
            model.train()
            for a, s in zip(adv.tolist(), seqs):          # one answer at a time: backward frees its graph
                lp = answer_logprob(enc, s, n_prompt)
                loss = -a * lp
                if args.kl:
                    with torch.no_grad(), model.disable_adapter():
                        ref = answer_logprob(enc, s, n_prompt)
                    loss = loss + args.kl * (lp - ref)       # k1 estimate of KL(policy || start) on the answer
                (loss / (args.batch * args.group)).backward()
                stats["loss"] += loss.item() / (args.batch * args.group)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        el = time.time() - t0
        rec = {"step": step, **{k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()},
               "lr": sched.get_last_lr()[0], "s_per_step": round(el / step, 1),
               "eta_m": round(el / step * (args.steps - step) / 60)}
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        if step % 10 == 0 or step == 1:
            print(json.dumps(rec), flush=True)
        if args.eval_every and (step % args.eval_every == 0 or step == args.steps):
            ck = out / f"ckpt-{step}"
            model.save_pretrained(ck)
            res = evaluate(step)[0]
            if res["all"] > best["all"]:
                best = {**res, "ckpt": ck.name}
    (out / "best.json").write_text(json.dumps(best, indent=1))
    print("best", json.dumps(best), flush=True)
    if best["ckpt"] == "init":
        print("no checkpoint beat the starting model on dev; not merging", flush=True)
        return
    del model, opt
    torch.cuda.empty_cache()
    base = AutoModelForImageTextToText.from_pretrained(args.init, dtype=torch.bfloat16, device_map="cuda")
    merged = PeftModel.from_pretrained(base, out / best["ckpt"]).merge_and_unload()
    merged.save_pretrained(out / "merged")
    proc.save_pretrained(out / "merged")
    print("saved", out / "merged", flush=True)


if __name__ == "__main__":
    main()
