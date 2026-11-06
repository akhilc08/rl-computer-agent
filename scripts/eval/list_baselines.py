"""Same element list, different decision model: Kev vs small fast LLMs vs an embedding match, timed on this Mac.

Every model gets the exact option text Kev sees (the element question of our records: "id: description" lines plus
"none") and must name one id. Generative models answer greedily with the id only (thinking off); their confidence
is the probability of the generated answer tokens. model2vec picks the option most similar to the command (no
"none"), as the speed floor. Latency is wall time per decision (prompt build + prefill + decode); the first 3
calls per model are warm-up and left out.

Runs in Kev's environment (mlx, mlx_lm, model2vec), one model at a time to keep memory low:
  .vendor/kev/.venv/bin/python scripts/eval/list_baselines.py --model lm:mlx-community/Qwen3-0.6B-4bit
  --model kev:jaredpalmer/kev-0.8b@v1.0 | lm:<mlx repo> | model2vec

On a GPU box (same prompt, transformers, bf16, greedy) use hf:<hub repo>, and --records_dir data/records_s --n 0 to
score the same fixed samples every other model is scored on:
  python scripts/eval/list_baselines.py --model hf:Qwen/Qwen3-0.6B --records_dir data/records_s --n 0 --sets ...
"""
import argparse
import json
import math
import random
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_kev import coverage_at, ece  # noqa: E402

SYSTEM = ("You operate a Mac for the user. Below is the user's command and a numbered list of on-screen elements, one "
          "per line as 'id: description'. Reply with only the id of the element to act on to carry out the command, "
          "or 'none' if no element fits. No other words.")


def sample(sets, n, records_dir="data/records"):
    out = []
    for s in sets:
        recs = [json.loads(l) for l in open(ROOT / records_dir / f"{s}.jsonl")]
        if s == "val":
            recs = [r for r in recs if r["_meta"]["source"] == "mac_labels"]
        out += [(s, r) for r in (random.Random(0).sample(recs, min(n, len(recs))) if n else recs)]
    return out


def answers(q):
    return set(q.get("target") or [q["label"]])


def kev_model(run):
    from rlca.kev_backend import ELEMENT_QUESTION, load_kev, score_options
    tok, model = load_kev(run)

    def decide(rec):
        q = rec["questions"]["element"]
        keys = list(q["criteria"])
        probs, _, _ = score_options(tok, model, rec["state"], [f"{k}: {v}" for k, v in q["criteria"].items()], ELEMENT_QUESTION)
        i = max(range(len(probs)), key=probs.__getitem__)
        return keys[i], probs[i], ""
    return decide


def lm_model(repo):
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    model, tok = load(repo)
    greedy = make_sampler(temp=0.0)

    def decide(rec):
        q = rec["questions"]["element"]
        user = rec["state"] + "\n\nElements:\n" + "\n".join(f"{k}: {v}" for k, v in q["criteria"].items()) + "\n\nAnswer with the id:"
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        text, logp = "", 0.0
        for r in stream_generate(model, tok, prompt, max_tokens=8, sampler=greedy):
            text += r.text
            logp += float(r.logprobs[r.token])
            if re.search(r"\b(none|[a-z]\d+)\b", text.lower()) and not text.lower().rstrip().endswith(tuple("0123456789")):
                break
        m = re.search(r"\b(none|[a-z]\d+)\b", text.lower())
        return (m.group(1) if m else None), math.exp(logp), text
    return decide


def hf_model(repo):
    """lm_model on CUDA with transformers: same prompt, greedy, thinking off, confidence = P(generated tokens)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16).cuda().eval()

    @torch.inference_mode()
    def decide(rec):
        q = rec["questions"]["element"]
        user = rec["state"] + "\n\nElements:\n" + "\n".join(f"{k}: {v}" for k, v in q["criteria"].items()) + "\n\nAnswer with the id:"
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        x = tok(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
        out = model.generate(**x, max_new_tokens=8, do_sample=False, return_dict_in_generate=True, output_scores=True,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        new = out.sequences[0, x["input_ids"].shape[1]:]
        logp = sum(float(torch.log_softmax(sc[0].float(), -1)[t]) for sc, t in zip(out.scores, new))
        torch.cuda.synchronize()
        text = tok.decode(new, skip_special_tokens=True)
        m = re.search(r"\b(none|[a-z]\d+)\b", text.lower())
        return (m.group(1) if m else None), math.exp(logp), text
    return decide


def model2vec_model():
    import numpy as np
    from model2vec import StaticModel
    from rlca.candidates import EMBED_MODEL
    enc = StaticModel.from_pretrained(EMBED_MODEL)

    def decide(rec):
        q = rec["questions"]["element"]
        keys = [k for k in q["criteria"] if k != "none"]
        command = rec["state"].split("User command:", 1)[-1].strip()
        v = enc.encode([command] + [q["criteria"][k] for k in keys])
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
        cos = v[1:] @ v[0]
        e = np.exp((cos - cos.max()) * 20)
        i = int(cos.argmax())
        return keys[i], float(e[i] / e.sum()), ""
    return decide


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sets", nargs="+", default=["val", "heldout_site", "eval_ssv2", "eval_sspro"])
    ap.add_argument("--n", type=int, default=100, help="records per set (0 = all)")
    ap.add_argument("--records_dir", default="data/records")
    ap.add_argument("--out", default=str(ROOT / "runs/list-baselines"))
    a = ap.parse_args()
    kind, _, name = a.model.partition(":")
    decide = {"kev": lambda: kev_model(name), "lm": lambda: lm_model(name), "hf": lambda: hf_model(name),
              "model2vec": model2vec_model}[kind]()
    rows = []
    for k, (s, rec) in enumerate(sample(a.sets, a.n, a.records_dir)):
        t0 = time.perf_counter()
        pick, conf, raw = decide(rec)
        ms = (time.perf_counter() - t0) * 1000
        rows.append({"set": s, "id": rec["_meta"]["id"], "pick": pick, "conf": conf, "hit": pick in answers(rec["questions"]["element"]),
                     "ms": round(ms, 1), "warmup": k < 3, "n_options": len(rec["questions"]["element"]["criteria"]), "raw": raw})
    timed = [r for r in rows if not r["warmup"]]
    per_set = {}
    for s in a.sets:
        v = [r for r in rows if r["set"] == s]
        confs, hits = [r["conf"] for r in v], [r["hit"] for r in v]
        per_set[s] = {"n": len(v), "acc": round(sum(hits) / len(v), 3), "ece": round(ece(confs, hits), 3),
                      "act_alone_at_90": round(coverage_at(confs, hits, 0.9), 3),
                      "unparsed": sum(r["pick"] is None for r in v),
                      "median_ms": statistics.median(r["ms"] for r in v if not r["warmup"]) if any(not r["warmup"] for r in v) else None}
    summary = {"model": a.model, "median_ms": statistics.median(r["ms"] for r in timed),
               "p90_ms": sorted(r["ms"] for r in timed)[int(0.9 * (len(timed) - 1))], "per_set": per_set}
    out = Path(a.out) / re.sub(r"[^\w.-]+", "_", a.model)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
