"""Run TypeSafe's Jev (hosted, via OpenRouter's Decisions API) on Kev-format records.

Each record's state and questions go to Jev as they are, minus the answer labels. Output matches Kev's
predictions.jsonl ({"id", "prediction": {"probabilities": {question: {option: p}}}, "ms", "usage"}), so the
same scoring applies. Reruns skip ids already written. Stops once spend passes --max_cost (dollars).
Public data only: the screens' text leaves the machine.

  python scripts/eval/run_jev.py --data data/records_s/eval_ssv2.jsonl --out runs/jev/eval_ssv2
Needs OPENROUTER_API_KEY in the environment or .env.
"""
import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
URL = "https://openrouter.ai/api/alpha/decisions"


def api_key():
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("OPENROUTER_API_KEY not set")


def request_body(rec, model):
    questions = {q: {k: v for k, v in spec.items() if k != "label"} for q, spec in rec["questions"].items()}
    return {"model": model, "state": rec["state"], "questions": questions}


def probabilities(answers):
    out = {}
    for q, a in answers.items():
        if a.get("type") == "choice":
            out[q] = a.get("probabilities") or {a["choice"]: 1.0}
        elif a.get("type") == "noul":
            out[q] = {"true": a["noul"], "false": 1 - a["noul"]}
        else:                       # refusal or an unexpected type: no answer
            out[q] = {}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="typesafe/jev-1.14")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_cost", type=float, default=1.5)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "predictions.jsonl"
    done = {json.loads(l)["id"] for l in open(pred_path)} if pred_path.exists() else set()
    recs = [json.loads(l) for l in open(args.data)]
    todo = [r for r in recs if r["_meta"]["id"] not in done][: args.limit or None]
    print(f"{len(recs)} records, {len(done)} done, {len(todo)} to do", flush=True)
    headers = {"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"}
    lock, spent, stop = threading.Lock(), [0.0], threading.Event()
    f = open(pred_path, "a")

    def one(rec):
        if stop.is_set():
            return
        body = request_body(rec, args.model)
        for attempt in range(6):
            t0 = time.perf_counter()
            try:
                r = requests.post(URL, headers=headers, json=body, timeout=120)
            except requests.RequestException as e:
                err = str(e)
            else:
                if r.status_code == 200:
                    d = r.json()
                    ms = round((time.perf_counter() - t0) * 1000)
                    row = {"id": rec["_meta"]["id"], "prediction": {"probabilities": probabilities(d["answers"])},
                           "ms": ms, "usage": d.get("usage"), "model": d.get("model")}
                    with lock:
                        spent[0] += (d.get("usage") or {}).get("cost") or 0
                        f.write(json.dumps(row) + "\n")
                        f.flush()
                        if spent[0] > args.max_cost:
                            stop.set()
                    return
                err = f"{r.status_code} {r.text[:200]}"
                if r.status_code not in (429, 500, 502, 503, 504):
                    break
            time.sleep(2 ** attempt)
        print(f"failed {rec['_meta']['id']}: {err}", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        for i, _ in enumerate(ex.map(one, todo)):
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(todo)} spent ${spent[0]:.4f}", flush=True)
    f.close()
    print(f"done, spent ${spent[0]:.4f}" + (" (stopped at --max_cost)" if stop.is_set() else ""), flush=True)


if __name__ == "__main__":
    main()
