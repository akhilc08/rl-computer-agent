#!/bin/bash
# Untrained evals, then SFT of Kev 0.8B, then trained evals. Logs to ~/pipeline.log, one line per stage.
set -u
cd ~/kev
export KEV_PAD_BUCKET=256
R=../rlca/data/records
SETS="eval_synthetic val heldout_site eval_ssv2 eval_omniact eval_sspro"
evals() {   # evals <run> <outdir>
  for s in $SETS; do
    echo "$(date +%H:%M) eval $2 $s start" >> ~/pipeline.log
    .venv/bin/python -m kev.benchmark --device cuda --run "$1" --data $R/$s.jsonl --out runs/$2/$s > ~/logs/$2-$s.log 2>&1 \
      && echo "$(date +%H:%M) eval $2 $s done" >> ~/pipeline.log || echo "$(date +%H:%M) eval $2 $s FAILED" >> ~/pipeline.log
  done
}
mkdir -p ~/logs
[ "${SKIP_BASE:-0}" = 1 ] || evals jaredpalmer/kev-0.8b@v1.0 eval-base-0.8b
echo "$(date +%H:%M) train 0.8b start" >> ~/pipeline.log
.venv/bin/python -m kev.train --device cuda --data $R/train_mix.jsonl --base Qwen/Qwen3.5-0.8B-Base \
  --init_from jaredpalmer/kev-0.8b@v1.0 --lr 2e-5 --batch 1 --accum 8 --max_state 4224 --perm_kl 0.1 \
  --dtype bf16 --checkpointing 1 --epochs 1 --out runs/sft-0.8b-v1 > ~/logs/train-0.8b.log 2>&1 \
  && echo "$(date +%H:%M) train 0.8b done" >> ~/pipeline.log || { echo "$(date +%H:%M) train 0.8b FAILED" >> ~/pipeline.log; exit 1; }
evals runs/sft-0.8b-v1 eval-sft-0.8b-v1
echo "$(date +%H:%M) all done" >> ~/pipeline.log
