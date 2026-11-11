#!/bin/bash
# Kev 0.8B SFT on the AX + ranked-icon sandbox records (data/records_pixel), then evals. One line per stage in ~/pipeline.log.
set -u
cd ~/kev
export KEV_PAD_BUCKET=256 KEV_EVAL_CONTEXT=serving
P=~/rlca/data/records_pixel R=~/rlca/data/records
log() { echo "$(date +%H:%M) $*" >> ~/pipeline.log; }
ev() {   # ev <run> <outdir> <data>
  n=$(basename $3 .jsonl); log "eval $2 $n start"
  .venv/bin/python -m kev.benchmark --device cuda --run "$1" --data $3 --out runs/$2/$n > ~/logs/$2-$n.log 2>&1 \
    && log "eval $2 $n done" || log "eval $2 $n FAILED"
}
log "train start"
.venv/bin/python -m kev.train --device cuda --data $P/train_mix.jsonl --base Qwen/Qwen3.5-0.8B-Base \
  --init_from jaredpalmer/kev-0.8b@v1.0 --lr 2e-5 --batch ${BATCH:-1} --accum ${ACCUM:-8} --max_state 4224 --perm_kl 0.1 \
  --dtype bf16 --checkpointing 1 --epochs 1 --save_every_minutes 60 --resume 1 --out runs/sft-0.8b-pixel > ~/logs/train-0.8b-pixel.log 2>&1 \
  && log "train done" || { log "train FAILED"; exit 1; }
for d in $P/val.jsonl $R/eval_ssv2.jsonl $R/eval_sspro.jsonl $R/eval_osworldg.jsonl $R/eval_omniact.jsonl; do
  ev runs/sft-0.8b-pixel eval-sft-0.8b-pixel $d
done
ev jaredpalmer/kev-0.8b@v1.0 eval-base-0.8b $P/val.jsonl
log "all done"
