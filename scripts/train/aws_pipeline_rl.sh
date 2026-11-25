#!/bin/bash
# RL on grounding, Qwen3-VL-2B: SFT on the train split minus the 8 RL dev apps, then GRPO on top, then MacBench v1.2
# (4,342-command screenshot subset) for both, so the only difference between the two scores is the RL stage.
# Data on the box: data/vlm_rl/{train,dev}.jsonl, the sandbox shots they point to, data/macbench/v1.2,
# data/macbench/subset_ids.txt. One line per stage in ~/rl.log.
#   STEPS=400 bash scripts/train/aws_pipeline_rl.sh
set -u
cd ~/rlca
PY=${PY:-~/vlmft/bin/python}
MP=3276800
B="--sets macbench --bench data/macbench/v1.2 --out runs/vlm-rl/mb12 --max_pixels $MP --ids_file data/macbench/subset_ids.txt"
log() { echo "$(date +%H:%M) $*" >> ~/rl.log; }
mkdir -p ~/logs runs/vlm-rl
log "sft start"
$PY scripts/train/vlm_sft.py --base Qwen/Qwen3-VL-2B-Instruct --data data/vlm_rl/train.jsonl --out runs/vlm-rl/qwen-sft \
  --per_seq 24 --max_pixels $MP > ~/logs/rl-sft.log 2>&1 && log "sft done" || { log "sft FAILED"; exit 1; }
log "grpo start"
$PY scripts/train/vlm_grpo.py --init runs/vlm-rl/qwen-sft/merged --out runs/vlm-rl/qwen-grpo --steps ${STEPS:-400} \
  > ~/logs/rl-grpo.log 2>&1 && log "grpo done" || { log "grpo FAILED"; exit 1; }
for t in qwen-sft qwen-grpo; do
  [ -d runs/vlm-rl/$t/merged ] || { log "eval $t skipped (no merge)"; continue; }
  log "eval $t start"
  $PY scripts/eval/vlm_grounding.py --model venus-2b --kind venus --repo runs/vlm-rl/$t/merged --tag $t $B \
    > ~/logs/rl-eval-$t.log 2>&1 && log "eval $t done" || log "eval $t FAILED"
done
log "all done"
