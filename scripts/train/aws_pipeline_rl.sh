#!/bin/bash
# RL on grounding, Qwen3-VL-2B: SFT on the train split minus the 8 RL dev apps, then GRPO on top, then MacBench v1.2
# (4,342-command screenshot subset) for both, so the only difference between the two scores is the RL stage.
# Data on the box: data/vlm_rl/{train,dev}.jsonl, the sandbox shots they point to, data/macbench/v1.2,
# data/macbench/subset_ids.txt. One line per stage in ~/rl.log.
#   FILTER=1 (default): before GRPO, sample 8 answers per training command with vLLM (~/vllm venv) and train only on
#     commands whose answers disagree (scripts/train/vlm_rl_filter.py) and at most 6 of 8 are right (7/8 ones often
#     resample as 8/8 and give no signal). FILTER=0 is run 1 (all commands, 400 steps).
#   STEPS=300 bash scripts/train/aws_pipeline_rl.sh
set -u
cd ~/rlca
PY=${PY:-~/vlmft/bin/python}
VPY=${VPY:-~/vllm/bin/python}
MP=3276800
FILTER=${FILTER:-1}
B="--sets macbench --bench data/macbench/v1.2 --out runs/vlm-rl/mb12 --max_pixels $MP --ids_file data/macbench/subset_ids.txt"
log() { echo "$(date +%H:%M) $*" >> ~/rl.log; }
mkdir -p ~/logs runs/vlm-rl
if [ -f runs/vlm-rl/qwen-sft/merged/config.json ]; then log "sft exists"; else
  log "sft start"
  $PY scripts/train/vlm_sft.py --base Qwen/Qwen3-VL-2B-Instruct --data data/vlm_rl/train.jsonl --out runs/vlm-rl/qwen-sft \
    --per_seq 24 --max_pixels $MP > ~/logs/rl-sft.log 2>&1 && log "sft done" || { log "sft FAILED"; exit 1; }
fi
if [ "$FILTER" = 1 ]; then
  DATA=data/vlm_rl/train_hard.jsonl; RUN=qwen-grpo-hard; STEPS=${STEPS:-300}; EVERY=75; EXTRA="--max_p_hit 0.75"
  if [ -f $DATA.stats.json ]; then log "filter exists"; else
    log "filter start"
    $VPY scripts/train/vlm_rl_filter.py --model runs/vlm-rl/qwen-sft/merged --out $DATA --max_pixels $MP \
      > ~/logs/rl-filter.log 2>&1 && log "filter done" || { log "filter FAILED"; exit 1; }
  fi
else
  DATA=data/vlm_rl/train.jsonl; RUN=qwen-grpo; STEPS=${STEPS:-400}; EVERY=50; EXTRA=""
fi
log "grpo $RUN start"
$PY scripts/train/vlm_grpo.py --init runs/vlm-rl/qwen-sft/merged --data $DATA --out runs/vlm-rl/$RUN --steps $STEPS \
  --eval_every $EVERY $EXTRA > ~/logs/rl-$RUN.log 2>&1 && log "grpo done" || { log "grpo FAILED"; exit 1; }
for t in $RUN qwen-sft; do
  [ -d runs/vlm-rl/$t/merged ] || { log "eval $t skipped (no merge)"; continue; }
  log "eval $t start"
  $PY scripts/eval/vlm_grounding.py --model venus-2b --kind venus --repo runs/vlm-rl/$t/merged --tag $t $B \
    > ~/logs/rl-eval-$t.log 2>&1 && log "eval $t done" || log "eval $t FAILED"
done
log "all done"
