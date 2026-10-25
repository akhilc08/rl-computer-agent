#!/bin/zsh
# Overnight run: explore the allowed apps and sites, then replay the screenshots through every
# element source for clean latency numbers. Keeps the Mac awake for the whole run.
#   scripts/data/overnight.sh [hours] [explore.py args, e.g. --arc_window ID]
set -e
cd "$(dirname "$0")/../.."
HOURS=${1:-6.5}
RUN=data/explore/$(date +%Y%m%d-%H%M%S)
mkdir -p "$RUN"
caffeinate -dims .venv/bin/python scripts/data/explore.py --hours "$HOURS" --out "$RUN" "${@:2}" 2>&1 | tee "$RUN/explore.log"
caffeinate -dims .venv/bin/python scripts/eval/replay_sources.py --run "$RUN" 2>&1 | tee "$RUN/replay.log"
echo "done: $RUN"
