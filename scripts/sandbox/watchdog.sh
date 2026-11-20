#!/bin/zsh
# Keep a sandbox collection alive until END (epoch secs): restart collect.py if it exits or its log
# stops moving for STALL secs. Each restart writes a new folder (<out>-partN), since shot ids restart at 0.
# Run inside the GUI session: sudo launchctl asuser 501 sudo -u ec2-user nohup zsh watchdog.sh END OUT ONLY &
# Extra collect.py arguments for restarts go in $EXTRA (e.g. EXTRA="--extra_sites f.txt --reseed_every 1000").
[[ "$RLCA_SANDBOX" == 1 ]] || { echo "refusing: sandbox only"; exit 1; }
END=$1 OUT=$2 ONLY=$3 STALL=${STALL:-3600}
cd ~/rlca
cur=$OUT part=1
while (( $(date +%s) < END - 600 )); do
  sleep 300
  log=$cur/collect.log
  age=$(( $(date +%s) - $(stat -f %m $log 2>/dev/null || echo 0) ))
  if pgrep -f scripts/sandbox/collect.py >/dev/null && (( age < STALL )); then continue; fi
  echo "$(date -u +%H:%M) restart: alive=$(pgrep -f scripts/sandbox/collect.py | wc -l | tr -d ' ') log_age=${age}s"
  pkill -f scripts/sandbox/collect.py; sleep 10; pkill -9 -f scripts/sandbox/collect.py
  hours=$(( (END - $(date +%s)) / 3600.0 ))
  part=$((part + 1)) cur=$OUT-part$part
  mkdir -p $cur
  nohup .venv/bin/python scripts/sandbox/collect.py --hours $hours --only $ONLY ${=EXTRA} --out $cur > $cur/collect.log 2>&1 &
done
echo "$(date -u +%H:%M) watchdog done"
