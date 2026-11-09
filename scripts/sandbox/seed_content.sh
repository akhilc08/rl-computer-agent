#!/bin/zsh
# Fill a throwaway Mac with made-up content (see seed_content.py). Refuses unless RLCA_SANDBOX=1.
#   RLCA_SANDBOX=1 scripts/sandbox/seed_content.sh
set -e
[[ "$RLCA_SANDBOX" == 1 ]] || { echo "refusing: set RLCA_SANDBOX=1 on a sandbox Mac only"; exit 1; }
cd "$(dirname "$0")/../.."
exec .venv/bin/python scripts/sandbox/seed_content.py "$@"
