#!/bin/zsh
# After the label_test batches are labelled: validate them, build MacBench v1, scan it for privacy, build the review page.
set -e
cd "${0:A:h}/../.."
.venv/bin/python scripts/sandbox/sandbox_label.py check --dir label_test
.venv/bin/python scripts/bench/macbench_build.py --out data/macbench/v1
.venv/bin/python scripts/bench/macbench_privacy.py --bench data/macbench/v1
.venv/bin/python scripts/bench/macbench_review_build.py --bench data/macbench/v1 --n 300
echo "review: open data/macbench/v1/review/index.html"
