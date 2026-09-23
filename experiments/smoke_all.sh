#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${PYTHONPATH:-.}"
"$PYTHON_BIN" experiments/run_trace_collection.py --dataset gsm8k --limit 2 --run-id smoke --seed 0
"$PYTHON_BIN" experiments/run_counterfactuals.py --run-id smoke --operator nullify --k 3 --seed 0
"$PYTHON_BIN" experiments/run_rewards.py --run-id smoke
"$PYTHON_BIN" experiments/run_student.py --run-id smoke
"$PYTHON_BIN" experiments/run_control.py --run-id smoke
"$PYTHON_BIN" experiments/analyze.py --run-id smoke
