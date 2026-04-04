#!/usr/bin/env bash
set -euo pipefail

echo "==> Running test suite"
pytest -q

echo
echo "==> Running quick end-to-end benchmark"
python run_benchmark.py --mode e2e --verbose 0 --save

echo
echo "Done. Check logs/ for benchmark artifacts."
