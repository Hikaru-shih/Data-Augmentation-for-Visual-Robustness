#!/usr/bin/env bash
set -euo pipefail
mkdir -p results/logs
for seed in 42 43 44; do
  run="cutmix_p05_resnet18/seed_${seed}/validation-v1"
  log="results/logs/cutmix_p05_seed_${seed}.log"
  if [[ ! -f "results/$run/clean_results.json" ]]; then
    python3 -u -m scripts.train --config configs/cutmix_p05.yaml --seed "$seed" --run-id validation-v1 2>&1 | tee -a "$log"
  fi
  if [[ ! -f "results/$run/robustness/metadata.json" ]]; then
    python3 -u -m scripts.evaluate_corruptions "$run" 2>&1 | tee -a "$log"
  fi
  python3 -m scripts.plot_history "$run"
  python3 -m scripts.plot_robustness "$run"
done
python3 -m scripts.compare_cutmix_probability
