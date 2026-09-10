#!/usr/bin/env bash
# Run from the repository root in the existing Python environment.
# Stop on any failure. Completed training/evaluation is reused, never overwritten.
set -euo pipefail
mkdir -p results/logs
for seed in 43 44; do
  for method in baseline mixup cutmix randaugment augmix; do
    run="${method}_resnet18/seed_${seed}/validation-v1"
    log="results/logs/${method}_seed_${seed}.log"
    if [[ ! -f "results/$run/clean_results.json" ]]; then
      python3 -u -m scripts.train --config "configs/$method.yaml" --seed "$seed" --run-id validation-v1 2>&1 | tee -a "$log"
    fi
    if [[ ! -f "results/$run/robustness/metadata.json" ]]; then
      python3 -u -m scripts.evaluate_corruptions "$run" 2>&1 | tee -a "$log"
    fi
  done
done
python3 -m scripts.summarize_multiseed
