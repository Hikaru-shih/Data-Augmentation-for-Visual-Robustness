#!/usr/bin/env bash
set -euo pipefail
# Validate the complete corruption archive and GPU before reserving any run.
python3 -m scripts.check_cifar100_data
python3 -m scripts.check_environment
mkdir -p results/logs
for seed in 42 43 44; do
  for method in baseline cutmix cutmix_p05; do
    run="cifar100_${method}_resnet18/seed_${seed}/validation-v1"
    log="results/logs/cifar100_${method}_seed_${seed}.log"
    if [[ ! -f "results/$run/clean_results.json" ]]; then
      python3 -u -m scripts.train --config "configs/cifar100_${method}.yaml" --seed "$seed" --run-id validation-v1 2>&1 | tee -a "$log"
    fi
    if [[ ! -f "results/$run/robustness/metadata.json" ]]; then
      python3 -u -m scripts.evaluate_corruptions "$run" 2>&1 | tee -a "$log"
    fi
    python3 -m scripts.plot_history "$run"
    python3 -m scripts.plot_robustness "$run"
  done
done
python3 -m scripts.compare_cutmix_probability --dataset cifar100
