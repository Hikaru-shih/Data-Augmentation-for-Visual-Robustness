# CutMix probability ablation — execution and change record

Implementation date: 2026-09-10. Status: implemented and tested, training not started by the assistant.

## Fixed design

New method: cutmix_p05_resnet18, probability=0.5, alpha=1.0. Seeds 42/43/44, run ID validation-v1. All other settings match configs/cutmix.yaml, including 100 epochs and the fixed validation split. Existing baseline_resnet18 and cutmix_resnet18 runs are controls. Missing probability in old CutMix configs is interpreted as 1.0 because the reviewed trainer always mixed every batch.

Primary endpoint: equally weighted mean accuracy over Gaussian and shot noise, five severities each, computed within each seed. Report p=0.5 minus p=1.0 paired differences, mean and sample SD (ddof=1). Also report baseline differences, clean accuracy and all-corruption mCA. Per-corruption outputs remain available in each run. This is an exploratory ablation motivated by prior inspection of CIFAR-10-C, not a new untouched confirmatory benchmark.

## What changed

- src/training/cutmix_trainer.py: batch probability gate; unselected batches use ordinary cross-entropy. Default p=1 uses no extra gate RNG draw, preserving the previous mixing random sequence. p=0 likewise avoids gate draws. A gate decision that selects a zero-area CutMix box still counts as applied; the count measures policy use, not changed pixels.
- src/datasets/splits.py: accepts CutMix probability without passing it into image transforms.
- scripts/train.py: validates and saves effective probability; adds batch counts and realized fraction to CutMix history; saves source archive hash in config and checkpoint.
- src/utils/provenance.py: archives exact Python, shell, YAML and test source bytes plus requirements.txt; records SHA-256 per file. Each new run stores source.zip and source_manifest.json. Dependencies themselves are not archived.
- configs/cutmix_p05.yaml: independent experiment; original configs and existing results were not rewritten.
- scripts/compare_cutmix_probability.py: checks evaluation provenance, method, seed, protocol, shared non-augmentation config, splits and complete histories; computes per-seed and paired statistics. Missing runs fail instead of silently producing partial statistics.
- scripts/run_cutmix_p05.sh: sequential training, evaluation, individual plots and final comparison. Logs are retained; failures stop the batch. Completed training/evaluation are skipped. Interrupted training still requires a new run identity or careful archiving of that incomplete run; there is no resume support.

## Run from WSL, repository root

```bash
bash scripts/run_cutmix_p05.sh
```

Only three new training runs are required. Outputs are under results/cutmix_p05_resnet18/seed_<seed>/validation-v1; checkpoints use the corresponding checkpoints path. Logs: results/logs/cutmix_p05_seed_<seed>.log. Comparison: results/analysis/cutmix-probability/{per_seed,mean_std,paired_deltas,paired_mean_std}.csv. mean_std files use two header rows (metric/statistic).

To repeat only comparison:

```bash
python3 -m scripts.compare_cutmix_probability
```

## Provenance limitations

Git records the implementation release; new runs record Git revision and working-tree status as well as actual source bytes. Existing seed 42/43/44 control runs predate source archives. Their saved configs, histories, environment information and evaluation hashes remain available, but their exact historical uncommitted source cannot be reconstructed with certainty. The release commit must not be presented as their original training commit. Keep this limitation in any report comparing old and new runs.

Before future code changes, retain the release commit and source archives. Do not edit source during a running experiment: imports and worker startup can otherwise observe different versions.
