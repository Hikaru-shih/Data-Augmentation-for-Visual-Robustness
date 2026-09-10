# Validation protocol: seed 42 results and multi-seed plan

## Completed seed 42

All five methods completed 100 epochs, clean evaluation, 75 corruption/severity evaluations, individual figures, pairwise baseline comparisons and the overall comparison.

Training uses 45,000 CIFAR-10 images and a fixed 5,000-image validation set (split seed 2026). Selection uses validation accuracy. Clean test accuracy is measured after selection. AugMix Transform excludes JSD consistency loss.

| Method | Clean accuracy (%) | mCA (%) | mCA delta vs baseline (pp) |
|---|---:|---:|---:|
| Baseline | 94.09 | 73.73 | 0.00 |
| Mixup | 95.23 | 79.10 | +5.37 |
| CutMix | 95.72 | 70.73 | -3.00 |
| RandAugment | 94.99 | 81.21 | +7.47 |
| AugMix Transform | 95.05 | 85.89 | +12.16 |

Source: results/comparisons/validation-v1-seed42/overall_comparison.csv. Deltas are computed before rounding. AugMix uses run ID validation-v2; its saved protocol is still validation-v1. The other methods use run ID validation-v1.

These are descriptive single-seed observations, not significance claims. Do not combine them with the old test-selected results in phase1_summary.md.

## Completed three-seed replication (2026-09-09)

Seeds 42, 43 and 44 are complete for all five methods. Values below are mean ± sample standard deviation (ddof=1) across three training seeds, in percent.

| Method | Clean accuracy (%) | mCA (%) | Mean mCA delta vs baseline (pp) |
|---|---:|---:|---:|
| Baseline | 94.42 ± 0.39 | 73.65 ± 0.26 | 0.00 |
| Mixup | 95.27 ± 0.15 | 78.54 ± 0.61 | +4.88 |
| CutMix | 95.77 ± 0.05 | 71.14 ± 0.46 | -2.51 |
| RandAugment | 95.03 ± 0.09 | 81.33 ± 0.34 | +7.68 |
| AugMix Transform | 94.83 ± 0.27 | 85.45 ± 0.44 | +11.80 |

Source: results/comparisons/multiseed-42-43-44/mean_std.csv and per_seed.csv. Deltas use unrounded means. The mCA ordering is identical in all three seeds: AugMix Transform > RandAugment > Mixup > Baseline > CutMix. CutMix has the highest clean accuracy in all three seeds. These are consistent descriptive observations across the tested seeds, not a formal statistical significance claim or evidence of a causal mechanism.

Next: analyze per-corruption and severity patterns across seeds, generate summary figures with variability, and then review related work to refine an extension hypothesis. No additional training is needed for those analyses. The execution instructions below are retained for reproducibility; the listed runs are already complete.

## Three-seed replication design (completed)

- Training seeds: 42, 43, 44 (all completed); five methods per seed.
- Keep validation seed 2026, split size, architecture, optimization, epochs and augmentation hyperparameters fixed. Do not tune based on the current corruption results during this replication.
- Reuse all seed 42 runs, including AugMix validation-v2. The 10 additional training runs have been completed.
- Report clean accuracy and mCA as mean and sample standard deviation across training seeds (ddof=1). Three seeds provide an initial variability estimate, not strong evidence of statistical significance.
- After replication, analyze corruption categories and severity, then check related literature before proposing an extension.

## Execute in WSL from the repository root

```bash
bash scripts/run_multiseed.sh
```

The script runs seeds 43 and 44 sequentially, trains and evaluates each method, then aggregates all 15 runs. It does not regenerate seed 42 or individual figures. Output logs are appended under results/logs/ so terminal history loss does not erase the logs. Keep the computer awake during training; logs do not provide checkpoint resume.

Completed training (clean_results.json present) and evaluation (metadata.json present) are skipped when restarting the batch. Final aggregation validates checkpoint and CSV hashes. If training is interrupted, the existing run directory still blocks restart: do not delete completed runs. Archive the incomplete run and adjust the batch and aggregation run mapping if choosing a new run ID. The trainer does not resume partial epochs.

To aggregate again after all runs finish:

```bash
python3 -m scripts.summarize_multiseed
```

Outputs: results/comparisons/multiseed-42-43-44/per_seed.csv and mean_std.csv. Aggregation refuses missing runs, changed settings across seeds, different split files, incomplete histories or invalid evaluation provenance. Missing runs are never silently excluded.
