# Validation protocol: completed experiments and current status

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

The original five-method corruption analysis and initial literature audit are complete. The CutMix p=0.5 ablation is also complete; see [its results and next steps](cutmix_probability_results.md). Its paired per-corruption and severity analysis is complete. Next, consolidate the report and predefine an independent generalization check before any new training. The execution instructions below are retained for reproducibility; the listed runs are already complete.

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

## CutMix probability ablation completed (verified 2026-09-14)

Three additional runs bring the completed validation-protocol experiments to 18. CutMix p=0.5 achieves clean accuracy 95.56 +/- 0.30%, mCA 73.56 +/- 0.67%, and Gaussian/shot noise accuracy 48.76 +/- 3.56%. Relative to p=1, paired noise accuracy improves by 13.21 +/- 5.60 percentage points. All three seeds improve noise accuracy and mCA, but noise remains below baseline in every seed. See [the full results](cutmix_probability_results.md) for definitions, paired statistics and limitations.

## Detailed probability analysis completed (2026-09-14)

See [ablation findings](cutmix_probability_results.md). The noise improvements are not uniform across noise types or monotonic with severity; impulse noise exceeds baseline on average across severities in all seeds, while Gaussian/shot remain below it. New tables and three figures are under results/analysis/cutmix-probability/.

## Report completed (2026-09-14)

See [research report](../../report/research_findings.md) for methods, results, paired ablation, figures and limitations. A [cross-dataset validation draft](transfer_validation_plan.md) proposes CIFAR-100 / CIFAR-100-C with three methods and three seeds. This is not implemented or started; current completed run count remains 18.

## CIFAR-100 implementation ready (2026-09-15)

The [transfer workflow](cifar100_execution.md) is implemented and passed 23 tests plus syntax checks. No new dataset was downloaded or scored in this step. Run preparation first, then the nine-run batch as documented. Completed experiment count remains 18 until those runs finish.
