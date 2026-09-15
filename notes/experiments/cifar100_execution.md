# CIFAR-100 transfer: implementation and execution record

Date: 2026-09-15. Implementation and synthetic tests complete. No CIFAR-100 model has been trained or scored by this change. The existing 18 CIFAR-10 runs remain separate.

## Commands (WSL, repository root, existing Python environment)

```bash
bash scripts/prepare_cifar100.sh
bash scripts/run_cifar100_transfer.sh
```

Run the second command only after the first succeeds. Preparation downloads the official approximately 2.9 GB CIFAR-100-C archive, verifies published MD5, extracts it, and downloads/verifies clean CIFAR-100 through torchvision. Allow roughly 6 GB for archive plus extracted corruptions, and additional space for clean data and runs. It does not score models. Interrupted archive downloads resume using the .part file. The tar is retained. Source: [Zenodo v1](https://zenodo.org/records/3555552), MD5 `11f0ed0f1191edbf9fa23466ae6021d3`.

The batch runs Baseline, CutMix p=1, CutMix p=0.5 for seeds 42/43/44, sequentially. Each run includes training, clean evaluation, corruption evaluation, history and robustness plots. At the end it produces aggregate statistics, paired deltas and detailed figures. Logs survive terminal closure under results/logs; a terminal closure may still terminate the process. Keep the machine awake. Completed stages are reused, but interrupted training does not resume and its directory will block a fresh run with the same identity. Inspect the incomplete run before changing names or archiving it.

Run keys: `cifar100_<baseline|cutmix|cutmix_p05>_resnet18/seed_<42|43|44>/validation-v1`. Saved protocol: `cifar100-transfer-v1`. Analysis output: `results/analysis/cifar100_cutmix-probability/`. No output shares the old CIFAR-10 paths.

To regenerate only analysis after all nine runs finish:

```bash
python3 -m scripts.compare_cutmix_probability --dataset cifar100
```

## Frozen settings and traceability

The design in [transfer_validation_plan.md](transfer_validation_plan.md) is implemented: 100 fine classes, exactly 50 validation examples per class, split seed 2026, 100 epochs, batch size 128, unchanged optimizer and CutMix alpha/probabilities. Mean/population SD are computed from the 45,000 unaugmented training images only, using float64 sums in batches; validation and test do not contribute. Values are written into the effective config and checkpoint. Training data pixels and ordered fine labels are SHA-256 hashed.

New runs save source.zip, source_manifest.json, source hash, Git metadata, exact split indices and effective config. Corruption evaluation validates fine-label identity against clean CIFAR-100 test labels before scoring, records all 16 used data-file hashes, and uses checkpoint normalization. Comparison requires matching configurations except augmentation, matching split hashes, and matching corruption data hashes. Baseline/p=1/p=0.5 are all trained anew on CIFAR-100.

Source snapshot and Git state record the implementation used; do not edit source during training. Known limitation: counts/label identity and checksums establish archive identity when using the verified preparation workflow; arbitrary renamed third-party arrays should not be substituted.

## Changes

- New src/datasets/cifar100.py: stratified split, train-only statistics, transforms and training data hash.
- src/datasets/splits.py and scripts/train.py: dataset selection, class-count validation, save resolved normalization before training, distinct protocol.
- src/datasets/cifar10c.py: parameterized class range/normalization and memory-mapped images; retains legacy 10-class defaults.
- scripts/evaluate_corruptions.py: model class count, corruption root and normalization follow checkpoint; CIFAR-100 data hashes and fine-label validation added.
- Three configs/cifar100_*.yaml, preparation/check/batch scripts, dataset option in probability comparison. Old configs and checkpoints are unchanged.
- tests/test_cifar100.py covers balanced disjoint deterministic split, exclusion of holdout from normalization, synthetic loaders, 100-class forward/backward, and corruption label 99/100 boundaries.

No hyperparameters were chosen by looking at CIFAR-100-C model scores. Performance and runtime remain unknown until the user runs the batch. The main endpoint is the paired p=0.5 minus p=1 Gaussian/shot noise difference, with clean accuracy and full mCA as secondary metrics. Report negative or inconsistent results too.
