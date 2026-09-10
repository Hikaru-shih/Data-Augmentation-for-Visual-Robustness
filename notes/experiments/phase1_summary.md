# Phase 1 — Overall Results

## Experimental Setup

- Dataset: CIFAR-10
- Robustness Benchmark: CIFAR-10-C
- Model: CIFAR-adapted ResNet-18
- Seed: 42
- Training Epochs: 100

Methods:
- Baseline
- Mixup
- CutMix
- RandAugment
- AugMix augmentation transform

## Overall Results

| Method | Clean Accuracy (%) | mCA (%) |
|---|---:|---:|
| Baseline | 95.04 | 72.47 |
| Mixup | 95.53 | 79.88 |
| CutMix | 96.09 | 72.28 |
| RandAugment | 95.45 | 82.03 |
| AugMix Transform | 95.24 | 85.36 |

## Initial Observations

- Clean accuracy is similar across all methods.
- Corruption robustness differs much more strongly than clean accuracy.
- CutMix achieves the highest clean accuracy, but does not improve mCA over the baseline.
- Mixup improves mCA substantially.
- RandAugment improves mCA further.
- AugMix Transform achieves the highest mCA in the current experiments.

## Important Notes

- These are single-seed results using seed 42.
- The current AugMix experiment uses torchvision.transforms.AugMix only.
- It does not include the Jensen-Shannon consistency loss from the original AugMix paper.
- Therefore, this experiment should be referred to as "AugMix augmentation transform", not a full reproduction of AugMix training. 
## Legacy evaluation protocol

These results used all 50,000 training images and selected the checkpoint with the highest clean test accuracy across 100 epochs. The test set therefore participated in model selection. These exploratory results are preserved and must not be pooled with the new validation-v1 runs, which train on 45,000 images and select on a fixed 5,000-image validation split.
