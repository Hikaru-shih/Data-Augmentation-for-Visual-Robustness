# Cross-dataset summary

Effects are paired within each dataset/seed; datasets are not pooled.
Noise endpoint = Gaussian and shot, averaged over five severities. SD is sample SD, not CI.

Positive p=0.5 minus p=1 corruption effects in all three seeds of both datasets: gaussian_noise, jpeg_compression, pixelate, shot_noise

| Dataset | Control | Metric | Mean delta (pp) | SD |
|---|---|---|---:|---:|
| CIFAR-10 | baseline | clean | +1.14 | 0.17 |
| CIFAR-10 | baseline | mca | -0.10 | 0.69 |
| CIFAR-10 | baseline | noise | -3.20 | 2.05 |
| CIFAR-10 | cutmix_p1 | clean | -0.21 | 0.31 |
| CIFAR-10 | cutmix_p1 | mca | +2.41 | 1.03 |
| CIFAR-10 | cutmix_p1 | noise | +13.21 | 5.60 |
| CIFAR-100 | baseline | clean | +2.03 | 0.46 |
| CIFAR-100 | baseline | mca | -0.19 | 0.43 |
| CIFAR-100 | baseline | noise | -5.30 | 1.44 |
| CIFAR-100 | cutmix_p1 | clean | -0.69 | 0.09 |
| CIFAR-100 | cutmix_p1 | mca | +1.32 | 0.70 |
| CIFAR-100 | cutmix_p1 | noise | +5.14 | 2.29 |

This is limited cross-task evidence within the CIFAR family and one architecture. No formal significance, equivalence or causal mechanism claim.
