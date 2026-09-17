# CutMix probability: detailed analysis

Accuracy and paired delta SD are computed across seeds 42,43,44 (ddof=1).
Severity is averaged within each seed for corruption-level tables. All 15 corruptions are included.
The predeclared primary noise endpoint includes Gaussian and shot only; impulse is secondary.

## p=0.5 minus CutMix p=1

Positive mean: 9/15; positive in all seeds: 5/15; negative in all seeds: 2/15.

| Corruption | Mean delta (pp) | Sample SD | Seed min | Seed max |
|---|---:|---:|---:|---:|
| gaussian_noise | +13.47 | 6.17 | +7.59 | +19.89 |
| shot_noise | +12.96 | 5.02 | +8.10 | +18.14 |
| impulse_noise | +5.00 | 2.42 | +2.29 | +6.94 |
| jpeg_compression | +3.40 | 0.54 | +2.77 | +3.72 |
| pixelate | +2.53 | 2.33 | +0.47 | +5.06 |
| defocus_blur | +0.39 | 2.40 | -1.21 | +3.15 |
| contrast | +0.32 | 0.36 | -0.05 | +0.67 |
| elastic_transform | +0.28 | 0.42 | -0.21 | +0.58 |
| zoom_blur | +0.09 | 3.04 | -1.80 | +3.59 |
| fog | -0.12 | 0.41 | -0.52 | +0.29 |
| brightness | -0.15 | 0.13 | -0.30 | -0.07 |
| frost | -0.22 | 0.98 | -1.05 | +0.86 |
| glass_blur | -0.28 | 5.18 | -3.41 | +5.70 |
| motion_blur | -0.48 | 0.94 | -1.57 | +0.07 |
| snow | -0.97 | 0.51 | -1.32 | -0.38 |

## p=0.5 minus Baseline

Positive mean: 8/15; positive in all seeds: 5/15; negative in all seeds: 5/15.

| Corruption | Mean delta (pp) | Sample SD | Seed min | Seed max |
|---|---:|---:|---:|---:|
| contrast | +4.84 | 0.40 | +4.41 | +5.22 |
| impulse_noise | +3.23 | 0.47 | +2.69 | +3.58 |
| snow | +3.06 | 1.31 | +2.16 | +4.57 |
| glass_blur | +1.41 | 3.94 | -2.45 | +5.42 |
| frost | +1.16 | 2.03 | -0.10 | +3.50 |
| brightness | +0.93 | 0.12 | +0.80 | +1.01 |
| fog | +0.81 | 0.32 | +0.48 | +1.12 |
| elastic_transform | +0.19 | 0.35 | -0.22 | +0.40 |
| motion_blur | -0.46 | 0.71 | -1.23 | +0.17 |
| defocus_blur | -1.41 | 1.35 | -2.22 | +0.15 |
| jpeg_compression | -2.63 | 0.50 | -3.03 | -2.07 |
| gaussian_noise | -2.83 | 2.22 | -5.25 | -0.89 |
| pixelate | -3.06 | 1.15 | -3.77 | -1.72 |
| zoom_blur | -3.14 | 1.48 | -4.17 | -1.44 |
| shot_noise | -3.56 | 1.88 | -5.62 | -1.94 |

## Noise severity: p=0.5 minus p=1

| Corruption | Severity | Mean delta (pp) | SD |
|---|---:|---:|---:|
| gaussian_noise | 1 | +10.34 | 2.30 |
| gaussian_noise | 2 | +18.12 | 8.30 |
| gaussian_noise | 3 | +15.95 | 8.57 |
| gaussian_noise | 4 | +12.90 | 6.44 |
| gaussian_noise | 5 | +10.06 | 5.33 |
| impulse_noise | 1 | +2.77 | 1.95 |
| impulse_noise | 2 | +7.17 | 4.29 |
| impulse_noise | 3 | +7.93 | 4.53 |
| impulse_noise | 4 | +4.51 | 4.82 |
| impulse_noise | 5 | +2.61 | 8.48 |
| shot_noise | 1 | +6.23 | 0.32 |
| shot_noise | 2 | +11.07 | 2.75 |
| shot_noise | 3 | +17.95 | 8.27 |
| shot_noise | 4 | +16.87 | 7.55 |
| shot_noise | 5 | +12.67 | 6.36 |

Descriptive exploratory comparisons; neither statistical significance nor a causal mechanism is established.
