# CutMix probability: detailed analysis

Accuracy and paired delta SD are computed across seeds 42,43,44 (ddof=1).
Severity is averaged within each seed for corruption-level tables. All 15 corruptions are included.
The predeclared primary noise endpoint includes Gaussian and shot only; impulse is secondary.

## p=0.5 minus CutMix p=1

Positive mean: 12/15; positive in all seeds: 8/15; negative in all seeds: 2/15.

| Corruption | Mean delta (pp) | Sample SD | Seed min | Seed max |
|---|---:|---:|---:|---:|
| shot_noise | +5.63 | 2.41 | +2.87 | +7.31 |
| gaussian_noise | +4.64 | 2.17 | +2.14 | +5.94 |
| impulse_noise | +3.43 | 3.89 | -0.40 | +7.37 |
| jpeg_compression | +2.61 | 1.39 | +1.36 | +4.10 |
| pixelate | +1.95 | 0.40 | +1.49 | +2.25 |
| elastic_transform | +0.77 | 0.29 | +0.44 | +1.01 |
| contrast | +0.73 | 0.44 | +0.23 | +1.06 |
| motion_blur | +0.69 | 0.35 | +0.31 | +0.99 |
| glass_blur | +0.59 | 0.55 | +0.01 | +1.11 |
| frost | +0.44 | 1.06 | -0.78 | +1.12 |
| defocus_blur | +0.03 | 0.24 | -0.20 | +0.27 |
| fog | +0.03 | 0.66 | -0.38 | +0.79 |
| brightness | -0.40 | 0.39 | -0.77 | +0.01 |
| snow | -0.61 | 0.63 | -1.32 | -0.12 |
| zoom_blur | -0.69 | 0.42 | -1.18 | -0.45 |

## p=0.5 minus Baseline

Positive mean: 11/15; positive in all seeds: 11/15; negative in all seeds: 4/15.

| Corruption | Mean delta (pp) | Sample SD | Seed min | Seed max |
|---|---:|---:|---:|---:|
| impulse_noise | +3.71 | 1.20 | +2.39 | +4.74 |
| snow | +3.34 | 0.91 | +2.58 | +4.34 |
| contrast | +2.48 | 0.70 | +1.81 | +3.22 |
| defocus_blur | +1.51 | 0.14 | +1.36 | +1.63 |
| brightness | +1.44 | 0.62 | +0.80 | +2.03 |
| fog | +0.99 | 0.41 | +0.53 | +1.29 |
| glass_blur | +0.92 | 0.79 | +0.15 | +1.72 |
| motion_blur | +0.91 | 0.14 | +0.78 | +1.05 |
| elastic_transform | +0.84 | 0.19 | +0.68 | +1.06 |
| zoom_blur | +0.45 | 0.34 | +0.10 | +0.77 |
| frost | +0.28 | 0.27 | +0.01 | +0.56 |
| pixelate | -4.34 | 0.49 | -4.67 | -3.78 |
| jpeg_compression | -4.83 | 0.80 | -5.72 | -4.15 |
| shot_noise | -5.27 | 1.35 | -6.55 | -3.86 |
| gaussian_noise | -5.33 | 1.55 | -6.53 | -3.58 |

## Noise severity: p=0.5 minus p=1

| Corruption | Severity | Mean delta (pp) | SD |
|---|---:|---:|---:|
| gaussian_noise | 1 | +8.85 | 4.01 |
| gaussian_noise | 2 | +6.06 | 3.28 |
| gaussian_noise | 3 | +3.66 | 1.55 |
| gaussian_noise | 4 | +2.73 | 1.34 |
| gaussian_noise | 5 | +1.91 | 0.94 |
| impulse_noise | 1 | +2.99 | 3.41 |
| impulse_noise | 2 | +6.18 | 6.65 |
| impulse_noise | 3 | +4.95 | 5.74 |
| impulse_noise | 4 | +2.28 | 2.31 |
| impulse_noise | 5 | +0.73 | 1.40 |
| shot_noise | 1 | +7.98 | 3.41 |
| shot_noise | 2 | +8.59 | 3.82 |
| shot_noise | 3 | +5.32 | 2.29 |
| shot_noise | 4 | +3.91 | 1.63 |
| shot_noise | 5 | +2.35 | 1.15 |

Descriptive exploratory comparisons; neither statistical significance nor a causal mechanism is established.
