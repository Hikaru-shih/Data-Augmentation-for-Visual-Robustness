# Corruption analysis

Seeds: 42, 43, 44. SD uses ddof=1, not standard error or a confidence interval.
Within each seed, corruptions/severities are equally weighted before across-seed statistics.
Deltas are paired by training seed. Category means are not averaged to calculate overall mCA.

## Category mapping

- noise: gaussian_noise, shot_noise, impulse_noise
- blur: defocus_blur, glass_blur, motion_blur, zoom_blur
- weather: snow, frost, fog, brightness
- digital: contrast, elastic_transform, pixelate, jpeg_compression

## CutMix: five lowest deltas

- shot_noise: -16.52 +/- 4.67 pp; seed range [-20.07, -11.23].
- gaussian_noise: -16.30 +/- 5.67 pp; seed range [-20.78, -9.93].
- jpeg_compression: -6.03 +/- 0.64 pp; seed range [-6.76, -5.57].
- pixelate: -5.58 +/- 1.34 pp; seed range [-6.79, -4.14].
- zoom_blur: -3.23 +/- 1.59 pp; seed range [-5.03, -2.01].

## AugMix Transform: five highest deltas

- gaussian_noise: +29.53 +/- 2.95 pp; seed range [+27.10, +32.82].
- impulse_noise: +27.03 +/- 3.38 pp; seed range [+24.24, +30.79].
- shot_noise: +24.01 +/- 2.63 pp; seed range [+21.96, +26.98].
- glass_blur: +14.81 +/- 2.28 pp; seed range [+13.16, +17.41].
- zoom_blur: +13.35 +/- 0.99 pp; seed range [+12.25, +14.17].

These are descriptive observations from three seeds; they do not establish significance or a causal mechanism.
