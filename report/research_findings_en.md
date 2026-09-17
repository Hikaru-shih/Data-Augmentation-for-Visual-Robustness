# Effects of Data Augmentation on Image Corruption Robustness: A Controlled Comparison and Analysis of CutMix Application Probability

Research Report

| Item | Information |
|---|---|
| Author | Pin-Kuang Shih |
| Department / Field of Study | Applied Mathematics & Computer Science |
| Course | Introduction to Pattern Recognition |
| Date | June 17, 2026 |
| Document version | Structured report draft — English edition |

## Abstract

This study compares five data augmentation strategies in terms of clean classification accuracy and image corruption robustness, and investigates sensitivity to the probability of applying CutMix. Using a CIFAR-adapted ResNet-18, a fixed validation split, and three training seeds, the CIFAR-10 comparison shows that AugMix Transform achieves the highest mean corruption accuracy, whereas CutMix achieves the highest clean accuracy. Reducing the batch-level CutMix application probability from 1.0 to 0.5 improves average accuracy under Gaussian and shot noise by 13.21 percentage points and overall mean corruption accuracy by 2.41 percentage points. Nevertheless, performance on the combined Gaussian/shot noise endpoint remains below the baseline. Further analysis reveals that the effects vary across corruption types and do not increase monotonically with severity. These findings support evaluating clean performance, individual corruption types, and augmentation application probability jointly, but do not establish a causal explanation involving frequency characteristics or other mechanisms.

On CIFAR-100, the same reduction in application probability improves the primary noise endpoint by a paired mean of 5.14 ± 2.29 percentage points, with positive differences for all three seeds, while mean clean accuracy decreases by 0.69 percentage points. The two datasets provide limited evidence of consistency across tasks, rather than evidence of improvement under every corruption or generalization to arbitrary real-world conditions.

**Keywords:** data augmentation; image corruption robustness; CutMix; CIFAR-10; CIFAR-100.

## Contents

1. Introduction
2. Related Work
3. Methodology
4. Results and Analysis
5. Discussion and Limitations
6. Conclusions and Future Work
7. References

Appendix A: Data Sources; Appendix B: Reproducibility and Version Records.

## 1. Introduction

### 1.1 Background and Motivation

Image classification models need to recognize unmodified test images and maintain performance when images are affected by noise, blur, or compression artifacts. In this study, classification accuracy on unmodified test images is termed *clean accuracy*, while performance under these distortions is referred to as *image corruption robustness*. Because these measures concern different input conditions, both are evaluated to determine whether improvements from data augmentation are consistent across conditions.

### 1.2 Research Questions and Contributions

This study addresses three questions: Does the ranking of augmentation methods by clean accuracy predict their ranking by corruption robustness? Are CutMix performance losses under particular corruptions sensitive to its batch-level application probability? Does this sensitivity also occur in another CIFAR classification task?

The study provides a five-method comparison under a shared architecture and training budget, a CutMix probability ablation across three seeds, and a CIFAR-100 validation with a direction-of-effect criterion specified before evaluation. Its contribution lies in controlled experimentation, reporting both positive and negative results, and traceable analysis. Method combinations, previously reported phenomena, and the setting p=0.5 are not presented as new algorithms.

### 1.3 Scope

The study includes 27 completed training runs: 18 on CIFAR-10 and nine on CIFAR-100. Archived interrupted runs are excluded. The work is positioned as a controlled empirical comparison and an analysis of sensitivity to experimental settings.

Table 1. Experimental stages and completed training runs.

| Stage | Dataset | Comparison | Completed runs |
|---|---|---|---:|
| Method comparison | CIFAR-10 | Five methods × three seeds | 15 |
| Application-probability ablation | CIFAR-10 | Three additional CutMix p=0.5 runs; existing controls reused | 3 |
| Cross-task validation | CIFAR-100 | Baseline, CutMix p=1, and CutMix p=0.5 × three seeds; all trained from scratch | 9 |

## 2. Related Work

Mixup trains on linear interpolations of examples and their labels. CutMix combines image regions and adjusts label weights according to their areas. RandAugment controls augmentation operations through a reduced parameter search space. See [Mixup](https://arxiv.org/abs/1710.09412), [CutMix](https://arxiv.org/abs/1905.04899), and [RandAugment](https://arxiv.org/abs/1909.13719).

AugMix combines mixtures of augmented images with a consistency objective. Its original evaluation compares clean performance and corruption robustness, including CIFAR-10-C results in which CutMix underperforms Standard training. Accordingly, the present study does not claim to discover this broad pattern for the first time. The implementation evaluated here uses only the AugMix transform and differs from the authors' default operation set. See the [AugMix paper](https://arxiv.org/pdf/1912.02781) and [torchvision 0.21 documentation](https://docs.pytorch.org/vision/0.21/generated/torchvision.transforms.AugMix.html).

The CutMix authors' CIFAR-100 example uses an application probability of 0.5, but its architecture and training schedule differ from those used here. That configuration provides a reference for the ablation, not evidence that p=0.5 is optimal in the present setting. The IPMix supplementary material also examines augmentation combinations; any future study of combinations must therefore distinguish its contribution from existing work. See the [official CutMix implementation](https://github.com/clovaai/CutMix-PyTorch) and [IPMix supplementary material, Section F](https://proceedings.neurips.cc/paper_files/paper/2023/file/c917d8b9e01427f3184d80ade22f4d1f-Supplemental-Conference.pdf).

## 3. Methodology

### 3.1 Data Splits and Model Training

For each dataset, the 50,000 training images are divided into 45,000 training images and 5,000 validation images using split seed 2026. CIFAR-10 uses a fixed random split. CIFAR-100 uses a stratified split, reserving 50 images from each fine class for validation. The training seeds are 42, 43, and 44.

The model is a CIFAR-adapted ResNet-18, with a 3×3 initial convolution at stride 1, no initial max-pooling layer, and an output layer for 10 or 100 classes. Each run is trained for 100 epochs with a batch size of 128, using SGD with an initial learning rate of 0.1, momentum of 0.9, weight decay of 0.0005, and a cosine learning-rate schedule. The checkpoint is selected by validation accuracy, with ties resolved in favor of the earlier epoch, and is then evaluated on the clean test set. This protocol compares methods under a common budget; it does not imply that each method has individually optimized hyperparameters.

### 3.2 Image Preprocessing and Normalization

CIFAR-10 uses fixed channel means of (0.4914, 0.4822, 0.4465) and standard deviations of (0.2470, 0.2435, 0.2616). For CIFAR-100, pixel-weighted channel means and population standard deviations are computed exclusively from the 45,000 unaugmented training images after splitting. These values are then fixed for validation, clean testing, and corruption evaluation, and are saved in the configurations and checkpoints. This preprocessing difference limits explanations of why effect magnitudes differ between the two tasks.

### 3.3 Augmentation Settings

The baseline uses random cropping and horizontal flipping. Mixup and CutMix both use alpha=1.0; the initial CutMix comparison applies CutMix to every batch. RandAugment uses num_ops=2 and magnitude=9. AugMix uses the torchvision 0.21 transform with severity=3, mixture_width=3, chain_depth=-1, alpha=1.0, and the default all_ops=True, without the JSD consistency loss. It is therefore termed *AugMix Transform* throughout this report, rather than a full reproduction of the original AugMix training procedure. Differences in operation sets are documented in the [literature audit](../notes/experiments/literature_audit.md).

### 3.4 Evaluation Metrics and Statistical Analysis

Corruption robustness is evaluated on CIFAR-10-C and CIFAR-100-C, respectively. Mean corruption accuracy (mCA) is the equally weighted mean accuracy across 15 corruption types and five severity levels. Conditions are first averaged within each training seed, after which the mean and sample standard deviation across the three seeds are calculated (ddof=1). This metric is accuracy, not reference-model-normalized mean corruption error (mCE).

For paired comparisons, the control result is subtracted from the p=0.5 result within each seed before aggregation across seeds. The 75 corruption–severity conditions are not treated as 75 independent training runs, and the two datasets are not pooled into a six-seed sample.

Throughout this report, ± denotes the sample standard deviation across three training seeds, not a confidence interval; pp denotes percentage points. Statistical values are calculated before rounding and displayed to two decimal places in the result tables.

The 15 corruptions are Gaussian, shot, and impulse noise; defocus, glass, motion, and zoom blur; snow, frost, fog, and brightness; contrast, elastic transform, pixelation, and JPEG compression. Overall mCA assigns equal weight to each corruption, rather than equal weight to four groups containing different numbers of corruptions. The evaluation uses the publicly distributed benchmark arrays. See the [benchmark authors' repository](https://github.com/hendrycks/robustness).

## 4. Results and Analysis

### 4.1 Five-Method Comparison on CIFAR-10

Table 2. CIFAR-10 performance across five methods (mean ± sample standard deviation, n=3).

| Method | Clean accuracy (%) | mCA (%) |
|---|---:|---:|
| Baseline | 94.42 ± 0.39 | 73.65 ± 0.26 |
| Mixup | 95.27 ± 0.15 | 78.54 ± 0.61 |
| CutMix p=1 | 95.77 ± 0.05 | 71.14 ± 0.46 |
| RandAugment | 95.03 ± 0.09 | 81.33 ± 0.34 |
| AugMix Transform | 94.83 ± 0.27 | 85.45 ± 0.44 |

The mCA ranking is identical for all three seeds: AugMix Transform, RandAugment, Mixup, Baseline, and CutMix. CutMix achieves the highest clean accuracy for every seed, showing that clean-accuracy rankings do not directly predict corruption-robustness rankings in this setting. AugMix Transform exceeds the baseline by 11.80 percentage points in mean mCA.

This broad pattern has precedent: Table 1 of the AugMix paper also reports higher CIFAR-10-C corruption error for CutMix than for Standard training. The present comparison is therefore positioned as replication under different experimental settings and more detailed analysis, without a novelty claim for the broad observation. See the [AugMix paper](https://arxiv.org/pdf/1912.02781).

### 4.2 CutMix Application-Probability Ablation

The initial comparison motivated the hypothesis that reducing the frequency of CutMix could mitigate the performance losses under Gaussian and shot noise. Three p=0.5 runs were added with the remaining experimental settings held fixed, reusing the baseline and p=1 controls. Before this ablation, the primary endpoint was specified as the average over the ten conditions formed by Gaussian and shot noise at five severity levels each. It excludes impulse noise. The exploratory origin of this endpoint is discussed in Section 5.

Table 3. CutMix application-probability ablation on CIFAR-10 (mean ± sample standard deviation, n=3).

| Method | Clean accuracy (%) | mCA (%) | Gaussian/shot noise (%) |
|---|---:|---:|---:|
| Baseline | 94.42 ± 0.39 | 73.65 ± 0.26 | 51.96 ± 2.69 |
| CutMix p=1 | 95.77 ± 0.05 | 71.14 ± 0.46 | 35.55 ± 2.50 |
| CutMix p=0.5 | 95.56 ± 0.30 | 73.56 ± 0.67 | 48.76 ± 3.56 |

Pairing results by seed, p=0.5 improves the primary noise endpoint over p=1 by +13.21 ± 5.60 pp and mCA by +2.41 ± 1.03 pp. Both differences are positive for all three seeds. The clean-accuracy difference is -0.21 ± 0.31 pp, with decreases for two seeds and an increase for one. Relative to the baseline, the primary noise endpoint remains lower by -3.20 ± 2.05 pp, with negative differences for all three seeds. Although the mean mCA difference from the baseline is only -0.10 pp, this does not establish statistical equivalence.

### 4.3 Corruption-Type and Severity Analysis

Reducing the CutMix application probability improves mean accuracy for nine of the 15 corruptions relative to p=1. Gaussian noise, shot noise, impulse noise, JPEG compression, and pixelation improve for all three seeds. The mean gains for Gaussian, shot, and impulse noise are 13.47, 12.96, and 5.00 pp, respectively. Conversely, snow and brightness decrease by means of 0.97 and 0.15 pp, with negative differences for every seed.

Relative to the baseline, Gaussian and shot noise remain worse for every seed after averaging across severity levels. Impulse noise, however, improves by a mean of 3.23 pp, with positive differences for all three seeds. The primary endpoint therefore cannot support a conclusion about all noise types. The mean improvement over p=1 peaks at severity 2 for Gaussian noise (18.12 pp) and severity 3 for shot noise (17.95 pp), providing no support for a monotonic increase in improvement with severity.

![Accuracy across severity levels for three noise types](../results/analysis/cutmix-probability/noise_severity.png)

Figure 1. Each point represents mean accuracy across three seeds. Error bars indicate sample standard deviations, not confidence intervals.

![Paired differences for individual corruption types](../results/analysis/cutmix-probability/corruption_deltas.png)

Figure 2. Differences are computed within each seed and then summarized by their mean and sample standard deviation. Positive values indicate higher accuracy for p=0.5 than for the corresponding control.

### 4.4 CIFAR-100 and Cross-Dataset Validation

The cross-dataset stage compares the baseline, p=1, and p=0.5. CIFAR-10 checkpoints are not used as CIFAR-100 controls. The primary endpoint remains the mean of the ten Gaussian/shot noise conditions. At the design stage, consistent improvement was defined as a positive paired difference for each of the three seeds; this criterion is descriptive and is not interpreted as a formal significance test.

Table 4. Performance of the three CIFAR-100 settings (mean ± sample standard deviation, n=3).

| Method | CIFAR-100 clean (%) | CIFAR-100 mCA (%) | Gaussian/shot noise (%) |
|---|---:|---:|---:|
| Baseline | 76.31 ± 0.41 | 47.78 ± 0.23 | 26.29 ± 0.59 |
| CutMix p=1 | 79.04 ± 0.09 | 46.26 ± 0.60 | 15.85 ± 1.47 |
| CutMix p=0.5 | 78.35 ± 0.10 | 47.58 ± 0.20 | 20.98 ± 1.00 |

Table 5. Within-dataset paired differences for p=0.5 relative to p=1 (mean ± sample standard deviation, n=3).

| Dataset | p=0.5−p=1 clean (pp) | mCA (pp) | Primary noise endpoint (pp) |
|---|---:|---:|---:|
| CIFAR-10 | -0.21 ± 0.31 | +2.41 ± 1.03 | +13.21 ± 5.60 |
| CIFAR-100 | -0.69 ± 0.09 | +1.32 ± 0.70 | +5.14 ± 2.29 |

The primary noise endpoint and mCA improve for all three seeds in both datasets. On CIFAR-100, clean accuracy decreases for every seed. The primary noise endpoint for p=0.5 remains below the baseline by -5.30 ± 1.44 pp, with negative differences for all three seeds. Thus, the cross-task result reproduces the direction of improvement relative to p=1, rather than recovery to baseline performance or improvement on every measure.

![Within-dataset paired effects across the two datasets](../results/analysis/cross_dataset/paired_effects.png)

Figure 3. Mean paired differences and sample standard deviations are calculated separately within each dataset. The comparison concerns the effect of changing the setting, rather than absolute accuracy across tasks. Different metrics use different vertical-axis scales.

![Paired effects by severity across the two datasets](../results/analysis/cross_dataset/noise_effects.png)

Figure 4. Per-severity differences for p=0.5 minus p=1, with sample standard deviations. On CIFAR-100, the mean Gaussian-noise improvement decreases from 8.85 pp at severity 1 to 1.91 pp at severity 5, while the shot-noise improvement peaks at severity 2 (8.59 pp). These patterns differ from the CIFAR-10 peaks at Gaussian-noise severity 2 and shot-noise severity 3, and do not support a shared monotonic relationship between severity and improvement.

Gaussian noise, shot noise, JPEG compression, and pixelation improve over p=1 for every seed in both datasets; snow decreases for every seed in both datasets. Impulse noise improves on average, but one CIFAR-100 seed has a negative difference after averaging over severity levels. These findings support reporting individual corruptions so that aggregate mCA does not obscure heterogeneous effects.

## 5. Discussion and Limitations

The results support a restricted conclusion: under the architecture, training budget, and datasets studied, CutMix corruption performance is sensitive to application probability. Reducing that probability mitigates some performance losses, but the effect depends on the corruption type. The results neither establish p=0.5 as optimal nor demonstrate reliance on high-frequency information, local textures, or boundaries.

The experiments cover one architecture, two CIFAR datasets, and three seeds per condition. Changing application probability also changes the random sequence used for augmentation sampling. Matching seeds provides a paired design; it does not imply that different methods encounter identical synthesized examples. Previously observed CIFAR-10-C results informed the ablation question, so the subsequent ablation is an exploratory extension rather than a confirmatory experiment on an untouched test set. No significance tests with multiple-comparison correction were performed.

The earlier controls retain configurations, environment records, split information, and evaluation hashes, but do not include snapshots of every uncommitted source file used during training. The added p=0.5 experiments preserve source.zip archives and per-file SHA-256 hashes. The implementation release is ac1ed9b, but it should not be treated as the original training commit for the earlier controls. Preliminary results obtained using test-based checkpoint selection are excluded from this report.

## 6. Conclusions and Future Work

Under the model and training budget examined here, improvements in clean accuracy do not guarantee improvements in corruption robustness. Lowering the CutMix application probability mitigates some corruption-related performance losses on both CIFAR tasks, but is accompanied by lower mean clean accuracy and does not restore the primary noise endpoint to baseline performance. The study provides a bounded sequence of empirical evidence: a method comparison, a configuration ablation, and validation of the direction of the effect across tasks.

Future work could specify a second architecture, more distinct data sources, and evaluation rules in advance to assess the scope of these findings. Mechanistic explanations would require direct measurements and intervention-based experimental designs. Additional accuracy tables alone cannot establish reliance on high-frequency information or causal mechanisms. The use of three seeds, datasets from the same CIFAR family, and different splitting and normalization procedures continues to limit generalization.

## 7. References

References below identify the works cited in the text and link to their original sources. Software documentation specifies the version used.

1. Zhang, H., Cisse, M., Dauphin, Y. N., and Lopez-Paz, D. (2017). [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412). arXiv:1710.09412.
2. Yun, S., Han, D., Oh, S. J., Chun, S., Choe, J., and Yoo, Y. (2019). [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://arxiv.org/abs/1905.04899). arXiv:1905.04899.
3. Cubuk, E. D., Zoph, B., Shlens, J., and Le, Q. V. (2019). [RandAugment: Practical automated data augmentation with a reduced search space](https://arxiv.org/abs/1909.13719). arXiv:1909.13719.
4. Hendrycks, D., Mu, N., Cubuk, E. D., Zoph, B., Gilmer, J., and Lakshminarayanan, B. (2019). [AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty](https://arxiv.org/abs/1912.02781). arXiv:1912.02781.
5. PyTorch contributors. [torchvision.transforms.AugMix](https://docs.pytorch.org/vision/0.21/generated/torchvision.transforms.AugMix.html). torchvision 0.21 documentation.
6. Clova AI. [CutMix-PyTorch](https://github.com/clovaai/CutMix-PyTorch). Official implementation repository.
7. [IPMix supplementary material](https://proceedings.neurips.cc/paper_files/paper/2023/file/c917d8b9e01427f3184d80ade22f4d1f-Supplemental-Conference.pdf). (2023). NeurIPS, Section F.
8. Hendrycks et al. [robustness](https://github.com/hendrycks/robustness). Official benchmark data and implementation repository.

## Appendix A: Data Sources

- [Five-method summary statistics](../results/comparisons/multiseed-42-43-44/mean_std.csv)
- [Paired ablation statistics](../results/analysis/cutmix-probability/paired_mean_std.csv)
- [Per-corruption differences](../results/analysis/cutmix-probability/corruption_delta.csv)
- [Per-severity differences](../results/analysis/cutmix-probability/corruption_severity_delta.csv)
- [Implementation and experiment records](../notes/experiments/cutmix_probability.md)
- [Cross-dataset paired statistics](../results/analysis/cross_dataset/paired_effects.csv)
- [Per-seed CIFAR-100 results](../results/analysis/cifar100_cutmix-probability/per_seed.csv)
- [Detailed cross-dataset findings](../notes/experiments/cross_dataset_results.md)

## Appendix B: Reproducibility and Version Records

From the project root, run `python3 -m scripts.compare_transfer` in the existing Python environment to verify and rebuild the analyses for both datasets and generate the combined figures. This command performs neither model training nor inference. File-integrity and numerical checks for the 27 completed runs are documented in the [report verification record](verification.md). The actual training version must be identified from each run's environment, configuration, and source archive; subsequent report versions must not be represented as training versions.
