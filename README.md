# Data Augmentation for Visual Robustness

Current extension: [CutMix probability ablation and change record](notes/experiments/cutmix_probability.md). Run `bash scripts/run_cutmix_p05.sh` in WSL for the three new p=0.5 seeds; existing controls are reused. New training runs preserve source archives and hashes in addition to Git metadata. See the record for historical provenance limitations.

## Run the experiments (validation-v1)

Use Python 3.10 or 3.11. Install the project dependencies from the repository root:

```sh
python -m venv .venv
# Linux / WSL: source .venv/bin/activate
# PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m scripts.check_environment
python -m unittest discover -s tests -v
```

The project pins PyTorch 2.6.0 and torchvision 0.21.0. GPU support depends on the installed PyTorch build and compatible driver; check_environment reports the actual device. Every new run records Python, PyTorch, torchvision, CUDA runtime, GPU, Git revision and working-tree status in environment.json. The original environment freeze is preserved in notes/experiments/legacy_environment_freeze.txt as historical evidence, not an installation requirements file. Its PyYAML 5.4.1 pin is replaced by 6.0.2 in the maintained requirements.

CIFAR-10 downloads automatically under dataset.data_dir. Obtain CIFAR-10-C separately and extract its .npy files, including labels.npy, into data/CIFAR-10-C (or the corresponding dataset.data_dir). The evaluator requires 50,000 uint8 images per corruption and 10,000 or 50,000 labels. Dataset binaries and checkpoints are excluded from Git; archive them separately together with run directories and evaluation metadata. Source, configs, tests, notes and result tables belong in version control.

```sh
python -m scripts.train --config configs/baseline.yaml --seed 42 --run-id validation-v1
python -m scripts.evaluate_corruptions baseline_resnet18/seed_42/validation-v1
python -m scripts.plot_history baseline_resnet18/seed_42/validation-v1
python -m scripts.plot_robustness baseline_resnet18/seed_42/validation-v1

python -m scripts.train --config configs/augmix.yaml --seed 42 --run-id validation-v1
python -m scripts.evaluate_corruptions augmix_resnet18/seed_42/validation-v1
python -m scripts.compare_all_experiments --runs baseline_resnet18/seed_42/validation-v1 augmix_resnet18/seed_42/validation-v1 --output-name validation-v1-seed42
```

The train_baseline, train_mixup, train_cutmix, train_randaugment and train_augmix modules remain wrappers for the shared trainer and accept --seed and --run-id. Use a fresh run ID for every rerun; existing directories are rejected. These scripts start fresh training and do not implement resume.

All methods share a fixed 45,000/5,000 training/validation split controlled by validation.seed (2026), independently of the training seed. Validation uses only tensor conversion and normalization. Checkpoints are selected by validation accuracy, with the earliest epoch winning ties; the clean test set is evaluated once after training. Exact split indices and the effective configuration are saved per run. RandAugment and AugMix parameters are read from YAML. AugMix here means **AugMix Transform without the JSD consistency loss**, not full AugMix training.

Corruption evaluation records the checkpoint SHA-256 and CSV hashes. Comparison scripts reject missing provenance or changed files/checkpoints; reevaluate a checkpoint to create valid metadata. A run must finish clean evaluation before corruption evaluation. Do not train into or replace a checkpoint while evaluating it.

The Current Results table below now reports the completed validation-protocol runs across seeds 42, 43 and 44. Historical test-selected results remain in notes/experiments/phase1_summary.md and must not be pooled with these results. Replication across seeds 43 and 44 is complete with hyperparameters unchanged. The next stage analyzes corruption categories and severity. See [the multi-seed plan](notes/experiments/validation_summary.md). Run `bash scripts/run_multiseed.sh` in WSL to reproduce the batch workflow; all 15 planned runs are now complete, so no retraining is needed. AugMix seed 42 uses run ID validation-v2; this is a retry name, not a different evaluation protocol.

> **Status:** Ongoing Independent Research Project
> **Topic:** Computer Vision / Deep Learning / Model Robustness / Data Augmentation

## 1. Overview

Deep neural networks can achieve high accuracy on standard image classification benchmarks, but their performance may degrade significantly when input images are affected by corruptions such as noise, blur, changes in brightness, or compression artifacts.

Data augmentation is widely used to improve model generalization. However, different augmentation strategies may provide different levels of robustness against unseen image corruptions.

This project begins with a **reproducibility study** of existing data augmentation methods and investigates how different augmentation strategies affect the corruption robustness of image classification models.

The project will first reproduce and compare established methods. Based on the experimental observations and related literature, a more specific research question or extension will be developed in a later stage.

---

## 2. Research Questions

The initial study focuses on the following questions:

1. How do different data augmentation strategies affect clean classification accuracy?
2. How do they affect robustness against unseen image corruptions?
3. Do different augmentation strategies perform differently across corruption categories?
4. Is there a trade-off between clean accuracy and corruption robustness?
5. What weaknesses or patterns can be observed that may lead to a further research question?

The final research question may be refined after the reproduction and literature-review stages.

---

## 3. Project Scope

### Dataset

Initial experiments will use:

* **CIFAR-10** — training and clean evaluation
* **CIFAR-10-C** — evaluation under common image corruptions

Possible extensions:

* CIFAR-100 / CIFAR-100-C
* Tiny ImageNet
* ImageNet-100

Larger datasets will only be considered if computational resources and project time allow.

### Baseline Model

Initial model:

* ResNet-18

Possible later comparison:

* Vision Transformer (ViT)
* DeiT
* other lightweight vision architectures

The first stage intentionally uses a simple and established architecture so that the effect of data augmentation can be studied independently.

---

## 4. Data Augmentation Methods

The initial reproduction study will compare several established augmentation strategies.

### Baseline

Standard augmentation:

* Random Crop
* Random Horizontal Flip

### Methods to Study

* Mixup
* CutMix
* RandAugment
* AugMix

Additional methods may be added after reviewing the related literature.

---

## 5. Corruption Robustness Evaluation

Models will be evaluated on both clean images and corrupted images.

Example corruption categories include:

### Noise

* Gaussian Noise
* Shot Noise
* Impulse Noise

### Blur

* Gaussian Blur
* Motion Blur
* Defocus Blur

### Weather / Environmental Effects

* Fog
* Frost
* Snow

### Digital / Image Processing Effects

* JPEG Compression
* Pixelation
* Contrast Changes
* Brightness Changes

Different corruption severity levels will also be considered.

---

## 6. Experimental Pipeline

```text
Literature Review
       │
       ▼
Reproduce Baseline
       │
       ▼
Train ResNet-18 on CIFAR-10
       │
       ├── Standard Augmentation
       ├── Mixup
       ├── CutMix
       ├── RandAugment
       └── AugMix
       │
       ▼
Clean Evaluation
       │
       ▼
CIFAR-10-C Evaluation
       │
       ▼
Compare Robustness Across Corruptions
       │
       ▼
Analyze Failure Patterns
       │
       ▼
Identify Research Gap / Observation
       │
       ▼
Design Extension or New Hypothesis
       │
       ▼
Additional Experiments
       │
       ▼
Final Analysis & Research Report
```

---

## 7. Evaluation Metrics

The initial metrics will include:

* Clean Accuracy
* Corrupted Accuracy
* Mean Corruption Accuracy
* Accuracy Drop

Additional robustness metrics may be introduced after reviewing established evaluation protocols in the literature.

Experiments should be repeated with controlled random seeds when computationally feasible.

---

## 8. Research Stages

### Phase 1 — Literature Review

**Goal:** Understand the existing research landscape.

Tasks:

* Read foundational papers on data augmentation
* Study robustness under image corruptions
* Understand CIFAR-10-C evaluation
* Study Mixup, CutMix, RandAugment, and AugMix
* Record research questions, assumptions, datasets, and evaluation methods
* Identify what has already been studied

**Deliverable:**

`notes/` containing structured paper notes and a related-work summary.

---

### Phase 2 — Baseline Reproduction

**Goal:** Build a reliable experimental pipeline.

Tasks:

* Set up PyTorch environment
* Prepare CIFAR-10
* Implement ResNet-18 training
* Implement standard augmentation
* Establish reproducible training configuration
* Evaluate clean accuracy

**Deliverable:**

A reproducible baseline training and evaluation pipeline.

---

### Phase 3 — Augmentation Reproduction

**Goal:** Reproduce established augmentation methods.

Implement and evaluate:

* Mixup
* CutMix
* RandAugment
* AugMix

Keep model architecture and training settings as consistent as possible to enable meaningful comparisons.

**Deliverable:**

Experimental results for all selected augmentation strategies.

---

### Phase 4 — Corruption Robustness Evaluation

**Goal:** Analyze model behavior under distribution shifts.

Tasks:

* Prepare CIFAR-10-C
* Evaluate all trained models
* Compare corruption categories
* Compare severity levels
* Measure robustness degradation
* Generate figures and tables

**Deliverable:**

A systematic robustness comparison.

---

### Phase 5 — Analysis and Research Question Refinement

**Goal:** Move from reproduction toward independent research.

Questions to investigate may include:

* Which augmentation strategies fail on specific corruption types?
* Are improvements consistent across corruption categories?
* Does stronger augmentation always improve robustness?
* Is robustness gained at the cost of clean accuracy?
* Are there systematic patterns behind successful augmentation strategies?

At this stage, existing literature will be checked again before claiming that an observation or hypothesis is novel.

**Deliverable:**

A clearly defined research question for the extension stage.

---

### Phase 6 — Research Extension

**Goal:** Design and evaluate a small independent extension.

The exact method will **not be predetermined** before the reproduction results are analyzed.

Possible directions may involve:

* adaptive augmentation
* corruption-aware augmentation
* frequency-aware augmentation
* augmentation combinations
* feature regularization
* knowledge distillation
* robustness-aware training objectives

These are candidate directions only and are not currently claimed as novel contributions.

**Deliverable:**

A proposed method or hypothesis supported by controlled experiments.

---

### Phase 7 — Final Analysis and Documentation

Tasks:

* Repeat important experiments
* Generate final figures and tables
* Analyze limitations
* Document unsuccessful experiments
* Write final report
* Clean and document source code
* Update README with final findings

**Final Deliverables:**

* Reproducible source code
* Experiment configurations
* Experimental results
* Figures and tables
* Research report
* Final GitHub documentation

---

## 9. Tentative Schedule

### Week 1–2 — Literature Review

* Read foundational papers
* Understand CIFAR-10-C
* Study augmentation methods
* Create paper notes
* Finalize baseline experiment design

### Week 3 — Environment & Baseline

* Create project environment
* Prepare datasets
* Implement training pipeline
* Train ResNet-18 baseline
* Verify reproducibility

### Week 4–5 — Augmentation Methods

* Mixup
* CutMix
* RandAugment
* AugMix
* Run controlled training experiments

### Week 6 — Robustness Evaluation

* Evaluate CIFAR-10-C
* Organize corruption categories
* Compare severity levels
* Generate initial tables and figures

### Week 7 — Analysis

* Analyze robustness patterns
* Compare clean vs. corrupted performance
* Identify interesting failure cases
* Return to literature to verify whether observations are already known

### Week 8 — Research Question

* Define a focused extension
* Formulate hypothesis
* Design controlled experiments
* Establish evaluation criteria

### Week 9–10 — Extension Experiments

* Implement proposed extension
* Run experiments
* Compare against reproduced baselines
* Perform ablation studies when appropriate

### Week 11 — Verification

* Repeat important experiments
* Check random-seed sensitivity
* Analyze failure cases
* Verify results

### Week 12 — Final Report

* Write research report
* Create final figures
* Document methodology
* Discuss limitations and future work
* Clean GitHub repository

---

## 10. Tentative Repository Structure

```text
data-augmentation-robustness/
│
├── README.md
├── requirements.txt
│
├── configs/
│   ├── baseline.yaml
│   ├── mixup.yaml
│   ├── cutmix.yaml
│   ├── randaugment.yaml
│   └── augmix.yaml
│
├── src/
│   ├── datasets/
│   ├── models/
│   ├── augmentations/
│   ├── training/
│   └── evaluation/
│
├── scripts/
│   ├── train.py
│   └── evaluate.py
│
├── experiments/
│
├── results/
│   ├── tables/
│   └── figures/
│
├── notes/
│   └── papers/
│
└── report/
```

The repository structure may change as the project develops.

---

## 11. Reproducibility

To make comparisons reliable, experiments should document:

* Dataset version
* Train/test split
* Model architecture
* Random seed
* Optimizer
* Learning rate
* Learning-rate schedule
* Batch size
* Number of epochs
* Augmentation parameters
* Software/library versions

Experiment configurations should be stored separately from source code whenever possible.

---

## 12. Research Integrity

This project begins as a reproduction and analysis of established research.

All implemented methods derived from previous work will be properly cited. Reproducing an existing method will not be presented as an original contribution.

Any proposed extension will undergo a related-work review before being described as novel.

Negative results and limitations will also be documented.

---

## 13. Expected Outcome

By the end of the project, the goal is to:

1. Understand modern data augmentation methods.
2. Develop practical experience in visual robustness research.
3. Reproduce established results using a controlled experimental pipeline.
4. Analyze model behavior under different image corruptions.
5. Identify a focused research question from experimental observations.
6. Design and evaluate a small research extension.
7. Produce a reproducible research repository and technical report.

The primary objective is not necessarily to achieve state-of-the-art performance, but to conduct a **careful, reproducible, and well-documented research study**.

---

## 14. Current Status

**Stage:** Phase 5 — Analysis and Research Question Refinement

### Completed

* [x] Set up the PyTorch environment
* [x] Prepare CIFAR-10 and CIFAR-10-C
* [x] Implement and train the ResNet-18 baseline
* [x] Evaluate clean accuracy
* [x] Implement Mixup, CutMix, RandAugment, and AugMix
* [x] Evaluate all methods on 15 CIFAR-10-C corruptions and 5 severity levels
* [x] Generate comparison tables and figures

### Current Results (validation protocol, seeds 42, 43, 44; mean ± sample SD)

| Method | Clean Accuracy | Mean Corruption Accuracy |
|---|---:|---:|
| Baseline | 94.42 ± 0.39% | 73.65 ± 0.26% |
| Mixup | 95.27 ± 0.15% | 78.54 ± 0.61% |
| CutMix | 95.77 ± 0.05% | 71.14 ± 0.46% |
| RandAugment | 95.03 ± 0.09% | 81.33 ± 0.34% |
| AugMix Transform | 94.83 ± 0.27% | 85.45 ± 0.44% |

The current results suggest that AugMix Transform provides the strongest corruption robustness, while CutMix achieves the highest clean accuracy but does not improve mean corruption accuracy over the baseline.

### Next Steps

* [x] Verify the results across multiple random seeds (42, 43, 44)
* [x] Calculate mean and sample standard deviation for clean accuracy and mean corruption accuracy
* [ ] Analyze performance by corruption category and severity
* [ ] Review related literature against the observed patterns
* [ ] Refine a focused research question for the extension stage
