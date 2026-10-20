# AbI-L: Abnormality Intensity Representation Learning for Progression-Aware Heel Spur Diagnosis

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of **Abnormality Intensity Representation Learning (AbI-L)** — a backbone-adaptive framework that explicitly structures the latent space to encode disease progression for severity-based medical image analysis.

---

##  Abstract

Severity-based medical image analysis requires representations that encode disease progression, yet deep networks optimise class separation, yielding latent spaces that may poorly reflect abnormality severity. This paper presents **AbI-L**, a backbone-adaptive representation learning framework that structures the latent space through five abnormality intensity descriptors: **concentration (AIC)**, **entropy (AIE)**, **local variation (AIV)**, **contrast (AICo)**, and **dispersion (AID)**. These descriptors are fused via descriptor-guided gating and optimised using an **Intensity Progression Loss** that explicitly enforces the severity sequence (μ_Normal < μ_HeelSpur < μ_Sever) on the latent class means.

AbI-L is evaluated on a three-class heel spur diagnosis task across **GhostNet**, **MobileNetV3-Large**, and **EfficientNet-B0**. Although classification accuracy remains competitive, AbI-L consistently yields more discriminative, compact, and progression-ordered latent representations, improves Quadratic Weighted Kappa (QWK) on capable backbones, enlarges centroid distances between clinically distinct grades, and produces sharper Grad-CAM attention. Statistical validation using bootstrap confidence intervals, Kruskal-Wallis analysis, and Bonferroni-corrected post-hoc comparisons confirms the framework's robustness. AbI-L operates purely on the latent space, making it adaptable to any CNN backbone.

**Keywords:** Deep Learning, Representation Learning, Disease Progression, Severity Grading, Heel Spur Diagnosis, Latent Space Structuring, Progression-Aware Loss, Grad-CAM Explainability.

---

The AbI-L framework consists of four core components:

### Ablation Configurations

| Ablation Arm | Descriptor Routing | Progression Loss | Core Objective |
|:---|:---|:---|:---|
| **A1_Backbone** | None | None | Baseline CNN (no AbI-L) |
| **A2_AbI_v1** | Concatenation | Hard Hinge `f(x)=max(0,x)` | Tests raw descriptors with strict constraint |
| **A2_AbI_NoGate** | Concatenation | Smooth Softplus `f(x)=(1/β)log(1+e^βx)` | Tests smooth constraint without gating |
| **A2_AbI_Gate** | Gated Residual | Smooth Softplus | **Full proposed framework** |

*(Note: β = 8.0, margin = 0.10 for all cases)*

---

## Core Contributions

### 1. Five Abnormality Intensity Descriptors

| Descriptor | Symbol | Statistical Property | Representation Significance |
|:---|:---:|:---|:---|
| **Concentration** | AIC | Weighted centroid | Where semantic activations are concentrated |
| **Entropy** | AIE | Information uncertainty | Diversity of abnormality-related responses |
| **Local Variation** | AIV | Mean absolute difference | Local irregularities in feature responses |
| **Contrast** | AICo | Log ratio of dominant/weak | Separation between strong and weak activations |
| **Dispersion** | AID | Standard deviation | Spread of latent activations |

### 2. Intensity Progression Loss

A mathematically formalised loss that enforces the severity sequence using Hard Hinge and Smooth Softplus penalties. Applied directly to the latent descriptors, not the classifier logits.

### 3. Backbone-Adaptive Framework

Projects any CNN backbone into a fixed 512-D latent space. Dynamically adapts to the statistical properties of each architecture.

### 4. Comprehensive Validation

Statistical analysis (Kruskal-Wallis, Bonferroni-corrected Mann-Whitney, Wilcoxon, Friedman), Grad-CAM explainability, retrieval purity, and latent space clustering metrics.

---

## Results Summary

### Classification Performance (Best Configuration: EfficientNet-B0 + NoGate)

| Metric | Value |
|:---|:---|
| **Accuracy** | 92.66 ± 0.55% |
| **Balanced Accuracy** | 93.10 ± 0.41% |
| **Macro F1-Score** | 93.25 ± 0.41% |
| **MCC** | 0.8847 ± 0.0075 |
| **Macro AUC** | 98.37 ± 0.22% |
| **QWK** | **0.9270 ± 0.0039** |

### Latent Space Quality (EfficientNet-B0 + NoGate)

| Metric | Baseline | AbI-L NoGate | Improvement |
|:---|:---:|:---:|:---:|
| Silhouette ↑ | 0.4893 | **0.5634** | +15.1% |
| Davies-Bouldin ↓ | 0.7555 | **0.5616** | −25.7% |
| Calinski-Harabasz ↑ | 723.80 | **1479.41** | +104.4% |
| kNN Consistency ↑ | 0.9337 | 0.9340 | +0.3% |

### Statistical Validation

- **Kruskal-Wallis:** All 5 descriptors significant (p < 0.001) across all 5 seeds
- **Mann-Whitney (Bonferroni):** **15/15 pass rate** — perfect statistical consistency across all descriptors, all classes, all seeds
- **Wilcoxon:** Large effect sizes (Cohen's d > 2 on QWK for best configurations)

### Grad-CAM Explainability

| Backbone | Attention Mechanism | Key Finding |
|:---|:---|:---|
| **GhostNet** | Ordinal Complexity Increase | Entropy monotonically increases: 0.839 → 0.855 → 0.861 |
| **MobileNetV3** | Severity Amplification | Activation area expands 2.6× for Sever cases |
| **EfficientNet-B0** | Clinical Localization | Activation area contracts to 2.94% (from 13.71%) for Sever |

##  Repository Structure
AbI-L/
├── AbI_pub_v2_Enhanced.py # Main training and evaluation script
├── requirements.txt # Python dependencies
├── README.md # This file
├── LICENSE # MIT License
├── docs/
│ └── architecture.png # Framework diagram
├── data/ # Dataset (user-provided)
│ ├── train/
│ ├── val/
│ └── test/
└── outputs/ # Generated results (created at runtime)
├── histories/
├── metrics/
├── reports/
├── gradcam/
├── gradcam_comparison/
├── latent_space/
├── statistical_tests/
├── descriptor_analysis/
├── progression/
├── ablation/
├── tables/
└── visualization_subsets/

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/AbI-L.git
cd AbI-L
pip install -r requirements.txt
torch>=2.0.0
torchvision>=0.15.0
3. Prepare the Dataset
Organize the dataset as described in Dataset Preparation.

4. Run the Full Pipeline
bash
python AbI_pub_v2_Enhanced.py
This will:

Train all 12 model configurations (3 backbones × 4 ablations)

Evaluate on 5 independent seeds (42–46)

Generate all figures (confusion matrices, ROC, Grad-CAM, t-SNE/UMAP)

Run all statistical tests (Kruskal-Wallis, Mann-Whitney, Wilcoxon, Friedman)

Save publication-ready tables (A–H)

Dataset Preparation
The dataset should be organized as:

text
data/
├── train/
│   ├── normal/
│   │   ├── img_001.jpg
│   │   └── ...
│   ├── heelspur/
│   │   ├── img_001.jpg
│   │   └── ...
│   └── sever/
│       ├── img_001.jpg
│       └── ...
├── val/
│   ├── normal/
│   ├── heelspur/
│   └── sever/
└── test/
    ├── normal/
    ├── heelspur/
    └── sever/
Dataset Source: Kaggle Heel Dataset

Dataset Statistics:

Total: 3,956 images

Normal: 1,842 images

Heel Spur: 1,316 images

Sever: 798 images

Split: 70% train / 15% validation / 15% test

Update the paths in AbI_pub_v2_Enhanced.py:

python
OUT_SPLIT = Path("/path/to/your/dataset")
SAVE_DIR  = Path("/path/to/your/outputs")
🏋️ Training & Evaluation
Full Pipeline (All Configurations)
bash
python AbI_pub_v2_Enhanced.py
Run a Single Configuration
python
from AbI_pub_v2_Enhanced import *

# Train EfficientNet-B0 + AbI-L NoGate, Seed 44
ab_flags = ABLATION_VARIANTS["A2_AbI_NoGate"]
result, history, preds, targets, probs, model = run_one_seed(
    seed=44,
    model_tag="EfficientNet_B0_A2_AbI_NoGate",
    backbone_name="EfficientNet_B0",
    ab_name="A2_AbI_NoGate",
    ab_flags=ab_flags,
    num_classes=3,
    class_names=["heelspur", "normal", "sever"],
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
)
Key Hyperparameters
python
# Training
IMG_SIZE      = 224
LR            = 3e-4
EPOCHS        = 60
PATIENCE      = 12
BATCH         = 16

# AbI-L
LATENT_DIM    = 512
PROG_MARGIN   = 0.10
PROG_BETA     = 8.0
GATE_HIDDEN   = 16

# Descriptor weights (annealing)
LAMBDA_V1     = dict(AIC=0.40, AIE=0.40, AIV=0.20, AID=0.20)
LAMBDA_FINAL  = dict(AIC=0.15, AIE=0.15, AIV=0.08, AID=0.08)

# Seeds
SEEDS = [42, 43, 44, 45, 46]
⚙️ Configuration
Key configuration options in AbI_pub_v2_Enhanced.py:

Parameter	Default	Description
IMG_SIZE	224	Input image resolution
LR	3e-4	Initial learning rate (AdamW)
EPOCHS	60	Maximum training epochs
PATIENCE	12	Early stopping patience
BATCH	16	Batch size
LATENT_DIM	512	Latent embedding dimension
PROG_MARGIN	0.10	Progression loss margin (m)
PROG_BETA	8.0	Smooth Softplus β
GATE_HIDDEN	16	Gated fusion hidden dimension
FOCAL_GAMMA	2.0	Focal loss γ
MIXUP_ALPHA	0.3	Mixup α
Output Structure
text
AbI_Results/
├── histories/              # Training curves (per model, per seed)
├── metrics/                # Confusion matrices, ROC, reliability diagrams
├── reports/                # Summary CSVs
├── gradcam/                # Individual Grad-CAM panels
├── gradcam_comparison/     # 4-arm comparison panels
├── latent_space/           # t-SNE and UMAP plots
├── statistical_tests/      # Kruskal-Wallis, Dunn, Nemenyi
├── descriptor_analysis/    # Per-class descriptor means
├── progression/            # Centroid distance metrics
├── ablation/               # Ablation summary plots
├── tables/                 # Publication Tables A–H
└── visualization_subsets/  # Fixed Grad-CAM evaluation subsets
Publication Tables (A–H)
Table	Content
A	Classification metrics (accuracy, balanced accuracy, F1, MCC, AUC, QWK)
B	Calibration metrics (ECE, Brier score)
C	Latent space metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz, kNN)
D	Progression metrics (centroid distances)
E	Kruskal-Wallis statistics (per descriptor, per model)
F	Grad-CAM metrics (activation area, entropy, contour complexity)
G	Retrieval purity metrics (Top-1, Top-5)
H	Bootstrap confidence intervals (95% CI for accuracy)
Statistical Validation
The pipeline runs the following statistical tests:

Test	Purpose	Output
Kruskal-Wallis	Omnibus test for descriptor separability	statistical_tests/kw_abi_validation.csv
Mann-Whitney (Bonferroni)	Pairwise post-hoc comparison	statistical_tests/*_dunn.csv
Wilcoxon	Paired comparison across seeds	reports/wilcoxon_comparisons.csv
Friedman + Nemenyi	Multi-arm comparison across ablations	statistical_tests/friedman_test.csv, statistical_tests/nemenyi_posthoc.csv
Bootstrap CI	95% confidence interval for accuracy	reports/bootstrap_results.csv
🔄 Reproducibility
All experiments are reproducible with:

Deterministic seed: torch.manual_seed(seed), np.random.seed(seed), random.seed(seed)

5 independent seeds: 42, 43, 44, 45, 46

Fixed train/val/test splits: 70/15/15

ImageNet normalization: mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225)

To reproduce the best configuration (EfficientNet-B0 + NoGate):

 Hardware Requirements
Training: Single NVIDIA GPU (16 GB+ VRAM recommended)

Tested on: NVIDIA DGX H200 (143 GB HBM)

Training time: ~8–12 hours for all 60 runs (12 configs × 5 seeds)

Inference time: ~15 ms per image on a single GPU

Citation
If you use this code or find it helpful in your research, please cite:

bibtex
@article{yourname2026abil,
  title   = {Abnormality Intensity Representation Learning for Progression-Aware Heel Spur Diagnosis},
  author  = {Your Name and Co-Authors},
  journal = {Journal Name},
  year    = {2026},
  volume  = {XX},
  pages   = {XXX--XXX},
  doi     = {10.XXXX/XXXXXX}
}
timm>=0.9.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.11.0
scikit-posthocs>=0.8.0
matplotlib>=3.7.0
seaborn>=0.12.0
umap-learn>=0.5.0
thop>=0.1.0
Pillow>=9.5.0
