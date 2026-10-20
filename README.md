
### Ablation Configurations

| Ablation Arm | Descriptor Routing | Progression Loss | Core Objective |
|:---|:---|:---|:---|
| **A1_Backbone** | None | None | Baseline CNN (no AbI-L) |
| **A2_AbI_v1** | Concatenation | Hard Hinge `f(x)=max(0,x)` | Tests raw descriptors with strict constraint |
| **A2_AbI_NoGate** | Concatenation | Smooth Softplus `f(x)=(1/β)log(1+e^βx)` | Tests smooth constraint without gating |
| **A2_AbI_Gate** | Gated Residual | Smooth Softplus | **Full proposed framework** |

*(Note: β = 8.0, margin = 0.10 for all cases)*

---

## 🔬 Core Contributions

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

## 📊 Results Summary

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


