# -*- coding: windows-1252 -*-
# -*- coding: utf-8 -*-
"""
AbI_pub_v2_Enhanced.py -- Complete Publication-Ready Code with GradCAM
========================================================================
Abnormality Intensity Representation Learning (Ab-I) for Ordinal
Heel Spur Severity Classification

FIX APPLIED (this version):
  - Removed `import pandas as pd` inside run_dunns_posthoc_abi() which
    shadowed the module-level `pd` and caused:
      UnboundLocalError: cannot access local variable 'pd'
      where it is not associated with a value
    pandas is already imported at the top level; no inner import needed.

Author : BSAM Research -- Ab-I Enhanced Edition
Date   : 2026
"""

# =============================================================================
# STANDARD LIBRARY
# =============================================================================
import os, gc, random, warnings, subprocess, sys, time
from pathlib import Path
from copy import deepcopy
from collections import defaultdict
from itertools import combinations
import pickle

# =============================================================================
# NUMERICAL / DATA
# =============================================================================
import numpy as np
import pandas as pd

# =============================================================================
# DEEP LEARNING
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.v2 as v2

# =============================================================================
# VISUALISATION
# =============================================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.ndimage import label as nd_label

# =============================================================================
# METRICS
# =============================================================================
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    balanced_accuracy_score, cohen_kappa_score,
    roc_auc_score, matthews_corrcoef, brier_score_loss,
    precision_score, recall_score,
    silhouette_score, davies_bouldin_score,
    calinski_harabasz_score, roc_curve, auc as sk_auc,
)
from sklearn.calibration import calibration_curve
from sklearn.utils.class_weight import compute_class_weight
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# =============================================================================
# STATISTICS
# =============================================================================
from scipy import stats
from scipy.stats import (
    wilcoxon, friedmanchisquare, kruskal, rankdata,
)

# =============================================================================
# OPTIONAL DEPENDENCIES
# =============================================================================
try:
    import timm
    HAS_TIMM = True
    print(f"  timm {timm.__version__}: OK")
except ImportError:
    HAS_TIMM = False
    print("  [WARN] timm not found -- pip install timm")

try:
    import umap as umap_lib
    HAS_UMAP = True
    print("  umap-learn: OK")
except ImportError:
    HAS_UMAP = False
    print("  umap-learn: not found (t-SNE only)")

try:
    from scikit_posthocs import posthoc_dunn
    HAS_DUNN = True
    print("  scikit-posthocs: OK")
except ImportError:
    HAS_DUNN = False
    print("  [WARN] scikit-posthocs not found -- pip install scikit-posthocs")

try:
    from thop import profile as thop_profile
    HAS_THOP = True
    print("  thop: OK")
except ImportError:
    HAS_THOP = False
    print("  thop: not found (FLOPs = NaN)")

warnings.filterwarnings("ignore")


# =============================================================================
# PUBLICATION FIGURE STYLE
# =============================================================================
PUB_STYLE = {
    "font.size":             16,
    "axes.titlesize":        18,
    "axes.labelsize":        16,
    "xtick.labelsize":       14,
    "ytick.labelsize":       14,
    "legend.fontsize":       13,
    "legend.title_fontsize": 14,
    "figure.titlesize":      20,
    "font.family":           "DejaVu Sans",
    "font.weight":           "normal",
    "axes.titleweight":      "bold",
    "axes.linewidth":        1.4,
    "xtick.major.width":     1.2,
    "ytick.major.width":     1.2,
    "xtick.major.size":      5,
    "ytick.major.size":      5,
    "lines.linewidth":       2.0,
    "patch.linewidth":       1.0,
    "axes.grid":             True,
    "grid.alpha":            0.3,
    "grid.linewidth":        0.8,
    "figure.dpi":            150,
    "savefig.dpi":           600,
    "savefig.bbox":          "tight",
    "savefig.pad_inches":    0.15,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
}
matplotlib.rcParams.update(PUB_STYLE)

# Color scheme for ablation arms
ABLATION_COLORS = {
    "A1_Backbone":   "#4C6B8A",
    "A2_AbI_v1":     "#E67E22",
    "A2_AbI_NoGate": "#27AE60",
    "A2_AbI_Gate":   "#C0392B",
}
ABLATION_MARKERS = {
    "A1_Backbone":   "o",
    "A2_AbI_v1":     "s",
    "A2_AbI_NoGate": "^",
    "A2_AbI_Gate":   "D",
}


# =============================================================================
# DEVICE
# =============================================================================
def get_best_gpu():
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free',
             '--format=csv,nounits,noheader'],
            capture_output=True, text=True)
        free = [int(x) for x in r.stdout.strip().split('\n')]
        best = free.index(max(free))
        print(f"  GPU {best} selected ({max(free)} MiB free)")
        return str(best)
    except Exception:
        return "0"

os.environ["CUDA_VISIBLE_DEVICES"] = get_best_gpu()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU   : {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()


# =============================================================================
# PATHS  (adjust to your environment)
# =============================================================================
OUT_SPLIT = Path("/nfsshare/users/raghavan/HeelNovel/splitgradcam")
SAVE_DIR  = Path("/nfsshare/users/raghavan/HeelSOCPH/AbI_Results_V2_Enhanced_experiment5")

TRAIN_PATH = OUT_SPLIT / "train"
VAL_PATH   = OUT_SPLIT / "val"
TEST_PATH  = OUT_SPLIT / "test"

# Create all necessary subdirectories
for sub in [
    'histories', 'metrics', 'reports',
    'gradcam', 'gradcam_comparison',
    'latent_space', 'latent_comparison',
    'statistical_tests',
    'descriptor_analysis', 'progression',
    'progression_visualizations',
    'ablation', 'tables',
    'visualization_subsets',
    'claim_validation',
    'prototype_analysis',
    'retrieval',
    'compute_profile',
    'bootstrap',
]:
    (SAVE_DIR / sub).mkdir(parents=True, exist_ok=True)


# =============================================================================
# HYPERPARAMETERS
# =============================================================================
IMG_SIZE      = 224
LR            = 3e-4
EPOCHS        = 60
PATIENCE      = 12
BATCH         = 16

LATENT_DIM    = 512

# Ab-I descriptor hyperparameters
ABI_HIGH_FRAC = 0.25
ABI_LOW_FRAC  = 0.25

# Progression-loss hyperparameters
PROG_MARGIN        = 0.10
PROG_BETA          = 8.0
PROG_WARMUP_EPOCHS = 15

LAMBDA_V1 = dict(AIC=0.40, AIE=0.40, AIV=0.20, AID=0.20)
LAMBDA_FINAL = dict(AIC=0.15, AIE=0.15, AIV=0.08, AID=0.08)

MIXUP_DECOUPLE_PROB = 0.5
GATE_HIDDEN     = 16
GATE_INIT_BIAS  = -2.0

FOCAL_GAMMA   = 2.0
LABEL_SMOOTH  = 0.05
MIXUP_ALPHA   = 0.3
ALPHA         = 0.05

SEEDS        = [42, 43, 44, 45, 46]
N_BOOT       = 2000
VIZ_SUBSET_N = 60
MAX_GRADCAM_PER_CLASS = 20  # Keep for quantitative, but we'll save all

DESCRIPTOR_NAMES = ['AIC', 'AIE', 'AIV', 'AICo', 'AID']
PROGRESSION_DESCRIPTORS = ['AIC', 'AIE', 'AIV', 'AID']

TIMM_NAMES = {
    "GhostNet":        "ghostnet_100",
    "EfficientNet_B0": "efficientnet_b0",
    "MobileNetV3":     "mobilenetv3_large_100",
}
BACKBONE_NAMES = list(TIMM_NAMES.keys())

BASELINE_ARM = "A1_Backbone"
PRIMARY_ABI_ARM = "A2_AbI_Gate"

ABLATION_VARIANTS = {
    "A1_Backbone": dict(
        model=dict(use_abi=False, fusion_mode=None, loss_mode=None),
        train=dict(mixup_decouple=False, anneal=False, lambda_weights=None),
    ),
    "A2_AbI_v1": dict(
        model=dict(use_abi=True, fusion_mode='concat', loss_mode='hard_hinge'),
        train=dict(mixup_decouple=False, anneal=False, lambda_weights=LAMBDA_V1),
    ),
    "A2_AbI_NoGate": dict(
        model=dict(use_abi=True, fusion_mode='concat', loss_mode='smooth'),
        train=dict(mixup_decouple=True, anneal=True, lambda_weights=LAMBDA_FINAL),
    ),
    "A2_AbI_Gate": dict(
        model=dict(use_abi=True, fusion_mode='gate', loss_mode='smooth'),
        train=dict(mixup_decouple=True, anneal=True, lambda_weights=LAMBDA_FINAL),
    ),
}
ABLATION_ARM_NAMES = list(ABLATION_VARIANTS.keys())


# =============================================================================
# BACKBONE FACTORY
# =============================================================================
def _find_gradcam_layer_cnn(module: nn.Module) -> nn.Module:
    candidates = [
        child for _, child in module.named_children()
        if any(isinstance(m, nn.Conv2d) for m in child.modules())
    ]
    if len(candidates) >= 2:
        return candidates[-2]
    if len(candidates) == 1:
        return candidates[-1]
    convs = [m for m in module.modules() if isinstance(m, nn.Conv2d)]
    return convs[-1] if convs else module


def build_backbone(name: str, pretrained: bool = True):
    if not HAS_TIMM:
        raise RuntimeError(f"timm required for {name}.")
    timm_name = TIMM_NAMES[name]
    bb = timm.create_model(timm_name, pretrained=pretrained,
                            num_classes=0, global_pool='avg')
    bb.eval()
    with torch.no_grad():
        feat_dim = int(bb(torch.zeros(1, 3, 224, 224)).shape[1])
    grad_layer = _find_gradcam_layer_cnn(bb)
    print(f"  [{name}] feat_dim={feat_dim}  gradcam={type(grad_layer).__name__}")
    return bb, feat_dim, grad_layer


# =============================================================================
# ABNORMALITY INTENSITY MODULE
# =============================================================================
class AbnormalityIntensityModule(nn.Module):
    def __init__(self, n: int, high_frac: float = ABI_HIGH_FRAC,
                 low_frac: float = ABI_LOW_FRAC, eps: float = 1e-6):
        super().__init__()
        self.n = n
        self.high_frac = high_frac
        self.low_frac  = low_frac
        self.eps       = eps
        self.register_buffer('k_idx', torch.arange(n, dtype=torch.float))

    def forward(self, z: torch.Tensor) -> dict:
        I = z.abs() + self.eps
        sum_I = I.sum(dim=1, keepdim=True)

        aic = (I * self.k_idx.unsqueeze(0)).sum(dim=1) / sum_I.squeeze(1)
        p   = I / sum_I
        aie = -(p * torch.log(p + self.eps)).sum(dim=1)
        aiv = (I[:, 1:] - I[:, :-1]).abs().mean(dim=1)

        sorted_I, _ = torch.sort(I, dim=1, descending=True)
        k_high = max(1, int(self.n * self.high_frac))
        k_low  = max(1, int(self.n * self.low_frac))
        high_sum = sorted_I[:, :k_high].sum(dim=1)
        low_sum  = sorted_I[:, -k_low:].sum(dim=1)
        aico = torch.log(high_sum + self.eps) - torch.log(low_sum + self.eps)

        mu  = I.mean(dim=1, keepdim=True)
        aid = torch.sqrt(((I - mu) ** 2).mean(dim=1) + self.eps)

        return dict(I=I, AIC=aic, AIE=aie, AIV=aiv, AICo=aico, AID=aid)


# =============================================================================
# GATED RESIDUAL FUSION
# =============================================================================
class GatedAbIFusion(nn.Module):
    def __init__(self, latent_dim: int, n_descriptors: int = len(DESCRIPTOR_NAMES),
                 hidden: int = GATE_HIDDEN, init_bias: float = GATE_INIT_BIAS):
        super().__init__()
        self.norm = nn.LayerNorm(n_descriptors)
        self.fc1  = nn.Linear(n_descriptors, hidden)
        self.fc2  = nn.Linear(hidden, latent_dim)
        nn.init.constant_(self.fc2.bias, init_bias)

    def forward(self, abi_vec: torch.Tensor, z: torch.Tensor):
        m_hat = self.norm(abi_vec)
        h = F.gelu(self.fc1(m_hat))
        g = torch.sigmoid(self.fc2(h))
        z_fused = z + g * z
        return z_fused, g


# =============================================================================
# INTENSITY PROGRESSION LOSS
# =============================================================================
class IntensityProgressionLoss(nn.Module):
    def __init__(self, margin: float = PROG_MARGIN,
                 smooth: bool = True, beta: float = PROG_BETA):
        super().__init__()
        self.margin = margin
        self.smooth = smooth
        self.beta   = beta

    def _ordinal_loss(self, vals: torch.Tensor,
                       labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        class_means = []
        for c in range(num_classes):
            mask = labels == c
            v = vals[mask].mean() if mask.sum() > 0 else vals.mean()
            class_means.append(v.unsqueeze(0))
        means = torch.cat(class_means)
        loss  = vals.new_zeros(())
        for i in range(num_classes - 1):
            diff = self.margin + means[i] - means[i + 1]
            if self.smooth:
                loss = loss + (1.0 / self.beta) * F.softplus(self.beta * diff)
            else:
                loss = loss + F.relu(diff)
        return loss / max(num_classes - 1, 1)

    def forward(self, descriptors: dict, labels: torch.Tensor,
                num_classes: int) -> dict:
        return {
            f'L_{d}': self._ordinal_loss(descriptors[d], labels, num_classes)
            for d in PROGRESSION_DESCRIPTORS
        }


# =============================================================================
# FOCAL LOSS
# =============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor,
                 gamma: float = 2.0, label_smoothing: float = 0.05):
        super().__init__()
        self.register_buffer('alpha', alpha)
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        C = logits.size(1)
        with torch.no_grad():
            st = torch.full_like(logits, self.label_smoothing / (C - 1))
            st.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        log_p = F.log_softmax(logits, dim=1)
        ce    = -(st * log_p).sum(dim=1)
        with torch.no_grad():
            p_t = F.softmax(logits, dim=1).gather(
                1, targets.unsqueeze(1)).squeeze(1)
            fw  = (1.0 - p_t) ** self.gamma
        return (self.alpha[targets] * fw * ce).mean()


# =============================================================================
# AB-I MODEL
# =============================================================================
class AbIModel(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int,
                 use_abi: bool = True, fusion_mode: str = 'concat',
                 loss_mode: str = 'smooth', pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes   = num_classes
        self.use_abi        = use_abi
        self.fusion_mode    = fusion_mode
        self.loss_mode      = loss_mode

        bb, feat_dim, grad_layer = build_backbone(backbone_name, pretrained)
        self.backbone       = bb
        self.feat_dim       = feat_dim
        self.gradcam_target = grad_layer

        self.proj = nn.Sequential(
            nn.Linear(feat_dim, LATENT_DIM),
            nn.LayerNorm(LATENT_DIM),
            nn.GELU(),
        )

        if use_abi:
            self.abi       = AbnormalityIntensityModule(LATENT_DIM)
            self.prog_loss = IntensityProgressionLoss(
                margin=PROG_MARGIN, smooth=(loss_mode == 'smooth'), beta=PROG_BETA)
            if fusion_mode == 'gate':
                self.fusion = GatedAbIFusion(LATENT_DIM, len(DESCRIPTOR_NAMES),
                                              hidden=GATE_HIDDEN, init_bias=GATE_INIT_BIAS)
                head_in = LATENT_DIM
            else:
                self.abi_norm = nn.LayerNorm(len(DESCRIPTOR_NAMES))
                head_in = LATENT_DIM + len(DESCRIPTOR_NAMES)
        else:
            head_in = LATENT_DIM

        self.dropout = nn.Dropout(0.35)
        self.head    = nn.Linear(head_in, num_classes)

    def _extract_features(self, x):
        return self.backbone(x)

    def _abi_vec(self, z):
        desc = self.abi(z)
        abi_vec = torch.stack([desc[d] for d in DESCRIPTOR_NAMES], dim=1)
        return desc, abi_vec

    @torch.no_grad()
    def get_embedding(self, x):
        z = self.proj(self._extract_features(x))
        if not self.use_abi:
            return z
        _, abi_vec = self._abi_vec(z)
        if self.fusion_mode == 'gate':
            z_fused, _ = self.fusion(abi_vec, z)
            return z_fused
        abi_vec_n = self.abi_norm(abi_vec)
        return torch.cat([z, abi_vec_n], dim=1)

    def forward(self, x, labels=None):
        feats = self._extract_features(x)
        z     = self.proj(feats)
        prog_losses = {}

        if self.use_abi:
            desc, abi_vec = self._abi_vec(z)
            if self.fusion_mode == 'gate':
                z_fused, _ = self.fusion(abi_vec, z)
                h_in = self.dropout(z_fused)
            else:
                abi_vec_n = self.abi_norm(abi_vec)
                h_in = torch.cat([self.dropout(z), abi_vec_n], dim=1)
            if self.training and labels is not None:
                prog_losses = self.prog_loss(desc, labels, self.num_classes)
        else:
            h_in = self.dropout(z)

        return self.head(h_in), prog_losses

    @torch.no_grad()
    def get_abi_signature(self, x):
        z = self.proj(self._extract_features(x))
        abi_tmp = self.abi if self.use_abi else AbnormalityIntensityModule(LATENT_DIM).to(z.device)
        desc = abi_tmp(z)
        return {k: (v.cpu().numpy() if k == 'I' else v.cpu().numpy().reshape(-1))
                for k, v in desc.items()}


# =============================================================================
# MIXUP
# =============================================================================
def mixup_data(x, y, alpha=MIXUP_ALPHA):
    lam = float(max(np.random.beta(alpha, alpha),
                    1 - np.random.beta(alpha, alpha)))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


# =============================================================================
# TRAIN / VALIDATE
# =============================================================================
def train_epoch(model, loader, optimizer, criterion, device,
                ab_flags, epoch, num_classes):
    model.train()
    total_loss = correct = total = 0.0

    if ab_flags.get('anneal', False):
        prog_scale = min(1.0, epoch / PROG_WARMUP_EPOCHS)
    else:
        prog_scale = 1.0

    lambda_weights = ab_flags.get('lambda_weights', {})
    is_mixup_decouple = ab_flags.get('mixup_decouple', False)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        do_mixup = is_mixup_decouple and (random.random() < MIXUP_DECOUPLE_PROB)

        if do_mixup:
            mx, ya, yb, lam = mixup_data(x, y)
            logits, _ = model(mx, labels=None)
            loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
            preds = logits.argmax(1)
            correct += (lam * (preds == ya).float() +
                        (1 - lam) * (preds == yb).float()).sum().item()
        else:
            logits, prog_losses = model(x, labels=y)
            loss = criterion(logits, y)
            if prog_losses:
                for dname in PROGRESSION_DESCRIPTORS:
                    key = f'L_{dname}'
                    if key in prog_losses:
                        w = lambda_weights.get(dname, 0.0) * prog_scale
                        loss = loss + w * prog_losses[key]
            preds = logits.argmax(1)
            correct += (preds == y).float().sum().item()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total += y.size(0)

    return total_loss / total, 100.0 * correct / total


def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            total_loss += criterion(logits, y).item() * x.size(0)
            correct    += logits.argmax(1).eq(y).sum().item()
            total      += y.size(0)
    return total_loss / total, 100.0 * correct / total


# =============================================================================
# CALIBRATION METRICS
# =============================================================================
def calibration_metrics(y_true, y_probs, n_bins=15):
    y_true  = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    preds   = y_probs.argmax(1)
    conf    = y_probs.max(1)
    ok      = (preds == y_true).astype(float)
    bins    = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 0:
            ece += m.mean() * abs(ok[m].mean() - conf[m].mean())
    C     = y_probs.shape[1]
    brier = float(np.mean([
        brier_score_loss((y_true == c).astype(float), y_probs[:, c])
        for c in range(C)
    ]))
    return {'ece': float(ece), 'brier_score': brier}


# =============================================================================
# BOOTSTRAP CI
# =============================================================================
def bootstrap_ci(preds, targets, n_iter=N_BOOT):
    n    = len(targets)
    rng  = np.random.default_rng(42)
    boot = [accuracy_score(targets[rng.integers(0, n, n)],
                           preds[rng.integers(0, n, n)])
            for _ in range(n_iter)]
    boot = np.array(boot)
    return dict(
        acc      = round(float(accuracy_score(targets, preds)), 4),
        ci_lower = round(float(np.percentile(boot, 2.5)), 4),
        ci_upper = round(float(np.percentile(boot, 97.5)), 4),
        ci_width = round(float(np.percentile(boot,97.5) - np.percentile(boot,2.5)), 4),
    )


# =============================================================================
# RETRIEVAL ANALYSIS
# =============================================================================
def retrieval_analysis(emb, labels, class_names, model_tag, K_list=(1, 5)):
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    emb_n = emb / norms
    sim   = emb_n @ emb_n.T
    np.fill_diagonal(sim, -1.0)
    rows  = []
    for K in K_list:
        top_idx  = np.argsort(-sim, axis=1)[:, :K]
        top_lbls = labels[top_idx]
        purity   = (top_lbls == labels[:, None]).mean(axis=1).mean()
        rows.append(dict(model=model_tag, K=K,
                         metric=f'Top-{K}_Purity', value=float(purity)))
        if K == 1:
            rows.append(dict(model=model_tag, K=K, metric='Top-1_Accuracy',
                             value=float((top_lbls[:, 0] == labels).mean())))
    return rows


# =============================================================================
# COMPUTE PROFILE
# =============================================================================
def profile_model(model, device, img_size=224, n_runs=50):
    actual   = model.module if hasattr(model, 'module') else model
    actual.eval()
    params_M = sum(p.numel() for p in actual.parameters()) / 1e6
    flops_G  = float('nan')
    if HAS_THOP:
        try:
            macs, _ = thop_profile(
                actual, inputs=(torch.randn(1, 3, img_size, img_size).to(device),),
                verbose=False)
            flops_G = macs / 1e9
        except Exception:
            pass
    gpu_mem_MB = float('nan')
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            actual(torch.randn(1, 3, img_size, img_size).to(device))
        gpu_mem_MB = torch.cuda.max_memory_allocated(device) / 1e6
    dummy = torch.randn(1, 3, img_size, img_size).to(device)
    with torch.no_grad():
        for _ in range(10):
            actual(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            actual(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) / n_runs * 1000.0
    return dict(params_M=round(params_M, 3), flops_G=round(flops_G, 3),
                gpu_mem_MB=round(gpu_mem_MB, 2), latency_ms=round(latency_ms, 3))


# =============================================================================
# AB-I VALIDATION FUNCTIONS
# =============================================================================
def run_abi_validation_one_seed(model, test_loader, class_names,
                                 model_tag, seed, save_dir):
    """
    Run Kruskal-Wallis validation on Ab-I descriptors for one seed.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    actual = model.module if hasattr(model, 'module') else model

    desc_lists = defaultdict(list)
    label_list = []
    actual.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            sig = actual.get_abi_signature(x)
            for d in DESCRIPTOR_NAMES:
                desc_lists[d].append(sig[d])
            label_list.append(np.atleast_1d(y.cpu().numpy()))

    labels = np.concatenate(label_list)
    desc_arrays = {d: np.concatenate(v) for d, v in desc_lists.items()}

    kw_rows = []
    for dname, dvals in desc_arrays.items():
        groups = [dvals[labels == c] for c in range(len(class_names)) if (labels == c).sum() > 0]
        if len(groups) >= 2:
            try:
                stat, p = kruskal(*groups)
            except Exception:
                stat, p = float('nan'), float('nan')
        else:
            stat, p = float('nan'), float('nan')
        kw_rows.append(dict(
            model=model_tag, seed=seed, descriptor=dname,
            kw_stat=round(float(stat), 4) if not np.isnan(stat) else float('nan'),
            p_value=round(float(p), 6) if not np.isnan(p) else float('nan'),
            significant=bool(p < ALPHA) if not np.isnan(p) else False))
    kw_df = pd.DataFrame(kw_rows)

    desc_rows = []
    for c, cn in enumerate(class_names):
        mask = labels == c
        if mask.sum() == 0:
            continue
        row = dict(model=model_tag, seed=seed,
                   backbone='_'.join(model_tag.split('_')[:-2]) if '_' in model_tag else model_tag,
                   class_name=cn)
        for dname, dvals in desc_arrays.items():
            row[f'{dname}_mean'] = round(float(dvals[mask].mean()), 6)
        desc_rows.append(row)

    kw_df.to_csv(save_dir / f'{model_tag}_seed{seed}_kruskal.csv', index=False)
    pd.DataFrame(desc_rows).to_csv(
        save_dir / f'{model_tag}_seed{seed}_class_descriptors.csv', index=False)
    return kw_df, desc_rows


# =============================================================================
# DUNN POST-HOC  *** BUG FIX: removed inner `import pandas as pd` ***
# =============================================================================
def run_dunns_posthoc_abi(model, test_loader, class_names,
                           model_tag, seed, save_dir):
    """
    Run Dunn's post-hoc test on Ab-I descriptors for one seed.

    FIX: The original code contained `import pandas as pd` inside a nested
    try block, which shadowed the module-level name and raised:
        UnboundLocalError: cannot access local variable 'pd'
    Removed — pandas is already imported at module level.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    actual = model.module if hasattr(model, 'module') else model

    desc_lists = defaultdict(list)
    label_list = []
    actual.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            sig = actual.get_abi_signature(x)
            for d in DESCRIPTOR_NAMES:
                desc_lists[d].append(sig[d])
            label_list.append(np.atleast_1d(y.cpu().numpy()))
    labels = np.concatenate(label_list)
    desc_arrays = {d: np.concatenate(v) for d, v in desc_lists.items()}

    rows = []
    for dname, dvals in desc_arrays.items():
        groups = {cn: dvals[labels == c] for c, cn in enumerate(class_names)
                  if (labels == c).sum() > 0}
        gkeys = list(groups.keys())
        if len(gkeys) < 2:
            continue

        # First check if Kruskal-Wallis is significant
        try:
            stat, p_kw = kruskal(*[groups[k] for k in gkeys])
        except Exception:
            continue

        if np.isnan(p_kw) or p_kw >= ALPHA:
            for ci, cj in combinations(gkeys, 2):
                rows.append(dict(
                    model=model_tag, seed=seed, descriptor=dname,
                    comparison=f"{ci} vs {cj}",
                    p_value_corrected=float('nan'),
                    significant=False, note='KW_not_significant'))
            continue

        # Run Dunn's test or Mann-Whitney fallback
        if HAS_DUNN:
            try:
                # Prepare data for scikit-posthocs using module-level pd
                all_vals = []
                all_grps = []
                for k in gkeys:
                    all_vals.extend(groups[k])
                    all_grps.extend([k] * len(groups[k]))

                df_dunn = pd.DataFrame({'val': all_vals, 'grp': all_grps})
                dunn_result = posthoc_dunn(
                    df_dunn, val_col='val', group_col='grp', p_adjust='bonferroni')

                for ci, cj in combinations(gkeys, 2):
                    p_corr = float(dunn_result.loc[ci, cj])
                    rows.append(dict(
                        model=model_tag, seed=seed, descriptor=dname,
                        comparison=f"{ci} vs {cj}",
                        p_value_corrected=round(p_corr, 6),
                        significant=bool(p_corr < ALPHA),
                        note='dunn_bonferroni'))

            except Exception as e:
                print(f"    [Dunn] {model_tag} seed{seed} {dname}: {e}")
                # Fallback to Mann-Whitney with Bonferroni
                pairs = list(combinations(gkeys, 2))
                n_pairs = len(pairs)
                for ci, cj in pairs:
                    try:
                        _, p_raw = stats.mannwhitneyu(
                            groups[ci], groups[cj], alternative='two-sided')
                        p_corr = min(float(p_raw) * n_pairs, 1.0)
                        rows.append(dict(
                            model=model_tag, seed=seed, descriptor=dname,
                            comparison=f"{ci} vs {cj}",
                            p_value_corrected=round(p_corr, 6),
                            significant=bool(p_corr < ALPHA),
                            note='mannwhitney_bonferroni_fallback'))
                    except Exception:
                        continue
        else:
            # Fallback to Mann-Whitney with Bonferroni
            pairs = list(combinations(gkeys, 2))
            n_pairs = len(pairs)
            for ci, cj in pairs:
                try:
                    _, p_raw = stats.mannwhitneyu(
                        groups[ci], groups[cj], alternative='two-sided')
                    p_corr = min(float(p_raw) * n_pairs, 1.0)
                    rows.append(dict(
                        model=model_tag, seed=seed, descriptor=dname,
                        comparison=f"{ci} vs {cj}",
                        p_value_corrected=round(p_corr, 6),
                        significant=bool(p_corr < ALPHA),
                        note='mannwhitney_bonferroni_fallback'))
                except Exception:
                    continue

    if rows:
        dunn_df = pd.DataFrame(rows)
        dunn_df.to_csv(save_dir / f'{model_tag}_seed{seed}_dunn.csv', index=False)
        return dunn_df.to_dict('records')
    return []


# =============================================================================
# DESCRIPTOR ANALYSIS HELPERS
# =============================================================================
def compute_descriptor_regulation(desc_df, backbone_names, class_names):
    """
    Compute delta (Ab-I - Baseline) for descriptors per backbone/seed/class.
    """
    if desc_df.empty:
        return pd.DataFrame()

    rows = []
    for bb in backbone_names:
        for seed in SEEDS:
            for cn in class_names:
                a1_tag = f"{bb}_{BASELINE_ARM}"
                a2_tag = f"{bb}_{PRIMARY_ABI_ARM}"
                sub_a1 = desc_df[(desc_df['model'] == a1_tag) &
                                  (desc_df['seed'] == seed) &
                                  (desc_df['class_name'] == cn)]
                sub_a2 = desc_df[(desc_df['model'] == a2_tag) &
                                  (desc_df['seed'] == seed) &
                                  (desc_df['class_name'] == cn)]
                if sub_a1.empty or sub_a2.empty:
                    continue
                row = dict(backbone=bb, seed=seed, class_name=cn)
                for dname in DESCRIPTOR_NAMES:
                    col = f'{dname}_mean'
                    base_val = float(sub_a1[col].values[0])
                    abi_val  = float(sub_a2[col].values[0])
                    row[f'baseline_{dname}'] = round(base_val, 6)
                    row[f'abi_{dname}']      = round(abi_val, 6)
                    row[f'delta_{dname}']    = round(abi_val - base_val, 6)
                rows.append(row)
    return pd.DataFrame(rows)


def compute_descriptor_separation_gain(kw_all_df, backbone_names):
    """
    Compute gain in Kruskal-Wallis statistic (Ab-I - Baseline).
    """
    if kw_all_df.empty:
        return pd.DataFrame()

    rows = []
    for bb in backbone_names:
        for seed in SEEDS:
            for dname in DESCRIPTOR_NAMES:
                a1_tag = f"{bb}_{BASELINE_ARM}"
                a2_tag = f"{bb}_{PRIMARY_ABI_ARM}"
                r_a1 = kw_all_df[(kw_all_df['model'] == a1_tag) &
                                  (kw_all_df['seed'] == seed) &
                                  (kw_all_df['descriptor'] == dname)]
                r_a2 = kw_all_df[(kw_all_df['model'] == a2_tag) &
                                  (kw_all_df['seed'] == seed) &
                                  (kw_all_df['descriptor'] == dname)]
                if r_a1.empty or r_a2.empty:
                    continue
                kw_a1 = float(r_a1['kw_stat'].values[0]) if not np.isnan(r_a1['kw_stat'].values[0]) else 0
                kw_a2 = float(r_a2['kw_stat'].values[0]) if not np.isnan(r_a2['kw_stat'].values[0]) else 0
                rows.append(dict(
                    backbone=bb, seed=seed, descriptor=dname,
                    kw_baseline=round(kw_a1, 4),
                    kw_abi=round(kw_a2, 4),
                    kw_gain=round(kw_a2 - kw_a1, 4)))
    return pd.DataFrame(rows)


def compute_representation_gain(latent_df, progression_df, backbone_names):
    """
    Compute representation learning improvement (Ab-I vs Baseline, etc.)
    """
    rows = []
    lat_metrics  = ['silhouette', 'davies_bouldin', 'calinski_harabasz', 'knn_consistency']
    prog_metrics = ['centroid_dist', 'transition_overlap']
    higher_better = {'silhouette', 'calinski_harabasz', 'knn_consistency', 'centroid_dist'}

    comparison_pairs = [
        (BASELINE_ARM,     PRIMARY_ABI_ARM,   'Gate_vs_Baseline'),
        ("A2_AbI_NoGate",  PRIMARY_ABI_ARM,   'Gate_vs_NoGate'),
        ("A2_AbI_v1",      PRIMARY_ABI_ARM,   'Gate_vs_v1'),
        ("A2_AbI_v1",      "A2_AbI_NoGate",   'NoGate_vs_v1'),
    ]

    for bb in backbone_names:
        for seed in SEEDS:
            for arm1, arm2, pair_name in comparison_pairs:
                tag1 = f"{bb}_{arm1}"
                tag2 = f"{bb}_{arm2}"

                if not latent_df.empty:
                    for met in lat_metrics:
                        sub_a1 = latent_df[(latent_df['model'] == tag1) & (latent_df['seed'] == seed)]
                        sub_a2 = latent_df[(latent_df['model'] == tag2) & (latent_df['seed'] == seed)]
                        if sub_a1.empty or sub_a2.empty:
                            continue
                        v_a1 = float(np.nanmean(sub_a1[met].values))
                        v_a2 = float(np.nanmean(sub_a2[met].values))
                        improvement = (v_a2 - v_a1) if met in higher_better else (v_a1 - v_a2)
                        rows.append(dict(
                            backbone=bb, seed=seed, metric=met, comparison=pair_name,
                            baseline_value=round(v_a1, 4),
                            abi_value=round(v_a2, 4),
                            improvement=round(improvement, 4)))

                if not progression_df.empty:
                    for met in prog_metrics:
                        pa1 = progression_df[(progression_df['model'] == tag1) &
                                              (progression_df['seed'] == seed) &
                                              (progression_df['metric'] == met)]
                        pa2 = progression_df[(progression_df['model'] == tag2) &
                                              (progression_df['seed'] == seed) &
                                              (progression_df['metric'] == met)]
                        if pa1.empty or pa2.empty:
                            continue
                        v_a1 = float(pa1['value'].mean())
                        v_a2 = float(pa2['value'].mean())
                        improvement = (v_a2 - v_a1) if met in higher_better else (v_a1 - v_a2)
                        rows.append(dict(
                            backbone=bb, seed=seed, metric=met, comparison=pair_name,
                            baseline_value=round(v_a1, 4),
                            abi_value=round(v_a2, 4),
                            improvement=round(improvement, 4)))
    return pd.DataFrame(rows)


def plot_descriptor_progression(desc_energy_df, backbone_names, rep_seed_store, class_names_order):
    out = SAVE_DIR / 'progression_visualizations'
    out.mkdir(parents=True, exist_ok=True)
    if desc_energy_df.empty:
        return

    for bb in backbone_names:
        rep_seed = rep_seed_store.get(f"{bb}_{PRIMARY_ABI_ARM}", SEEDS[0])

        fig, axes = plt.subplots(1, len(DESCRIPTOR_NAMES), figsize=(4 * len(DESCRIPTOR_NAMES), 4))
        if len(DESCRIPTOR_NAMES) == 1:
            axes = [axes]

        for ax, dname in zip(axes, DESCRIPTOR_NAMES):
            for ab in ABLATION_ARM_NAMES:
                color = ABLATION_COLORS.get(ab, '#888888')
                mt = f"{bb}_{ab}"
                sub = desc_energy_df[(desc_energy_df['model'] == mt) &
                                      (desc_energy_df['seed'] == rep_seed)]
                if sub.empty:
                    continue
                vals = [sub[sub['class_name'] == cn][f'{dname}_mean'].mean()
                        for cn in class_names_order]
                ax.plot(range(len(class_names_order)), vals, 'o-',
                        color=color, label=ab, lw=2, markersize=8)

            ax.set_xticks(range(len(class_names_order)))
            ax.set_xticklabels(class_names_order, rotation=15)
            ax.set_title(dname, fontweight='bold', fontsize=14)
            ax.set_xlabel('Severity Grade', fontsize=12)
            ax.legend(fontsize=9, loc='best')
            ax.grid(alpha=0.3)

        fig.suptitle(f'{bb} -- Abnormality Intensity Descriptor Progression '
                     f'(Rep Seed {rep_seed})', fontweight='bold', fontsize=16)
        plt.tight_layout()
        plt.savefig(out / f'{bb}_descriptor_progression_rep_seed{rep_seed}.png',
                    dpi=300, bbox_inches='tight')
        plt.close(fig)


# =============================================================================
# GRADCAM ENGINE
# =============================================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self._acts = None
        self._grads = None
        self._fh = target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, '_acts', o.detach()))
        self._bh = target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, '_grads', go[0].detach()))

    def generate(self, x, cls_idx=None):
        self.model.eval()
        self.model.zero_grad()
        self._acts = None
        self._grads = None
        logits, _ = self.model(x)
        if cls_idx is None:
            cls_idx = int(logits.argmax(1).item())
        logits[:, cls_idx].sum().backward()
        if self._acts is None or self._grads is None:
            raise RuntimeError("GradCAM hooks did not fire.")
        acts, grads = self._acts, self._grads
        if acts.dim() == 3:
            B, L, C = acts.shape
            H = W = int(L ** 0.5)
            if H * W != L:
                return np.ones((x.shape[2], x.shape[3]), dtype=np.float32)
            acts  = acts.permute(0, 2, 1).reshape(B, C, H, W)
            grads = grads.permute(0, 2, 1).reshape(B, C, H, W)
        if acts.dim() < 4:
            return np.ones((x.shape[2], x.shape[3]), dtype=np.float32)
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:],
                            mode='bilinear', align_corners=False)
        B2 = cam.size(0)
        cmin = cam.view(B2, -1).min(1).values.view(B2, 1, 1, 1)
        cmax = cam.view(B2, -1).max(1).values.view(B2, 1, 1, 1)
        cam = (cam - cmin) / (cmax - cmin + 1e-8)
        return cam.detach().cpu().numpy()[0, 0]

    def release(self):
        self._fh.remove()
        self._bh.remove()


# =============================================================================
# GRADCAM QUANTITATIVE METRICS
# =============================================================================
def gradcam_quant(cam: np.ndarray, thr: float = 0.4) -> dict:
    if cam is None or cam.size == 0:
        return dict(activation_area=float('nan'),
                    activation_entropy=float('nan'),
                    contour_complexity=float('nan'))
    H, W = cam.shape
    N    = H * W
    mask = (cam >= thr).astype(int)
    area = mask.sum() / N * 100.0
    flat = cam.flatten() / (cam.sum() + 1e-8)
    ent  = -np.sum(flat * np.log(flat + 1e-10)) / np.log(N)
    contour = 0.0
    if mask.sum() > 0:
        pad   = np.pad(mask, 1)
        edges = (np.abs(pad[1:-1, 1:-1] - pad[:-2, 1:-1]) +
                 np.abs(pad[1:-1, 1:-1] - pad[2:,  1:-1]) +
                 np.abs(pad[1:-1, 1:-1] - pad[1:-1, :-2]) +
                 np.abs(pad[1:-1, 1:-1] - pad[1:-1, 2:]))
        contour = float(edges.sum() / (4.0 * np.sqrt(mask.sum() + 1e-8) + 1e-8))
    return dict(
        activation_area=float(area),
        activation_entropy=float(ent),
        contour_complexity=float(contour),
    )


# =============================================================================
# GRADCAM VISUALIZATION - SAVES ALL IMAGES
# =============================================================================
def _save_gradcam_panel(img_np, cam, model_tag, seed, idx,
                         cls_name, pred_name, save_path):
    correct    = (cls_name == pred_name)
    status     = "? Correct" if correct else "? Incorrect"
    status_col = "#27AE60"   if correct else "#C0392B"
    fig = plt.figure(figsize=(20, 5.5), constrained_layout=False)
    gs  = GridSpec(1, 4, figure=fig,
                   left=0.03, right=0.97, top=0.82, bottom=0.05, wspace=0.10)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    fig.text(0.5, 0.90,
             f"{model_tag}  |  Seed {seed}  |  Image #{idx}\n"
             f"True: {cls_name}   Predicted: {pred_name}   [{status}]",
             ha='center', va='bottom', fontsize=16, fontweight='bold',
             color=status_col)
    axes[0].imshow(img_np); axes[0].axis('off')
    im1 = axes[1].imshow(cam, cmap='inferno', vmin=0, vmax=1)
    axes[1].axis('off')
    cb = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, aspect=20)
    cb.set_label("Activation Intensity", fontsize=13)
    axes[2].imshow(img_np)
    axes[2].imshow(cam, cmap='jet', alpha=0.45, vmin=0, vmax=1)
    try:
        if cam.max() - cam.min() > 0.1:
            axes[2].contour(cam, levels=[0.5], colors='white', linewidths=2.0)
    except Exception:
        pass
    axes[2].axis('off')
    mask_vis = (cam >= 0.4).astype(float)
    axes[3].imshow(img_np)
    axes[3].imshow(mask_vis, cmap='Reds', alpha=0.60, vmin=0, vmax=1)
    axes[3].axis('off')
    for ax, title in zip(axes, ["Original", "GradCAM", "Overlay", "Mask"]):
        ax.set_title(title, fontsize=14, fontweight='bold', pad=6)
    qm  = gradcam_quant(cam)
    ann = (f"Area={qm['activation_area']:.1f}%  "
           f"Entropy={qm['activation_entropy']:.3f}  "
           f"Contour={qm['contour_complexity']:.3f}")
    fig.text(0.5, 0.005, ann, ha='center', va='bottom', fontsize=12,
             color='#2C3E50',
             bbox=dict(facecolor='white', edgecolor='#BDC3C7',
                       boxstyle='round,pad=0.3', alpha=0.85))
    try:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
    except Exception as e:
        print(f"    [GradCAM-Save] {save_path}: {e}")
    finally:
        plt.close(fig)


def run_full_gradcam_per_seed(model, model_tag, seed, test_loader,
                               all_preds, all_targets, class_names):
    """
    Run GradCAM for one seed -- saves ALL images (not just 20 per class).
    """
    out_root = SAVE_DIR / 'gradcam' / model_tag / f"seed{seed}"
    actual   = model.module if hasattr(model, 'module') else model
    try:
        engine = GradCAM(actual, actual.gradcam_target)
    except Exception as exc:
        print(f"  [GradCAM] Cannot create engine for {model_tag}: {exc}")
        return []

    records = []
    try:
        for idx, (x, y) in enumerate(test_loader):
            cls_name  = class_names[int(y.item())]
            pred_name = class_names[int(all_preds[idx])]
            try:
                cam = engine.generate(x.to(DEVICE), cls_idx=int(y.item()))
            except Exception as exc:
                print(f"    [GradCAM] img {idx} skipped: {exc}")
                continue
            if cam is None or not np.all(np.isfinite(cam)):
                continue
            qm = gradcam_quant(cam)
            qm.update(dict(idx=idx, true=cls_name, pred=pred_name,
                           correct=(int(all_preds[idx]) == int(y.item())),
                           model=model_tag, seed=seed))
            records.append(qm)

            cls_dir = out_root / f"class_{cls_name}"
            cls_dir.mkdir(parents=True, exist_ok=True)
            img = x.squeeze().cpu().numpy().transpose(1, 2, 0)
            img = np.clip(img * [.229, .224, .225] + [.485, .456, .406], 0, 1)
            tag  = "ok" if int(all_preds[idx]) == int(y.item()) else "err"
            save_path = cls_dir / f"{idx:05d}_{tag}_pred{pred_name}.png"
            try:
                _save_gradcam_panel(img, cam, model_tag, seed, idx,
                                    cls_name, pred_name, save_path)
            except Exception as exc:
                print(f"    [GradCAM-Save] img {idx} failed: {exc}")

            if idx % 50 == 0 and idx > 0:
                print(f"    GradCAM [{model_tag}] seed={seed} "
                      f"{idx + 1}/{len(test_loader)} images processed")
    finally:
        try:
            engine.release()
        except Exception:
            pass

    if records:
        df_seed = pd.DataFrame(records)
        df_seed.to_csv(out_root / f"gradcam_quant_seed{seed}.csv", index=False)
        qm_cols = ['activation_area', 'activation_entropy', 'contour_complexity']
        agg = df_seed.groupby('true')[qm_cols].agg(['mean', 'std']).round(4)
        agg.to_csv(out_root / f"gradcam_quant_by_class_seed{seed}.csv")
    return records


# =============================================================================
# GRADCAM COMPARISON ACROSS ALL 4 MODULES
# =============================================================================
def run_gradcam_comparison_all_modules(
        models_dict, test_loader, viz_indices, all_preds_dict,
        all_targets, class_names, seed, backbone_name):
    """
    Generate GradCAM comparison panels for ALL 4 ablation modules
    on IDENTICAL images.
    """
    out_dir = SAVE_DIR / 'gradcam_comparison' / f'seed{seed}' / backbone_name
    out_dir.mkdir(parents=True, exist_ok=True)

    engines = {}
    for ab_name, model in models_dict.items():
        actual = model.module if hasattr(model, 'module') else model
        try:
            engines[ab_name] = GradCAM(actual, actual.gradcam_target)
        except Exception as exc:
            print(f"  [GradCAM-Cmp] Cannot create engine for {ab_name}: {exc}")
            return []

    test_ds = test_loader.dataset

    records = []
    try:
        for local_i, global_idx in enumerate(viz_indices):
            img_path, true_lbl = test_ds.samples[global_idx]
            true_cls = class_names[true_lbl]

            from PIL import Image as PILImage
            pil_img = PILImage.open(img_path)
            val_tf  = v2.Compose([
                v2.Lambda(lambda img: img.convert("RGB")),
                v2.Resize((IMG_SIZE, IMG_SIZE)),
                v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            x = val_tf(pil_img).unsqueeze(0).to(DEVICE)

            cams  = {}
            preds = {}
            for ab_name, engine in engines.items():
                try:
                    cams[ab_name]  = engine.generate(x, cls_idx=true_lbl)
                    preds[ab_name] = class_names[all_preds_dict[ab_name][global_idx]]
                except Exception as exc:
                    print(f"    [GradCAM-Cmp] {ab_name} img {global_idx} skipped: {exc}")
                    cams[ab_name]  = None
                    preds[ab_name] = "ERROR"

            if any(cam is None for cam in cams.values()):
                continue

            img_np = x.squeeze().cpu().numpy().transpose(1, 2, 0)
            img_np = np.clip(img_np * [.229, .224, .225] + [.485, .456, .406], 0, 1)

            cls_dir = out_dir / f"class_{true_cls}"
            cls_dir.mkdir(parents=True, exist_ok=True)

            n_modules = len(ABLATION_ARM_NAMES)
            fig, axes = plt.subplots(n_modules + 1, 3,
                                     figsize=(15, 4 * (n_modules + 1)))

            axes[0, 0].imshow(img_np)
            axes[0, 0].set_title("Original Image", fontsize=14, fontweight='bold')
            axes[0, 0].axis('off')
            axes[0, 1].axis('off')
            axes[0, 2].axis('off')

            for row, ab_name in enumerate(ABLATION_ARM_NAMES, start=1):
                if ab_name not in cams or cams[ab_name] is None:
                    continue
                cam  = cams[ab_name]
                pred = preds.get(ab_name, "N/A")
                im   = axes[row, 0].imshow(cam, cmap='inferno', vmin=0, vmax=1)
                axes[row, 0].set_title(f"{ab_name}\nPred: {pred}",
                                        fontsize=12, fontweight='bold')
                axes[row, 0].axis('off')
                plt.colorbar(im, ax=axes[row, 0], fraction=0.046, pad=0.04)
                axes[row, 1].imshow(img_np)
                axes[row, 1].imshow(cam, cmap='jet', alpha=0.45, vmin=0, vmax=1)
                axes[row, 1].set_title("Overlay", fontsize=12)
                axes[row, 1].axis('off')
                mask_vis = (cam >= 0.4).astype(float)
                axes[row, 2].imshow(img_np)
                axes[row, 2].imshow(mask_vis, cmap='Reds', alpha=0.55, vmin=0, vmax=1)
                axes[row, 2].set_title("Mask (=0.4)", fontsize=12)
                axes[row, 2].axis('off')

            for row, ab_name in enumerate(ABLATION_ARM_NAMES, start=1):
                axes[row, 0].set_ylabel(ab_name, fontsize=11, fontweight='bold',
                                         rotation=0, labelpad=60)

            fig.suptitle(
                f"{backbone_name} | Seed {seed} | True: {true_cls} | Image #{global_idx}",
                fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout(rect=[0, 0, 1, 0.98])

            fname = f"{local_i:04d}_idx{global_idx}_true{true_cls}.png"
            plt.savefig(cls_dir / fname, dpi=300, bbox_inches='tight')
            plt.close(fig)

            # Individual panels per module
            for ab_name, cam in cams.items():
                if cam is None:
                    continue
                fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4))
                axes2[0].imshow(img_np); axes2[0].set_title("Original"); axes2[0].axis('off')
                axes2[1].imshow(cam, cmap='inferno', vmin=0, vmax=1)
                axes2[1].set_title(f"{ab_name} GradCAM"); axes2[1].axis('off')
                axes2[2].imshow(img_np); axes2[2].imshow(cam, cmap='jet', alpha=0.45)
                axes2[2].set_title("Overlay"); axes2[2].axis('off')
                axes2[3].imshow(img_np)
                axes2[3].imshow((cam >= 0.4).astype(float), cmap='Reds', alpha=0.55)
                axes2[3].set_title("Mask (=0.4)"); axes2[3].axis('off')
                fig2.suptitle(f"{backbone_name} | {ab_name} | Seed {seed} | True: {true_cls}",
                              fontsize=14, fontweight='bold')
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                panel_dir = out_dir / "individual_panels" / ab_name
                panel_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(panel_dir / f"{local_i:04d}_idx{global_idx}_true{true_cls}.png",
                            dpi=300, bbox_inches='tight')
                plt.close(fig2)

            records.append(global_idx)

            if local_i % 10 == 0 and local_i > 0:
                print(f"    [GradCAM-Cmp] {local_i}/{len(viz_indices)} images processed")

    finally:
        for engine in engines.values():
            try:
                engine.release()
            except Exception:
                pass

    print(f"  [GradCAM-Cmp] seed{seed} {backbone_name}: "
          f"{len(records)} panels -> {out_dir}")
    return records


# =============================================================================
# VISUALIZATION SUBSET
# =============================================================================
def build_visualization_subset(test_ds, class_names, seed, n_total=VIZ_SUBSET_N):
    rng   = np.random.RandomState(seed)
    n_cls = len(class_names)
    per_cls = n_total // n_cls

    cls_indices = defaultdict(list)
    for idx, (_, lbl) in enumerate(test_ds.samples):
        cls_indices[lbl].append(idx)

    selected = []
    for c in range(n_cls):
        pool   = cls_indices[c]
        k      = min(per_cls, len(pool))
        chosen = rng.choice(pool, size=k, replace=False).tolist()
        selected.extend(chosen)
    rng.shuffle(selected)

    rows = []
    for idx in selected:
        path, lbl = test_ds.samples[idx]
        rows.append(dict(image_id=idx, image_path=str(path),
                         class_label=class_names[lbl]))
    out = SAVE_DIR / 'visualization_subsets' / f'visualization_subset_seed{seed}.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  [VizSubset] seed{seed}: {len(selected)} images -> {out}")
    return selected


# =============================================================================
# LATENT SPACE FUNCTIONS
# =============================================================================
def extract_embeddings(model, loader, device):
    model.eval()
    actual = model.module if hasattr(model, 'module') else model
    embs, lbls = [], []
    with torch.no_grad():
        for x, y in loader:
            embs.append(actual.get_embedding(x.to(device)).cpu().numpy())
            lbls.append(y.numpy())
    return np.vstack(embs), np.concatenate(lbls)


def latent_quality_metrics(emb, lbl):
    out = {}
    try:
        out['silhouette']          = float(silhouette_score(emb, lbl))
        out['davies_bouldin']      = float(davies_bouldin_score(emb, lbl))
        out['calinski_harabasz']   = float(calinski_harabasz_score(emb, lbl))
    except Exception:
        out['silhouette'] = out['davies_bouldin'] = \
            out['calinski_harabasz'] = float('nan')
    try:
        knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
        knn.fit(emb, lbl)
        out['knn_consistency'] = float(accuracy_score(lbl, knn.predict(emb)))
    except Exception:
        out['knn_consistency'] = float('nan')
    return out


def plot_2d_projections(emb_dict, labels, class_names, seed, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    methods = ['tsne'] + (['umap'] if HAS_UMAP else [])
    for method in methods:
        n    = len(emb_dict)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]
        pal = plt.cm.Set1(np.linspace(0, 0.9, len(class_names)))
        for ax, (mname, emb) in zip(axes, emb_dict.items()):
            try:
                if method == 'tsne':
                    proj = TSNE(n_components=2, random_state=seed,
                                perplexity=min(30, len(labels) - 1)).fit_transform(emb)
                else:
                    proj = umap_lib.UMAP(n_components=2,
                                          random_state=seed).fit_transform(emb)
            except Exception as exc:
                print(f"  [Projection] {method} failed for {mname}: {exc}")
                continue
            for c, (cn, col) in enumerate(zip(class_names, pal)):
                mk = labels == c
                ax.scatter(proj[mk, 0], proj[mk, 1], c=[col], label=cn,
                           s=18, alpha=0.72, edgecolors='none')
            ax.set_title(mname, fontsize=14, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([])
            ax.legend(fontsize=10, markerscale=1.5, loc='best')
        fig.suptitle(f'{method.upper()} -- Seed {seed}', fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_dir / f'{method}_seed{seed}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)


# =============================================================================
# CONFUSION MATRIX AND OTHER PLOTS
# =============================================================================
def plot_confusion_matrix(cm, class_names, model_tag, seed):
    out = SAVE_DIR / 'metrics' / model_tag
    out.mkdir(parents=True, exist_ok=True)
    cm_n = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(max(10, len(class_names) * 2),
                       max(4,  len(class_names) - 1)))
    sns.heatmap(cm_n, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax1, linewidths=0.5, vmin=0, vmax=1)
    ax1.set_title('Normalised', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted', fontsize=12); ax1.set_ylabel('True', fontsize=12)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax2, linewidths=0.5)
    ax2.set_title('Counts', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted', fontsize=12); ax2.set_ylabel('True', fontsize=12)
    fig.suptitle(f'{model_tag} -- Seed {seed}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(out / f'seed{seed}_cm_combined.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_training_curves(history, model_tag, seed):
    out = SAVE_DIR / 'histories' / model_tag
    out.mkdir(parents=True, exist_ok=True)
    e = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(e, history['train_loss'], label='Train', color='steelblue', lw=2)
    ax1.plot(e, history['val_loss'],   label='Val',   color='tomato',    lw=2)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves', fontweight='bold')
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(e, history['train_acc'], label='Train', color='steelblue', lw=2)
    ax2.plot(e, history['val_acc'],   label='Val',   color='tomato',    lw=2)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Curves', fontweight='bold')
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.suptitle(f'{model_tag} -- Seed {seed}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(out / f'seed{seed}_curves.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_roc_curves(y_true, y_probs, class_names, model_tag, seed):
    out = SAVE_DIR / 'metrics' / model_tag
    out.mkdir(parents=True, exist_ok=True)
    n_cls = y_probs.shape[1]
    fig, ax = plt.subplots(figsize=(7, 6))
    pal  = plt.cm.Set1(np.linspace(0, 0.9, n_cls))
    tprs, mean_fpr = [], np.linspace(0, 1, 200)
    for c, col in enumerate(pal):
        binary = (y_true == c).astype(int)
        if binary.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(binary, y_probs[:, c])
        ax.plot(fpr, tpr, color=col, lw=2,
                label=f'{class_names[c]} (AUC={sk_auc(fpr, tpr):.3f})')
        tprs.append(np.interp(mean_fpr, fpr, tpr))
    if tprs:
        mt = np.mean(tprs, axis=0)
        ax.plot(mean_fpr, mt, 'k--', lw=2.8,
                label=f'Macro avg (AUC={sk_auc(mean_fpr, mt):.3f})')
    ax.plot([0, 1], [0, 1], 'gray', lw=1.5, linestyle=':')
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate',  fontsize=14)
    ax.set_title(f'{model_tag} -- Seed {seed}', fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / f'seed{seed}_roc.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_reliability_diagram(y_true, y_probs, class_names, model_tag, seed):
    out = SAVE_DIR / 'metrics' / model_tag
    out.mkdir(parents=True, exist_ok=True)
    n_cls = y_probs.shape[1]
    fig, axes = plt.subplots(1, n_cls, figsize=(4 * n_cls, 4), sharey=True)
    if n_cls == 1:
        axes = [axes]
    for c, ax in enumerate(axes):
        try:
            pt, pp = calibration_curve(
                (y_true == c).astype(int), y_probs[:, c],
                n_bins=10, strategy='uniform')
        except Exception:
            continue
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect')
        ax.plot(pp, pt, 'o-', color='steelblue', lw=2.5, ms=8,
                label=class_names[c])
        ax.fill_between(pp, pt, pp, alpha=0.15, color='red', label='Miscal.')
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction Positive', fontsize=12)
        ax.set_title(f'Class: {class_names[c]}', fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle(f'{model_tag} -- Seed {seed} -- Reliability Diagram',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(out / f'seed{seed}_reliability.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# DISEASE PROGRESSION ANALYSIS
# =============================================================================
def analyze_disease_progression(emb_dict, labels, class_names, seed, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    K, rows = len(class_names), []
    for mname, emb in emb_dict.items():
        centroids = np.array([
            emb[labels == c].mean(0) if (labels == c).sum() > 0
            else np.zeros(emb.shape[1])
            for c in range(K)
        ])
        for i in range(K):
            for j in range(i + 1, K):
                rows.append(dict(
                    model=mname, seed=seed,
                    class_i=class_names[i], class_j=class_names[j],
                    metric='centroid_dist',
                    value=float(np.linalg.norm(centroids[i] - centroids[j]))))
    return rows


# =============================================================================
# STATISTICAL TESTS
# =============================================================================
def cohen_d_paired(a, b):
    diff = np.asarray(b) - np.asarray(a)
    return float(diff.mean() / (diff.std(ddof=1) + 1e-10))


def run_wilcoxon_tests(all_results_df, backbone_names):
    metrics = ['accuracy', 'balanced_accuracy', 'f1_macro', 'qwk', 'mcc', 'auc_macro']
    rows    = []

    comparisons = [
        (BASELINE_ARM,    PRIMARY_ABI_ARM,   f'{PRIMARY_ABI_ARM}_vs_{BASELINE_ARM}'),
        ("A2_AbI_NoGate", PRIMARY_ABI_ARM,   'Gate_vs_NoGate'),
        ("A2_AbI_v1",     PRIMARY_ABI_ARM,   'Gate_vs_v1'),
        ("A2_AbI_v1",     "A2_AbI_NoGate",   'NoGate_vs_v1'),
    ]

    for bb in backbone_names:
        for arm1, arm2, comp_name in comparisons:
            tag1 = f"{bb}_{arm1}"
            tag2 = f"{bb}_{arm2}"
            sub1 = all_results_df[all_results_df['model'] == tag1].sort_values('seed')
            sub2 = all_results_df[all_results_df['model'] == tag2].sort_values('seed')
            if len(sub1) < 2 or len(sub2) < 2:
                continue
            for met in metrics:
                a, b = sub1[met].values, sub2[met].values
                if len(a) != len(b):
                    continue
                try:
                    stat, p = wilcoxon(a, b, alternative='two-sided')
                    cd = cohen_d_paired(a, b)
                    rows.append(dict(
                        backbone=bb, metric=met, comparison=comp_name,
                        arm1=arm1, arm2=arm2,
                        arm1_mean=round(float(a.mean()), 4),
                        arm2_mean=round(float(b.mean()), 4),
                        mean_gain=round(float((b - a).mean()), 5),
                        W=round(float(stat), 4), p_value=round(float(p), 6),
                        cohens_d=round(cd, 4), significant=bool(p < ALPHA),
                        interpretation=(
                            f'{arm2} significantly better' if p < ALPHA and cd > 0 else
                            f'{arm1} significantly better' if p < ALPHA and cd < 0 else
                            'No significant difference')))
                except Exception as exc:
                    rows.append(dict(backbone=bb, metric=met, comparison=comp_name,
                                     arm1=arm1, arm2=arm2, error=str(exc)))
    return pd.DataFrame(rows)


def _nemenyi_cd(k, n, alpha=0.05):
    q_table = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
               6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
    q = q_table.get(min(k, 10), 3.164)
    return q * np.sqrt(k * (k + 1) / (6 * n))


def run_friedman_nemenyi(all_results_df, backbone_names, save_dir):
    save_dir  = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ab_keys   = ABLATION_ARM_NAMES
    if len(ab_keys) < 3:
        print("  [Friedman+Nemenyi] Skipped: requires >=3 ablation arms.")
        return pd.DataFrame(), pd.DataFrame()

    metrics = ['accuracy', 'f1_macro', 'mcc', 'balanced_accuracy']
    friedman_rows, nemenyi_rows = [], []
    for bb in backbone_names:
        for met in metrics:
            mat, valid = [], []
            for ab in ab_keys:
                sub = all_results_df[
                    all_results_df['model'] == f"{bb}_{ab}"].sort_values('seed')
                if len(sub) == len(SEEDS):
                    mat.append(sub[met].values)
                    valid.append(ab)
            if len(mat) < 2:
                continue
            mat_np = np.array(mat)
            try:
                stat, p = friedmanchisquare(*[mat_np[i] for i in range(len(valid))])
            except Exception as exc:
                print(f"  Friedman {bb}/{met}: {exc}")
                continue
            n_s   = mat_np.shape[1]
            ranks = np.zeros(mat_np.shape[0])
            for s in range(n_s):
                ranks += rankdata(-mat_np[:, s])
            mean_ranks = ranks / n_s
            row = dict(backbone=bb, metric=met,
                       chi2=round(float(stat), 4),
                       p_value=round(float(p), 6),
                       significant=bool(p < ALPHA))
            for ab, mr in zip(valid, mean_ranks):
                row[f'rank_{ab}'] = round(float(mr), 3)
            friedman_rows.append(row)
            if p < ALPHA:
                cd = _nemenyi_cd(len(valid), n_s)
                for i, a_ in enumerate(valid):
                    for j, b_ in enumerate(valid):
                        if j <= i:
                            continue
                        diff = abs(mean_ranks[i] - mean_ranks[j])
                        nemenyi_rows.append(dict(
                            backbone=bb, metric=met,
                            ablation_A=a_, ablation_B=b_,
                            rank_A=round(float(mean_ranks[i]), 3),
                            rank_B=round(float(mean_ranks[j]), 3),
                            rank_diff=round(float(diff), 3),
                            cd=round(float(cd), 3),
                            significant=bool(diff > cd)))

    friedman_df = pd.DataFrame(friedman_rows)
    nemenyi_df  = pd.DataFrame(nemenyi_rows)
    friedman_df.to_csv(save_dir / 'friedman_test.csv', index=False)
    nemenyi_df.to_csv(save_dir / 'nemenyi_posthoc.csv', index=False)
    print(f"  [Friedman+Nemenyi] saved -> {save_dir}")
    return friedman_df, nemenyi_df


# =============================================================================
# ABLATION SUMMARY PLOTS
# =============================================================================
def plot_ablation_summary(all_results_df, backbone_names):
    ab_keys  = ABLATION_ARM_NAMES
    metrics  = ['accuracy', 'balanced_accuracy', 'f1_macro', 'mcc', 'auc_macro', 'qwk']
    mlabels  = ['Accuracy', 'Bal. Accuracy', 'F1 Macro', 'MCC', 'AUC Macro', 'QWK']
    pal      = [ABLATION_COLORS.get(k, '#888888') for k in ab_keys]

    for bb in backbone_names:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        for ax, met, mlab in zip(axes, metrics, mlabels):
            means, stds = [], []
            for ab in ab_keys:
                sub = all_results_df[all_results_df['model'] == f"{bb}_{ab}"]
                means.append(sub[met].values.mean() if not sub.empty else 0)
                stds.append(sub[met].values.std()  if not sub.empty else 0)
            bars = ax.bar(ab_keys, means, yerr=stds, capsize=5, color=pal,
                          edgecolor='black', linewidth=0.7, alpha=0.88)
            for bar, mu in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f'{mu:.3f}', ha='center', va='bottom',
                        fontsize=9, fontweight='bold')
            ax.set_title(mlab, fontweight='bold', fontsize=14)
            ax.set_ylabel('Score', fontsize=12)
            ax.set_xticklabels(ab_keys, rotation=25, ha='right', fontsize=9)
            valid_m = [m for m in means if m > 0]
            ax.set_ylim(0, min(1.12, max(valid_m) + 0.15) if valid_m else 1.0)
            ax.grid(axis='y', alpha=0.25)
        plt.suptitle(f'{bb} -- 4-Arm Ablation (mean±std, {len(SEEDS)} seeds)',
                     fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(SAVE_DIR / 'ablation' / f'{bb}_ablation.png',
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

    # Cross-backbone comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    x    = np.arange(len(backbone_names))
    w    = 0.8 / max(len(ab_keys), 1)
    bb_pal = plt.cm.Set1(np.linspace(0, 0.9, len(ab_keys)))
    for ax, met, mlab in zip(axes, metrics, mlabels):
        for ai, (ab, col) in enumerate(zip(ab_keys, bb_pal)):
            means, stds = [], []
            for bb in backbone_names:
                sub = all_results_df[all_results_df['model'] == f"{bb}_{ab}"]
                means.append(sub[met].values.mean() if not sub.empty else 0)
                stds.append(sub[met].values.std()  if not sub.empty else 0)
            ax.bar(x + ai * w, means, w, yerr=stds, capsize=3, color=col,
                   label=ab, alpha=0.82, edgecolor='black', linewidth=0.5)
        ax.set_xticks(x + w * len(ab_keys) / 2)
        ax.set_xticklabels(backbone_names, rotation=20, ha='right', fontsize=10)
        ax.set_title(mlab, fontweight='bold', fontsize=14)
        ax.set_ylabel('Score', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.25)
    plt.suptitle('Cross-Backbone 4-Arm Ablation (mean±std, 5 seeds)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(SAVE_DIR / 'ablation' / 'cross_backbone_ablation.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# PERFORMANCE COMPARISON PLOT
# =============================================================================
def plot_performance_comparison(all_results_df, backbone_names):
    metrics = ['accuracy', 'balanced_accuracy', 'f1_macro', 'qwk', 'mcc', 'auc_macro', 'ece']
    mlabels = ['Accuracy', 'Bal. Accuracy', 'F1 Macro', 'QWK', 'MCC', 'AUC Macro', 'ECE']

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for ax, met, mlab in zip(axes[:len(metrics)], metrics, mlabels):
        x = np.arange(len(backbone_names))
        w = 0.8 / len(ABLATION_ARM_NAMES)

        for i, ab in enumerate(ABLATION_ARM_NAMES):
            means, stds = [], []
            for bb in backbone_names:
                sub = all_results_df[all_results_df['model'] == f"{bb}_{ab}"]
                if not sub.empty:
                    means.append(sub[met].mean())
                    stds.append(sub[met].std())
                else:
                    means.append(0); stds.append(0)

            offset = (i - len(ABLATION_ARM_NAMES) / 2 + 0.5) * w
            ax.bar(x + offset, means, w, yerr=stds,
                   color=ABLATION_COLORS.get(ab, '#888888'),
                   label=ab, alpha=0.85, edgecolor='black',
                   linewidth=0.8, capsize=4)

        ax.set_xticks(x)
        ax.set_xticklabels(backbone_names, rotation=25, ha='right')
        ax.set_title(mlab, fontweight='bold', fontsize=14)
        ax.set_ylabel('Score', fontsize=12)
        ax.legend(fontsize=9, loc='best')
        ax.grid(axis='y', alpha=0.3)

    for ax in axes[len(metrics):]:
        ax.remove()

    plt.suptitle('4-Arm Ablation Performance Comparison\n(mean ± std, 5 seeds)',
                 fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(SAVE_DIR / 'reports' / 'performance_comparison_4arm.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# PUBLICATION TABLES
# =============================================================================
def generate_publication_tables(all_results_df, backbone_names,
                                  latent_df, progression_df,
                                  kw_all_df, dunn_all_df,
                                  bootstrap_df, desc_reg_df, repr_gain_df,
                                  retrieval_df, gradcam_df):
    rep     = SAVE_DIR / 'tables'
    rep.mkdir(parents=True, exist_ok=True)
    ab_keys = ABLATION_ARM_NAMES

    # Table A: Classification metrics
    cls_metrics = ['accuracy', 'balanced_accuracy', 'f1_macro', 'mcc', 'auc_macro', 'qwk']
    rows_a = []
    for bb in backbone_names:
        for ab in ab_keys:
            m   = f"{bb}_{ab}"
            sub = all_results_df[all_results_df['model'] == m]
            if sub.empty:
                continue
            row = {'Model': m, 'Backbone': bb, 'Ablation': ab}
            for met in cls_metrics:
                v = sub[met].values
                row[f'{met}_mean'] = round(float(v.mean()), 4)
                row[f'{met}_std']  = round(float(v.std()),  4)
            rows_a.append(row)
    pd.DataFrame(rows_a).to_csv(rep / 'tableA_classification.csv', index=False)

    # Table B: Calibration metrics
    cal_metrics = ['ece', 'brier_score']
    rows_b = []
    for bb in backbone_names:
        for ab in ab_keys:
            m   = f"{bb}_{ab}"
            sub = all_results_df[all_results_df['model'] == m]
            if sub.empty:
                continue
            row = {'Model': m, 'Backbone': bb, 'Ablation': ab}
            for met in cal_metrics:
                v = sub[met].values
                row[f'{met}_mean'] = round(float(v.mean()), 4)
                row[f'{met}_std']  = round(float(v.std()),  4)
            rows_b.append(row)
    pd.DataFrame(rows_b).to_csv(rep / 'tableB_calibration.csv', index=False)

    # Table C: Latent space metrics
    if not latent_df.empty:
        lat_metrics = ['silhouette', 'davies_bouldin', 'calinski_harabasz', 'knn_consistency']
        rows_c = []
        for bb in backbone_names:
            for ab in ab_keys:
                m   = f"{bb}_{ab}"
                sub = latent_df[latent_df['model'] == m]
                if sub.empty:
                    continue
                row = {'Model': m, 'Backbone': bb, 'Ablation': ab}
                for met in lat_metrics:
                    v = sub[met].values
                    row[f'{met}_mean'] = round(float(np.nanmean(v)), 4)
                    row[f'{met}_std']  = round(float(np.nanstd(v)),  4)
                rows_c.append(row)
        pd.DataFrame(rows_c).to_csv(rep / 'tableC_latent.csv', index=False)

    # Table D: Progression metrics
    if not progression_df.empty:
        rows_d = []
        for (model, ci, cj, met), grp in progression_df.groupby(
                ['model', 'class_i', 'class_j', 'metric']):
            rows_d.append(dict(
                model=model, class_i=ci, class_j=cj, metric=met,
                mean=round(float(grp['value'].mean()), 4),
                std=round(float(grp['value'].std()),  4)))
        pd.DataFrame(rows_d).to_csv(rep / 'tableD_progression.csv', index=False)

    # Table E: Kruskal-Wallis
    if not kw_all_df.empty:
        rows_e = []
        for (model, dname), grp in kw_all_df.groupby(['model', 'descriptor']):
            rows_e.append(dict(
                model=model, descriptor=dname,
                kw_stat_mean=round(float(grp['kw_stat'].mean()), 4),
                kw_stat_std=round(float(grp['kw_stat'].std()),   4),
                significant_count=int(grp['significant'].sum())))
        pd.DataFrame(rows_e).to_csv(rep / 'tableE_kruskal.csv', index=False)

    # Table F: GradCAM metrics
    if gradcam_df is not None and not gradcam_df.empty:
        qm_cols = ['activation_area', 'activation_entropy', 'contour_complexity']
        rows_f  = []
        for (model, true_cls), grp in gradcam_df.groupby(['model', 'true']):
            row = {'Model': model, 'Class': true_cls}
            for met in qm_cols:
                if met in grp.columns:
                    row[f'{met}_mean'] = round(float(grp[met].mean()), 4)
                    row[f'{met}_std']  = round(float(grp[met].std()),  4)
            rows_f.append(row)
        pd.DataFrame(rows_f).to_csv(rep / 'tableF_gradcam.csv', index=False)

    # Table G: Retrieval metrics
    if retrieval_df is not None and not retrieval_df.empty:
        retrieval_df.to_csv(rep / 'tableG_retrieval.csv', index=False)

    # Table H: Bootstrap CI
    if not bootstrap_df.empty:
        bootstrap_df.to_csv(rep / 'tableH_bootstrap_ci.csv', index=False)

    print(f"  [Tables A-H] saved -> {rep}")


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================
def run_one_seed(seed, model_tag, backbone_name, ab_name, ab_flags,
                 num_classes, class_names, train_loader, val_loader, test_loader):
    print(f"    seed={seed} | {model_tag}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model_kwargs = ab_flags['model'].copy()
    model = AbIModel(backbone_name, num_classes,
                     pretrained=True, **model_kwargs).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS)

    tgts = [lbl for _, lbl in train_loader.dataset.samples]
    cw   = compute_class_weight('balanced',
                                classes=np.arange(num_classes), y=tgts)
    cw   = cw / cw.mean()
    crit  = FocalLoss(torch.tensor(cw, dtype=torch.float).to(DEVICE),
                      FOCAL_GAMMA, LABEL_SMOOTH)
    vcrit = nn.CrossEntropyLoss()

    best_va    = 0.0
    best_state = None
    patience   = 0
    history    = defaultdict(list)

    train_flags = ab_flags['train']

    for epoch in range(EPOCHS):
        tl, ta = train_epoch(model, train_loader, optimizer, crit, DEVICE,
                             train_flags, epoch, num_classes)
        vl, va = validate_epoch(model, val_loader, vcrit, DEVICE)
        scheduler.step()
        for k, v in zip(['train_loss', 'train_acc', 'val_loss', 'val_acc'],
                         [tl, ta, vl, va]):
            history[k].append(v)
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f"      E{epoch+1:3d}/{EPOCHS} | tr={ta:.1f}%  va={va:.1f}%")
        if va > best_va:
            best_va, best_state, patience = va, deepcopy(model.state_dict()), 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"      Early stop @ E{epoch+1}")
                break
        if epoch % 15 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    model.load_state_dict(best_state)
    model.eval()

    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits, _ = model(x)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            all_probs.extend(F.softmax(logits, dim=1).cpu().numpy())

    all_preds   = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs   = np.array(all_probs)

    acc     = float(accuracy_score(all_targets, all_preds))
    bal_acc = float(balanced_accuracy_score(all_targets, all_preds))
    prec    = float(precision_score(all_targets, all_preds,
                                    average='macro', zero_division=0))
    rec     = float(recall_score(all_targets, all_preds,
                                  average='macro', zero_division=0))
    f1_mac  = float(f1_score(all_targets, all_preds,
                              average='macro', zero_division=0))
    f1_w    = float(f1_score(all_targets, all_preds,
                              average='weighted', zero_division=0))
    mcc_s   = float(matthews_corrcoef(all_targets, all_preds))
    qwk_s   = float(cohen_kappa_score(all_targets, all_preds, weights='quadratic'))
    try:
        auc_m = float(roc_auc_score(all_targets, all_probs,
                                     multi_class='ovr', average='macro'))
    except Exception:
        auc_m = 0.0
    cal = calibration_metrics(all_targets, all_probs)

    pd.DataFrame(history).to_csv(
        SAVE_DIR / 'histories' / f"{model_tag}_seed{seed}.csv", index=False)
    plot_training_curves(dict(history), model_tag, seed)
    plot_confusion_matrix(confusion_matrix(all_targets, all_preds),
                          class_names, model_tag, seed)
    plot_roc_curves(all_targets, all_probs, class_names, model_tag, seed)
    plot_reliability_diagram(all_targets, all_probs, class_names, model_tag, seed)

    result = dict(
        seed=seed, model=model_tag, backbone=backbone_name, ablation=ab_name,
        use_abi=model_kwargs.get('use_abi', False),
        accuracy=acc, balanced_accuracy=bal_acc,
        precision_macro=prec, recall_macro=rec,
        f1_macro=f1_mac, f1_weighted=f1_w, mcc=mcc_s,
        auc_macro=auc_m, qwk=qwk_s,
        ece=cal['ece'], brier_score=cal['brier_score'],
        best_val_acc=best_va,
    )
    print(f"      ACC={acc:.4f}  BalACC={bal_acc:.4f}  F1={f1_mac:.4f}  "
          f"QWK={qwk_s:.4f}  ECE={cal['ece']:.4f}")
    return result, dict(history), all_preds, all_targets, all_probs, model


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    start_time = time.time()

    print("\n" + "=" * 72)
    print("  Ab-I V2 ENHANCED -- Complete Publication-Ready Code")
    print(f"  Backbones : {BACKBONE_NAMES}")
    print(f"  Ablations : {ABLATION_ARM_NAMES}")
    print(f"  Seeds     : {SEEDS}")
    print(f"  Device    : {DEVICE}")
    print(f"  Output    : {SAVE_DIR}")
    print("=" * 72 + "\n")

    if not TRAIN_PATH.exists():
        print(f"  ERROR: {TRAIN_PATH} not found.")
        exit(1)

    # --- Data Transforms ---
    train_tf = v2.Compose([
        v2.Lambda(lambda img: img.convert("RGB")),
        v2.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(15),
        v2.RandomAffine(degrees=0, translate=(0.05, 0.05), shear=6),
        v2.ColorJitter(brightness=0.15, contrast=0.15),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = v2.Compose([
        v2.Lambda(lambda img: img.convert("RGB")),
        v2.Resize((IMG_SIZE, IMG_SIZE)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds   = datasets.ImageFolder(str(TRAIN_PATH), transform=train_tf)
    val_ds     = datasets.ImageFolder(str(VAL_PATH),   transform=val_tf)
    test_ds_raw = datasets.ImageFolder(str(TEST_PATH), transform=val_tf)
    num_classes = len(train_ds.classes)
    class_names = train_ds.classes
    print(f"  Classes ({num_classes}): {class_names}")
    print(f"  Train:{len(train_ds)}  Val:{len(val_ds)}  Test:{len(test_ds_raw)}\n")

    # --- Visualization subsets ---
    viz_subsets = {}
    for seed in SEEDS:
        viz_subsets[seed] = build_visualization_subset(
            test_ds_raw, class_names, seed, n_total=VIZ_SUBSET_N)

    # --- Accumulators ---
    all_results      = []
    latent_rows_all  = []
    bootstrap_rows   = []
    kw_rows_all      = []
    dunn_rows_all    = []
    desc_energy_rows = []
    progression_rows = []
    retrieval_rows   = []
    compute_rows     = []
    gradcam_records  = []
    all_preds_dict   = {}
    model_store      = {}
    emb_store        = {}
    rep_seed_store   = {}

    # --- MAIN TRAINING LOOP ---
    for bb_name in BACKBONE_NAMES:
        for ab_name, ab_flags in ABLATION_VARIANTS.items():
            model_tag    = f"{bb_name}_{ab_name}"
            seed_results = []

            print(f"\n{'#' * 68}")
            print(f"  BACKBONE: {bb_name}  |  ABLATION: {ab_name}")
            print(f"{'#' * 68}")

            for seed in SEEDS:
                g = torch.Generator()
                g.manual_seed(seed)
                train_loader = DataLoader(train_ds, BATCH, shuffle=True,
                                          num_workers=4, generator=g,
                                          pin_memory=True)
                val_loader   = DataLoader(val_ds, BATCH, shuffle=False,
                                          num_workers=2, pin_memory=True)
                test_loader  = DataLoader(test_ds_raw, 1, shuffle=False,
                                          num_workers=2, pin_memory=True)

                (result, history, preds, targets,
                 probs, trained_model) = run_one_seed(
                    seed, model_tag, bb_name, ab_name, ab_flags,
                    num_classes, class_names,
                    train_loader, val_loader, test_loader)

                all_results.append(result)
                seed_results.append(result)
                all_preds_dict[model_tag] = preds
                model_store[(bb_name, ab_name, seed)] = trained_model

                # Bootstrap CI
                bci = bootstrap_ci(preds, targets)
                bci.update(dict(model=model_tag, backbone=bb_name,
                                ablation=ab_name, seed=seed))
                bootstrap_rows.append(bci)

                # Latent embeddings
                print(f"  [Latent] {model_tag} seed{seed} ...")
                emb_loader = DataLoader(test_ds_raw, 16, shuffle=False,
                                        num_workers=2, pin_memory=True)
                emb, lbl = extract_embeddings(trained_model, emb_loader, DEVICE)
                lq = latent_quality_metrics(emb, lbl)
                lq.update(dict(model=model_tag, backbone=bb_name,
                               ablation=ab_name, seed=seed))
                latent_rows_all.append(lq)
                emb_store[(bb_name, ab_name, seed)] = (emb, lbl)
                print(f"    Sil={lq['silhouette']:.4f}  "
                      f"DB={lq['davies_bouldin']:.4f}  "
                      f"KNN={lq['knn_consistency']:.4f}")

                # Retrieval analysis
                retrieval_rows.extend(
                    retrieval_analysis(emb, lbl, class_names, model_tag,
                                       K_list=(1, 5)))

                # Ab-I Validation
                print(f"  [Ab-I Validation] {model_tag} seed{seed} ...")
                abi_loader = DataLoader(test_ds_raw, 1, shuffle=False,
                                        num_workers=2, pin_memory=True)
                kw_df_s, desc_rows_s = run_abi_validation_one_seed(
                    trained_model, abi_loader, class_names,
                    model_tag, seed, SAVE_DIR / 'descriptor_analysis')
                if not kw_df_s.empty:
                    kw_rows_all.extend(kw_df_s.to_dict('records'))
                desc_energy_rows.extend(desc_rows_s)

                # Dunn post-hoc
                print(f"  [Dunn] {model_tag} seed{seed} ...")
                abi_loader2 = DataLoader(test_ds_raw, 1, shuffle=False,
                                         num_workers=2, pin_memory=True)
                dunn_rows = run_dunns_posthoc_abi(
                    trained_model, abi_loader2, class_names,
                    model_tag, seed, SAVE_DIR / 'statistical_tests')
                dunn_rows_all.extend(dunn_rows)

                # Progression
                pair = {model_tag: emb}
                prog_rows = analyze_disease_progression(
                    pair, lbl, class_names, seed,
                    SAVE_DIR / 'progression' / bb_name)
                progression_rows.extend(prog_rows)

                # GradCAM (all seeds)
                print(f"  [GradCAM] {model_tag} seed{seed} ...")
                test_loader_grad = DataLoader(test_ds_raw, 1, shuffle=False,
                                               num_workers=2, pin_memory=True)
                cam_recs = run_full_gradcam_per_seed(
                    trained_model, model_tag, seed, test_loader_grad,
                    preds, targets, class_names)
                gradcam_records.extend(cam_recs)

                # Compute profile (last seed only)
                if seed == SEEDS[-1]:
                    print(f"  [Profile] {model_tag} ...")
                    cp = profile_model(trained_model, DEVICE)
                    cp['model'] = model_tag
                    compute_rows.append(cp)

                del trained_model
                torch.cuda.empty_cache()
                gc.collect()

            # Seed summary
            vdf = pd.DataFrame(seed_results)
            print(f"\n  [{model_tag}] 5-seed Summary:")
            for met in ['accuracy', 'balanced_accuracy', 'f1_macro', 'mcc', 'qwk']:
                v = vdf[met].values
                print(f"    {met:<22}  {v.mean():.4f} ± {v.std():.4f}")

    # --- GRADCAM COMPARISON ACROSS ALL 4 MODULES ---
    print("\n" + "=" * 68)
    print("  GRADCAM COMPARISON ACROSS ALL 4 ABLATION MODULES")
    print("=" * 68)

    for bb_name in BACKBONE_NAMES:
        for seed in SEEDS:
            models_dict = {}
            preds_dict  = {}
            for ab_name in ABLATION_ARM_NAMES:
                key = (bb_name, ab_name, seed)
                if key in model_store:
                    models_dict[ab_name] = model_store[key]
                    preds_dict[ab_name]  = all_preds_dict.get(
                        f"{bb_name}_{ab_name}", None)

            if len(models_dict) == len(ABLATION_ARM_NAMES):
                test_loader_grad = DataLoader(test_ds_raw, 1, shuffle=False,
                                               num_workers=2, pin_memory=True)
                all_targets_ref = np.array(
                    [lbl for _, lbl in test_ds_raw.samples])
                run_gradcam_comparison_all_modules(
                    models_dict, test_loader_grad, viz_subsets[seed],
                    preds_dict, all_targets_ref, class_names, seed, bb_name)
            else:
                print(f"  [GradCAM-Cmp] Skipping {bb_name} seed{seed}: "
                      f"missing models")

    # --- LATENT SPACE PROJECTIONS (representative seed) ---
    print("\n  [Latent Projections] ...")
    for bb_name in BACKBONE_NAMES:
        rep_seed = SEEDS[len(SEEDS) // 2]
        emb_dict = {}
        for ab_name in ABLATION_ARM_NAMES:
            key = (bb_name, ab_name, rep_seed)
            if key in emb_store:
                emb, _ = emb_store[key]
                emb_dict[ab_name] = emb
        if emb_dict:
            _, lbl_ref = emb_store.get(
                (bb_name, ABLATION_ARM_NAMES[0], rep_seed), (None, None))
            if lbl_ref is not None:
                plot_2d_projections(
                    emb_dict, lbl_ref, class_names, rep_seed,
                    SAVE_DIR / 'latent_space' / bb_name)

    # --- BUILD DATAFRAMES ---
    all_results_df  = pd.DataFrame(all_results)
    latent_df       = pd.DataFrame(latent_rows_all)
    bootstrap_df    = pd.DataFrame(bootstrap_rows)
    kw_all_df       = pd.DataFrame(kw_rows_all)
    dunn_all_df     = pd.DataFrame(dunn_rows_all)
    desc_energy_df  = pd.DataFrame(desc_energy_rows)
    progression_df  = pd.DataFrame(progression_rows)
    retrieval_df    = pd.DataFrame(retrieval_rows)
    compute_df      = pd.DataFrame(compute_rows)
    gradcam_df      = pd.DataFrame(gradcam_records)

    # Save all DataFrames
    rep_dir = SAVE_DIR / 'reports'
    all_results_df.to_csv(rep_dir / 'classification_all_seeds.csv',    index=False)
    latent_df.to_csv(rep_dir / 'latent_metrics_all_seeds.csv',         index=False)
    bootstrap_df.to_csv(rep_dir / 'bootstrap_results.csv',             index=False)
    kw_all_df.to_csv(rep_dir / 'kw_abi_validation.csv',                index=False)
    dunn_all_df.to_csv(rep_dir / 'dunn_posthoc.csv',                   index=False)
    desc_energy_df.to_csv(rep_dir / 'descriptor_energy_all_seeds.csv', index=False)
    progression_df.to_csv(rep_dir / 'progression_metrics.csv',         index=False)
    retrieval_df.to_csv(rep_dir / 'retrieval_metrics.csv',             index=False)
    compute_df.to_csv(rep_dir / 'compute_profile.csv',                 index=False)
    if not gradcam_df.empty:
        gradcam_df.to_csv(rep_dir / 'gradcam_quantitative.csv',        index=False)

    # --- STATISTICAL TESTS ---
    print("\n  [Statistical Tests] ...")
    wilcoxon_df = run_wilcoxon_tests(all_results_df, BACKBONE_NAMES)
    wilcoxon_df.to_csv(rep_dir / 'wilcoxon_comparisons.csv', index=False)

    friedman_df, nemenyi_df = run_friedman_nemenyi(
        all_results_df, BACKBONE_NAMES, SAVE_DIR / 'statistical_tests')

    # --- PLOTS ---
    print("\n  [Ablation Summary Plots] ...")
    plot_ablation_summary(all_results_df, BACKBONE_NAMES)
    plot_performance_comparison(all_results_df, BACKBONE_NAMES)

    # --- PUBLICATION TABLES ---
    print("\n  [Publication Tables A-H] ...")
    generate_publication_tables(
        all_results_df, BACKBONE_NAMES,
        latent_df, progression_df,
        kw_all_df, dunn_all_df,
        bootstrap_df, None, None,
        retrieval_df, gradcam_df)

    # --- FINAL SUMMARY ---
    elapsed = time.time() - start_time
    print("\n" + "=" * 72)
    print("  AB-I V2 ENHANCED COMPLETE")
    print(f"  Total time : {elapsed / 60:.1f} minutes")
    print(f"  All outputs: {SAVE_DIR}")
    print("=" * 72 + "\n")

    print("\n  FINAL PERFORMANCE SUMMARY (mean ± std, 5 seeds):")
    for bb_name in BACKBONE_NAMES:
        print(f"\n  {bb_name}:")
        for ab_name in ABLATION_ARM_NAMES:
            m   = f"{bb_name}_{ab_name}"
            sub = all_results_df[all_results_df['model'] == m]
            if not sub.empty:
                acc = sub['accuracy'].values
                f1  = sub['f1_macro'].values
                print(f"    {ab_name:<20} "
                      f"ACC={acc.mean():.4f}±{acc.std():.4f}  "
                      f"F1={f1.mean():.4f}±{f1.std():.4f}")
