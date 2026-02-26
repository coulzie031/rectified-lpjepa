# Comparison: LeJEPA (Antoine) vs Rectified LpJEPA

## Overview

Both projects implement a self-supervised learning method on medical images (MedMNIST). Antoine's repo (**LeJEPA**) is the original version. Rectified LpJEPA is a **theoretical extension** that introduces a richer target distribution and a modified architecture to enforce sparsity in the learned representations.

---

## 1. The Loss Function — Core Change

This is the most important difference between the two projects. Everything else follows from it.

### LeJEPA (Antoine) — SIGReg
Antoine uses a loss called **SIGReg** (Statistical Independence with Gaussian Regularization), imported from the external `lejepa` library:

```python
from src.losses.sigreg import invariance_loss
sig = sigreg(flat_proj)
loss = cfg["lamb"] * sig + (1 - cfg["lamb"]) * inv
```

SIGReg is a **univariate statistical test** (Epps-Pulley test) applied along random slices. It compares the feature distribution to a **Gaussian**, measuring whether each dimension is independent. It is a **one-sample** test: the features are compared against a fixed theoretical distribution.

### Rectified LpJEPA — RDMReg
The Rectified version replaces SIGReg with **RDMReg** (Rectified Distribution Matching Regularization), implemented from scratch in `src/losses/rdmreg.py`:

```python
from src.losses.rdmreg import rectified_lpjepa_loss
total, inv, reg = rectified_lpjepa_loss(proj, lamb_inv=..., lamb_rdm=..., p=p, mu=mu, ...)
```

RDMReg is a **Sliced 2-Wasserstein distance** between the features and samples drawn from a target distribution called the **RGG** (Rectified Generalized Gaussian). It is a **two-sample** test: samples are explicitly drawn from the target distribution and compared to the learned features.

### Why this change?

| | SIGReg (LeJEPA) | RDMReg (Rectified LpJEPA) |
|---|---|---|
| Target distribution | Gaussian | RGG = ReLU(GN_p(mu, sigma)) |
| Test type | One-sample (statistical) | Two-sample (Wasserstein) |
| Sparsity | Implicit (dense by default) | Explicit and controllable |
| Parameters | `lamb`, `knots`, `num_slices` | `lamb_inv`, `lamb_rdm`, `p`, `mu` |
| External dependency | `lejepa` (pip) | 100% from scratch |

The **RGG** is a generalized Gaussian rectified by a ReLU. It lives in the positive orthant (all values ≥ 0) and naturally produces sparse features: a fraction of dimensions is exactly 0, controlled by `mu`. The more negative `mu` is, the sparser the features.

---

## 2. Projector Architecture — Added Final ReLU

### LeJEPA (Antoine)
The projector is a standard MLP from `torchvision` with no final activation:
```python
from torchvision.ops import MLP
self.proj = MLP(embed_dim, [mid_dim, proj_dim], norm_layer=nn.BatchNorm1d)
# Output: values in ]-∞, +∞[
```
Features can be negative or positive — their distribution has no sign constraint.

### Rectified LpJEPA
The projector is a custom `RectifiedMLP` with a **mandatory final ReLU**:
```python
self.net = nn.Sequential(
    nn.Linear(...), nn.BatchNorm1d(...), nn.ReLU(),
    nn.Linear(...), nn.BatchNorm1d(...), nn.ReLU(),
    nn.Linear(...),
    nn.ReLU()  # ← key innovation
)
# Output: values in [0, +∞[
```

This final ReLU is **not trivial**: it guarantees that features are non-negative, which is essential for the distribution matching with the RGG to be theoretically sound. Without this ReLU, the method would be mathematically incorrect since the RGG only has support on positive values.

Additionally, the Rectified LpJEPA projector is deeper: **3 layers** (vs 2 in Antoine's), with `proj_hidden_dim` configurable independently from `proj_dim`.

---

## 3. Optimizer — AdamW vs LARS

### LeJEPA (Antoine)
Uses **AdamW**, a standard optimizer:
```python
opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=5e-2)
```
AdamW is simple and robust, well-suited for small batch sizes.

### Rectified LpJEPA
Uses **LARS** (Layer-wise Adaptive Rate Scaling), a custom optimizer implemented from scratch:
```python
opt = LARS(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], eta=0.001)
```
LARS adapts the learning rate **layer by layer** based on the ratio between weight norms and gradient norms. It is designed for large batch sizes (≥ 256) and is standard in modern self-supervised learning methods (SimCLR, VICReg, etc.).

---

## 4. Target Distribution — Gaussian vs RGG

This is the theoretical core of the difference.

**LeJEPA** aligns features to a **standard Gaussian** via SIGReg. The learned representations are dense: all dimensions carry information.

**Rectified LpJEPA** aligns features to an **RGG** parameterized by:
- `p`: the generalized Gaussian parameter (p=1 → Laplace, p=2 → Gaussian)
- `mu`: the mean shift (controls sparsity)
- `sigma`: the scale (automatically computed for unit variance)

With default parameters (p=1.0, mu=0.0), roughly **50% of dimensions** will be zero. With mu=-2.0, **98% of dimensions** will be zero. This makes representations more compact and potentially more interpretable.

---

## 5. Hyperparameters — configs/default.yaml

| Parameter | LeJEPA (Antoine) | Rectified LpJEPA |
|---|---|---|
| `proj_dim` | 64 | 512 |
| `proj_hidden_dim` | — | 512 |
| `epochs` | 1 (test) | 200 |
| `lr` | 0.002 | 0.03 |
| `lamb` | 0.02 | — |
| `lamb_inv` | — | 25.0 |
| `lamb_rdm` | — | 125.0 |
| `p` | — | 1.0 |
| `mu` | — | 0.0 |
| `num_projections` | 1024 | 8192 |
| `warmup_epochs` | — | 10 |
| `dataset` | OrganA+OrganB+OrganT | OrganA+OrganB+OrganT |

`proj_dim` goes from 64 to **512** — representations are 8× larger, giving the model more capacity to encode information. `num_projections` goes from 1024 to **8192** for a more precise Wasserstein test (at the cost of slower training).

---

## 6. Training Metrics

### LeJEPA (Antoine)
Displays only: `loss` (combination of inv + sig)

### Rectified LpJEPA
Additionally displays:
- **`inv`**: invariance loss alone
- **`rdm`**: RDMReg loss alone
- **`l0`**: fraction of exactly-zero dimensions (strict sparsity)
- **`l1`**: L1/L2 ratio (global sparsity measure)

These metrics allow direct monitoring of whether the target sparsity is being achieved throughout training.

---

## Summary in One Sentence

LeJEPA learns dense representations aligned to a Gaussian via a statistical test; Rectified LpJEPA learns **sparse** representations aligned to a rectified Laplace distribution via a Wasserstein distance, with a final ReLU in the projector to ensure theoretical correctness.
