"""
rdmreg.py — Rectified Distribution Matching Regularization (RDMReg)

Replaces SIGReg (LeJEPA) with a two-sample Sliced 2-Wasserstein test
against the Rectified Generalized Gaussian (RGG) distribution.

Key differences vs SIGReg:
- One-sample (SIGReg) → Two-sample (RDMReg): we explicitly sample from the RGG target
- Gaussian target (SIGReg) → RGG target (RDMReg): encodes explicit L0 sparsity
- Epps-Pulley char. function test → Sliced Wasserstein distance

Reference: Rectified LpJEPA (Kuang et al., 2026) — arXiv:2602.01456
"""

import math
import torch
from torch.distributions.laplace import Laplace


# =============================================================================
# 1. Sigma determination
# =============================================================================

def determine_sigma_gn(p: float) -> float:
    """
    Compute sigma such that GN_p(0, sigma) has unit variance *before* rectification.
        sigma = sqrt(Gamma(1/p)) / (p^(1/p) * sqrt(Gamma(3/p)))

    This is the default mode (sigma_GN) used in the paper.
    """
    return math.sqrt(math.gamma(1.0 / p)) / (p ** (1.0 / p) * math.sqrt(math.gamma(3.0 / p)))


# =============================================================================
# 2. Generalized Gaussian (GN_p) sampler
# =============================================================================

def sample_generalized_gaussian(
    shape,
    p: float,
    loc: float = 0.0,
    scale: float = 1.0,
    device: str = "cpu",
    dtype=torch.float32,
) -> torch.Tensor:
    """
    Sample from the Generalized Gaussian GN_p(loc, scale).

    PDF ∝ exp(-|x - loc|^p / (p * scale^p))
    - p = 1 → Laplace  (fast path via torch.distributions.Laplace)
    - p = 2 → Gaussian (fast path via torch.randn)
    - else  → general  (via Gamma trick, slower)
    """
    if p == 1.0:
        dist = Laplace(
            loc=torch.tensor(loc, device=device, dtype=dtype),
            scale=torch.tensor(scale, device=device, dtype=dtype),
        )
        return dist.sample(shape)

    elif p == 2.0:
        return loc + scale * torch.randn(shape, device=device, dtype=dtype)

    else:
        # General case: sign * (p * Gamma(1/p))^(1/p), then affine shift
        sign = 2.0 * torch.empty(shape, device=device, dtype=dtype).bernoulli_(0.5) - 1.0
        gamma_dist = torch.distributions.Gamma(concentration=1.0 / p, rate=1.0)
        g = gamma_dist.sample(shape).to(device=device, dtype=dtype)
        x = sign * (p * g).pow(1.0 / p)
        return loc + scale * x


# =============================================================================
# 3. Rectified Generalized Gaussian (RGG) sampler
# =============================================================================

def sample_rgg(
    shape,
    p: float = 1.0,
    mu: float = 0.0,
    sigma: float = None,
    device: str = "cpu",
    dtype=torch.float32,
) -> torch.Tensor:
    """
    Sample from the Rectified Generalized Gaussian: z = ReLU(GN_p(mu, sigma)).

    The RGG is the key target distribution of Rectified LpJEPA. It:
    - Encodes explicit L0 sparsity via the Dirac mass at 0
    - Preserves max-entropy guarantees (rescaled by Rényi information dimension)
    - Allows direct control of expected L0 norm via {mu, sigma, p}

    Args:
        shape:  (B, D) shape of samples to draw
        p:      Lp norm parameter (1.0 = Rectified Laplace [default], 2.0 = Rectified Gaussian)
        mu:     mean shift — lower mu → more zeros → more sparse
                  mu=0    → ~50% sparsity (symmetric around 0)
                  mu=-1   → ~84% sparsity
                  mu=-2   → ~98% sparsity
        sigma:  scale parameter. If None, uses determine_sigma_gn(p) for unit-variance GN_p.
        device: torch device string
        dtype:  torch dtype
    """
    if sigma is None:
        sigma = determine_sigma_gn(p)
    gn_samples = sample_generalized_gaussian(shape, p=p, loc=mu, scale=sigma,
                                              device=device, dtype=dtype)
    return torch.relu(gn_samples)


# =============================================================================
# 4. Invariance Loss (multi-view, unchanged from LeJEPA)
# =============================================================================

def invariance_loss(proj: torch.Tensor) -> torch.Tensor:
    """
    Multi-view invariance loss: MSE between each view and the mean of all views.

    Args:
        proj: (V, B, D) — V views, batch size B, feature dim D
    """
    mean_proj = proj.mean(dim=0, keepdim=True)   # (1, B, D)
    return (mean_proj - proj).square().mean()


# =============================================================================
# 5. RDMReg Loss — Sliced 2-Wasserstein against RGG
# =============================================================================

def _rdmreg_single_view(
    z: torch.Tensor,
    p: float,
    mu: float,
    sigma: float,
    num_projections: int,
) -> torch.Tensor:
    """
    Sliced 2-Wasserstein distance between projected features z and RGG samples.

    Algorithm (from the paper):
        1. Sample N random unit vectors c_i from the L2 sphere
        2. Project z and RGG samples onto each c_i  → 1D distributions
        3. Sort both 1D projections along batch dim
        4. MSE between sorted projections = W2^2 for that direction
        5. Average over all N directions

    Args:
        z:               (B, D) ReLU-activated features for ONE view
        p, mu, sigma:    RGG distribution parameters
        num_projections: number of random projection directions (default 8192)
    """
    B, D = z.shape
    device, dtype = z.device, z.dtype

    # Step 1 — sample RGG target (same batch size as features for two-sample test)
    target = sample_rgg((B, D), p=p, mu=mu, sigma=sigma, device=device, dtype=dtype)

    # Step 2 — random projections on the unit L2 sphere
    projs = torch.randn(num_projections, D, device=device, dtype=dtype)
    projs = projs / projs.norm(dim=1, keepdim=True)   # (N, D)

    # Step 3 — project: (B, N)
    proj_z = z @ projs.T
    proj_t = target @ projs.T

    # Step 4 — sort along batch dimension
    proj_z_sorted, _ = torch.sort(proj_z, dim=0)   # (B, N)
    proj_t_sorted, _ = torch.sort(proj_t, dim=0)   # (B, N)

    # Step 5 — Sliced W2^2
    return torch.mean((proj_z_sorted - proj_t_sorted) ** 2)


def rdmreg_loss(
    proj: torch.Tensor,
    p: float = 1.0,
    mu: float = 0.0,
    sigma: float = None,
    num_projections: int = 8192,
) -> torch.Tensor:
    """
    RDMReg loss averaged over all V views.

    Args:
        proj:            (V, B, D) ReLU-activated projected features (all views)
        p:               Lp norm parameter for the GG distribution
        mu:              mean shift (controls L0 sparsity of target)
        sigma:           scale parameter (None → auto unit-variance)
        num_projections: number of random Cramér-Wold projection directions
    """
    if sigma is None:
        sigma = determine_sigma_gn(p)

    V = proj.shape[0]
    view_losses = torch.stack([
        _rdmreg_single_view(proj[v], p=p, mu=mu, sigma=sigma,
                            num_projections=num_projections)
        for v in range(V)
    ])
    return view_losses.mean()


# =============================================================================
# 6. Full Rectified LpJEPA Loss
# =============================================================================

def rectified_lpjepa_loss(
    proj: torch.Tensor,
    lamb_inv: float = 25.0,
    lamb_rdm: float = 125.0,
    p: float = 1.0,
    mu: float = 0.0,
    sigma: float = None,
    num_projections: int = 8192,
):
    """
    Full Rectified LpJEPA objective:

        L = lamb_inv * L_invariance + lamb_rdm * L_RDMReg

    Where:
        - L_invariance: pulls views of the same sample together (feature alignment)
        - L_RDMReg:     pushes feature distribution towards RGG (sparsity + max-entropy)

    Args:
        proj:            (V, B, D) ReLU-activated projected features
        lamb_inv:        invariance loss weight (default 25.0)
        lamb_rdm:        RDMReg loss weight     (default 125.0)
        p:               Lp norm parameter (1.0 = Rectified Laplace)
        mu:              mean shift (0.0 = ~50% sparsity, -2.0 = ~98% sparsity)
        sigma:           scale parameter (None = auto)
        num_projections: number of random projections for Cramér-Wold

    Returns:
        (total_loss, inv_loss, reg_loss)
    """
    inv = invariance_loss(proj)
    reg = rdmreg_loss(proj, p=p, mu=mu, sigma=sigma, num_projections=num_projections)
    total = lamb_inv * inv + lamb_rdm * reg
    return total, inv, reg


# =============================================================================
# 7. Sparsity Metrics
# =============================================================================

@torch.no_grad()
def l0_sparsity(z: torch.Tensor, eps: float = 1e-8) -> float:
    """
    L0 sparsity: fraction of feature dimensions that are (near-)zero per sample.
    Returns value in [0, 1]: 1.0 = all zeros (complete sparsity).
    """
    return (z.abs() < eps).float().mean().item()


@torch.no_grad()
def l1_sparsity(z: torch.Tensor, eps: float = 1e-12) -> float:
    """
    L1/L2 sparsity ratio per sample: lower = sparser relative to L2 norm.
    (1/D) * (||z||_1 / ||z||_2)^2  in [0, 1]
    """
    D = z.shape[1]
    l1_norms = torch.linalg.norm(z, ord=1, dim=1)
    l2_norms = torch.linalg.norm(z, ord=2, dim=1)
    return ((1.0 / D) * (l1_norms / (l2_norms + eps)) ** 2).mean().item()
