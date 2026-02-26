"""
encoder.py — Rectified LpJEPA Encoder

Key change vs LeJEPA: the MLP projector ends with a ReLU activation.
This is a deliberate architectural choice — NOT just a nonlinearity:
    z = ReLU(MLP(backbone(x)))
It ensures features are non-negative, matching the RGG target distribution
(which lives in the positive orthant). Without this ReLU, the method is incorrect.

Reference: Rectified LpJEPA (Kuang et al., 2026) — Section 4, arXiv:2602.01456
"""

import torch
import torch.nn as nn
import timm


# =============================================================================
# Backbone: TinyCNN
# =============================================================================

class TinyCNN(nn.Module):
    """Lightweight CNN encoder for quick MedMNIST experiments (3-layer conv)."""

    def __init__(self, in_chans: int = 3, embed_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_chans, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, embed_dim)
        self.num_features = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).flatten(1))


# =============================================================================
# Projector: RectifiedMLP
# =============================================================================

class RectifiedMLP(nn.Module):
    """
    3-layer MLP projector with MANDATORY final ReLU.

    Architecture:
        Linear(in) → BN → ReLU
        Linear(hid) → BN → ReLU
        Linear(out) → ReLU   ← this last ReLU is the key innovation

    The final ReLU maps features to the positive orthant, which is required
    for the RGG distribution matching to be theoretically correct.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),  # ← deliberate final ReLU — essential for Rectified LpJEPA
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# Full Encoder: RectifiedLpJEPAEncoder
# =============================================================================

class RectifiedLpJEPAEncoder(nn.Module):
    """
    Full encoder for Rectified LpJEPA.

    Pipeline:
        (B, V, C, H, W)
          → backbone: (B*V, embed_dim)
          → RectifiedMLP: (B*V, proj_dim)   [with final ReLU]
          → reshape: proj (V, B, proj_dim)  [for multi-view loss]
                     emb  (B, V, embed_dim) [for downstream eval]

    The projector output is guaranteed non-negative (positive orthant),
    matching the support of the RGG target distribution.
    """

    def __init__(self, backbone: str, proj_dim: int, proj_hidden_dim: int = None):
        super().__init__()

        # --- Backbone ---
        if backbone == "tiny_cnn":
            self.backbone = TinyCNN(embed_dim=128)
        else:
            self.backbone = timm.create_model(
                backbone,
                num_classes=0,
                global_pool="avg",
                pretrained=False,
                in_chans=3,
            )

        embed_dim = self.backbone.num_features
        hidden_dim = proj_hidden_dim or max(proj_dim * 2, embed_dim)

        # --- Projector (with final ReLU) ---
        self.proj = RectifiedMLP(embed_dim, hidden_dim, proj_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, V, C, H, W)
        Returns:
            emb:  (B, V, embed_dim) — backbone embeddings (for linear probe / knn)
            proj: (V, B, proj_dim)  — non-negative projections (for RDMReg loss)
        """
        B, V = x.shape[:2]
        feats = self.backbone(x.flatten(0, 1))              # (B*V, embed_dim)
        emb = feats.view(B, V, -1)                           # (B, V, embed_dim)
        proj = self.proj(feats).view(B, V, -1).transpose(0, 1)  # (V, B, proj_dim)
        return emb, proj


# Backward-compatibility alias so probe_medmnist.py keeps working unchanged
LeJEPAEncoder = RectifiedLpJEPAEncoder
