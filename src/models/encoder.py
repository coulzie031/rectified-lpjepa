import torch
import torch.nn as nn
from torchvision.ops import MLP
import timm


class TinyCNN(nn.Module):
    """Lightweight encoder for quick MedMNIST experiments."""

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
        x = self.features(x)
        x = x.flatten(1)
        return self.head(x)


class LeJEPAEncoder(nn.Module):
    def __init__(self, backbone: str, proj_dim: int):
        super().__init__()
        if backbone == "tiny_cnn":
            self.backbone = TinyCNN(embed_dim=128)
        else:
            self.backbone = timm.create_model(
                backbone, num_classes=0, global_pool="avg", pretrained=False, in_chans=3
            )
        embed_dim = self.backbone.num_features
        mid_dim = max(proj_dim * 2, embed_dim)
        self.proj = MLP(embed_dim, [mid_dim, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x: torch.Tensor):
        # x: (B, V, C, H, W)
        B, V = x.shape[:2]
        feats = self.backbone(x.flatten(0, 1))
        emb = feats.view(B, V, -1)
        proj = self.proj(feats).view(B, V, -1).transpose(0, 1)
        return emb, proj
