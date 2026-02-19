import torch
import torch.nn as nn

def invariance_loss(proj: torch.Tensor) -> torch.Tensor:
    # proj: (V, B, D)
    return (proj.mean(0) - proj).square().mean()
