import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """Spectral regularizer used in LeJEPA."""

    def __init__(self, knots: int = 17, t_max: float = 3.0, proj_samples: int = 256):
        super().__init__()
        assert knots >= 2, "knots must be >= 2"
        self.knots = knots
        self.t_max = t_max
        self.proj_samples = proj_samples
        t = torch.linspace(0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("weights", weights * window)
        self.register_buffer("phi", window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        # proj: (V, B, D)
        A = torch.randn(proj.size(-1), self.proj_samples, device=proj.device)
        A = A / A.norm(p=2, dim=0)
        x_t = (proj.transpose(0, 1) @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


def invariance_loss(proj: torch.Tensor) -> torch.Tensor:
    # proj: (V, B, D)
    return (proj.mean(0) - proj).square().mean()
