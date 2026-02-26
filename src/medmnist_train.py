"""
medmnist_train.py — Rectified LpJEPA pretraining on MedMNIST

Key changes vs LeJEPA version:
  1. Replaced lejepa.SlicingUnivariateTest (SIGReg, one-sample) 
     with rdmreg_loss (RDMReg, two-sample Sliced Wasserstein)
  2. Replaced LeJEPAEncoder with RectifiedLpJEPAEncoder (final ReLU in projector)
  3. Replaced single lambda with lamb_inv + lamb_rdm (separate loss weights)
  4. Added L0/L1 sparsity metrics logging
  5. Replaced AdamW with LARS (recommended for SSL by the paper)
  6. Added p, mu, num_projections to config

Reference: Rectified LpJEPA (Kuang et al., 2026) — arXiv:2602.01456
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.losses.rdmreg import (
    rectified_lpjepa_loss,
    determine_sigma_gn,
    l0_sparsity,
    l1_sparsity,
)
from src.models.encoder import RectifiedLpJEPAEncoder
from src.datasets.medmnist_data import get_medmnist_datasets, MultiViewMedMNIST


# =============================================================================
# LARS Optimizer (recommended for SSL, from the paper)
# =============================================================================

class LARS(torch.optim.Optimizer):
    """
    Layer-wise Adaptive Rate Scaling optimizer.
    Recommended for self-supervised learning with large batch sizes.
    """

    def __init__(self, params, lr=0.03, momentum=0.9, weight_decay=1e-4,
                 eta=0.02, eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        eta=eta, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            wd = float(group["weight_decay"])
            momentum = group["momentum"]
            eta = group["eta"]
            eps = group["eps"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad

                if p.ndim != 1 and not group.get("exclude_bias_n_norm", False):
                    p_norm = torch.norm(p)
                    g_norm = torch.norm(d_p)
                    if p_norm != 0 and g_norm != 0:
                        lars_lr = eta * p_norm / (g_norm + p_norm * wd + eps)
                        d_p = d_p.add(p, alpha=wd).mul(lars_lr)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.clone(d_p).detach()
                else:
                    state["momentum_buffer"].mul_(momentum).add_(d_p)

                p.add_(state["momentum_buffer"], alpha=-lr)


# =============================================================================
# Logging helpers
# =============================================================================

def make_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(event: dict):
        with log_path.open("a") as f:
            json.dump(event, f)
            f.write("\n")

    return _log


def build_run_record(cfg: dict, model: RectifiedLpJEPAEncoder) -> dict:
    return {
        "config": cfg,
        "method": "Rectified LpJEPA",
        "model": {
            "backbone": cfg.get("backbone"),
            "proj_dim": cfg.get("proj_dim"),
            "proj_hidden_dim": cfg.get("proj_hidden_dim"),
            "num_features": getattr(model.backbone, "num_features", None),
        },
        "distribution": {
            "type": "Rectified Generalized Gaussian (RGG)",
            "p": cfg.get("p", 1.0),
            "mu": cfg.get("mu", 0.0),
            "sigma": determine_sigma_gn(cfg.get("p", 1.0)),
        },
        "cmd": " ".join(sys.argv),
    }


# =============================================================================
# Pretraining loop
# =============================================================================

def pretrain(cfg, model, loader, device, log_event=None):
    """
    Main Rectified LpJEPA pretraining loop.

    Loss = lamb_inv * L_invariance + lamb_rdm * L_RDMReg
    """

    # --- Optimizer: LARS with cosine schedule ---
    opt = LARS(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 1e-4),
    )
    total_steps = cfg["epochs"] * len(loader)
    warmup_steps = cfg.get("warmup_epochs", 10) * len(loader)
    schedule = torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=0.2, total_iters=warmup_steps
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, total_steps - warmup_steps)
            ),
        ],
        milestones=[warmup_steps],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    # --- RGG distribution parameters ---
    p = cfg.get("p", 1.0)
    mu = cfg.get("mu", 0.0)
    sigma = determine_sigma_gn(p)
    num_projections = cfg.get("num_projections", 8192)
    lamb_inv = cfg.get("lamb_inv", 25.0)
    lamb_rdm = cfg.get("lamb_rdm", 125.0)

    print(f"\n{'='*60}")
    print(f"  Rectified LpJEPA Pretraining")
    print(f"  Distribution: RGG(p={p}, mu={mu}, sigma={sigma:.4f})")
    print(f"  Loss weights: lamb_inv={lamb_inv}, lamb_rdm={lamb_rdm}")
    print(f"  Projections:  {num_projections}")
    print(f"  Epochs:       {cfg['epochs']}  |  Batch: {cfg['batch_size']}")
    print(f"{'='*60}\n")

    model.train()
    for epoch in range(cfg["epochs"]):
        running_total = running_inv = running_rdm = 0.0
        prog = tqdm(loader, desc=f"epoch {epoch+1:03d}/{cfg['epochs']}", leave=False)

        for views, _ in prog:
            views = views.to(device, non_blocking=True)   # (B, V, C, H, W)

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                _, proj = model(views)   # proj: (V, B, proj_dim) — already ReLU-activated

                total_loss, inv_loss, rdm_loss = rectified_lpjepa_loss(
                    proj,
                    lamb_inv=lamb_inv,
                    lamb_rdm=lamb_rdm,
                    p=p,
                    mu=mu,
                    sigma=sigma,
                    num_projections=num_projections,
                )

            opt.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(opt)
            scaler.update()
            schedule.step()

            running_total += total_loss.item()
            running_inv += inv_loss.item()
            running_rdm += rdm_loss.item()
            prog.set_postfix(
                loss=f"{total_loss.item():.4f}",
                inv=f"{inv_loss.item():.4f}",
                rdm=f"{rdm_loss.item():.4f}",
            )

        n = max(1, len(loader))
        mean_total = running_total / n
        mean_inv = running_inv / n
        mean_rdm = running_rdm / n

        # --- Sparsity metrics (on last batch, no overhead) ---
        with torch.no_grad():
            last_z = proj[0]   # first view, shape (B, D)
            l0 = l0_sparsity(last_z)
            l1 = l1_sparsity(last_z)

        curr_lr = opt.param_groups[0]["lr"]
        print(
            f"[epoch {epoch+1:03d}] "
            f"loss={mean_total:.4f}  inv={mean_inv:.4f}  rdm={mean_rdm:.4f}  "
            f"l0={l0:.3f}  l1={l1:.3f}  lr={curr_lr:.5f}"
        )

        if log_event:
            log_event({
                "phase": "pretrain",
                "epoch": epoch + 1,
                "loss": mean_total,
                "inv_loss": mean_inv,
                "rdm_loss": mean_rdm,
                "l0_sparsity": l0,
                "l1_sparsity": l1,
                "lr": curr_lr,
            })


# =============================================================================
# Config & CLI
# =============================================================================

def load_config(path: Path, overrides: dict) -> dict:
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    if cfg.get("device", "auto") == "auto":
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


def parse_args():
    p = argparse.ArgumentParser(description="Rectified LpJEPA pretraining on MedMNIST")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--dataset", help="Override dataset key")
    p.add_argument("--backbone", help="Override backbone")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--views", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--device", help="cpu / cuda / auto")
    p.add_argument("--output-dir")
    # Rectified LpJEPA specific
    p.add_argument("--p", type=float, help="Lp norm parameter (1.0=Laplace, 2.0=Gaussian)")
    p.add_argument("--mu", type=float, help="Mean shift: lower → more sparsity")
    p.add_argument("--lamb-inv", type=float, help="Invariance loss weight")
    p.add_argument("--lamb-rdm", type=float, help="RDMReg loss weight")
    p.add_argument("--num-projections", type=int, help="Number of Cramér-Wold projections")
    p.add_argument("--proj-dim", type=int, help="Projector output dimension")
    return p.parse_args()


def save_artifacts(model, cfg, save_dir: Path, run_record: dict):
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"backbone": model.backbone.state_dict(), "proj": model.proj.state_dict()},
        save_dir / "encoder.pt",
    )
    with (save_dir / "config.json").open("w") as f:
        json.dump(cfg, f, indent=2)
    with (save_dir / "run_record.json").open("w") as f:
        json.dump(run_record, f, indent=2)
    print(f"\nSaved encoder + config to {save_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    overrides = {
        "dataset": args.dataset,
        "backbone": args.backbone,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "views": args.views,
        "lr": args.lr,
        "device": args.device,
        "output_dir": args.output_dir,
        "p": args.p,
        "mu": args.mu,
        "lamb_inv": args.lamb_inv,
        "lamb_rdm": args.lamb_rdm,
        "num_projections": args.num_projections,
        "proj_dim": args.proj_dim,
    }
    cfg = load_config(Path(args.config), overrides)
    device = torch.device(cfg["device"])

    # --- Data ---
    info, train_ds_raw, _, _ = get_medmnist_datasets(cfg["dataset"], cfg["data_dir"])
    train_ssl = MultiViewMedMNIST(
        train_ds_raw, V=cfg["views"], image_size=cfg["image_size"], train=True
    )
    ssl_loader = DataLoader(
        train_ssl,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        drop_last=True,
        pin_memory=True,
    )

    # --- Model ---
    model = RectifiedLpJEPAEncoder(
        backbone=cfg["backbone"],
        proj_dim=cfg.get("proj_dim", 512),
        proj_hidden_dim=cfg.get("proj_hidden_dim", None),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # --- Logging ---
    save_dir = Path(cfg["output_dir"]) / cfg["dataset"] / cfg["backbone"]
    log_event = make_logger(save_dir / "logs.jsonl")
    run_record = build_run_record(cfg, model)
    log_event({
        "phase": "config",
        "config": cfg,
        "cmd": run_record["cmd"],
        "dataset": {
            "name": info.get("name", cfg["dataset"]),
            "train_examples": len(train_ds_raw),
        },
        "run_record": run_record,
    })

    # --- Pretrain ---
    pretrain(cfg, model, ssl_loader, device, log_event=log_event)

    # --- Save ---
    save_artifacts(model, cfg, save_dir, run_record)


if __name__ == "__main__":
    main()
