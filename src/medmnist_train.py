import argparse
import json
import sys
from pathlib import Path
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.losses.sigreg import SIGReg, invariance_loss
from src.models.encoder import LeJEPAEncoder
from src.datasets.medmnist_data import get_medmnist_datasets, MultiViewMedMNIST


def make_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(event: dict):
        with log_path.open("a") as f:
            json.dump(event, f)
            f.write("\n")

    return _log


def build_run_record(cfg: dict, model: LeJEPAEncoder, sigreg: SIGReg) -> dict:
    return {
        "config": cfg,
        "model": {
            "backbone": cfg.get("backbone"),
            "proj_dim": cfg.get("proj_dim"),
            "num_features": getattr(model.backbone, "num_features", None),
        },
        "sigreg": {
            "knots": getattr(sigreg, "knots", None),
            "t_max": getattr(sigreg, "t_max", None),
            "proj_samples": getattr(sigreg, "proj_samples", None),
        },
        "cmd": " ".join(sys.argv),
    }


def pretrain(cfg, model, sigreg, loader, device, log_event=None):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=5e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    total_steps = cfg["epochs"] * len(loader)
    warmup = max(10, len(loader))
    schedule = torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.2, total_iters=warmup),
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps - warmup)),
        ],
        milestones=[warmup],
    )

    model.train()
    sigreg.train()
    for epoch in range(cfg["epochs"]):
        running = 0.0
        prog = tqdm(loader, desc=f"pretrain epoch {epoch+1}/{cfg['epochs']}", leave=False)
        for views, _ in prog:
            views = views.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                emb, proj = model(views)
                inv = invariance_loss(proj)
                sig = sigreg(proj)
                loss = cfg["lamb"] * sig + (1 - cfg["lamb"]) * inv
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            schedule.step()
            running += loss.item()
            prog.set_postfix(loss=loss.item())
        mean_loss = running / max(1, len(loader))
        print(f"[pretrain] epoch {epoch+1}/{cfg['epochs']} loss={mean_loss:.4f}")
        if log_event:
            log_event({"phase": "pretrain", "epoch": epoch + 1, "loss": mean_loss})


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
    p = argparse.ArgumentParser(description="LeJEPA pretraining on MedMNIST")
    p.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    p.add_argument("--dataset", help="Override dataset key")
    p.add_argument("--backbone", help="Override backbone")
    p.add_argument("--epochs", type=int, help="Override epochs")
    p.add_argument("--batch-size", type=int, help="Override SSL batch size")
    p.add_argument("--views", type=int, help="Override number of views")
    p.add_argument("--lr", type=float, help="Override learning rate")
    p.add_argument("--device", help="cpu/cuda/auto")
    p.add_argument("--output-dir", help="Override output directory")
    return p.parse_args()


def save_artifacts(model, cfg, save_dir: Path, run_record: dict):
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"backbone": model.backbone.state_dict(), "proj": model.proj.state_dict()}, save_dir / "encoder.pt")
    with (save_dir / "config.json").open("w") as f:
        json.dump(cfg, f, indent=2)
    with (save_dir / "run_record.json").open("w") as f:
        json.dump(run_record, f, indent=2)


def main():
    args = parse_args()
    cfg_path = Path(args.config)
    overrides = {
        "dataset": args.dataset,
        "backbone": args.backbone,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "views": args.views,
        "lr": args.lr,
        "device": args.device,
        "output_dir": args.output_dir,
    }
    cfg = load_config(cfg_path, overrides)

    device = torch.device(cfg["device"])
    info, train_ds_raw, _, _ = get_medmnist_datasets(cfg["dataset"], cfg["data_dir"])
    train_ssl = MultiViewMedMNIST(train_ds_raw, V=cfg["views"], image_size=cfg["image_size"], train=True)

    ssl_loader = DataLoader(train_ssl, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg.get("num_workers", 4), drop_last=True, pin_memory=True)

    model = LeJEPAEncoder(cfg["backbone"], cfg["proj_dim"]).to(device)
    sigreg = SIGReg().to(device)

    save_dir = Path(cfg["output_dir"]) / cfg["dataset"] / cfg["backbone"]
    log_event = make_logger(save_dir / "logs.jsonl")
    run_record = build_run_record(cfg, model, sigreg)
    log_event({
        "phase": "config",
        "config": cfg,
        "cmd": run_record["cmd"],
        "dataset": {
            "name": info.get("name", cfg["dataset"]),
            "train_examples": len(train_ds_raw),
        },
    })

    print("Starting pretraining...")
    pretrain(cfg, model, sigreg, ssl_loader, device, log_event=log_event)

    save_artifacts(model, cfg, save_dir, run_record)
    print(f"Saved encoder and run_record to {save_dir}")


if __name__ == "__main__":
    main()
