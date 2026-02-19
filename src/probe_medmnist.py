import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from src.datasets.medmnist_data import MultiViewMedMNIST, get_medmnist_datasets
from src.models.encoder import LeJEPAEncoder
from src.utils.eval import evaluate_probe, linear_probe

DEFAULT_DATASETS: List[str] = ["organamnist", "organcmnist", "organsmnist"]


def load_run_config(run_dir: Path) -> Dict:
    config_path = run_dir / "config.json"
    if config_path.exists():
        with config_path.open("r") as f:
            return json.load(f)
    return {}


def infer_device(preferred: Optional[str]) -> str:
    if not preferred or preferred == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return preferred


def parse_args():
    parser = argparse.ArgumentParser(description="Linear probe MedMNIST datasets from a saved LeJEPA encoder")
    parser.add_argument("--run-dir", required=True, help="Directory containing encoder.pt and optional config.json")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="List of MedMNIST dataset keys (default: OrganA/B/T)",
    )
    parser.add_argument("--backbone", help="Override backbone if config.json missing")
    parser.add_argument("--proj-dim", type=int, help="Override projector dim if config.json missing")
    parser.add_argument("--data-dir", help="MedMNIST data directory (overrides config)")
    parser.add_argument("--image-size", type=int, help="Resize edge length (default from config or 64)")
    parser.add_argument("--eval-batch-size", type=int, help="Batch size for probe dataloaders")
    parser.add_argument("--probe-epochs", type=int, default=5, help="Linear probe epochs (default 5)")
    parser.add_argument("--device", help="cpu/cuda/auto")
    parser.add_argument("--num-workers", type=int, help="DataLoader workers (default 4)")
    parser.add_argument("--output-json", help="Optional path to save probe metrics as JSON")
    return parser.parse_args()


def build_model(run_dir: Path, cfg: Dict, args) -> Tuple[torch.nn.Module, torch.device]:
    backbone = args.backbone or cfg.get("backbone")
    proj_dim = args.proj_dim or cfg.get("proj_dim", 64)
    if not backbone:
        raise ValueError("Backbone must be provided via --backbone or config.json")
    device = torch.device(infer_device(args.device or cfg.get("device")))
    model = LeJEPAEncoder(backbone, proj_dim).to(device)
    ckpt_path = Path(run_dir) / "encoder.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing encoder checkpoint at {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.backbone.load_state_dict(state["backbone"])
    model.proj.load_state_dict(state["proj"])
    return model, device


def make_loaders(dataset_name: str, data_dir: str, image_size: int, batch_size: int, num_workers: int):
    info, train_ds, val_ds, test_ds = get_medmnist_datasets(dataset_name, data_dir)
    num_classes = int(info.get("n_classes", len(info.get("label", []))))
    train_mv = MultiViewMedMNIST(train_ds, V=1, image_size=image_size, train=False)
    val_mv = MultiViewMedMNIST(val_ds, V=1, image_size=image_size, train=False)
    test_mv = MultiViewMedMNIST(test_ds, V=1, image_size=image_size, train=False)
    common = dict(num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_mv, batch_size=batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_mv, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_mv, batch_size=batch_size, shuffle=False, **common)
    return num_classes, train_loader, val_loader, test_loader


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    cfg = load_run_config(run_dir)

    data_dir = args.data_dir or cfg.get("data_dir", "./data")
    image_size = args.image_size or cfg.get("image_size", 64)
    batch_size = args.eval_batch_size or cfg.get("eval_batch_size", 512)
    num_workers = args.num_workers or cfg.get("num_workers", 4)
    probe_epochs = args.probe_epochs or cfg.get("probe_epochs", 5)

    model, device = build_model(run_dir, cfg, args)

    results = []
    for dataset_name in args.datasets:
        print(f"\n=== Linear probe: {dataset_name} ===")
        num_classes, train_loader, val_loader, test_loader = make_loaders(
            dataset_name, data_dir, image_size, batch_size, num_workers
        )
        probe = linear_probe(
            model,
            train_loader,
            val_loader,
            num_classes,
            device,
            epochs=probe_epochs,
        )
        test_acc = evaluate_probe(model, probe, test_loader, device)
        print(f"[test] dataset={dataset_name} acc={test_acc:.4f}")
        results.append({"dataset": dataset_name, "test_acc": test_acc})

    if args.output_json:
        output_path = Path(args.output_json)
    else:
        output_path = run_dir / "probe_results.json"
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved probe metrics to {output_path}")


if __name__ == "__main__":
    main()
