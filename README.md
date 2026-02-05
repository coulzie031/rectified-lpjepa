# LeJEPA on MedMNIST

Clean layout with configs and src:
- `configs/default.yaml` – hyperparameters and paths (use `--config` to point elsewhere; CLI flags override keys)
- `src/medmnist_train.py` – main training script (pretrain, probe, radius-kNN, saving)
- `medmnist_lejepa.py` – earlier single-file version (kept for reference)

## Quick start
1) Install deps (PyTorch + CUDA if available):
```bash
pip install torch torchvision timm medmnist pyyaml tqdm
```

2) Run with defaults (PathMNIST, tiny CNN):
```bash
python -m src.medmnist_train --config configs/default.yaml
```

3) Override on CLI, e.g., use ResNet18 and more epochs:
```bash
python -m src.medmnist_train --config configs/default.yaml \
  --backbone resnet18 --epochs 20 --views 6 --lr 1e-3
```

Artifacts are saved to `outputs/<dataset>/<backbone>/` (`encoder.pt`, `probe.pt`, `config.json`).

## What it does
- **Pretraining**: LeJEPA objective = SIGReg + invariance across `views` with a tiny CNN by default (swap any timm backbone).
- **Linear probe**: small head on frozen features, short training loop with tqdm progress bars.
- **Radius kNN**: vote inside a Euclidean ball; fallback weighted top-k when empty.

Tune `views`, `epochs`, `lamb`, and backbone to balance speed vs. accuracy; small MedMNIST tasks run comfortably on a single GPU.
