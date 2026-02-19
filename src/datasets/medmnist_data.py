from typing import Dict, List, Tuple

import medmnist
from medmnist import INFO
import torch
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms
from PIL import Image


# Map user-facing aliases onto the canonical MedMNIST identifiers.
_DATASET_ALIASES: Dict[str, str] = {
    "organa": "organamnist",
    "organ_a": "organamnist",
    "organamnist": "organamnist",
    "organb": "organcmnist",
    "organ_b": "organcmnist",
    "organbmnist": "organcmnist",
    "organcmnist": "organcmnist",
    "organt": "organsmnist",
    "organ_t": "organsmnist",
    "organtmnist": "organsmnist",
    "organsmnist": "organsmnist",
}


def _to_pil(img):
    if isinstance(img, Image.Image):
        return img
    return Image.fromarray(img)


def _to_3ch(t: torch.Tensor) -> torch.Tensor:
    if t.shape[0] == 3:
        return t
    return t.repeat(3, 1, 1)


class MultiViewMedMNIST(Dataset):
    def __init__(self, base_ds, V: int, image_size: int, train: bool):
        self.base_ds = base_ds
        self.V = V
        base_tx = [transforms.ToTensor(), transforms.Lambda(_to_3ch), transforms.Resize(image_size)]
        if train and V > 1:
            aug = [
                transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.0),
                transforms.RandomHorizontalFlip(),
            ]
            normalize = [transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
            self.tx = transforms.Compose(aug + base_tx + normalize)
        else:
            normalize = [transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
            self.tx = transforms.Compose(base_tx + normalize)

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        img, label = self.base_ds[idx]
        img = _to_pil(img)
        if self.V > 1:
            views = torch.stack([self.tx(img) for _ in range(self.V)])
        else:
            views = self.tx(img).unsqueeze(0)
        label = torch.as_tensor(label).long()
        return views, label


def resolve_dataset_names(name: str) -> List[str]:
    raw_names: List[str] = [chunk.strip() for chunk in name.split("+") if chunk.strip()]
    resolved = [_DATASET_ALIASES.get(chunk.lower(), chunk.lower()) for chunk in raw_names]
    if not resolved:
        raise ValueError("No MedMNIST dataset specified.")
    return resolved


def load_single_medmnist(name: str, data_dir: str) -> Tuple[dict, Dataset, Dataset, Dataset]:
    info = INFO[name]
    DataClass = getattr(medmnist, info["python_class"])
    train_ds = DataClass(split="train", download=True, root=data_dir)
    val_ds = DataClass(split="val", download=True, root=data_dir)
    test_ds = DataClass(split="test", download=True, root=data_dir)
    return info, train_ds, val_ds, test_ds


def get_medmnist_datasets(name: str, data_dir: str) -> Tuple[dict, Dataset, Dataset, Dataset]:
    resolved = resolve_dataset_names(name)
    if len(resolved) == 1:
        return load_single_medmnist(resolved[0], data_dir)

    infos = []
    train_parts = []
    val_parts = []
    test_parts = []
    for dataset_name in resolved:
        info, train_ds, val_ds, test_ds = load_single_medmnist(dataset_name, data_dir)
        infos.append({"name": dataset_name, "n_classes": info.get("n_classes"), "label": info.get("label")})
        train_parts.append(train_ds)
        val_parts.append(val_ds)
        test_parts.append(test_ds)

    merged_info = {
        "name": "+".join(resolved),
        "n_classes": sum((entry.get("n_classes") or 0) for entry in infos),
        "components": infos,
        "resolved": resolved,
    }

    return merged_info, ConcatDataset(train_parts), ConcatDataset(val_parts), ConcatDataset(test_parts)
