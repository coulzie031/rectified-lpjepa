from typing import Tuple

import medmnist
from medmnist import INFO
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


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


def get_medmnist_datasets(name: str, data_dir: str) -> Tuple[dict, Dataset, Dataset, Dataset]:
    info = INFO[name]
    DataClass = getattr(medmnist, info["python_class"])
    train_ds = DataClass(split="train", download=True, root=data_dir)
    val_ds = DataClass(split="val", download=True, root=data_dir)
    test_ds = DataClass(split="test", download=True, root=data_dir)
    return info, train_ds, val_ds, test_ds
