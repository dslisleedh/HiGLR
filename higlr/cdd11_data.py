"""CDD11_all sampling/pixel protocol from MoCE-IR 94203ce; independent implementation.

Reference: src/data/dataset_utils.py:CDD11 and utils/image_utils.py at
https://github.com/eduardzamfir/MoCE-IR/tree/94203ce86052fbcbfb711e17138b178193284db3
Requested differences: crop256, identity enabled, rain/snow modes0/1/4/5.
No extra degradation synthesis or image resizing.
"""
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

CLASSES = tuple(sorted(("low", "haze", "rain", "snow", "low_haze", "low_rain",
                        "low_snow", "haze_rain", "haze_snow", "low_haze_rain", "low_haze_snow")))


def augment(array, mode):
    if mode not in range(8):
        raise ValueError("Expected augmentation mode 0..7")
    result = np.rot90(array, k=mode // 2)
    if mode % 2:
        result = np.flipud(result)
    return result.copy()


def augmentation_modes(kind):
    return (0, 1, 4, 5) if {'rain', 'snow'}.intersection(kind.split('_')) else tuple(range(8))


def tensor(array):
    return torch.from_numpy(array.transpose(2, 0, 1).copy()).float().div_(255)


class CDD11Dataset(Dataset):
    def __init__(self, root, patch_size=256):
        self.root = Path(root).resolve(strict=True)
        self.patch_size = patch_size
        train = self.root / 'train'
        if {p.name for p in train.iterdir() if p.is_dir()} != set(CLASSES) | {'clear'}:
            raise ValueError("Expected clear and exactly 11 CDD11 degradation directories")
        self.clean = sorted((train / 'clear').glob('*.png'))
        names = [p.name for p in self.clean]
        if len(names) != 1183:
            raise ValueError(f"Expected 1183 clean training images, got {len(names)}")
        self.degraded = {}
        for name in CLASSES:
            paths = sorted((train / name).glob('*.png'))
            if [p.name for p in paths] != names:
                raise ValueError(f"CDD11 pair mismatch: {name}")
            self.degraded[name] = paths * 2

    def __len__(self):
        return sum(map(len, self.degraded.values()))

    def __getitem__(self, index):
        # Intentionally ignore the sampler index, as the reference loader does.
        kind = random.choice(CLASSES)
        path = random.choice(self.degraded[kind])
        clean_path = self.root / 'train' / 'clear' / path.name
        with Image.open(path) as image:
            degraded = np.array(image.convert('RGB'))
        with Image.open(clean_path) as image:
            clean = np.array(image.convert('RGB'))
        if degraded.shape != clean.shape:
            raise ValueError(f"Unpaired geometry: {path}")
        h, w = clean.shape[:2]
        p = self.patch_size
        if min(h, w) < p:
            raise ValueError(f"Undersized CDD11 image; resizing is not authorized: {path}")
        top, left = random.randint(0, h - p), random.randint(0, w - p)
        mode = random.choice(augmentation_modes(kind))
        crop = lambda x: tensor(augment(x[top:top+p, left:left+p], mode))
        return [str(clean_path), CLASSES.index(kind)], crop(degraded), crop(clean)
