"""Paired FT256 short-side resizing, preserving the established rain axis policy."""
import math
from types import MethodType
import numpy as np
from PIL import Image
POLICY = (
    'after base16 trim, if shortest side <256, resize bicubic with preserved '
    'aspect ratio so shortest side is 256; use the same geometry for paired '
    'input/target, then aligned random 256 crop; synthesize denoising noise after crop'
)


def resize_short_side(image, size):
    height, width = image.shape[:2]
    if height < 1 or width < 1:
        raise ValueError('empty image')
    if min(height, width) >= size:
        return image
    scale = size / min(height, width)
    new_height = max(size, math.ceil(height * scale))
    new_width = max(size, math.ceil(width * scale))
    return np.asarray(Image.fromarray(image).resize(
        (new_width, new_height), resample=Image.Resampling.BICUBIC))


class ResizeThenTransform:
    def __init__(self, transform, size):
        self.transform, self.size = transform, size

    def __call__(self, image):
        return self.transform(resize_short_side(image, self.size))


def resized_pair_crop(self, degraded, clean):
    if degraded.shape != clean.shape:
        raise ValueError(f'paired shapes differ: {degraded.shape}, {clean.shape}')
    size = self.args.patch_size
    return type(self)._crop_patch(
        self, resize_short_side(degraded, size), resize_short_side(clean, size))


def prepare_finetune(dataset):
    dataset.crop_transform=ResizeThenTransform(dataset.crop_transform,256)
    dataset._crop_patch=MethodType(resized_pair_crop,dataset)
    return dataset
