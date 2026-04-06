from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


EXPECTED_DIM = 256
IMAGE_SIZE = 224
CROP_INSET_RATIO = 0.08
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class CropInset:
    def __init__(self, ratio: float = CROP_INSET_RATIO) -> None:
        self.ratio = ratio

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        inset_x = int(width * self.ratio)
        inset_y = int(height * self.ratio)
        left = min(inset_x, max(0, width - 1))
        top = min(inset_y, max(0, height - 1))
        right = max(left + 1, width - inset_x)
        bottom = max(top + 1, height - inset_y)
        return image.crop((left, top, right, bottom))


def crop_inset(image: Image.Image, ratio: float = CROP_INSET_RATIO) -> Image.Image:
    return CropInset(ratio=ratio)(image)


def prepare_base_image(image: Image.Image, image_size: int = IMAGE_SIZE) -> Image.Image:
    base_image = crop_inset(image.convert("RGB"))
    return base_image.resize((image_size, image_size), resample=Image.Resampling.BILINEAR)


def preprocess_image(image: Image.Image, image_size: int = IMAGE_SIZE) -> np.ndarray:
    base_image = prepare_base_image(image, image_size=image_size)
    array = np.asarray(base_image, dtype=np.float32) / 255.0
    normalized = (array - MEAN) / STD
    chw = np.transpose(normalized, (2, 0, 1))
    return np.expand_dims(chw.astype(np.float32, copy=False), axis=0)


def preprocess_image_path(image_path: Path, image_size: int = IMAGE_SIZE) -> np.ndarray:
    with Image.open(image_path) as image:
        return preprocess_image(image, image_size=image_size)
