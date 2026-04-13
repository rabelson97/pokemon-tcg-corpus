from __future__ import annotations

import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from torch import nn
from torch.utils.data import Dataset
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from embedder_contract import CropInset


@dataclass(frozen=True)
class ManifestRecord:
    card_id: str
    name: str
    set_name: str | None
    number: str | None
    image_path: str
    image_url: str
    locale: str | None = None
    upstream_id: str | None = None
    set_id: str | None = None
    equivalence_key: str | None = None


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def same_art_key(record: ManifestRecord) -> str:
    equivalence_key = normalize_text(record.equivalence_key)
    if equivalence_key:
        return f"equiv:{equivalence_key}"
    upstream_id = normalize_text(record.upstream_id)
    if upstream_id:
        return f"upstream:{upstream_id}"
    return f"name:{normalize_text(record.name)}"


def load_manifest(path: str | Path) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            records.append(
                ManifestRecord(
                    card_id=row["card_id"],
                    name=row["name"],
                    set_name=row.get("set_name"),
                    number=row.get("number"),
                    image_path=row["image_path"],
                    image_url=row["image_url"],
                    locale=row.get("locale"),
                    upstream_id=row.get("upstream_id"),
                    set_id=row.get("set_id"),
                    equivalence_key=row.get("equivalence_key"),
                )
            )
    return records


def split_records(
    records: list[ManifestRecord],
    val_fraction: float,
    seed: int,
) -> tuple[list[ManifestRecord], list[ManifestRecord]]:
    by_locale: dict[str, list[ManifestRecord]] = {}
    for record in records:
        locale = (record.locale or "unknown").strip() or "unknown"
        by_locale.setdefault(locale, []).append(record)

    rng = random.Random(seed)
    train_records: list[ManifestRecord] = []
    val_records: list[ManifestRecord] = []
    for locale in sorted(by_locale):
        locale_records = list(by_locale[locale])
        rng.shuffle(locale_records)
        val_count = max(1, int(len(locale_records) * val_fraction))
        val_records.extend(locale_records[:val_count])
        train_records.extend(locale_records[val_count:])

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    return train_records, val_records


class RandomStreamOverlay:
    def __call__(self, image: Image.Image) -> Image.Image:
        array = np.array(image).copy()
        height, width = array.shape[:2]

        if random.random() < 0.65:
            overlay_height = max(18, int(height * random.uniform(0.08, 0.18)))
            overlay_width = max(40, int(width * random.uniform(0.22, 0.45)))
            x = random.randint(0, max(0, width - overlay_width))
            y = random.randint(int(height * 0.55), max(int(height * 0.80), height - overlay_height))
            array[y : y + overlay_height, x : x + overlay_width] = 0

        if random.random() < 0.55:
            rail_width = max(18, int(width * random.uniform(0.10, 0.18)))
            x = width - rail_width
            y = int(height * 0.10)
            h = int(height * 0.80)
            array[y : y + h, x:width] = 0

        return Image.fromarray(array)


def _apply_specular_glare(array: np.ndarray, rng: random.Random) -> np.ndarray:
    height, width = array.shape[:2]
    xs = np.linspace(0.0, 1.0, width, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    center_x = rng.uniform(0.2, 0.8)
    center_y = rng.uniform(0.15, 0.85)
    sigma_x = rng.uniform(0.05, 0.18)
    sigma_y = rng.uniform(0.04, 0.14)
    angle = rng.uniform(-1.0, 1.0)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    x_shifted = xx - center_x
    y_shifted = yy - center_y
    rot_x = (x_shifted * cos_a) - (y_shifted * sin_a)
    rot_y = (x_shifted * sin_a) + (y_shifted * cos_a)
    glare = np.exp(-0.5 * ((rot_x / sigma_x) ** 2 + (rot_y / sigma_y) ** 2))
    intensity = rng.uniform(90.0, 180.0)
    tint = np.asarray(
        [
            rng.uniform(0.92, 1.0),
            rng.uniform(0.92, 1.0),
            rng.uniform(0.86, 0.98),
        ],
        dtype=np.float32,
    )
    array = array.astype(np.float32)
    array += glare[..., None] * intensity * tint
    return np.clip(array, 0.0, 255.0).astype(np.uint8)


def _apply_sleeve_scratches(image: Image.Image, rng: random.Random) -> Image.Image:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    scratch_count = rng.randint(3, 9)
    for _ in range(scratch_count):
        x0 = rng.randint(0, width)
        y0 = rng.randint(0, height)
        x1 = x0 + rng.randint(-width // 6, width // 6)
        y1 = y0 + rng.randint(height // 8, height // 2)
        alpha = rng.randint(18, 52)
        color = (255, 255, 255, alpha)
        draw.line((x0, y0, x1, y1), fill=color, width=rng.randint(1, 3))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def _apply_thumb_occlusion(image: Image.Image, rng: random.Random) -> Image.Image:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    skin_tones = [
        (232, 190, 172, 230),
        (210, 158, 132, 230),
        (166, 112, 83, 230),
        (120, 79, 58, 230),
    ]
    tone = skin_tones[rng.randrange(len(skin_tones))]
    if rng.random() < 0.6:
        ellipse_w = int(width * rng.uniform(0.18, 0.34))
        ellipse_h = int(height * rng.uniform(0.14, 0.24))
        x0 = rng.randint(-ellipse_w // 4, width - ellipse_w // 2)
        y0 = rng.randint(int(height * 0.68), height - ellipse_h // 2)
        draw.ellipse((x0, y0, x0 + ellipse_w, y0 + ellipse_h), fill=tone)
    else:
        ellipse_w = int(width * rng.uniform(0.14, 0.24))
        ellipse_h = int(height * rng.uniform(0.28, 0.42))
        x0 = rng.choice([rng.randint(-ellipse_w // 3, 0), rng.randint(width - ellipse_w, width - ellipse_w // 3)])
        y0 = rng.randint(int(height * 0.35), int(height * 0.62))
        draw.ellipse((x0, y0, x0 + ellipse_w, y0 + ellipse_h), fill=tone)
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def make_stream_artifact_image(image: Image.Image, rng: random.Random) -> Image.Image:
    image = image.convert("RGB")
    array = np.array(image).copy()
    if rng.random() < 0.72:
        array = _apply_specular_glare(array, rng)
    image = Image.fromarray(array)

    if rng.random() < 0.55:
        image = _apply_sleeve_scratches(image, rng)
    if rng.random() < 0.45:
        image = _apply_thumb_occlusion(image, rng)
    if rng.random() < 0.60:
        image = RandomStreamOverlay()(image)

    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.72, 1.28))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.72, 1.30))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.72, 1.22))
    if rng.random() < 0.35:
        image = ImageEnhance.Sharpness(image).enhance(rng.uniform(0.35, 0.9))
    if rng.random() < 0.55:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.25, 2.2)))

    return image


class StreamArtifactAugment:
    def __call__(self, image: Image.Image) -> Image.Image:
        return make_stream_artifact_image(image, random)


def build_stream_train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            CropInset(),
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomApply([StreamArtifactAugment()], p=0.9),
            transforms.ColorJitter(brightness=0.42, contrast=0.38, saturation=0.28, hue=0.08),
            transforms.RandomPerspective(distortion_scale=0.24, p=0.45),
            transforms.RandomAffine(
                degrees=9,
                translate=(0.10, 0.10),
                scale=(0.84, 1.10),
                shear=6,
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.15, 2.4))], p=0.5),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.35, scale=(0.02, 0.16), ratio=(0.3, 2.5), value=0),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            CropInset(),
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


class CardImageDataset(Dataset):
    def __init__(
        self,
        records: list[ManifestRecord],
        label_map: dict[str, int],
        transform: transforms.Compose,
    ) -> None:
        self.records = records
        self.label_map = label_map
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        return self.transform(image), self.label_map[record.card_id]


class CardInferenceDataset(Dataset):
    def __init__(self, records: list[ManifestRecord], transform: transforms.Compose) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        return self.transform(image), record.card_id


class TwoViewCardDataset(Dataset):
    def __init__(self, records: list[ManifestRecord], transform: transforms.Compose) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        return self.transform(image), self.transform(image), record.card_id


class CardEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        feature_dim = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.embedding = nn.Linear(feature_dim, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(images)
        embeddings = torch.nn.functional.normalize(self.embedding(features), dim=1)
        logits = self.classifier(embeddings)
        return embeddings, logits


def create_label_map(records: Iterable[ManifestRecord]) -> dict[str, int]:
    ids = sorted({record.card_id for record in records})
    return {card_id: index for index, card_id in enumerate(ids)}


def count_records_by_locale(records: Iterable[ManifestRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        locale = (record.locale or "unknown").strip() or "unknown"
        counts[locale] = counts.get(locale, 0) + 1
    return counts
