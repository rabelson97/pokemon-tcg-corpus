from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
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


def build_stream_train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            CropInset(),
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomApply([RandomStreamOverlay()], p=0.75),
            transforms.ColorJitter(brightness=0.35, contrast=0.3, saturation=0.25, hue=0.06),
            transforms.RandomPerspective(distortion_scale=0.16, p=0.35),
            transforms.RandomAffine(
                degrees=7,
                translate=(0.08, 0.08),
                scale=(0.88, 1.08),
                shear=4,
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.15, 2.0))], p=0.4),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.4, 2.2), value=0),
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
