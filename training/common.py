from __future__ import annotations

import json
import random
import re
import sys
from dataclasses import dataclass
from io import BytesIO
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


@dataclass(frozen=True)
class StreamAugmentProfile:
    name: str
    specular_glare_prob: float = 0.72
    sleeve_scratches_prob: float = 0.55
    thumb_occlusion_prob: float = 0.45
    stream_overlay_prob: float = 0.60
    lower_corner_occlusion_prob: float = 0.0
    brightness_range: tuple[float, float] = (0.72, 1.28)
    contrast_range: tuple[float, float] = (0.72, 1.30)
    color_range: tuple[float, float] = (0.72, 1.22)
    sharpness_prob: float = 0.35
    sharpness_range: tuple[float, float] = (0.35, 0.9)
    gaussian_blur_prob: float = 0.55
    gaussian_blur_radius: tuple[float, float] = (0.25, 2.2)
    motion_blur_prob: float = 0.0
    motion_blur_length: tuple[int, int] = (5, 13)
    partial_crop_prob: float = 0.0
    partial_crop_trim: tuple[float, float] = (0.05, 0.18)
    jpeg_artifact_prob: float = 0.0
    jpeg_quality: tuple[int, int] = (28, 65)
    sensor_noise_prob: float = 0.0
    sensor_noise_sigma: tuple[float, float] = (4.0, 12.0)


BASE_STREAM_AUGMENT_PROFILE = StreamAugmentProfile(name="baseline")
TARGETED_STREAM_AUGMENT_PROFILE = StreamAugmentProfile(
    name="targeted_v1",
    motion_blur_prob=0.45,
    partial_crop_prob=0.45,
    jpeg_artifact_prob=0.35,
    sensor_noise_prob=0.28,
)
TARGETED_STREAM_AUGMENT_PROFILE_V2 = StreamAugmentProfile(
    name="targeted_v2",
    motion_blur_prob=0.28,
    motion_blur_length=(3, 9),
    partial_crop_prob=0.22,
    partial_crop_trim=(0.03, 0.10),
    jpeg_artifact_prob=0.25,
    jpeg_quality=(42, 78),
    sensor_noise_prob=0.16,
    sensor_noise_sigma=(2.5, 7.0),
)
TARGETED_STREAM_AUGMENT_PROFILE_V3 = StreamAugmentProfile(
    name="targeted_v3",
    motion_blur_prob=0.24,
    motion_blur_length=(3, 7),
    partial_crop_prob=0.18,
    partial_crop_trim=(0.02, 0.08),
    jpeg_artifact_prob=0.22,
    jpeg_quality=(46, 80),
    sensor_noise_prob=0.14,
    sensor_noise_sigma=(2.0, 6.0),
    lower_corner_occlusion_prob=0.32,
)


STREAM_AUGMENT_PROFILES = {
    BASE_STREAM_AUGMENT_PROFILE.name: BASE_STREAM_AUGMENT_PROFILE,
    TARGETED_STREAM_AUGMENT_PROFILE.name: TARGETED_STREAM_AUGMENT_PROFILE,
    TARGETED_STREAM_AUGMENT_PROFILE_V2.name: TARGETED_STREAM_AUGMENT_PROFILE_V2,
    TARGETED_STREAM_AUGMENT_PROFILE_V3.name: TARGETED_STREAM_AUGMENT_PROFILE_V3,
}


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
            region = array[y : y + overlay_height, x : x + overlay_width].astype(np.float32)
            region *= random.uniform(0.18, 0.42)
            array[y : y + overlay_height, x : x + overlay_width] = region.astype(np.uint8)

        if random.random() < 0.55:
            rail_width = max(18, int(width * random.uniform(0.10, 0.18)))
            x = width - rail_width
            y = int(height * 0.10)
            h = int(height * 0.80)
            region = array[y : y + h, x:width].astype(np.float32)
            region *= random.uniform(0.10, 0.35)
            array[y : y + h, x:width] = region.astype(np.uint8)

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


def _apply_lower_corner_occlusion(image: Image.Image, rng: random.Random) -> Image.Image:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    color = (18, 18, 18, rng.randint(120, 185))
    target = rng.choice(["lower_left", "lower_right", "bottom_band"])
    if target == "lower_left":
        box_w = int(width * rng.uniform(0.16, 0.34))
        box_h = int(height * rng.uniform(0.06, 0.14))
        x0 = rng.randint(0, max(0, int(width * 0.08)))
        y0 = rng.randint(int(height * 0.78), max(int(height * 0.90), height - box_h))
        draw.rounded_rectangle((x0, y0, x0 + box_w, y0 + box_h), radius=8, fill=color)
    elif target == "lower_right":
        box_w = int(width * rng.uniform(0.18, 0.36))
        box_h = int(height * rng.uniform(0.06, 0.14))
        x0 = rng.randint(int(width * 0.58), max(int(width * 0.74), width - box_w))
        y0 = rng.randint(int(height * 0.76), max(int(height * 0.90), height - box_h))
        draw.rounded_rectangle((x0, y0, x0 + box_w, y0 + box_h), radius=8, fill=color)
    else:
        band_h = int(height * rng.uniform(0.06, 0.10))
        y0 = rng.randint(int(height * 0.84), max(int(height * 0.92), height - band_h))
        draw.rounded_rectangle((int(width * 0.10), y0, int(width * 0.90), y0 + band_h), radius=10, fill=color)
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def _apply_motion_blur(image: Image.Image, rng: random.Random, length_range: tuple[int, int]) -> Image.Image:
    width, height = image.size
    length = max(2, rng.randint(length_range[0], length_range[1]))
    axis = rng.choice(["horizontal", "vertical", "diag_down", "diag_up"])
    acc = np.zeros((height, width, 3), dtype=np.float32)
    base = np.asarray(image, dtype=np.float32)
    shifts: list[tuple[int, int]] = []
    for step in range(-length // 2, (length // 2) + 1):
        if axis == "horizontal":
            shifts.append((step, 0))
        elif axis == "vertical":
            shifts.append((0, step))
        elif axis == "diag_down":
            shifts.append((step, step))
        else:
            shifts.append((step, -step))
    for dx, dy in shifts:
        shifted = image.filter(ImageFilter.GaussianBlur(radius=0.5))
        canvas = Image.new("RGB", image.size, tuple(int(channel) for channel in np.asarray(image).mean(axis=(0, 1)).tolist()))
        canvas.paste(shifted, (dx, dy))
        acc += np.asarray(canvas, dtype=np.float32)
    acc /= max(float(len(shifts)), 1.0)
    blended = np.clip((0.25 * base) + (0.75 * acc), 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(blended)


def _apply_partial_crop(image: Image.Image, rng: random.Random, trim_range: tuple[float, float]) -> Image.Image:
    width, height = image.size
    min_trim, max_trim = trim_range
    left_trim = int(width * rng.uniform(0.0, max_trim if rng.random() < 0.5 else min_trim))
    right_trim = int(width * rng.uniform(0.0, max_trim if rng.random() < 0.5 else min_trim))
    top_trim = int(height * rng.uniform(0.0, max_trim if rng.random() < 0.5 else min_trim))
    bottom_trim = int(height * rng.uniform(0.0, max_trim if rng.random() < 0.5 else min_trim))
    crop_left = min(left_trim, width - 8)
    crop_top = min(top_trim, height - 8)
    crop_right = max(crop_left + 8, width - right_trim)
    crop_bottom = max(crop_top + 8, height - bottom_trim)
    cropped = image.crop((crop_left, crop_top, crop_right, crop_bottom))
    return cropped.resize((width, height), Image.Resampling.BICUBIC)


def _apply_jpeg_artifacts(image: Image.Image, rng: random.Random, quality_range: tuple[int, int]) -> Image.Image:
    quality = rng.randint(quality_range[0], quality_range[1])
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=False)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _apply_sensor_noise(image: Image.Image, rng: random.Random, sigma_range: tuple[float, float]) -> Image.Image:
    sigma = rng.uniform(sigma_range[0], sigma_range[1])
    array = np.asarray(image, dtype=np.float32)
    noise = np.asarray([rng.gauss(0.0, sigma) for _ in range(array.size)], dtype=np.float32).reshape(array.shape)
    array = np.clip(array + noise, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(array)


def make_stream_artifact_image(
    image: Image.Image,
    rng: random.Random,
    profile: StreamAugmentProfile = BASE_STREAM_AUGMENT_PROFILE,
) -> Image.Image:
    image = image.convert("RGB")
    array = np.array(image).copy()
    if rng.random() < profile.specular_glare_prob:
        array = _apply_specular_glare(array, rng)
    image = Image.fromarray(array)

    if rng.random() < profile.sleeve_scratches_prob:
        image = _apply_sleeve_scratches(image, rng)
    if rng.random() < profile.thumb_occlusion_prob:
        image = _apply_thumb_occlusion(image, rng)
    if rng.random() < profile.stream_overlay_prob:
        image = RandomStreamOverlay()(image)
    if rng.random() < profile.lower_corner_occlusion_prob:
        image = _apply_lower_corner_occlusion(image, rng)
    if rng.random() < profile.partial_crop_prob:
        image = _apply_partial_crop(image, rng, profile.partial_crop_trim)

    image = ImageEnhance.Brightness(image).enhance(rng.uniform(*profile.brightness_range))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(*profile.contrast_range))
    image = ImageEnhance.Color(image).enhance(rng.uniform(*profile.color_range))
    if rng.random() < profile.sharpness_prob:
        image = ImageEnhance.Sharpness(image).enhance(rng.uniform(*profile.sharpness_range))
    if rng.random() < profile.gaussian_blur_prob:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(*profile.gaussian_blur_radius)))
    if rng.random() < profile.motion_blur_prob:
        image = _apply_motion_blur(image, rng, profile.motion_blur_length)
    if rng.random() < profile.jpeg_artifact_prob:
        image = _apply_jpeg_artifacts(image, rng, profile.jpeg_quality)
    if rng.random() < profile.sensor_noise_prob:
        image = _apply_sensor_noise(image, rng, profile.sensor_noise_sigma)

    return image


class StreamArtifactAugment:
    def __init__(self, profile: StreamAugmentProfile = BASE_STREAM_AUGMENT_PROFILE) -> None:
        self.profile = profile

    def __call__(self, image: Image.Image) -> Image.Image:
        return make_stream_artifact_image(image, random, self.profile)


def resolve_stream_augment_profile(name: str) -> StreamAugmentProfile:
    profile = STREAM_AUGMENT_PROFILES.get(name)
    if profile is None:
        supported = ", ".join(sorted(STREAM_AUGMENT_PROFILES))
        raise ValueError(f"Unknown augment profile '{name}'. Expected one of: {supported}")
    return profile


def build_stream_train_transform(
    image_size: int,
    augment_profile: StreamAugmentProfile = BASE_STREAM_AUGMENT_PROFILE,
) -> transforms.Compose:
    return transforms.Compose(
        [
            CropInset(),
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomApply([StreamArtifactAugment(augment_profile)], p=0.9),
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
