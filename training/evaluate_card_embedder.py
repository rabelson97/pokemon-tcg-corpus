#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from common import ManifestRecord, load_manifest, split_records


EXPECTED_DIM = 256
IMAGE_SIZE = 224
CROP_INSET_RATIO = 0.08
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class RetrievalMetrics:
    sample_count: int
    recall_at_1: float
    recall_at_5: float
    median_top1_score: float
    mean_top1_score: float
    failure_examples: list[dict[str, object]]


def crop_inset(image: Image.Image) -> Image.Image:
    width, height = image.size
    inset_x = int(width * CROP_INSET_RATIO)
    inset_y = int(height * CROP_INSET_RATIO)
    left = min(inset_x, max(0, width - 1))
    top = min(inset_y, max(0, height - 1))
    right = max(left + 1, width - inset_x)
    bottom = max(top + 1, height - inset_y)
    return image.crop((left, top, right, bottom))


def preprocess(image: Image.Image) -> np.ndarray:
    image = crop_inset(image.convert("RGB"))
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    normalized = (array - MEAN) / STD
    chw = np.transpose(normalized, (2, 0, 1))
    return np.expand_dims(chw.astype(np.float32, copy=False), axis=0)


def make_stream_like(image: Image.Image, rng: random.Random) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size

    if rng.random() < 0.8:
        angle = rng.uniform(-7.0, 7.0)
        image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))

    if rng.random() < 0.9:
        translate_x = int(width * rng.uniform(-0.06, 0.06))
        translate_y = int(height * rng.uniform(-0.06, 0.06))
        image = ImageOps.expand(image, border=max(abs(translate_x), abs(translate_y)) + 8, fill=(0, 0, 0))
        image = image.crop(
            (
                max(0, translate_x),
                max(0, translate_y),
                max(0, translate_x) + width,
                max(0, translate_y) + height,
            )
        )

    array = np.array(image)
    if rng.random() < 0.7:
        overlay_h = max(18, int(height * rng.uniform(0.08, 0.18)))
        overlay_w = max(40, int(width * rng.uniform(0.22, 0.45)))
        x = rng.randint(0, max(0, width - overlay_w))
        y = rng.randint(int(height * 0.55), max(int(height * 0.80), height - overlay_h))
        array[y : y + overlay_h, x : x + overlay_w] = 0
    if rng.random() < 0.5:
        rail_w = max(18, int(width * rng.uniform(0.10, 0.18)))
        x = max(0, width - rail_w)
        y = int(height * 0.10)
        h = int(height * 0.80)
        array[y : y + h, x:width] = 0
    image = Image.fromarray(array)

    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.75, 1.20))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.75, 1.25))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.80, 1.20))

    if rng.random() < 0.5:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 1.5)))

    return image


def embed_images(session: ort.InferenceSession, input_name: str, images: Iterable[Image.Image]) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for image in images:
        tensor = preprocess(image)
        vector = np.asarray(session.run(None, {input_name: tensor})[0][0], dtype=np.float32)
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        outputs.append(vector)
    return np.stack(outputs, axis=0)


def retrieve_metrics(
    query_vectors: np.ndarray,
    query_records: list[ManifestRecord],
    reference_vectors: np.ndarray,
    reference_records: list[ManifestRecord],
    *,
    top_k: int = 5,
) -> RetrievalMetrics:
    sims = query_vectors @ reference_vectors.T
    top1_hits = 0
    top5_hits = 0
    top1_scores: list[float] = []
    failures: list[dict[str, object]] = []

    ref_ids = [record.card_id for record in reference_records]
    for query_index, query_record in enumerate(query_records):
        row = sims[query_index]
        top_indices = np.argsort(row)[::-1][:top_k]
        top_cards = [(ref_ids[idx], float(row[idx])) for idx in top_indices]
        top1_scores.append(top_cards[0][1])
        if top_cards[0][0] == query_record.card_id:
            top1_hits += 1
        if any(card_id == query_record.card_id for card_id, _ in top_cards):
            top5_hits += 1
        else:
            failures.append(
                {
                    "query_card_id": query_record.card_id,
                    "query_name": query_record.name,
                    "top_candidates": [
                        {"card_id": card_id, "score": score}
                        for card_id, score in top_cards
                    ],
                }
            )

    sample_count = len(query_records)
    return RetrievalMetrics(
        sample_count=sample_count,
        recall_at_1=top1_hits / max(1, sample_count),
        recall_at_5=top5_hits / max(1, sample_count),
        median_top1_score=float(np.median(np.asarray(top1_scores, dtype=np.float32))),
        mean_top1_score=float(np.mean(np.asarray(top1_scores, dtype=np.float32))),
        failure_examples=failures[:20],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a candidate embedder against the repository contract.")
    parser.add_argument("--manifest", default="training/data/full/manifest.jsonl")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stream-query-count", type=int, default=1)
    args = parser.parse_args()

    records = load_manifest(args.manifest)
    _, val_records = split_records(records, val_fraction=args.val_fraction, seed=args.seed)
    session = ort.InferenceSession(str(Path(args.model).resolve()), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_shape = session.get_outputs()[0].shape
    if len(output_shape) < 2 or int(output_shape[-1]) != EXPECTED_DIM:
        raise SystemExit(f"Unexpected ONNX output shape: {output_shape}")

    reference_images = [Image.open(record.image_path).convert("RGB") for record in val_records]
    reference_vectors = embed_images(session, input_name, reference_images)

    exact_vectors = embed_images(session, input_name, [Image.open(record.image_path).convert("RGB") for record in val_records])
    exact_metrics = retrieve_metrics(exact_vectors, val_records, reference_vectors, val_records)

    rng = random.Random(args.seed)
    stream_records: list[ManifestRecord] = []
    stream_images: list[Image.Image] = []
    for record in val_records:
        for _ in range(max(1, args.stream_query_count)):
            image = Image.open(record.image_path).convert("RGB")
            stream_images.append(make_stream_like(image, rng))
            stream_records.append(record)
    stream_vectors = embed_images(session, input_name, stream_images)
    stream_metrics = retrieve_metrics(stream_vectors, stream_records, reference_vectors, val_records)

    model_bytes = Path(args.model).read_bytes()
    result = {
        "model_path": str(Path(args.model).resolve()),
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "embedding_dim": EXPECTED_DIM,
        "image_size": IMAGE_SIZE,
        "crop_inset_ratio": CROP_INSET_RATIO,
        "normalization": {
            "mean": MEAN.tolist(),
            "std": STD.tolist(),
        },
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "reference_count": len(val_records),
        "stream_query_count": max(1, args.stream_query_count),
        "exact_metrics": asdict(exact_metrics),
        "stream_metrics": asdict(stream_metrics),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
