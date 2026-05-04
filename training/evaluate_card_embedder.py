#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import onnxruntime as ort
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from embedder_contract import CROP_INSET_RATIO, EXPECTED_DIM, IMAGE_SIZE, MEAN, STD, preprocess_image
from common import (
    ManifestRecord,
    count_records_by_locale,
    load_manifest,
    make_stream_artifact_image,
    normalize_text,
    same_art_key,
    split_records,
)


@dataclass(frozen=True)
class BenchmarkFrame:
    clip_id: str
    frame_index: int
    timestamp_millis: int
    record: ManifestRecord
    perturbation_seed: int

@dataclass
class RetrievalMetrics:
    sample_count: int
    recall_at_1: float
    recall_at_5: float
    mean_reciprocal_rank: float
    mean_rank: float
    median_top1_score: float
    mean_top1_score: float
    mean_top1_margin: float
    same_art_top1_confusion_rate: float
    failure_examples: list[dict[str, object]]


def preprocess(image: Image.Image) -> np.ndarray:
    return preprocess_image(image, image_size=IMAGE_SIZE)


def make_stream_like(image: Image.Image, rng: random.Random) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    if rng.random() < 0.85:
        angle = rng.uniform(-9.0, 9.0)
        image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
    if rng.random() < 0.95:
        translate_x = int(width * rng.uniform(-0.09, 0.09))
        translate_y = int(height * rng.uniform(-0.09, 0.09))
        padded = Image.new("RGB", (width + 32, height + 32), (0, 0, 0))
        padded.paste(image, (16 + translate_x, 16 + translate_y))
        image = padded.crop((16, 16, 16 + width, 16 + height))
    return make_stream_artifact_image(image, rng)


def embed_images(session: ort.InferenceSession, input_name: str, images: Iterable[Image.Image]) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for image in images:
        tensor = preprocess(image)
        vector = np.asarray(session.run(None, {input_name: tensor})[0][0], dtype=np.float32)
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        outputs.append(vector)
    return np.stack(outputs, axis=0)


def load_benchmark_frames(path: str | Path) -> list[BenchmarkFrame]:
    frames: list[BenchmarkFrame] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            frames.append(
                BenchmarkFrame(
                    clip_id=row["clip_id"],
                    frame_index=int(row["frame_index"]),
                    timestamp_millis=int(row["timestamp_millis"]),
                    perturbation_seed=int(row["perturbation_seed"]),
                    record=ManifestRecord(
                        card_id=row["card_id"],
                        name=row["name"],
                        set_name=row.get("set_name"),
                        number=row.get("number"),
                        image_path=row["image_path"],
                        image_url=row.get("image_url") or "",
                        locale=row.get("locale"),
                        upstream_id=row.get("upstream_id"),
                        set_id=row.get("set_id"),
                        equivalence_key=row.get("same_art_key") or row.get("equivalence_key"),
                    ),
                )
            )
    return frames


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
    reciprocal_rank_sum = 0.0
    rank_sum = 0.0
    top1_scores: list[float] = []
    top1_margins: list[float] = []
    same_art_confusions = 0
    failures: list[dict[str, object]] = []

    ref_ids = [record.card_id for record in reference_records]
    ref_id_to_record = {record.card_id: record for record in reference_records}
    for query_index, query_record in enumerate(query_records):
        row = sims[query_index]
        ranked_indices = np.argsort(row)[::-1]
        top_indices = np.argsort(row)[::-1][:top_k]
        top_cards = [(ref_ids[idx], float(row[idx])) for idx in top_indices]
        top1_scores.append(top_cards[0][1])
        if len(top_cards) > 1:
            top1_margins.append(top_cards[0][1] - top_cards[1][1])
        else:
            top1_margins.append(top_cards[0][1])
        correct_rank = next((rank + 1 for rank, idx in enumerate(ranked_indices.tolist()) if ref_ids[idx] == query_record.card_id), len(reference_records) + 1)
        reciprocal_rank_sum += 1.0 / correct_rank
        rank_sum += float(correct_rank)
        if top_cards[0][0] == query_record.card_id:
            top1_hits += 1
        else:
            predicted = ref_id_to_record[top_cards[0][0]]
            if same_art_key(predicted) == same_art_key(query_record) or normalize_text(predicted.name) == normalize_text(query_record.name):
                same_art_confusions += 1
        if any(card_id == query_record.card_id for card_id, _ in top_cards):
            top5_hits += 1
        else:
            failures.append(
                {
                    "query_card_id": query_record.card_id,
                    "query_name": query_record.name,
                    "query_locale": query_record.locale,
                    "query_set_id": query_record.set_id,
                    "correct_rank": correct_rank,
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
        mean_reciprocal_rank=reciprocal_rank_sum / max(1, sample_count),
        mean_rank=rank_sum / max(1, sample_count),
        median_top1_score=float(np.median(np.asarray(top1_scores, dtype=np.float32))),
        mean_top1_score=float(np.mean(np.asarray(top1_scores, dtype=np.float32))),
        mean_top1_margin=float(np.mean(np.asarray(top1_margins, dtype=np.float32))),
        same_art_top1_confusion_rate=same_art_confusions / max(1, sample_count),
        failure_examples=failures[:20],
    )


def metrics_by_locale(
    query_vectors: np.ndarray,
    query_records: list[ManifestRecord],
    reference_vectors: np.ndarray,
    reference_records: list[ManifestRecord],
    *,
    top_k: int = 5,
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    locales = sorted({(record.locale or "unknown").strip() or "unknown" for record in query_records})
    for locale in locales:
        indices = [index for index, record in enumerate(query_records) if (record.locale or "unknown").strip() == locale]
        locale_query_vectors = query_vectors[indices]
        locale_query_records = [query_records[index] for index in indices]
        metrics = retrieve_metrics(
            locale_query_vectors,
            locale_query_records,
            reference_vectors,
            reference_records,
            top_k=top_k,
        )
        result[locale] = {
            "sample_count": metrics.sample_count,
            "recall_at_1": metrics.recall_at_1,
            "recall_at_5": metrics.recall_at_5,
            "mean_reciprocal_rank": metrics.mean_reciprocal_rank,
            "mean_rank": metrics.mean_rank,
            "mean_top1_score": metrics.mean_top1_score,
            "median_top1_score": metrics.median_top1_score,
            "mean_top1_margin": metrics.mean_top1_margin,
            "same_art_top1_confusion_rate": metrics.same_art_top1_confusion_rate,
        }
    return result


def same_art_confusion_examples(
    query_vectors: np.ndarray,
    query_records: list[ManifestRecord],
    reference_vectors: np.ndarray,
    reference_records: list[ManifestRecord],
    *,
    limit: int = 20,
) -> list[dict[str, object]]:
    sims = query_vectors @ reference_vectors.T
    ref_ids = [record.card_id for record in reference_records]
    ref_records = {record.card_id: record for record in reference_records}
    examples: list[dict[str, object]] = []
    for query_index, query_record in enumerate(query_records):
        ranked = np.argsort(sims[query_index])[::-1]
        top_card_id = ref_ids[int(ranked[0])]
        if top_card_id == query_record.card_id:
            continue
        predicted = ref_records[top_card_id]
        if same_art_key(predicted) != same_art_key(query_record) and normalize_text(predicted.name) != normalize_text(query_record.name):
            continue
        examples.append(
            {
                "query_card_id": query_record.card_id,
                "query_name": query_record.name,
                "query_locale": query_record.locale,
                "query_set_id": query_record.set_id,
                "predicted_card_id": predicted.card_id,
                "predicted_name": predicted.name,
                "predicted_locale": predicted.locale,
                "predicted_set_id": predicted.set_id,
                "score": float(sims[query_index][int(ranked[0])]),
            }
        )
        if len(examples) >= limit:
            break
    return examples


def benchmark_topk_rows(
    query_vectors: np.ndarray,
    benchmark_frames: list[BenchmarkFrame],
    reference_vectors: np.ndarray,
    reference_records: list[ManifestRecord],
    *,
    top_k: int,
) -> list[dict[str, object]]:
    sims = query_vectors @ reference_vectors.T
    ref_ids = [record.card_id for record in reference_records]
    rows: list[dict[str, object]] = []
    for index, frame in enumerate(benchmark_frames):
        ranked_indices = np.argsort(sims[index])[::-1]
        top_indices = ranked_indices[:top_k]
        correct_rank = next((rank + 1 for rank, ref_index in enumerate(ranked_indices.tolist()) if ref_ids[ref_index] == frame.record.card_id), len(reference_records) + 1)
        rows.append(
            {
                "clip_id": frame.clip_id,
                "frame_index": frame.frame_index,
                "timestamp_millis": frame.timestamp_millis,
                "card_id": frame.record.card_id,
                "correct_rank": correct_rank,
                "top_candidates": [
                    {
                        "card_id": ref_ids[ref_index],
                        "score": float(sims[index][ref_index]),
                    }
                    for ref_index in top_indices.tolist()
                ],
            }
        )
    return rows


def clip_metrics(topk_rows: list[dict[str, object]], *, top_k: int) -> dict[str, float | int | None]:
    by_clip: dict[str, list[dict[str, object]]] = {}
    for row in topk_rows:
        by_clip.setdefault(str(row["clip_id"]), []).append(row)
    clip_count = len(by_clip)
    first_top1_frames: list[int] = []
    first_topk_frames: list[int] = []
    full_clip_topk_hits = 0
    for rows in by_clip.values():
        rows = sorted(rows, key=lambda row: int(row["frame_index"]))
        top1_frame = next((int(row["frame_index"]) for row in rows if int(row["correct_rank"]) == 1), None)
        topk_frame = next((int(row["frame_index"]) for row in rows if int(row["correct_rank"]) <= top_k), None)
        if top1_frame is not None:
            first_top1_frames.append(top1_frame)
        if topk_frame is not None:
            first_topk_frames.append(topk_frame)
            full_clip_topk_hits += 1
    return {
        "clip_count": clip_count,
        "clip_topk_success_rate": full_clip_topk_hits / max(1, clip_count),
        "mean_first_top1_frame": float(np.mean(np.asarray(first_top1_frames, dtype=np.float32))) if first_top1_frames else None,
        "mean_first_topk_frame": float(np.mean(np.asarray(first_topk_frames, dtype=np.float32))) if first_topk_frames else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a candidate embedder against the repository contract.")
    parser.add_argument("--manifest", default="training/data/full/manifest.jsonl")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stream-query-count", type=int, default=1)
    parser.add_argument("--benchmark-manifest")
    parser.add_argument("--benchmark-topk-output")
    parser.add_argument(
        "--benchmark-reference-scope",
        choices=["manifest", "validation"],
        default="manifest",
        help="Reference set for benchmark queries. Use 'manifest' for valid retrieval metrics; 'validation' preserves the old split-only behavior.",
    )
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

    benchmark_result = None
    if args.benchmark_manifest:
        frames = load_benchmark_frames(args.benchmark_manifest)
        benchmark_reference_records = records if args.benchmark_reference_scope == "manifest" else val_records
        benchmark_reference_images = [Image.open(record.image_path).convert("RGB") for record in benchmark_reference_records]
        benchmark_reference_vectors = embed_images(session, input_name, benchmark_reference_images)
        benchmark_images = []
        for frame in frames:
            image = Image.open(frame.record.image_path).convert("RGB")
            benchmark_images.append(make_stream_like(image, random.Random(frame.perturbation_seed)))
        benchmark_vectors = embed_images(session, input_name, benchmark_images)
        benchmark_records = [frame.record for frame in frames]
        benchmark_metrics = retrieve_metrics(
            benchmark_vectors,
            benchmark_records,
            benchmark_reference_vectors,
            benchmark_reference_records,
        )
        topk_rows = benchmark_topk_rows(
            benchmark_vectors,
            frames,
            benchmark_reference_vectors,
            benchmark_reference_records,
            top_k=5,
        )
        benchmark_result = {
            "benchmark_manifest": str(Path(args.benchmark_manifest).resolve()),
            "reference_scope": args.benchmark_reference_scope,
            "reference_count": len(benchmark_reference_records),
            "metrics": asdict(benchmark_metrics),
            "clip_metrics": clip_metrics(topk_rows, top_k=5),
        }
        if args.benchmark_topk_output:
            benchmark_output_path = Path(args.benchmark_topk_output).resolve()
            benchmark_output_path.parent.mkdir(parents=True, exist_ok=True)
            benchmark_output_path.write_text(json.dumps(topk_rows, indent=2), encoding="utf-8")

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
        "manifest_counts": count_records_by_locale(records),
        "validation_counts": count_records_by_locale(val_records),
        "reference_count": len(val_records),
        "stream_query_count": max(1, args.stream_query_count),
        "exact_metrics": asdict(exact_metrics),
        "stream_metrics": asdict(stream_metrics),
        "exact_metrics_by_locale": metrics_by_locale(exact_vectors, val_records, reference_vectors, val_records),
        "stream_metrics_by_locale": metrics_by_locale(stream_vectors, stream_records, reference_vectors, val_records),
        "same_art_confusion_examples": same_art_confusion_examples(
            stream_vectors,
            stream_records,
            reference_vectors,
            val_records,
        ),
        "benchmark": benchmark_result,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
