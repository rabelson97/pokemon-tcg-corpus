#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

from common import load_manifest, same_art_key


REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


def run_command(*args: str) -> None:
    command = [str(PYTHON), *args]
    print("$", " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def create_subset_manifest(source: Path, destination: Path, *, limit: int, seed: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    records = load_manifest(source)
    by_group = {}
    for record in records:
        by_group.setdefault(same_art_key(record), []).append(record)
    hard_groups = [group for group in by_group.values() if len(group) > 1]
    singleton_records = [group[0] for group in by_group.values() if len(group) == 1]
    rng.shuffle(hard_groups)
    rng.shuffle(singleton_records)

    selected_records = []
    hard_target = min(len(hard_groups), max(0, limit // 4))
    for group in hard_groups[:hard_target]:
        take = min(len(group), max(2, limit // max(1, hard_target * 4)))
        selected_records.extend(rng.sample(group, k=min(take, len(group))))
        if len(selected_records) >= limit:
            break
    if len(selected_records) < limit:
        remaining = [record for record in singleton_records if record not in selected_records]
        selected_records.extend(remaining[: limit - len(selected_records)])
    if len(selected_records) < limit:
        fallback = [record for group in hard_groups[hard_target:] for record in group if record not in selected_records]
        rng.shuffle(fallback)
        selected_records.extend(fallback[: limit - len(selected_records)])
    selected_records = selected_records[:limit]

    with destination.open("w", encoding="utf-8") as dst:
        for record in selected_records:
            dst.write(
                json.dumps(
                    {
                        "card_id": record.card_id,
                        "name": record.name,
                        "set_name": record.set_name,
                        "number": record.number,
                        "image_path": record.image_path,
                        "image_url": record.image_url,
                        "locale": record.locale,
                        "upstream_id": record.upstream_id,
                        "set_id": record.set_id,
                        "equivalence_key": record.equivalence_key,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train/export/evaluate the augmented baseline embedder candidate.")
    parser.add_argument("--manifest", default="training/data/full/manifest.jsonl")
    parser.add_argument("--output-dir", default="training/benchmarks/matrix")
    parser.add_argument("--benchmark-manifest")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--subset-size", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not PYTHON.exists():
        raise SystemExit(f"Expected virtualenv python at {PYTHON}")

    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.subset_size:
        manifest_subset = output_dir / f"manifest_subset_{args.subset_size}.jsonl"
        create_subset_manifest(manifest_path, manifest_subset, limit=args.subset_size, seed=args.seed)
        manifest_path = manifest_subset

    benchmark_manifest = Path(args.benchmark_manifest).resolve() if args.benchmark_manifest else output_dir / "synthetic_benchmark.jsonl"
    benchmark_summary = output_dir / "synthetic_benchmark.summary.json"
    if not benchmark_manifest.exists():
        run_command(
            "training/build_benchmark_manifest.py",
            "--manifest",
            str(manifest_path),
            "--output",
            str(benchmark_manifest),
            "--summary-json",
            str(benchmark_summary),
            "--max-clips",
            "256" if not args.subset_size else str(min(128, max(16, args.subset_size // 4))),
            "--seed",
            str(args.seed),
        )

    configs = [
        {
            "name": "baseline_mobilenet",
        },
    ]

    summary_rows = []
    for config in configs:
        checkpoint = output_dir / f"{config['name']}.pt"
        onnx_model = output_dir / f"{config['name']}.onnx"
        eval_json = output_dir / f"{config['name']}.eval.json"
        benchmark_topk_json = output_dir / f"{config['name']}.benchmark.topk.json"

        run_command(
            "training/train_retrieval.py",
            "--manifest",
            str(manifest_path),
            "--output",
            str(checkpoint),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--seed",
            str(args.seed),
        )
        run_command(
            "training/export_card_embedder_onnx.py",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(onnx_model),
        )
        run_command(
            "training/evaluate_card_embedder.py",
            "--manifest",
            str(manifest_path),
            "--model",
            str(onnx_model),
            "--output",
            str(eval_json),
            "--benchmark-manifest",
            str(benchmark_manifest),
            "--benchmark-topk-output",
            str(benchmark_topk_json),
            "--seed",
            str(args.seed),
        )

        evaluation = load_json(eval_json)
        benchmark = evaluation.get("benchmark") or {}
        benchmark_metrics = benchmark.get("metrics") or {}
        clip_metrics = benchmark.get("clip_metrics") or {}
        summary_rows.append(
            {
                "name": config["name"],
                "backbone": "mobilenet_v3_small",
                "training_strategy": "baseline",
                "exact_recall_at_1": evaluation["exact_metrics"]["recall_at_1"],
                "stream_recall_at_1": evaluation["stream_metrics"]["recall_at_1"],
                "stream_recall_at_5": evaluation["stream_metrics"]["recall_at_5"],
                "stream_mrr": evaluation["stream_metrics"]["mean_reciprocal_rank"],
                "stream_same_art_confusion": evaluation["stream_metrics"]["same_art_top1_confusion_rate"],
                "benchmark_recall_at_1": benchmark_metrics.get("recall_at_1"),
                "benchmark_recall_at_5": benchmark_metrics.get("recall_at_5"),
                "benchmark_mrr": benchmark_metrics.get("mean_reciprocal_rank"),
                "benchmark_clip_topk_success_rate": clip_metrics.get("clip_topk_success_rate"),
                "benchmark_mean_first_top1_frame": clip_metrics.get("mean_first_top1_frame"),
                "benchmark_mean_first_topk_frame": clip_metrics.get("mean_first_topk_frame"),
                "checkpoint": str(checkpoint),
                "onnx_model": str(onnx_model),
                "evaluation_json": str(eval_json),
                "benchmark_topk_json": str(benchmark_topk_json),
            }
        )

    summary = {
        "manifest": str(manifest_path),
        "benchmark_manifest": str(benchmark_manifest),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "subset_size": args.subset_size,
        "results": summary_rows,
    }
    summary_path = output_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
