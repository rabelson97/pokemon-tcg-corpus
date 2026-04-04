#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote a validated candidate embedder to the production models/ path.")
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--evaluation-json", required=True)
    parser.add_argument("--output-model", default="models/card_embedder.onnx")
    parser.add_argument("--output-manifest", default="models/card_embedder.manifest.json")
    parser.add_argument("--min-exact-recall-at-1", type=float, default=0.99)
    parser.add_argument("--min-stream-recall-at-1", type=float, default=0.20)
    parser.add_argument("--min-stream-recall-at-5", type=float, default=0.60)
    args = parser.parse_args()

    candidate_model = Path(args.candidate_model).resolve()
    evaluation_path = Path(args.evaluation_json).resolve()
    if not candidate_model.exists():
        raise SystemExit(f"Candidate model not found: {candidate_model}")
    if not evaluation_path.exists():
        raise SystemExit(f"Evaluation JSON not found: {evaluation_path}")

    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    exact = evaluation["exact_metrics"]
    stream = evaluation["stream_metrics"]
    checks = {
        "exact_recall_at_1": (float(exact["recall_at_1"]), args.min_exact_recall_at_1),
        "stream_recall_at_1": (float(stream["recall_at_1"]), args.min_stream_recall_at_1),
        "stream_recall_at_5": (float(stream["recall_at_5"]), args.min_stream_recall_at_5),
    }
    failures = [
        f"{name}={actual:.4f} < {threshold:.4f}"
        for name, (actual, threshold) in checks.items()
        if actual < threshold
    ]
    if failures:
        raise SystemExit("Candidate model failed promotion thresholds: " + ", ".join(failures))

    model_bytes = candidate_model.read_bytes()
    model_sha = hashlib.sha256(model_bytes).hexdigest()
    if model_sha != evaluation["model_sha256"]:
        raise SystemExit("Candidate model hash does not match evaluation JSON")

    output_model = Path(args.output_model).resolve()
    output_manifest = Path(args.output_manifest).resolve()
    output_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate_model, output_model)

    manifest = {
        "model_name": "cardhawk:card_embedder.onnx",
        "status": "production-ready",
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "model_sha256": model_sha,
        "candidate_source": str(candidate_model),
        "evaluation_source": str(evaluation_path),
        "embedding_dim": int(evaluation["embedding_dim"]),
        "image_size": int(evaluation["image_size"]),
        "crop_inset_ratio": float(evaluation["crop_inset_ratio"]),
        "normalization": evaluation["normalization"],
        "metrics": {
            "exact_recall_at_1": float(exact["recall_at_1"]),
            "exact_recall_at_5": float(exact["recall_at_5"]),
            "stream_recall_at_1": float(stream["recall_at_1"]),
            "stream_recall_at_5": float(stream["recall_at_5"]),
            "stream_mean_top1_score": float(stream["mean_top1_score"]),
            "stream_median_top1_score": float(stream["median_top1_score"]),
        },
        "thresholds": {
            "min_exact_recall_at_1": args.min_exact_recall_at_1,
            "min_stream_recall_at_1": args.min_stream_recall_at_1,
            "min_stream_recall_at_5": args.min_stream_recall_at_5,
        },
    }
    output_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
