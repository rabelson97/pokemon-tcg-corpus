#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate that the production embedder is explicitly promoted and hash-locked.")
    parser.add_argument("--model", default="models/card_embedder.onnx")
    parser.add_argument("--manifest", default="models/card_embedder.manifest.json")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    manifest_path = Path(args.manifest).resolve()
    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path}")
    if not manifest_path.exists():
        raise SystemExit(f"Manifest file not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "production-ready":
        raise SystemExit(f"Model manifest status must be production-ready, got: {manifest.get('status')}")

    actual_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    expected_sha = manifest.get("model_sha256")
    if actual_sha != expected_sha:
        raise SystemExit(f"Model sha256 mismatch: expected {expected_sha}, got {actual_sha}")

    metrics = manifest.get("metrics") or {}
    thresholds = manifest.get("thresholds") or {}
    comparisons = [
        ("exact_recall_at_1", "min_exact_recall_at_1"),
        ("stream_recall_at_1", "min_stream_recall_at_1"),
        ("stream_recall_at_5", "min_stream_recall_at_5"),
    ]
    failures = []
    for metric_key, threshold_key in comparisons:
        metric_value = metrics.get(metric_key)
        threshold_value = thresholds.get(threshold_key)
        if metric_value is None or threshold_value is None:
            failures.append(f"Missing {metric_key} or {threshold_key} in manifest")
            continue
        if float(metric_value) < float(threshold_value):
            failures.append(f"{metric_key}={metric_value} < {threshold_key}={threshold_value}")
    if failures:
        raise SystemExit("Manifest thresholds not satisfied: " + ", ".join(failures))

    print(json.dumps({"model": str(model_path), "sha256": actual_sha, "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
