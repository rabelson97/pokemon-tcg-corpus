# Training

Local training utilities for the corpus models.

This directory is for model experiments and exports. It is not required for the
daily prices workflow. Release embeddings are built with the promoted ONNX model
in `models/`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r training/requirements.txt
```

## Retrieval Embedder

Build an English manifest:

```bash
python scripts/build_training_manifest.py \
  --output training/data/full/manifest.jsonl \
  --image-dir training/data/full/images \
  --locales en \
  --summary-json training/data/full/manifest.summary.json
```

Train a candidate:

```bash
python training/train_retrieval.py \
  --manifest training/data/full/manifest.jsonl \
  --output training/checkpoints/card_retrieval_candidate.pt
```

Export to ONNX:

```bash
python training/export_card_embedder_onnx.py \
  --checkpoint training/checkpoints/card_retrieval_candidate.pt \
  --output training/exports/card_embedder_candidate.onnx
```

Evaluate the candidate:

```bash
python training/evaluate_card_embedder.py \
  --manifest training/data/full/manifest.jsonl \
  --model training/exports/card_embedder_candidate.onnx \
  --output training/exports/card_embedder_candidate.eval.json
```

Promote only after evaluation:

```bash
python training/promote_card_embedder.py \
  --candidate-model training/exports/card_embedder_candidate.onnx \
  --evaluation-json training/exports/card_embedder_candidate.eval.json
```

Promotion updates:

- `models/card_embedder.onnx`
- `models/card_embedder.manifest.json`

After promotion, rebuild embeddings:

```bash
python scripts/rebuild_embeddings_local.py
```

## Augmentation Samples

Render sample augmentations before committing to a new training profile:

```bash
python training/render_augment_samples.py \
  --manifest training/data/full/manifest.jsonl \
  --augment-profile targeted_v1 \
  --output-dir training/augment_samples/targeted_v1
```

## Detector Tools

Prepare frames from local video:

```bash
python training/prepare_detector_frames.py \
  --video /path/to/session.mp4 \
  --output-dir training/data/detector/session_001 \
  --fps 0.5 \
  --max-frames 250
```

Train a detector from a YOLO dataset:

```bash
python training/train_detector.py \
  --data training/data/roboflow/pokemon-card-identification-v1/pokemon-card-identification.v1i.yolov8/data.yaml \
  --model yolov8n.pt \
  --epochs 30 \
  --imgsz 960 \
  --batch 16
```

Export detector ONNX:

```bash
python training/export_detector_onnx.py \
  --checkpoint training/detector_runs/path/to/best.pt \
  --output training/exports/card_detector_candidate.onnx
```

## Rules

- Do not copy candidate models directly into `models/`.
- Do not promote a candidate without an evaluation JSON.
- Keep manifests English-only unless the release scope changes.
- Keep generated training data, checkpoints, and exports out of git unless a
  specific artifact is intentionally promoted.
