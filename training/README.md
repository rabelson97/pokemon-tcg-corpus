# Training Pipelines

This directory contains two distinct tracks:

1. Retrieval embedder training and promotion for `models/card_embedder.onnx`
2. Detector training and ONNX export for card localization

The retrieval embedder is the production-critical asset used by both:

- repository inference and downstream consumers of the exported ONNX model
- the `embeddings.db` release pipeline

Because of that, a candidate model must never be copied directly into `models/` without evaluation and promotion.

## Retrieval embedder workflow

Train a retrieval checkpoint:

```bash
python3 training/train_retrieval.py \
  --manifest training/data/full/manifest.jsonl \
  --output training/checkpoints/card_retrieval_candidate.pt
```

Export a candidate ONNX model:

```bash
python3 training/export_card_embedder_onnx.py \
  --checkpoint training/checkpoints/card_retrieval_candidate.pt \
  --output training/exports/card_embedder_candidate.onnx
```

Evaluate the candidate against the repository contract:

```bash
python3 training/evaluate_card_embedder.py \
  --manifest training/data/full/manifest.jsonl \
  --model training/exports/card_embedder_candidate.onnx \
  --output training/exports/card_embedder_candidate.eval.json
```

Promote only if evaluation passes:

```bash
python3 training/promote_card_embedder.py \
  --candidate-model training/exports/card_embedder_candidate.onnx \
  --evaluation-json training/exports/card_embedder_candidate.eval.json
```

Promotion writes:

- `models/card_embedder.onnx`
- `models/card_embedder.manifest.json`

The release workflow validates that manifest before building `embeddings.db.zip`.

## Detector workflow

This `training/` directory also includes detector work:

1. `prepare_detector_frames.py`
- Extracts frames from a local stream recording with `ffmpeg`
- Converts them into detector-friendly JPEGs
- Skips near-duplicate frames using an average-hash dedupe pass
- Creates `images/`, `labels/`, and `frames_manifest.jsonl` so you can start annotating immediately

2. `train_detector.py`
- Fine-tunes a YOLO detector on a Roboflow-style YOLO export
- Intended for the "where is the card?" stage of a card-recognition pipeline
- Produces `best.pt` / `last.pt` weights in `training/detector_runs/...`

3. `export_detector_onnx.py`
- Exports a trained YOLO detector checkpoint to ONNX

## Quick start

Install detector dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r training/requirements.txt
```

## Preparing detector data from stream recordings

If you have a Whatnot screen recording, this is the fastest way to build a detector-labeling set without taking hundreds of manual screenshots.

Extract and dedupe frames:

```bash
python3 training/prepare_detector_frames.py \
  --video /path/to/whatnot-session.mp4 \
  --output-dir training/data/detector/session_001 \
  --fps 0.5 \
  --max-frames 250
```

That will create:

- `training/data/detector/session_001/images/`
- `training/data/detector/session_001/labels/`
- `training/data/detector/session_001/frames_manifest.jsonl`

Recommended settings:

- `--fps 0.5` for one frame every 2 seconds
- `--fps 1.0` if the stream changes cards quickly
- `--max-frames 150-300` for an initial labeling batch

Labeling guidance for the first detector pass:

- Draw one box around the main visible card or slab you want tracked
- Use the outer visible border of the card or slab, not the art box
- Treat raw cards and slabs as the same class for the first version
- Skip frames where the card is too occluded to label confidently

## Training a detector from the Roboflow export

You already downloaded a ready-made Roboflow dataset export. To train a baseline detector from it:

```bash
python3 training/train_detector.py \
  --data training/data/roboflow/pokemon-card-identification-v1/pokemon-card-identification.v1i.yolov8/data.yaml \
  --model yolov8n.pt \
  --epochs 30 \
  --imgsz 960 \
  --batch 16
```

Notes:

- `yolov8n.pt` is the fastest baseline.
- `yolov8s.pt` is a reasonable next step if `n` underfits.
- On Apple Silicon, `--device mps` is the default in the script.
- This detector solves localization only. Identity still comes from the SQLite embeddings corpus built by the repo-level embeddings workflow.

## Exporting detector weights to ONNX

```bash
python3 training/export_detector_onnx.py \
  --weights training/detector_runs/pokemon_card_detector/weights/best.pt \
  --imgsz 960
```

## Notes

- This is intentionally a first-pass detector stack, not a finished production recognizer.
- A complete consumer pipeline would typically use:
  - detector/tracker
  - ONNX embedder inference on the detected crop
  - nearest-neighbor retrieval against the published SQLite embeddings corpus
  - temporal stabilization
  - local price lookup
- CardHawk's current shipped path is embedding-first. Treat OCR as an optional experiment, not the primary runtime identifier.
