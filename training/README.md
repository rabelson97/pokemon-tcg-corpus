# Detector Pipeline

The custom embedding training pipeline has been removed. Card embeddings are now built directly from a public pretrained `open_clip` model in [build_embeddings_db.py](/Users/rabelson/Documents/GitHub/pokemon-tcg-corpus/scripts/build_embeddings_db.py).

This `training/` directory is now only for detector work:

1. `prepare_detector_frames.py`
- Extracts frames from a local stream recording with `ffmpeg`
- Converts them into detector-friendly JPEGs
- Skips near-duplicate frames using an average-hash dedupe pass
- Creates `images/`, `labels/`, and `frames_manifest.jsonl` so you can start annotating immediately

2. `train_detector.py`
- Fine-tunes a YOLO detector on a Roboflow-style YOLO export
- Intended for the "where is the card?" stage of the app pipeline
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
- The app side should eventually use:
  - detector/tracker
  - embedding lookup against the published SQLite corpus
  - ANN lookup against exported embeddings
  - OCR as reranking/disambiguation
