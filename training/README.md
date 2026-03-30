# Training Pipeline

This is the first pass at turning the published corpus into a trained visual retrieval model instead of relying only on ORB descriptors.

## What this gives you

1. `prepare_dataset.py`
- Reads `index.json` from a corpus release zip
- Downloads the card images referenced by `imageUrl`
- Writes a `manifest.jsonl` with card ids, names, and local image paths

2. `train_embedding.py`
- Trains a simple classification-backed embedding model
- Useful as a baseline, but not the preferred path for stream retrieval

3. `train_retrieval.py`
- Trains a contrastive retrieval model with two augmented views per card
- Uses stream-like augmentation so the model sees blur, overlays, warp, and occlusion during training
- This is the recommended first model for Whatnot-style retrieval

4. `export_embeddings.py`
- Runs the trained model over the full corpus
- Writes an `.npz` file containing:
  - `card_ids`
  - `embeddings`

5. `export_onnx.py`
- Exports the trained embedding branch to ONNX for Android inference

6. `package_embedding_corpus.py`
- Reads an existing corpus zip plus exported embeddings
- Writes a new corpus zip that includes `embeddings.bin`
- Updates `index.json` with `embeddingIndex` metadata so the Android app can preload it

That packaged corpus is what the Android app can actually consume.

## Quick start

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r training/requirements.txt
```

Prepare the dataset from the latest corpus zip:

```bash
python3 training/prepare_dataset.py \
  --bundle pokemon_tcg_corpus_v3.zip \
  --output-dir training/data/full
```

Train a retrieval checkpoint:

```bash
python3 training/train_retrieval.py \
  --manifest training/data/full/manifest.jsonl \
  --output training/checkpoints/card_retrieval.pt \
  --epochs 8
```

Export embeddings:

```bash
python3 training/export_embeddings.py \
  --manifest training/data/full/manifest.jsonl \
  --checkpoint training/checkpoints/card_retrieval.pt \
  --output training/exports/card_embeddings.npz
```

Export the ONNX model for the app:

```bash
python3 training/export_onnx.py \
  --checkpoint training/checkpoints/card_retrieval.pt \
  --output training/exports/card_embedder.onnx
```

Package a new corpus release with embeddings:

```bash
python3 training/package_embedding_corpus.py \
  --base-bundle pokemon_tcg_corpus_v3.zip \
  --embedding-npz training/exports/card_embeddings.npz \
  --output pokemon_tcg_corpus_v4.zip \
  --version v4
```

## Notes

- This is intentionally a first-pass training stack, not a finished production recognizer.
- `train_retrieval.py` already injects stream-style noise and overlay simulation. The next improvement is adding slab glare, signer ink, and real captured frames from Whatnot sessions.
- The app side should eventually use:
  - detector/tracker
  - embedding inference on the tracked crop
  - ANN lookup against exported embeddings
  - OCR as reranking/disambiguation
