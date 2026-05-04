# Embeddings Update Runbook

Use this when you are trying to improve CardHawk retrieval without getting lost in the full training pipeline.

## Decide Which Path You Need

### Path A: Rebuild the embeddings database only

Use this when:

- you changed card metadata handling
- you changed which locales are included
- you changed image URLs or upstream card records
- you want a fresh `embeddings.db` from the already-promoted production embedder

You do **not** need to retrain the model for this path.

Run:

```bash
python3 scripts/rebuild_embeddings_local.py
```

Outputs:

- `build/local-embeddings/embeddings.db`
- `build/local-embeddings/embeddings.db.zip`
- `build/local-embeddings/embeddings-build-summary.json`

Useful smoke test:

```bash
python3 scripts/rebuild_embeddings_local.py --limit 500 --min-row-count 100
```

### Path B: Retrain and promote a new embedder model

Use this only when the current ONNX embedder itself is the problem.

Run in order:

```bash
python3 scripts/build_training_manifest.py \
  --output training/data/full/manifest.jsonl \
  --image-dir training/data/full/images \
  --locales en,ja \
  --summary-json training/data/full/manifest.summary.json

python3 training/train_retrieval.py \
  --manifest training/data/full/manifest.jsonl \
  --output training/checkpoints/card_retrieval_candidate.pt

python3 training/export_card_embedder_onnx.py \
  --checkpoint training/checkpoints/card_retrieval_candidate.pt \
  --output training/exports/card_embedder_candidate.onnx

python3 training/evaluate_card_embedder.py \
  --manifest training/data/full/manifest.jsonl \
  --model training/exports/card_embedder_candidate.onnx \
  --output training/exports/card_embedder_candidate.eval.json

python3 training/promote_card_embedder.py \
  --candidate-model training/exports/card_embedder_candidate.onnx \
  --evaluation-json training/exports/card_embedder_candidate.eval.json
```

After promotion, rebuild the database:

```bash
python3 scripts/rebuild_embeddings_local.py
```

## Recommended Workflow For CardHawk Problems

For most live-recognition failures, use this order:

1. Reproduce the failure in CardHawk and save screenshots or session clips.
2. Decide whether the error is:
   - stale UI / pipeline logic in `cardhawk`
   - missing or weak metadata in `pokemon-tcg-corpus`
   - a true embedder-quality issue
3. If the promoted model is still correct but the corpus needs a fresh rebuild, run Path A.
4. If the promoted model cannot separate the target cards even with correct metadata, use Path B.

## Current Seel / Holo-Type Problem

For wrong-printing mistakes such as holo vs non-holo:

- first check whether the corpus actually distinguishes the printings cleanly
- if metadata or upstream IDs are wrong, fix corpus data and rebuild with Path A
- if the corpus is correct but the embedder still collapses those printings together, gather a replay/eval slice and train a better candidate with Path B

Do not jump to model retraining before verifying the corpus rows and print-variant labeling.

## What To Automate With AI

The easiest automation targets are:

1. failure triage:
   - have Codex inspect a screenshot/log pair and classify whether the bug belongs in `cardhawk` or `pokemon-tcg-corpus`
2. local rebuild:
   - use `python3 scripts/rebuild_embeddings_local.py` instead of reconstructing the builder command
3. replay-driven experiments:
   - keep a folder of known-bad Whatnot screenshots/clips and ask Codex to convert them into a concrete checklist of corpus/runtime fixes

## Release Note

This runbook is for local rebuilds and local decision-making.

The GitHub Actions workflow still remains the release path for published `embeddings-latest` and versioned `embeddings-v*` assets.
