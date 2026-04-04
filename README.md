# Pokemon TCG Corpus

Precomputed ORB feature descriptors for all English Pokemon TCG cards (~20,000 cards), used for local image-based card identification.

This repo also includes detector tooling in [training/README.md](/Users/rabelson/Documents/GitHub/pokemon-tcg-corpus/training/README.md) for preparing and training a card-localization model.

## SQLite Assets For CardHawk

The repo now includes GitHub Actions workflows that publish the SQLite assets consumed by CardHawk:

- `prices-latest` publishes `prices.db.zip`
- `embeddings-latest` publishes `embeddings.db.zip`
- versioned `embeddings-v*` releases preserve rollback history for embeddings builds

Relevant entry points:

- [scripts/build_prices_db.py](/Users/rabelson/Documents/GitHub/pokemon-tcg-corpus/scripts/build_prices_db.py)
- [scripts/build_embeddings_db.py](/Users/rabelson/Documents/GitHub/pokemon-tcg-corpus/scripts/build_embeddings_db.py)
- [scripts/prune_embeddings_releases.py](/Users/rabelson/Documents/GitHub/pokemon-tcg-corpus/scripts/prune_embeddings_releases.py)
- [.github/workflows/prices.yml](/Users/rabelson/Documents/GitHub/pokemon-tcg-corpus/.github/workflows/prices.yml)
- [.github/workflows/embeddings.yml](/Users/rabelson/Documents/GitHub/pokemon-tcg-corpus/.github/workflows/embeddings.yml)

Workflow prerequisites:

- GitHub Actions secret: `POKEMONTCG_API_KEY`
- The embeddings workflow uses the exact CardHawk runtime ONNX embedder from [card_embedder.onnx](/Users/rabelson/Documents/GitHub/pokemon-tcg-corpus/models/card_embedder.onnx) and mirrors the app preprocessing contract.
- The embeddings workflow is incremental by default: it downloads the current `embeddings-latest` asset when available and only computes vectors for missing `card_id` rows unless `force_rebuild` is set.
- The embeddings workflow now requires a promoted production model manifest at [models/card_embedder.manifest.json](/Users/rabelson/Documents/GitHub/pokemon-tcg-corpus/models/card_embedder.manifest.json). Release builds must not use ad hoc exports directly from `training/exports/`.

## Format

Each release contains `pokemon_tcg_corpus_v{version}.zip` with three files:

- **`index.json`** — Card metadata (id, name, set, rarity, searchable OCR text) plus descriptor byte offsets
- **`descriptors.bin`** — Concatenated raw ORB descriptor bytes (200 features × 32 bytes per card)
- **`coarse_index.bin`** — One compact fixed-length coarse signature per card for fast full-corpus shortlist retrieval

## Details

- Source: [pokemontcg.io](https://pokemontcg.io) (English cards only)
- ORB features: 200 per card, HARRIS_SCORE, 8-level pyramid
- Image size at descriptor extraction: 480×680 grayscale
- Descriptor format: `uint8`, shape `(descriptorCount, 32)`, stored at `descriptorOffset` bytes into `descriptors.bin`
- Coarse index format: `uint8`, one 32-byte signature per card, aligned to card order in `index.json`
- Search text includes the card name plus lightweight textual metadata such as set name, attacks, rules, and flavor text for OCR-assisted reranking

## Training

See [training/README.md](/Users/rabelson/Documents/GitHub/pokemon-tcg-corpus/training/README.md) for:

- retrieval embedder training, evaluation, ONNX export, and promotion
- detector frame preparation
- YOLO detector training
- detector ONNX export

## License

This repository's original work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). If you reuse or redistribute the corpus packaging, indexes, build scripts, or derived original materials from this repo, you must provide attribution to the repository author.

Suggested attribution:

`Pokemon TCG Corpus by Rabelson (https://github.com/rabelson97/pokemon-tcg-corpus), licensed CC BY 4.0`

Important carve-out:

- Pokemon names, card art, logos, trademarks, and other third-party IP are not owned by this repository author and are not newly licensed by this repository.
- Source card data and images remain subject to their original upstream terms and rights.
- The CC BY 4.0 license here is intended to cover this repo's original compilation, indexing, training/export scripts, and other original repository-authored material.
