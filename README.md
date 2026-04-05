# Pokemon TCG Corpus

Release source for CardHawk's Pokemon card retrieval corpus and local price database.

The current repository outputs are:

- `embeddings.db.zip`: card metadata plus normalized embedding vectors built with the promoted ONNX embedder
- `prices.db.zip`: local card market-price snapshot keyed by Pokemon card id
- training tooling for the retrieval embedder and a detector-localization model

This repo is upstream data/model infrastructure. Code is the source of truth for the live app runtime, but the current CardHawk consumer path is:

`detect -> embed -> retrieve -> stabilize -> price -> publish`

In practice that means the app:

- bundles `card_detector.onnx` and `card_embedder.onnx`
- downloads or syncs `embeddings.db` and `prices.db`
- computes a card embedding from the detected crop
- retrieves nearest-neighbor candidates from the local embeddings database
- applies temporal stabilization before showing price data from the local prices database

This repo also includes detector tooling in [training/README.md](/Users/rabelson/Documents/GitHub/pokemon-tcg-corpus/training/README.md).

## SQLite Assets

The repo includes GitHub Actions workflows that publish standalone SQLite assets built from this corpus:

- `prices-latest` publishes `prices.db.zip`
- `embeddings-latest` publishes `embeddings.db.zip`
- versioned `embeddings-v*` releases preserve rollback history for embeddings builds

Relevant entry points:

- [scripts/build_prices_db.py](/Users/rabelson/Documents/GitHub/pokemon-tcg-corpus/scripts/build_prices_db.py)
- [scripts/build_embeddings_db.py](/Users/rabelson/Documents/GitHub/pokemon-tcg-corpus/scripts/build_embeddings_db.py)
- [scripts/prune_embeddings_releases.py](/Users/rabelson/Documents/GitHub/pokemon-tcg-corpus/scripts/prune_embeddings_releases.py)
- [.github/workflows/prices.yml](/Users/rabelson/Documents/GitHub/pokemon-tcg-corpus/.github/workflows/prices.yml)
- [.github/workflows/build-embeddings-db.yml](/Users/rabelson/Documents/GitHub/pokemon-tcg-corpus/.github/workflows/build-embeddings-db.yml)

Workflow prerequisites:

- GitHub Actions secret: `POKEMONTCG_API_KEY`
- The embeddings workflow uses the promoted ONNX embedder at [models/card_embedder.onnx](/Users/rabelson/Documents/GitHub/pokemon-tcg-corpus/models/card_embedder.onnx).
- The embeddings build applies the same crop inset, resize, and normalization contract that downstream CardHawk runtime code expects from this embedder family.
- The embeddings workflow is incremental by default: it downloads the current `embeddings-latest` asset when available and only computes vectors for missing `card_id` rows unless `force_rebuild` is set.
- The embeddings workflow now requires a promoted production model manifest at [models/card_embedder.manifest.json](/Users/rabelson/Documents/GitHub/pokemon-tcg-corpus/models/card_embedder.manifest.json). Release builds must not use ad hoc exports directly from `training/exports/`.

## Release Format

### `embeddings.db.zip`

SQLite database with:

- `cards`
  - `id`
  - `name`
  - `set_code`
  - `set_name`
  - `card_number`
  - `rarity`
- `embeddings`
  - `card_id`
  - `model_name`
  - `dim`
  - `vector_blob`

Current build contract:

- one normalized float32 embedding vector per card
- embedding dimension: `256`
- model name written by the builder: `cardhawk:card_embedder.onnx`
- incremental rebuild support from an existing compatible `embeddings.db`

### `prices.db.zip`

SQLite database with:

- `prices`
  - `card_id`
  - `market_price`
  - `updated_at`

## Details

- Source: [pokemontcg.io](https://pokemontcg.io) (English cards only)
- Retrieval source image: canonical Pokemon card art from `images.large` with fallback to `images.small`
- Embedder preprocessing: crop inset ratio `0.08`, resize to `224x224`, ImageNet-style mean/std normalization
- Embedding storage: little-endian float32 blob, one row per `card_id`
- Prices source: `tcgplayer.prices` market data from pokemontcg.io, reduced to one local `market_price` per card
- The app currently uses local retrieval and local price lookup after a stable match. This repo is not the place to document app-only thresholds or UI behavior.

## Training

See [training/README.md](/Users/rabelson/Documents/GitHub/pokemon-tcg-corpus/training/README.md) for:

- retrieval embedder training, evaluation, ONNX export, and promotion
- detector frame preparation
- YOLO detector training
- detector ONNX export

## Consumer Notes

- CardHawk currently uses embedding retrieval as the primary identification path.
- OCR may still be useful for offline experiments or future disambiguation work, but it is not the current primary runtime identifier.
- If the app runtime contract changes, update this README in the same change so this repo does not drift into planning-doc fiction.

## License

This repository's original work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). If you reuse or redistribute the corpus packaging, indexes, build scripts, or derived original materials from this repo, you must provide attribution to the repository author.

Suggested attribution:

`Pokemon TCG Corpus by Rabelson (https://github.com/rabelson97/pokemon-tcg-corpus), licensed CC BY 4.0`

Important carve-out:

- Pokemon names, card art, logos, trademarks, and other third-party IP are not owned by this repository author and are not newly licensed by this repository.
- Source card data and images remain subject to their original upstream terms and rights.
- The CC BY 4.0 license here is intended to cover this repo's original compilation, indexing, training/export scripts, and other original repository-authored material.
