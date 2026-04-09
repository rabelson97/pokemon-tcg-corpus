# Pokemon TCG Corpus

Release source for CardHawk's locale-first Pokemon card retrieval corpus and local price database.

Current release rollout scope: `en`, `ja`, and `fr`.

The current repository outputs are:

- `embeddings.db.zip`: locale-first card metadata plus normalized embedding vectors built with the promoted ONNX embedder
- `prices.db.zip`: market-aware local price snapshot keyed by locale-first Pokemon card id
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

- `prices-latest` publishes `prices.db.zip` when the rebuilt DB content differs from the currently published asset
- `embeddings-latest` publishes `embeddings.db.zip`
- versioned `embeddings-v*` releases preserve rollback history for embeddings builds

Relevant entry points:

- [scripts/build_prices_db.py](/Users/rabelson/Documents/GitHub/pokemon-tcg-corpus/scripts/build_prices_db.py)
- [scripts/build_embeddings_db.py](/Users/rabelson/Documents/GitHub/pokemon-tcg-corpus/scripts/build_embeddings_db.py)
- [scripts/build_training_manifest.py](/Users/rabelson/Documents/GitHub/pokemon-tcg-corpus/scripts/build_training_manifest.py)
- [scripts/prune_embeddings_releases.py](/Users/rabelson/Documents/GitHub/pokemon-tcg-corpus/scripts/prune_embeddings_releases.py)
- [.github/workflows/prices.yml](/Users/rabelson/Documents/GitHub/pokemon-tcg-corpus/.github/workflows/prices.yml)
- [.github/workflows/build-embeddings-db.yml](/Users/rabelson/Documents/GitHub/pokemon-tcg-corpus/.github/workflows/build-embeddings-db.yml)

Workflow notes:

- The embeddings workflow uses the promoted ONNX embedder at [models/card_embedder.onnx](/Users/rabelson/Documents/GitHub/pokemon-tcg-corpus/models/card_embedder.onnx).
- The embeddings build applies the same crop inset, resize, and normalization contract that downstream CardHawk runtime code expects from this embedder family.
- The embeddings workflow is a fresh locale-first rebuild from TCGdex, not an incremental extension of the old English-only corpus.
- The current automated release workflows publish the Phase B locale set: `en`, `ja`, and `fr`.
- The embeddings workflow requires a promoted production model manifest at [models/card_embedder.manifest.json](/Users/rabelson/Documents/GitHub/pokemon-tcg-corpus/models/card_embedder.manifest.json). Release builds must not use ad hoc exports directly from `training/exports/`.
- TCGdex card payloads expose an asset base URL in `image`; the builders normalize that to the final localized binary art path at `.../high.webp`.
- Some upstream cards still have no localized art URL. Manifest and embeddings builds skip those rows explicitly and report the skipped counts and reasons by locale.
- The prices workflow writes `prices-build-summary.json` with a per-locale source coverage audit, provider transport diagnostics, and a deterministic content hash, then skips release upload when that hash matches the current `prices-latest` asset.
- The prices workflow can use the existing `POKEMONTCG_API_KEY` secret for higher-rate US price fetches while keeping the published SQLite contract unchanged.
- The April 2026 English USD provider audit and deterministic gap sample live in [docs/english_price_provider_comparison.md](/Users/rabelson/Documents/GitHub/pokemon-tcg-corpus/docs/english_price_provider_comparison.md) and were generated with [scripts/build_english_price_gap_sample.py](/Users/rabelson/Documents/GitHub/pokemon-tcg-corpus/scripts/build_english_price_gap_sample.py).

## Release Format

### `embeddings.db.zip`

SQLite database with:

- `cards`
  - `id`
  - `locale`
  - `upstream_id`
  - `set_id`
  - `set_name`
  - `card_number`
  - `name`
  - `rarity`
  - `image_url`
  - `equivalence_key`
- `embeddings`
  - `card_id`
  - `model_name`
  - `dim`
  - `vector_blob`
- `card_equivalents`
  - `card_id`
  - `equivalence_key`
  - `upstream_source`
  - `upstream_id`
  - `locale`
  - `set_id`
  - `local_id`

Current build contract:

- one normalized float32 embedding vector per card
- embedding dimension: `256`
- model name written by the builder: `cardhawk:card_embedder.onnx`
- canonical card id format: `{game}:{locale}:{set_id}:{local_id}`

### `prices.db.zip`

SQLite database with:

- `prices`
  - `card_id`
  - `market_code`
  - `currency_code`
  - `source_name`
  - `low_price`
  - `market_price`
  - `high_price`
  - `updated_at`
  - `is_primary`

## Details

- Source: [TCGdex](https://api.tcgdex.net) locale-scoped REST API
- Retrieval source image: canonical localized Pokemon card art from normalized TCGdex asset URLs ending in `/high.webp`
- Embedder preprocessing: crop inset ratio `0.08`, resize to `224x224`, ImageNet-style mean/std normalization
- Embedding storage: little-endian float32 blob, one row per `card_id`
- Prices source: US `tcgplayer` rows are selected from PokemonTCG.io's English card feed when the matched card's `tcgplayer.updatedAt` is present and within the builder freshness window; `pricing.cardmarket` from TCGdex remains the EU fallback/reference row
- Price row contract: exactly one `is_primary = 1` row for each `card_id` present in `prices`, with `cardmarket` promoted to primary when `tcgplayer` is missing
- Prices build audit: per locale, the builder reports cards with `tcgplayer`, cards with `cardmarket`, cards with both, cards with neither, and which source ended up primary
- Prices build metadata: `prices-build-summary.json` also records provider transport counts plus PokemonTCG.io fetch/match/staleness diagnostics so release audits can distinguish source selection from database shape
- The app currently uses local retrieval and local price lookup after a stable match. This repo is not the place to document app-only thresholds or UI behavior.

## Training

See [training/README.md](/Users/rabelson/Documents/GitHub/pokemon-tcg-corpus/training/README.md) for:

- multilingual retrieval embedder manifest generation, training, evaluation, ONNX export, and promotion
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
