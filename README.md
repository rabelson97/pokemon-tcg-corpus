# Pokemon TCG Corpus

Public build repository for an English-only Pokemon TCG card corpus.

This repo builds and publishes two SQLite assets:

- `embeddings.db.zip`: English card metadata plus visual embedding vectors.
- `prices.db.zip`: English card price rows keyed by the same card ids.

The repository is intentionally self-contained. The code, workflows, and release
assets are the public contract. Private application details and provider
credentials are not documented here.

## Scope

Current release scope is English cards only.

The corpus is built from public card metadata and public card image URLs. Every
scan-eligible card in the embeddings database must have a verified public HTTPS
image URL before the database is published. Cards without usable image art are
blocked rather than filled with placeholder images.

The prices database is built against the card universe from the current
`embeddings-latest` release. Every English card in that universe must have a
primary USD row before the prices release can publish. When a public USD market
price is unavailable for a known edge case, the row is source-labeled so the
database distinguishes "no USD market" from an actual market price.

## Published Releases

GitHub Actions publishes rolling release assets:

- `embeddings-latest`: latest `embeddings.db.zip`
- `prices-latest`: latest `prices.db.zip`
- `embeddings-v*`: versioned embedding releases for rollback history

The workflows only upload a new asset when the rebuilt database content differs
from the current release.

## Database Shape

### `embeddings.db`

Tables:

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
  - `image_url_low`
  - `equivalence_key`
  - `hp`
  - `types`
- `embeddings`
  - `card_id`
  - `model_name`
  - `variant_idx`
  - `variant_tag`
  - `dim`
  - `vector_blob`
- `embeddings_int8`
  - `card_id`
  - `variant_idx`
  - `variant_tag`
  - `dim`
  - `vector_int8`
- `cards_fts`
  - `id`
  - `locale`
  - `name`
  - `set_name`
  - `set_id`
  - `card_number`
  - `rarity`
- `card_equivalents`
  - `card_id`
  - `equivalence_key`
  - `upstream_source`
  - `upstream_id`
  - `locale`
  - `set_id`
  - `local_id`

Current embedding contract:

- four normalized float32 variant vectors per scan-eligible card
- one int8 vector per card variant
- one FTS5 row per card for local name and metadata search
- embedding dimension: `256`
- model name: written by the embeddings builder
- canonical id format: `pokemon:en:{set_id}:{local_id}`
- primary image field: `image_url`
- optional lower-resolution image field: `image_url_low`

### `prices.db`

Table:

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

Current price contract:

- exactly one `is_primary = 1` row per priced card id
- English release builds require full USD coverage for the embeddings universe
- USD sources keep their provenance in `source_name`
- fallback rows are not relabeled as another provider

Recognized USD source labels include:

- `tcgplayer`
- `pokemonpricetracker`
- `poketrace`
- `pkmngg`
- `pricecharting`
- `scrydex`
- `limitless`
- `no_usd_market`

## Build Entry Points

Main scripts:

- `scripts/build_embeddings_db.py`
- `scripts/rebuild_embeddings_local.py`
- `scripts/build_prices_db.py`
- `scripts/build_training_manifest.py`
- `scripts/prune_embeddings_releases.py`

Main workflows:

- `.github/workflows/build-embeddings-db.yml`
- `.github/workflows/prices.yml`

Local smoke builds:

```bash
python scripts/rebuild_embeddings_local.py --limit 500 --min-row-count 100
python scripts/build_prices_db.py --output build/local-prices/prices.db --locales en --min-row-count 100
```

Release builds are expected to run in GitHub Actions because they rely on the
published release assets, caches, and private workflow configuration. Do not
commit provider credentials or paste them into docs.

## Image Resolution

The embeddings pipeline resolves card images before it reuses old embeddings or
writes metadata. Image resolution prefers stable public HTTPS URLs and validates
that image bytes can actually be fetched. `image-fallbacks.json` is the
git-tracked registry for verified image URL overrides.

Important behavior:

- no placeholder/card-back images for scan-eligible cards
- no embedding row without real card art
- no metadata upsert for a scan-eligible card whose image URL cannot be resolved
- cached image bytes can speed up rebuilds but do not replace the public URL
  requirement

## Price Resolution

The daily prices workflow starts from the current embeddings card universe and
then resolves USD rows in this order:

1. Fresh exact PokemonTCG.io / TCGplayer data.
2. Reused same-day rows when a manual rebuild is rerun.
3. Dynamic pkmn.gg set payloads.
4. Explicit scraped USD fallbacks for known public pages.
5. Optional keyed providers when configured and healthy.
6. Explicit `no_usd_market` rows for known cards with no reachable free USD
   market price.

The fallback stack is designed to degrade gracefully. If a provider is
unauthorized, rate-limited, or unavailable, the build records diagnostics and
moves on to the next source instead of publishing incomplete data.

## Training

Training code lives under `training/`. It supports:

- retrieval embedder manifest generation
- retrieval model training and evaluation
- ONNX export and promotion
- detector frame preparation
- detector training and ONNX export

See `training/README.md` for the current local training commands.

## Maintenance Rules

- Keep this README aligned with the actual workflows and SQLite schema.
- Keep the repository documented as an English-only public corpus unless the
  release workflows are changed to publish additional locales.
- Do not document private application architecture here.
- Do not commit, print, or describe credential values.
- Treat generated release databases as build artifacts; source scripts and
  tracked registries are the maintainable source of truth.

## License

This repository's original work is licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Suggested attribution:

`Pokemon TCG Corpus by Rabelson (https://github.com/rabelson97/pokemon-tcg-corpus), licensed CC BY 4.0`

Important carve-out:

- Pokemon names, card art, logos, trademarks, and other third-party IP are not
  owned by this repository author and are not newly licensed by this repository.
- Source card data and images remain subject to their original upstream terms
  and rights.
- The CC BY 4.0 license covers this repo's original compilation, indexing,
  training/export scripts, and other repository-authored material.
