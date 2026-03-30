# Pokemon TCG Corpus

Precomputed ORB feature descriptors for all English Pokemon TCG cards (~20,000 cards), used for local image-based card identification.

This repo now also includes a first-pass model training scaffold in [training/README.md](/Users/rabelson/Documents/GitHub/pokemon-tcg-corpus/training/README.md) for building a learned visual embedding from the published corpus images.

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

- preparing image data from a published corpus zip
- training a MobileNetV3-based embedding model
- exporting corpus embeddings for ANN lookup

## License

This repository's original work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). If you reuse or redistribute the corpus packaging, indexes, build scripts, or derived original materials from this repo, you must provide attribution to the repository author.

Suggested attribution:

`Pokemon TCG Corpus by Rabelson (https://github.com/rabelson97/pokemon-tcg-corpus), licensed CC BY 4.0`

Important carve-out:

- Pokemon names, card art, logos, trademarks, and other third-party IP are not owned by this repository author and are not newly licensed by this repository.
- Source card data and images remain subject to their original upstream terms and rights.
- The CC BY 4.0 license here is intended to cover this repo's original compilation, indexing, training/export scripts, and other original repository-authored material.
