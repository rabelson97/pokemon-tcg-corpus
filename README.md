# Pokemon TCG Corpus

Precomputed ORB feature descriptors for all English Pokemon TCG cards (~20,000 cards), used for local image-based card identification.

## Format

Each release contains `pokemon_tcg_corpus_v{version}.zip` with two files:

- **`index.json`** — Card metadata (id, name, set, rarity) plus descriptor byte offsets
- **`descriptors.bin`** — Concatenated raw ORB descriptor bytes (200 features × 32 bytes per card)

## Details

- Source: [pokemontcg.io](https://pokemontcg.io) (English cards only)
- ORB features: 200 per card, HARRIS_SCORE, 8-level pyramid
- Image size at descriptor extraction: 480×680 grayscale
- Descriptor format: `uint8`, shape `(descriptorCount, 32)`, stored at `descriptorOffset` bytes into `descriptors.bin`
