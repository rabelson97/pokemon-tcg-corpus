#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np


def quantize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    clipped = np.clip((embeddings + 1.0) * 0.5, 0.0, 1.0)
    return np.rint(clipped * 255.0).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description="Package exported embeddings into a new corpus release zip.")
    parser.add_argument("--base-bundle", required=True, help="Existing pokemon_tcg_corpus_v*.zip")
    parser.add_argument("--embedding-npz", required=True, help="Output from export_embeddings.py")
    parser.add_argument("--output", required=True, help="New corpus zip path")
    parser.add_argument("--version", required=True, help="New corpus version string, e.g. v4")
    args = parser.parse_args()

    base_bundle = Path(args.base_bundle)
    output_path = Path(args.output)
    embedding_npz = Path(args.embedding_npz)

    with zipfile.ZipFile(base_bundle) as archive:
        index = json.loads(archive.read("index.json"))
        descriptors_bytes = archive.read("descriptors.bin")
        coarse_bytes = archive.read("coarse_index.bin")

    npz = np.load(embedding_npz)
    card_ids = npz["card_ids"]
    embeddings = npz["embeddings"].astype(np.float32)
    embedding_by_id = {str(card_id): embeddings[idx] for idx, card_id in enumerate(card_ids)}

    ordered = []
    for card in index["cards"]:
        card_id = card["id"]
        if card_id not in embedding_by_id:
            raise SystemExit(f"Missing embedding for card {card_id}")
        ordered.append(embedding_by_id[card_id])

    ordered_matrix = np.stack(ordered, axis=0)
    quantized = quantize_embeddings(ordered_matrix)
    embedding_bytes = quantized.tobytes()

    index["version"] = args.version
    index["embeddingIndex"] = {
        "version": 1,
        "dimension": int(ordered_matrix.shape[1]),
        "fileName": "embeddings.bin",
        "quantized": True,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("index.json", json.dumps(index, ensure_ascii=True))
        archive.writestr("descriptors.bin", descriptors_bytes)
        archive.writestr("coarse_index.bin", coarse_bytes)
        archive.writestr("embeddings.bin", embedding_bytes)

    metadata_path = output_path.with_suffix(".embedding-index.json")
    metadata_path.write_text(
        json.dumps(
            {
                "version": args.version,
                "card_count": int(ordered_matrix.shape[0]),
                "dimension": int(ordered_matrix.shape[1]),
                "quantized": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {output_path}")
    print(f"wrote {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
