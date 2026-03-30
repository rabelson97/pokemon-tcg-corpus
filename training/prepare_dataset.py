#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path


def download_image(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "PokemonTCGTrainingPrep/1.0"})
    with urllib.request.urlopen(request) as response:
        destination.write_bytes(response.read())


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a training dataset from a corpus release zip.")
    parser.add_argument("--bundle", required=True, help="Path to pokemon_tcg_corpus_v*.zip")
    parser.add_argument("--output-dir", required=True, help="Directory to write images + manifest")
    parser.add_argument("--limit", type=int, help="Optional max number of cards to download")
    parser.add_argument("--image-cache-dir", help="Optional existing image cache dir keyed by '<card_id>.img'")
    args = parser.parse_args()

    bundle_path = Path(args.bundle)
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    manifest_path = output_dir / "manifest.jsonl"
    images_dir.mkdir(parents=True, exist_ok=True)
    image_cache_dir = Path(args.image_cache_dir).resolve() if args.image_cache_dir else None

    with zipfile.ZipFile(bundle_path) as archive:
        index = json.loads(archive.read("index.json"))

    cards = index["cards"]
    if args.limit:
        cards = cards[: args.limit]

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for idx, card in enumerate(cards, start=1):
            if image_cache_dir is not None:
                cached_path = image_cache_dir / f"{card['id'].replace('/', '_')}.img"
                if not cached_path.exists():
                    raise SystemExit(f"Missing cached image for {card['id']} at {cached_path}")
                image_path = cached_path
            else:
                image_path = images_dir / f"{card['id'].replace('/', '_')}.png"
                if not image_path.exists():
                    download_image(card["imageUrl"], image_path)

            row = {
                "card_id": card["id"],
                "name": card["name"],
                "set_name": card.get("setName"),
                "number": card.get("number"),
                "image_url": card["imageUrl"],
                "image_path": str(image_path.resolve()),
            }
            manifest.write(json.dumps(row, ensure_ascii=True) + "\n")

            if idx % 500 == 0:
                print(f"prepared {idx} cards")

    print(f"Wrote manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
