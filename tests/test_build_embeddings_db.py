from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_embeddings_db  # noqa: E402
import tcgdex_api  # noqa: E402


class NormalizeCardImageUrlsTests(unittest.TestCase):
    def test_prefers_tcgdex_asset_root_and_derives_low_variant(self) -> None:
        image_url, image_url_low = tcgdex_api.normalize_card_image_urls(
            {
                "image": "https://assets.tcgdex.net/ja/SV/SV6/001",
                "images": {
                    "small": "https://images.pokemontcg.io/base1/1.png",
                    "large": "https://images.pokemontcg.io/base1/1_hires.png",
                },
            }
        )

        self.assertEqual("https://assets.tcgdex.net/ja/SV/SV6/001/high.webp", image_url)
        self.assertEqual("https://assets.tcgdex.net/ja/SV/SV6/001/low.webp", image_url_low)

    def test_falls_back_to_pokemontcgio_images_when_tcgdex_image_missing(self) -> None:
        image_url, image_url_low = tcgdex_api.normalize_card_image_urls(
            {
                "images": {
                    "small": "https://images.pokemontcg.io/base1/1.png",
                    "large": "https://images.pokemontcg.io/base1/1_hires.png",
                }
            }
        )

        self.assertEqual("https://images.pokemontcg.io/base1/1_hires.png", image_url)
        self.assertEqual("https://images.pokemontcg.io/base1/1.png", image_url_low)

    def test_single_size_url_keeps_low_null(self) -> None:
        image_url, image_url_low = tcgdex_api.normalize_image_urls("https://example.com/card.png")

        self.assertEqual("https://example.com/card.png", image_url)
        self.assertIsNone(image_url_low)


class InsertEmbeddingsTests(unittest.TestCase):
    def test_insert_new_embeddings_writes_image_url_low(self) -> None:
        class FakeSession:
            def run(self, _output_names: object, _inputs: object) -> list[np.ndarray]:
                return [np.asarray([np.ones(build_embeddings_db.EXPECTED_DIM, dtype=np.float32)])]

        card = {
            "id": "pokemon:en:base1:1",
            "locale": "en",
            "upstream_id": "base1-1",
            "set_id": "base1",
            "set_name": "Base Set",
            "card_number": "1",
            "name": "Alakazam",
            "rarity": "Rare Holo",
            "image_url": "https://assets.tcgdex.net/en/base/base1/1/high.webp",
            "image_url_low": "https://assets.tcgdex.net/en/base/base1/1/low.webp",
            "equivalence_key": "pokemon:xlocale:test",
            "upstream_source": "tcgdex",
            "hp": "80",
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_db = Path(tmp_dir) / "embeddings.db"
            record = build_embeddings_db.DownloadedCard(card=card, image_path=Path(tmp_dir) / "unused.img")

            with (
                mock.patch.object(
                    build_embeddings_db,
                    "load_onnx_session",
                    return_value=(FakeSession(), "input", build_embeddings_db.EXPECTED_DIM, 0.0),
                ),
                mock.patch.object(
                    build_embeddings_db,
                    "preprocess_for_embedder",
                    return_value=np.zeros((1, 3, 224, 224), dtype=np.float32),
                ),
            ):
                inserted, _model_load_seconds, _elapsed = build_embeddings_db.insert_new_embeddings(
                    output_db,
                    [record],
                    model_path=Path("unused.onnx"),
                )

            self.assertEqual(1, inserted)
            with sqlite3.connect(output_db) as connection:
                row = connection.execute(
                    "SELECT image_url, image_url_low FROM cards WHERE id = ?;",
                    (card["id"],),
                ).fetchone()
                self.assertEqual(
                    (
                        "https://assets.tcgdex.net/en/base/base1/1/high.webp",
                        "https://assets.tcgdex.net/en/base/base1/1/low.webp",
                    ),
                    row,
                )
