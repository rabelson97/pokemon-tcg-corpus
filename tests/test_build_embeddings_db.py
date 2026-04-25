from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image


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
    @staticmethod
    def _make_card(card_id: str, name: str = "Alakazam") -> dict[str, object]:
        return {
            "id": card_id,
            "locale": "en",
            "upstream_id": f"{card_id}-up",
            "set_id": "base1",
            "set_name": "Base Set",
            "card_number": "1",
            "name": name,
            "rarity": "Rare Holo",
            "image_url": "https://assets.tcgdex.net/en/base/base1/1/high.webp",
            "image_url_low": "https://assets.tcgdex.net/en/base/base1/1/low.webp",
            "equivalence_key": "pokemon:xlocale:test",
            "upstream_source": "tcgdex",
            "hp": "80",
        }

    @staticmethod
    def _patched_session(call_log: list[int] | None = None):
        """ONNX session that returns a deterministic vector seeded by the variant
        index so we can assert variants produce different embeddings."""

        class FakeSession:
            def __init__(self) -> None:
                self.calls = 0

            def run(self, _output_names: object, _inputs: object) -> list[np.ndarray]:
                index = self.calls
                self.calls += 1
                if call_log is not None:
                    call_log.append(index)
                vector = np.zeros(build_embeddings_db.EXPECTED_DIM, dtype=np.float32)
                vector[index % build_embeddings_db.EXPECTED_DIM] = 1.0
                return [np.asarray([vector])]

        return FakeSession()

    def test_insert_new_embeddings_writes_image_url_low_and_k_variants(self) -> None:
        card = self._make_card("pokemon:en:base1:1")

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_db = Path(tmp_dir) / "embeddings.db"
            record = build_embeddings_db.DownloadedCard(card=card, image_path=Path(tmp_dir) / "unused.img")

            with (
                mock.patch.object(
                    build_embeddings_db,
                    "load_onnx_session",
                    return_value=(self._patched_session(), "input", build_embeddings_db.EXPECTED_DIM, 0.0),
                ),
                mock.patch.object(
                    build_embeddings_db,
                    "base_pil_for_card",
                    return_value=Image.new("RGB", (224, 224), color=(127, 127, 127)),
                ),
            ):
                inserted, _model_load_seconds, _elapsed = build_embeddings_db.insert_new_embeddings(
                    output_db,
                    [record],
                    model_path=Path("unused.onnx"),
                )

            # Returned count is cards inserted, not variant rows.
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

                variant_count = connection.execute(
                    "SELECT COUNT(*) FROM embeddings WHERE card_id = ?;",
                    (card["id"],),
                ).fetchone()[0]
                self.assertEqual(build_embeddings_db.VARIANT_K, variant_count)

                tags = sorted(
                    str(value)
                    for (value,) in connection.execute(
                        "SELECT variant_tag FROM embeddings WHERE card_id = ? ORDER BY variant_idx;",
                        (card["id"],),
                    ).fetchall()
                )
                self.assertEqual(sorted(build_embeddings_db.VARIANT_TAGS), tags)

    def test_variants_produce_distinct_blobs(self) -> None:
        card = self._make_card("pokemon:en:base1:2", name="Charmeleon")

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_db = Path(tmp_dir) / "embeddings.db"
            record = build_embeddings_db.DownloadedCard(card=card, image_path=Path(tmp_dir) / "unused.img")

            with (
                mock.patch.object(
                    build_embeddings_db,
                    "load_onnx_session",
                    return_value=(self._patched_session(), "input", build_embeddings_db.EXPECTED_DIM, 0.0),
                ),
                mock.patch.object(
                    build_embeddings_db,
                    "base_pil_for_card",
                    return_value=Image.new("RGB", (224, 224), color=(127, 127, 127)),
                ),
            ):
                build_embeddings_db.insert_new_embeddings(
                    output_db,
                    [record],
                    model_path=Path("unused.onnx"),
                )

            with sqlite3.connect(output_db) as connection:
                blobs = [
                    row[0]
                    for row in connection.execute(
                        "SELECT vector_blob FROM embeddings WHERE card_id = ? ORDER BY variant_idx;",
                        (card["id"],),
                    ).fetchall()
                ]
            # Each variant feeds a distinct fake-session call → distinct blob.
            self.assertEqual(build_embeddings_db.VARIANT_K, len(blobs))
            self.assertEqual(len({bytes(blob) for blob in blobs}), build_embeddings_db.VARIANT_K)


class RenderVariantTests(unittest.TestCase):
    def test_clean_variant_returns_input_unchanged(self) -> None:
        base = Image.new("RGB", (224, 224), color=(50, 100, 150))
        out = build_embeddings_db.render_variant(base, 0, __import__("random").Random(0))
        self.assertEqual(base.tobytes(), out.tobytes())

    def test_known_variants_change_pixels(self) -> None:
        rng = __import__("random").Random(0)
        # Use a non-uniform image so blur / glare actually change pixel values.
        array = (np.linspace(0, 255, 224 * 224 * 3, dtype=np.float32).reshape(224, 224, 3)).astype(np.uint8)
        base = Image.fromarray(array)
        for variant_idx in (1, 2, 3):
            out = build_embeddings_db.render_variant(base, variant_idx, rng)
            self.assertNotEqual(base.tobytes(), out.tobytes(), f"variant_idx={variant_idx} did not change image")
            self.assertEqual(base.size, out.size)

    def test_seeds_are_deterministic_per_card_and_variant(self) -> None:
        seed_a = build_embeddings_db.card_variant_seed("pokemon:en:base1:1", 1)
        seed_b = build_embeddings_db.card_variant_seed("pokemon:en:base1:1", 1)
        seed_c = build_embeddings_db.card_variant_seed("pokemon:en:base1:1", 2)
        seed_d = build_embeddings_db.card_variant_seed("pokemon:en:base1:2", 1)
        self.assertEqual(seed_a, seed_b)
        self.assertNotEqual(seed_a, seed_c)
        self.assertNotEqual(seed_a, seed_d)
