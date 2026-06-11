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


class TcgDexCacheTests(unittest.TestCase):
    def test_card_listing_uses_live_api_even_when_detail_cache_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "detail-cache.jsonl"
            cache_path.write_text(
                '{"locale":"en","upstream_id":"cached-old","payload":{"id":"cached-old"}}\n',
                encoding="utf-8",
            )
            tcgdex_api.set_detail_cache_path(cache_path)

            with (
                mock.patch.dict("os.environ", {"TCGDEX_USE_DETAIL_CACHE_AS_LISTING": ""}),
                mock.patch.object(
                    tcgdex_api,
                    "api_get_json",
                    side_effect=[[{"id": "live-new"}], []],
                ) as api_get_json,
            ):
                briefs = tcgdex_api.fetch_card_briefs("en", items_per_page=100)

        self.assertEqual([{"id": "live-new"}], briefs)
        self.assertEqual(1, api_get_json.call_count)


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
                source_url = connection.execute(
                    "SELECT image_url FROM embedding_sources WHERE card_id = ? AND model_name = ?;",
                    (card["id"], build_embeddings_db.MODEL_NAME),
                ).fetchone()[0]
                self.assertEqual("https://assets.tcgdex.net/en/base/base1/1/high.webp", source_url)

                fts_row = connection.execute(
                    """
                    SELECT id, locale, name, set_name, set_id, card_number, rarity
                    FROM cards_fts
                    WHERE cards_fts MATCH ?;
                    """,
                    ("alak*",),
                ).fetchone()
                self.assertEqual(
                    (
                        "pokemon:en:base1:1",
                        "en",
                        "Alakazam",
                        "Base Set",
                        "base1",
                        "1",
                        "Rare Holo",
                    ),
                    fts_row,
                )

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

    def test_insert_int8_embeddings_writes_all_variants(self) -> None:
        card = self._make_card("pokemon:en:base1:3", name="Blastoise")

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
                row_count, total_bytes = build_embeddings_db.insert_int8_embeddings(connection)
                self.assertEqual(build_embeddings_db.VARIANT_K, row_count)
                self.assertEqual(build_embeddings_db.VARIANT_K * build_embeddings_db.EXPECTED_DIM, total_bytes)
                rows = connection.execute(
                    """
                    SELECT variant_idx, variant_tag, length(vector_int8)
                    FROM embeddings_int8
                    WHERE card_id = ?
                    ORDER BY variant_idx;
                    """,
                    (card["id"],),
                ).fetchall()

            self.assertEqual(
                [(idx, tag, build_embeddings_db.EXPECTED_DIM) for idx, tag in enumerate(build_embeddings_db.VARIANT_TAGS)],
                rows,
            )

    def test_validate_embeddings_db_requires_cards_fts(self) -> None:
        card = self._make_card("pokemon:en:base1:4", name="Pikachu")

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
                connection.execute("DROP TABLE cards_fts;")

            with self.assertRaisesRegex(RuntimeError, "cards_fts table is missing"):
                build_embeddings_db.validate_embeddings_db(output_db, min_row_count=1)

    def test_rebuild_int8_embeddings_recreates_table_without_inference(self) -> None:
        card = self._make_card("pokemon:en:base1:4", name="Venusaur")

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_db = Path(tmp_dir) / "embeddings.db"
            summary_json = Path(tmp_dir) / "summary.json"
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
                connection.execute("DROP TABLE embeddings_int8;")
                connection.execute(
                    """
                    CREATE TABLE embeddings_int8 (
                      card_id TEXT PRIMARY KEY,
                      dim INTEGER NOT NULL,
                      vector_int8 BLOB NOT NULL
                    );
                    """
                )
                connection.execute("PRAGMA user_version=6;")

            summary = build_embeddings_db.rebuild_int8_embeddings(output_db, summary_json=summary_json)

            self.assertEqual("int8_rebuilt", summary["status"])
            self.assertEqual(build_embeddings_db.VARIANT_K, summary["int8_row_count"])
            with sqlite3.connect(output_db) as connection:
                user_version = connection.execute("PRAGMA user_version;").fetchone()[0]
                self.assertEqual(build_embeddings_db.DB_USER_VERSION, user_version)
                columns = [row[1] for row in connection.execute("PRAGMA table_info(embeddings_int8);").fetchall()]
                self.assertIn("variant_idx", columns)
                self.assertIn("variant_tag", columns)
                row_count = connection.execute("SELECT COUNT(*) FROM embeddings_int8;").fetchone()[0]
                self.assertEqual(build_embeddings_db.VARIANT_K, row_count)
            self.assertTrue(summary_json.exists())

    def test_copy_seed_embeddings_reuses_complete_variant_rows(self) -> None:
        card = self._make_card("pokemon:en:base1:5", name="Mewtwo")

        with tempfile.TemporaryDirectory() as tmp_dir:
            seed_db = Path(tmp_dir) / "seed.db"
            output_db = Path(tmp_dir) / "output.db"
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
                    seed_db,
                    [record],
                    model_path=Path("unused.onnx"),
                )

            updated_card = dict(card)
            updated_card["name"] = "Mewtwo Updated"
            with sqlite3.connect(output_db) as connection:
                build_embeddings_db.init_db(connection)

            reused_ids = build_embeddings_db.copy_seed_embeddings(output_db, seed_db, [updated_card])

            self.assertEqual({card["id"]}, reused_ids)
            with sqlite3.connect(output_db) as connection:
                name = connection.execute("SELECT name FROM cards WHERE id = ?;", (card["id"],)).fetchone()[0]
                embedding_count = connection.execute("SELECT COUNT(*) FROM embeddings WHERE card_id = ?;", (card["id"],)).fetchone()[0]
                source_url = connection.execute(
                    "SELECT image_url FROM embedding_sources WHERE card_id = ? AND model_name = ?;",
                    (card["id"], build_embeddings_db.MODEL_NAME),
                ).fetchone()[0]

            self.assertEqual("Mewtwo Updated", name)
            self.assertEqual(build_embeddings_db.VARIANT_K, embedding_count)
            self.assertEqual(card["image_url"], source_url)

    def test_copy_seed_embeddings_skips_seed_without_source_metadata(self) -> None:
        card = self._make_card("pokemon:en:base1:6", name="Mew")

        with tempfile.TemporaryDirectory() as tmp_dir:
            seed_db = Path(tmp_dir) / "seed.db"
            output_db = Path(tmp_dir) / "output.db"
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
                    seed_db,
                    [record],
                    model_path=Path("unused.onnx"),
                )

            with sqlite3.connect(seed_db) as connection:
                connection.execute("DROP TABLE embedding_sources;")
            with sqlite3.connect(output_db) as connection:
                build_embeddings_db.init_db(connection)

            reused_ids = build_embeddings_db.copy_seed_embeddings(output_db, seed_db, [card])

            self.assertEqual(set(), reused_ids)
            with sqlite3.connect(output_db) as connection:
                card_count = connection.execute("SELECT COUNT(*) FROM cards;").fetchone()[0]
                embedding_count = connection.execute("SELECT COUNT(*) FROM embeddings;").fetchone()[0]

            self.assertEqual(0, card_count)
            self.assertEqual(0, embedding_count)

    def test_copy_seed_embeddings_skips_when_image_url_changed(self) -> None:
        card = self._make_card("pokemon:en:base1:7", name="Mewtwo ex")

        with tempfile.TemporaryDirectory() as tmp_dir:
            seed_db = Path(tmp_dir) / "seed.db"
            output_db = Path(tmp_dir) / "output.db"
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
                    seed_db,
                    [record],
                    model_path=Path("unused.onnx"),
                )

            updated_card = dict(card)
            updated_card["image_url"] = "https://assets.tcgdex.net/en/base/base1/6/high.webp"
            with sqlite3.connect(output_db) as connection:
                build_embeddings_db.init_db(connection)

            reused_ids = build_embeddings_db.copy_seed_embeddings(output_db, seed_db, [updated_card])

            self.assertEqual(set(), reused_ids)
            with sqlite3.connect(output_db) as connection:
                card_count = connection.execute("SELECT COUNT(*) FROM cards;").fetchone()[0]
                embedding_count = connection.execute("SELECT COUNT(*) FROM embeddings;").fetchone()[0]

            self.assertEqual(0, card_count)
            self.assertEqual(0, embedding_count)

    def test_fetch_supplementary_skips_without_api_key(self) -> None:
        result = build_embeddings_db.fetch_supplementary_pokemontcgio_cards(
            [], api_key="",
        )
        self.assertEqual([], result)

    def test_fetch_supplementary_deduplicates_by_set_and_number(self) -> None:
        existing = [
            {
                "id": "pokemon:en:sm2:60",
                "set_id": "sm2",
                "card_number": "60",
                "locale": "en",
                "name": "Tapu Lele-GX",
            },
        ]
        sets_payload = {
            "data": [
                {"id": "sm2", "name": "Guardians Rising", "total": 2, "printedTotal": 2},
            ],
        }
        cards_payload = {
            "totalCount": 2,
            "data": [
                {
                    "id": "sm2-60",
                    "number": "60",
                    "name": "Tapu Lele-GX",
                    "set": {"id": "sm2", "name": "Guardians Rising"},
                    "rarity": "Rare Holo GX",
                    "images": {"large": "https://example.com/sm2-60.png", "small": "https://example.com/sm2-60-sm.png"},
                    "artist": "5ban Graphics",
                    "hp": "170",
                    "supertype": "Pokémon",
                    "types": ["Psychic"],
                },
                {
                    "id": "sm2-60a",
                    "number": "60a",
                    "name": "Tapu Lele-GX",
                    "set": {"id": "sm2", "name": "Guardians Rising"},
                    "rarity": "Rare Ultra",
                    "images": {"large": "https://example.com/sm2-60a.png"},
                    "artist": "5ban Graphics",
                    "hp": "170",
                    "supertype": "Pokémon",
                    "types": ["Psychic"],
                },
            ],
        }

        call_count = [0]
        def fake_api_get_json(path, *, params=None, **kwargs):
            call_count[0] += 1
            if "/sets" in path:
                return sets_payload
            return cards_payload

        with mock.patch.object(
            build_embeddings_db,
            "pio_api_get_json",
            side_effect=fake_api_get_json,
        ), mock.patch.object(
            build_embeddings_db,
            "pio_resolve_api_key",
            return_value="fake-key",
        ):
            result = build_embeddings_db.fetch_supplementary_pokemontcgio_cards(
                existing, api_key="fake-key",
            )

        self.assertEqual(1, len(result))
        self.assertEqual("pokemon:en:sm2:60a", result[0]["id"])
        self.assertEqual("sm2", result[0]["set_id"])
        self.assertEqual("60a", result[0]["card_number"])
        self.assertEqual("pokemontcgio", result[0]["upstream_source"])
        self.assertEqual("https://example.com/sm2-60a.png", result[0]["image_url"])


class ImageFallbackTests(unittest.TestCase):
    @staticmethod
    def _missing_image_card() -> dict[str, object]:
        return {
            "id": "pokemon:en:mep:023",
            "locale": "en",
            "upstream_id": "mep-023",
            "set_id": "mep",
            "set_name": "MEP Black Star Promos",
            "card_number": "023",
            "name": "Mega Charizard X ex",
            "rarity": "None",
            "image_url": "",
            "image_url_low": None,
            "equivalence_key": "pokemon:xlocale:mep-023",
            "upstream_source": "tcgdex",
            "hp": "360",
            "types": ["Fire"],
        }

    @staticmethod
    def _write_probe_image(_url: str, destination: Path) -> None:
        Image.new("RGB", (480, 672), color=(180, 40, 40)).save(destination, format="PNG")

    def test_web_fallback_resolution_is_cached_with_source_url(self) -> None:
        card = self._missing_image_card()
        fallback = build_embeddings_db.ImageResolution(
            url="https://example.com/mep-023.png",
            source="web_search_unverified",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            with (
                mock.patch.object(build_embeddings_db, "resolve_fallback_image", return_value=fallback),
                mock.patch.object(build_embeddings_db, "download_binary", side_effect=self._write_probe_image),
            ):
                ready, skipped, _seconds, image_sources = build_embeddings_db.ensure_images(
                    [card],
                    cache_dir,
                    download_workers=1,
                    allow_web_image_fallback=True,
                )

            self.assertEqual([], skipped)
            self.assertEqual(1, len(ready))
            self.assertEqual("https://example.com/mep-023.png", card["image_url"])
            self.assertEqual({"pokemon:en:mep:023": "web_search_unverified"}, image_sources)
            self.assertEqual(
                {
                    "pokemon:en:mep:023": {
                        "url": "https://example.com/mep-023.png",
                        "source": "web_search_unverified",
                    }
                },
                build_embeddings_db.load_fallback_manifest(cache_dir),
            )

    def test_cached_fallback_reuses_manifest_url_not_file_url(self) -> None:
        card = self._missing_image_card()

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            image_path = cache_dir / f"{tcgdex_api.sanitize_card_id(str(card['id']))}.img"
            self._write_probe_image("unused", image_path)
            build_embeddings_db.write_fallback_manifest(
                cache_dir,
                {
                    "pokemon:en:mep:023": {
                        "url": "https://example.com/mep-023.png",
                        "source": "web_search_unverified",
                    }
                },
            )

            with mock.patch.object(build_embeddings_db, "resolve_fallback_image") as resolve_mock:
                ready, skipped, _seconds, image_sources = build_embeddings_db.ensure_images(
                    [card],
                    cache_dir,
                    download_workers=1,
                    allow_web_image_fallback=True,
                )

            resolve_mock.assert_not_called()
            self.assertEqual([], skipped)
            self.assertEqual(1, len(ready))
            self.assertEqual("https://example.com/mep-023.png", card["image_url"])
            self.assertEqual({"pokemon:en:mep:023": "web_search_unverified:cached"}, image_sources)

    def test_cached_image_without_public_url_is_rejected_for_release_metadata(self) -> None:
        card = self._missing_image_card()

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            image_path = cache_dir / f"{tcgdex_api.sanitize_card_id(str(card['id']))}.img"
            self._write_probe_image("unused", image_path)

            with mock.patch.object(build_embeddings_db, "resolve_fallback_image", return_value=None) as resolve_mock:
                ready, skipped, _seconds, image_sources = build_embeddings_db.ensure_images(
                    [card],
                    cache_dir,
                    download_workers=1,
                    allow_web_image_fallback=False,
                )

            resolve_mock.assert_called_once()
            self.assertEqual([], ready)
            self.assertEqual(1, len(skipped))
            self.assertEqual("missing_image_url", skipped[0].reason)
            self.assertEqual({}, image_sources)

    def test_file_url_fallback_manifest_is_ignored(self) -> None:
        card = self._missing_image_card()

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            build_embeddings_db.write_fallback_manifest(
                cache_dir,
                {
                    "pokemon:en:mep:023": {
                        "url": "file:///home/runner/work/card.png",
                        "source": "bad_local_path",
                    }
                },
            )

            with mock.patch.object(build_embeddings_db, "resolve_fallback_image", return_value=None):
                ready, skipped, _seconds, image_sources = build_embeddings_db.ensure_images(
                    [card],
                    cache_dir,
                    download_workers=1,
                    allow_web_image_fallback=False,
                )

            self.assertEqual([], ready)
            self.assertEqual(1, len(skipped))
            self.assertEqual("missing_image_url", skipped[0].reason)
            self.assertEqual({}, image_sources)

    def test_pokemontcgio_fallback_rejects_matching_art_with_wrong_number(self) -> None:
        card = self._missing_image_card()
        card["illustrator"] = "Saboteri"
        candidate = {
            "id": "me2-13",
            "name": "Mega Charizard X ex",
            "number": "13",
            "hp": "360",
            "artist": "Saboteri",
            "images": {"large": "https://images.pokemontcg.io/me2/13_hires.png"},
        }

        mock_api = mock.Mock(
            search_card_by_set_and_number=mock.Mock(return_value=None),
            search_cards_by_name=mock.Mock(return_value=[candidate]),
        )
        with mock.patch.dict(
            sys.modules,
            {
                "pokemontcgio_api": mock_api,
                "scripts.pokemontcgio_api": mock_api,
            },
        ):
            fallback = build_embeddings_db.resolve_fallback_image(
                card,
                allow_web_image_fallback=False,
            )

        self.assertIsNone(fallback)

    def test_pokemontcgio_fallback_allows_matching_art_with_missing_illustrator_or_hp(self) -> None:
        card = self._missing_image_card()
        # Card from upstream is missing illustrator and HP (common for Trainer cards or promos)
        card["illustrator"] = None
        card["hp"] = None
        candidate = {
            "id": "mep-023",
            "name": "Mega Charizard X ex",
            "number": "023",
            "hp": "360",
            "artist": "Saboteri",
            "images": {"large": "https://images.pokemontcg.io/mep/023_hires.png"},
        }

        mock_api = mock.Mock(
            search_card_by_set_and_number=mock.Mock(return_value=None),
            search_cards_by_name=mock.Mock(return_value=[candidate]),
        )
        with mock.patch.dict(
            sys.modules,
            {
                "pokemontcgio_api": mock_api,
                "scripts.pokemontcgio_api": mock_api,
            },
        ):
            fallback = build_embeddings_db.resolve_fallback_image(
                card,
                allow_web_image_fallback=False,
            )

        self.assertIsNotNone(fallback)
        self.assertEqual("https://images.pokemontcg.io/mep/023_hires.png", fallback.url)
        self.assertEqual("pokemontcgio_name_artist_hp", fallback.source)

    def test_pokemontcgio_fallback_passes_card_number_to_api(self) -> None:
        card = self._missing_image_card()
        card["card_number"] = "023"

        mock_search = mock.Mock(return_value=[])
        mock_api = mock.Mock(
            search_card_by_set_and_number=mock.Mock(return_value=None),
            search_cards_by_name=mock_search,
        )
        with mock.patch.dict(
            sys.modules,
            {
                "pokemontcgio_api": mock_api,
                "scripts.pokemontcgio_api": mock_api,
            },
        ):
            build_embeddings_db.resolve_fallback_image(
                card,
                allow_web_image_fallback=False,
            )

        mock_search.assert_called_once_with("Mega Charizard X ex", number="23")

    def test_pokemontcgio_identity_fallback_handles_provider_alias_and_name_punctuation(self) -> None:
        card = self._missing_image_card()
        card.update(
            {
                "id": "pokemon:en:sm3.5:10",
                "set_id": "sm3.5",
                "card_number": "10",
                "name": "Entei GX",
                "hp": "180",
                "illustrator": "5ban Graphics",
            }
        )
        candidate = {
            "id": "sm35-10",
            "name": "Entei-GX",
            "number": "10",
            "hp": "180",
            "artist": "5ban Graphics",
            "images": {"large": "https://images.pokemontcg.io/sm35/10_hires.png"},
        }

        mock_api = mock.Mock(
            search_card_by_set_and_number=mock.Mock(side_effect=lambda set_id, number: candidate if (set_id, number) == ("sm35", "10") else None),
            search_cards_by_name=mock.Mock(return_value=[]),
        )
        with mock.patch.dict(
            sys.modules,
            {
                "pokemontcgio_api": mock_api,
                "scripts.pokemontcgio_api": mock_api,
            },
        ):
            fallback = build_embeddings_db.resolve_fallback_image(
                card,
                allow_web_image_fallback=False,
            )

        self.assertIsNotNone(fallback)
        self.assertEqual("https://images.pokemontcg.io/sm35/10_hires.png", fallback.url)
        self.assertEqual("pokemontcgio_set_number_name", fallback.source)



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
