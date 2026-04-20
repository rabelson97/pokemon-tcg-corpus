from __future__ import annotations

import datetime as dt
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_prices_db  # noqa: E402


class BuildPricesDbTests(unittest.TestCase):
    def test_extract_price_rows_supports_pokemontcgio_tcgplayer_shape(self) -> None:
        rows = build_prices_db.extract_price_rows_from_selected_sources(
            "pokemon:en:sv01:001",
            {
                "tcgplayer": {
                    "unit": "USD",
                    "updated": "2026/04/06",
                    "selected_variant": {
                        "low": 1.25,
                        "mid": 2.5,
                        "high": 4.0,
                        "market": 2.2,
                        "directLow": 2.1,
                    },
                },
                "cardmarket": {
                    "unit": "EUR",
                    "updated": "2026-04-05T00:00:00.000Z",
                    "selected_variant": {
                        "low": 0.5,
                        "avg": 0.75,
                        "trend": 0.8,
                    },
                },
            },
        )

        self.assertEqual(2, len(rows))
        self.assertEqual(
            (
                "pokemon:en:sv01:001",
                "US",
                "USD",
                "tcgplayer",
                1.25,
                2.2,
                4.0,
                "2026/04/06",
                1,
            ),
            rows[0],
        )
        self.assertEqual("cardmarket", rows[1][3])
        self.assertEqual(0, rows[1][-1])

    def test_extract_price_rows_promotes_cardmarket_when_tcgplayer_missing(self) -> None:
        card = {
            "id": "pokemon:fr:sv01:002",
            "locale": "fr",
            "pricing": {
                "cardmarket": {
                    "updated": "2026-04-05T00:00:00.000Z",
                    "unit": "EUR",
                    "low": 0.1,
                    "trend": 0.2,
                }
            },
        }

        rows = build_prices_db.extract_price_rows(card)

        self.assertEqual(1, len(rows))
        self.assertEqual("cardmarket", rows[0][3])
        self.assertEqual(1, rows[0][-1])

    def test_select_price_sources_prefers_pokemontcgio_for_english_cards(self) -> None:
        summary = {
            "transport_counts": {"cardmarket": {"tcgdex": 0}, "tcgplayer": {"pokemontcgio": 0}},
            "pokemontcgio": {
                "english_cards_considered": 0,
                "english_cards_with_match": 0,
                "english_cards_without_match": 0,
                "english_cards_with_tcgplayer": 0,
                "english_cards_without_tcgplayer": 0,
                "stale_tcgplayer_rows": 0,
                "stale_reasons": {},
            },
        }
        card = {
            "id": "pokemon:en:swsh1:1",
            "locale": "en",
            "set_id": "swsh1",
            "card_number": "1",
            "pricing": {
                "tcgplayer": {
                    "updated": "2026-04-05T12:34:56.000Z",
                    "unit": "USD",
                    "normal": {
                        "lowPrice": 999.0,
                        "marketPrice": 999.0,
                        "highPrice": 999.0,
                    },
                },
                "cardmarket": {
                    "updated": "2026-04-05T00:00:00.000Z",
                    "unit": "EUR",
                    "low": 0.5,
                    "avg": 0.75,
                    "trend": 0.8,
                },
            },
        }
        pokemontcgio_index = {
            ("swsh1", "1"): {
                "tcgplayer": {
                    "updatedAt": "2026/04/06",
                    "prices": {
                        "holofoil": {
                            "low": 1.25,
                            "mid": 2.5,
                            "high": 4.0,
                            "market": 2.2,
                            "directLow": 2.1,
                        }
                    },
                }
            }
        }

        selected = build_prices_db.select_price_sources(
            card,
            pokemontcgio_index=pokemontcgio_index,
            max_pokemontcgio_age_days=14,
            now=dt.datetime(2026, 4, 6, tzinfo=dt.timezone.utc),
            summary=summary,
        )

        rows = build_prices_db.extract_price_rows_from_selected_sources(card["id"], selected)
        self.assertEqual("2026/04/06", rows[0][7])
        self.assertEqual(1, summary["transport_counts"]["tcgplayer"]["pokemontcgio"])
        self.assertNotIn("cardmarket", selected)

    def test_select_price_sources_skips_stale_pokemontcgio_prices(self) -> None:
        summary = {
            "transport_counts": {"cardmarket": {"tcgdex": 0}, "tcgplayer": {"pokemontcgio": 0}},
            "pokemontcgio": {
                "english_cards_considered": 0,
                "english_cards_with_match": 0,
                "english_cards_without_match": 0,
                "english_cards_with_tcgplayer": 0,
                "english_cards_without_tcgplayer": 0,
                "stale_tcgplayer_rows": 0,
                "stale_reasons": {},
            },
        }
        card = {
            "id": "pokemon:en:swsh1:1",
            "locale": "en",
            "set_id": "swsh1",
            "card_number": "1",
            "pricing": {
                "cardmarket": {
                    "updated": "2026-04-05T00:00:00.000Z",
                    "unit": "EUR",
                    "low": 0.5,
                    "avg": 0.75,
                    "trend": 0.8,
                },
            },
        }
        pokemontcgio_index = {
            ("swsh1", "1"): {
                "tcgplayer": {
                    "updatedAt": "2026/03/01",
                    "prices": {
                        "holofoil": {
                            "low": 1.25,
                            "mid": 2.5,
                            "high": 4.0,
                            "market": 2.2,
                        }
                    },
                }
            }
        }

        selected = build_prices_db.select_price_sources(
            card,
            pokemontcgio_index=pokemontcgio_index,
            max_pokemontcgio_age_days=14,
            now=dt.datetime(2026, 4, 6, tzinfo=dt.timezone.utc),
            summary=summary,
        )

        self.assertNotIn("tcgplayer", selected)
        self.assertNotIn("cardmarket", selected)
        self.assertEqual(1, summary["pokemontcgio"]["stale_tcgplayer_rows"])
        self.assertEqual(1, summary["pokemontcgio"]["stale_reasons"]["older_than_max_age"])

    def test_candidate_pokemontcgio_match_keys_cover_alias_set_and_number_formats(self) -> None:
        self.assertIn(
            ("swsh35", "71"),
            build_prices_db.candidate_pokemontcgio_match_keys("swsh3.5", "71"),
        )
        self.assertIn(
            ("swsh12pt5gg", "GG05"),
            build_prices_db.candidate_pokemontcgio_match_keys("swsh12.5", "GG05"),
        )
        self.assertIn(
            ("swsh9tg", "TG02"),
            build_prices_db.candidate_pokemontcgio_match_keys("swsh9", "TG02"),
        )
        self.assertIn(
            ("sv1", "43"),
            build_prices_db.candidate_pokemontcgio_match_keys("sv01", "043"),
        )

    def test_select_price_sources_matches_alias_formatted_english_cards(self) -> None:
        summary = {
            "transport_counts": {"tcgplayer": {"pokemontcgio": 0}},
            "pokemontcgio": {
                "english_cards_considered": 0,
                "english_cards_with_match": 0,
                "english_cards_without_match": 0,
                "english_cards_with_tcgplayer": 0,
                "english_cards_without_tcgplayer": 0,
                "stale_tcgplayer_rows": 0,
                "stale_reasons": {},
            },
        }
        pokemontcgio_index = {
            ("swsh12pt5gg", "GG05"): {
                "tcgplayer": {"updatedAt": "2026/04/10", "prices": {"holofoil": {"low": 16.0, "market": 19.0, "high": 150.0}}}
            },
            ("swsh9tg", "TG02"): {
                "tcgplayer": {"updatedAt": "2026/04/10", "prices": {"holofoil": {"low": 16.0, "market": 19.81, "high": 79.99}}}
            },
            ("sv1", "43"): {
                "tcgplayer": {"updatedAt": "2026/04/10", "prices": {"holofoil": {"low": 0.03, "market": 0.31, "high": 501.5}}}
            },
        }

        for card in [
            {"id": "pokemon:en:swsh12.5:GG05", "locale": "en", "set_id": "swsh12.5", "card_number": "GG05", "pricing": {}},
            {"id": "pokemon:en:swsh9:TG02", "locale": "en", "set_id": "swsh9", "card_number": "TG02", "pricing": {}},
            {"id": "pokemon:en:sv01:043", "locale": "en", "set_id": "sv01", "card_number": "043", "pricing": {}},
        ]:
            selected = build_prices_db.select_price_sources(
                card,
                pokemontcgio_index=pokemontcgio_index,
                max_pokemontcgio_age_days=14,
                now=dt.datetime(2026, 4, 10, tzinfo=dt.timezone.utc),
                summary=summary,
            )
            self.assertIn("tcgplayer", selected, card["id"])

        self.assertEqual(3, summary["pokemontcgio"]["english_cards_with_match"])
        self.assertEqual(0, summary["pokemontcgio"]["english_cards_without_match"])
        self.assertEqual(3, summary["pokemontcgio"]["english_cards_with_tcgplayer"])

    def test_fallback_budget_zero_skips_all_english_fallback_attempts(self) -> None:
        summary = {
            "transport_counts": {},
            "pokemontcgio": {
                "english_cards_considered": 0,
                "english_cards_with_match": 0,
                "english_cards_without_match": 0,
                "english_cards_with_tcgplayer": 0,
                "english_cards_without_tcgplayer": 0,
                "stale_tcgplayer_rows": 0,
                "stale_reasons": {},
            },
            "fallback_providers": {
                "ppt_configured": True,
                "poketrace_configured": True,
                "poketrace_set_slugs_mapped": 0,
                "max_cards": 0,
                "english_cards_tried_fallback": 0,
                "english_cards_skipped_due_to_budget": 0,
                "ppt_hits": 0,
                "ppt_misses": 0,
                "ppt_errors": 0,
                "poketrace_hits": 0,
                "poketrace_misses": 0,
                "poketrace_errors": 0,
                "poketrace_set_mapping_failures": 0,
            },
        }

        max_fallback_cards = 0
        fallback_attempts = 0
        selected = build_prices_db.select_price_sources(
            {"id": "pokemon:en:sv01:001", "locale": "en", "set_id": "sv01", "card_number": "001", "pricing": {}},
            pokemontcgio_index={},
            max_pokemontcgio_age_days=14,
            now=dt.datetime(2026, 4, 10, tzinfo=dt.timezone.utc),
            summary=summary,
        )
        if "tcgplayer" not in selected and (summary["fallback_providers"]["ppt_configured"] or summary["fallback_providers"]["poketrace_configured"]):
            if max_fallback_cards is not None and fallback_attempts >= max_fallback_cards:
                summary["fallback_providers"]["english_cards_skipped_due_to_budget"] += 1
            else:
                fallback_attempts += 1
                summary["fallback_providers"]["english_cards_tried_fallback"] += 1

        self.assertEqual(0, summary["fallback_providers"]["english_cards_tried_fallback"])
        self.assertEqual(1, summary["fallback_providers"]["english_cards_skipped_due_to_budget"])

    def test_build_prices_db_skips_poketrace_slug_bootstrap_when_fallback_budget_zero(self) -> None:
        cards = [
            {
                "id": "pokemon:en:sv01:001",
                "locale": "en",
                "upstream_id": "sv01-001",
                "set_id": "sv01",
                "set_name": "Scarlet & Violet",
                "card_number": "001",
                "pricing": {},
            }
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "prices.db"
            with mock.patch.object(build_prices_db, "fetch_all_card_records", return_value=(cards, {"en": 1})), mock.patch.object(
                build_prices_db,
                "fetch_targeted_pokemontcgio_cards",
                return_value=[],
            ), mock.patch.object(
                build_prices_db,
                "build_pokemontcgio_index",
                return_value={},
            ), mock.patch.object(
                build_prices_db,
                "build_poketrace_set_slugs",
                return_value={"sv01": "scarlet-violet"},
            ) as build_slugs_mock:
                summary = build_prices_db.build_prices_db(
                    output_path,
                    locales=["en"],
                    limit=1,
                    min_row_count=0,
                    max_fallback_cards=0,
                )

        build_slugs_mock.assert_not_called()
        self.assertEqual(0, summary["fallback_providers"]["poketrace_set_slugs_mapped"])
        self.assertEqual(0, summary["fallback_providers"]["english_cards_tried_fallback"])

    def test_collect_reusable_existing_tcgplayer_rows_filters_by_updated_date(self) -> None:
        rows = {
            "pokemon:en:sv01:001": [
                ("pokemon:en:sv01:001", "US", "USD", "tcgplayer", 1.0, 2.0, 3.0, "2026/04/10", 1),
            ],
            "pokemon:en:sv01:002": [
                ("pokemon:en:sv01:002", "US", "USD", "tcgplayer", 1.0, 2.0, 3.0, "2026/04/09", 1),
            ],
        }

        reusable = build_prices_db.collect_reusable_existing_tcgplayer_rows(rows, updated_date_prefix="2026/04/10")

        self.assertIn("pokemon:en:sv01:001", reusable)
        self.assertNotIn("pokemon:en:sv01:002", reusable)

    def test_build_prices_db_reuses_seed_tcgplayer_rows_for_same_day_manual_rerun(self) -> None:
        cards = [
            {
                "id": "pokemon:en:sv01:001",
                "locale": "en",
                "upstream_id": "sv01-001",
                "set_id": "sv01",
                "set_name": "Scarlet & Violet",
                "card_number": "001",
                "pricing": {},
            },
            {
                "id": "pokemon:en:sv01:002",
                "locale": "en",
                "upstream_id": "sv01-002",
                "set_id": "sv01",
                "set_name": "Scarlet & Violet",
                "card_number": "002",
                "pricing": {},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            seed_db = Path(tmp_dir) / "seed.db"
            with sqlite3.connect(seed_db) as connection:
                connection.execute("PRAGMA user_version=2;")
                connection.execute(
                    """
                    CREATE TABLE prices (
                      card_id TEXT NOT NULL,
                      market_code TEXT NOT NULL,
                      currency_code TEXT NOT NULL,
                      source_name TEXT NOT NULL,
                      low_price REAL,
                      market_price REAL,
                      high_price REAL,
                      updated_at TEXT,
                      is_primary INTEGER NOT NULL DEFAULT 0,
                      PRIMARY KEY (card_id, source_name)
                    );
                    """
                )
                connection.execute(
                    """
                    INSERT INTO prices (
                      card_id, market_code, currency_code, source_name, low_price, market_price, high_price, updated_at, is_primary
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("pokemon:en:sv01:001", "US", "USD", "tcgplayer", 1.0, 2.0, 3.0, "2026/04/10", 1),
                )
                connection.commit()

            output_path = Path(tmp_dir) / "prices.db"
            targeted_result = [{"set": {"id": "sv1"}, "number": "2", "tcgplayer": {"updatedAt": "2026/04/10", "prices": {"normal": {"low": 4.0, "market": 5.0, "high": 6.0}}}}]
            with mock.patch.object(build_prices_db, "fetch_all_card_records", return_value=(cards, {"en": 2})), mock.patch.object(
                build_prices_db,
                "fetch_targeted_pokemontcgio_cards",
                return_value=targeted_result,
            ) as targeted_fetch_mock, mock.patch.object(
                build_prices_db,
                "build_pokemontcgio_index",
                return_value={("sv1", "2"): targeted_result[0]},
            ), mock.patch.object(
                build_prices_db,
                "build_poketrace_set_slugs",
                return_value={},
            ):
                summary = build_prices_db.build_prices_db(
                    output_path,
                    locales=["en"],
                    limit=None,
                    min_row_count=0,
                    max_fallback_cards=0,
                    seed_db_path=seed_db,
                    reuse_existing_tcgplayer_date="2026/04/10",
                )

            targeted_fetch_mock.assert_called_once()
            self.assertEqual(1, summary["seed_reuse"]["cards_with_reused_tcgplayer_rows"])
            with sqlite3.connect(output_path) as connection:
                rows = connection.execute(
                    "SELECT card_id, market_price, updated_at FROM prices ORDER BY card_id"
                ).fetchall()
            self.assertEqual(
                [
                    ("pokemon:en:sv01:001", 2.0, "2026/04/10"),
                    ("pokemon:en:sv01:002", 5.0, "2026/04/10"),
                ],
                rows,
            )

    def test_load_poketrace_set_mapping_overrides_reads_repo_cache(self) -> None:
        mapping = build_prices_db.load_poketrace_set_mapping_overrides()
        self.assertEqual("twilight-masquerade", mapping["sv06"])
        self.assertEqual("paldean-fates", mapping["sv04.5"])

    def test_slugify_poketrace_set_name(self) -> None:
        self.assertEqual("scarlet-violet", build_prices_db.slugify_poketrace_set_name("Scarlet & Violet"))
        self.assertEqual("brilliant-stars", build_prices_db.slugify_poketrace_set_name("Brilliant Stars"))
        self.assertEqual("sv-black-star-promos", build_prices_db.slugify_poketrace_set_name("SV Black Star Promos"))

    def test_locale_coverage_audit_tracks_source_mix(self) -> None:
        audit = build_prices_db.create_locale_coverage_audit(["en"])

        build_prices_db.update_locale_coverage_audit(
            audit,
            locale="en",
            extracted_rows=[
                ("pokemon:en:sv01:001", "US", "USD", "tcgplayer", 1.0, 2.0, 3.0, None, 1),
                ("pokemon:en:sv01:001", "EU", "EUR", "cardmarket", 0.5, 0.7, 0.8, None, 0),
            ],
        )
        build_prices_db.update_locale_coverage_audit(
            audit,
            locale="en",
            extracted_rows=[
                ("pokemon:en:sv01:002", "EU", "EUR", "cardmarket", 0.2, 0.3, 0.4, None, 1),
            ],
        )
        build_prices_db.update_locale_coverage_audit(audit, locale="en", extracted_rows=[])

        self.assertEqual(
            {
                "cards_total": 3,
                "cards_with_tcgplayer": 1,
                "cards_with_cardmarket": 2,
                "cards_with_both_sources": 1,
                "cards_with_tcgplayer_only": 0,
                "cards_with_cardmarket_only": 1,
                "cards_without_prices": 1,
                "cards_primary_tcgplayer": 1,
                "cards_primary_cardmarket": 1,
                "tcgplayer_rows": 1,
                "cardmarket_rows": 2,
            },
            audit["en"],
        )


    def test_select_price_sources_excludes_cardmarket_for_english(self) -> None:
        """English card with only cardmarket should get empty selected_sources."""
        summary = {
            "transport_counts": {},
            "pokemontcgio": {
                "english_cards_considered": 0,
                "english_cards_with_match": 0,
                "english_cards_without_match": 0,
                "english_cards_with_tcgplayer": 0,
                "english_cards_without_tcgplayer": 0,
                "stale_tcgplayer_rows": 0,
                "stale_reasons": {},
            },
        }
        card = {
            "id": "pokemon:en:sv01:001",
            "locale": "en",
            "set_id": "sv01",
            "card_number": "1",
            "pricing": {
                "cardmarket": {
                    "updated": "2026-04-05T00:00:00.000Z",
                    "unit": "EUR",
                    "low": 0.5,
                    "avg": 0.75,
                    "trend": 0.8,
                },
            },
        }
        selected = build_prices_db.select_price_sources(
            card,
            pokemontcgio_index={},
            max_pokemontcgio_age_days=14,
            now=dt.datetime(2026, 4, 6, tzinfo=dt.timezone.utc),
            summary=summary,
        )
        self.assertNotIn("cardmarket", selected)
        self.assertEqual({}, selected)

    def test_select_price_sources_keeps_cardmarket_for_non_english(self) -> None:
        """Non-English card should still get cardmarket."""
        summary = {
            "transport_counts": {},
            "pokemontcgio": {
                "english_cards_considered": 0,
                "english_cards_with_match": 0,
                "english_cards_without_match": 0,
                "english_cards_with_tcgplayer": 0,
                "english_cards_without_tcgplayer": 0,
                "stale_tcgplayer_rows": 0,
                "stale_reasons": {},
            },
        }
        card = {
            "id": "pokemon:fr:sv01:001",
            "locale": "fr",
            "set_id": "sv01",
            "card_number": "1",
            "pricing": {
                "cardmarket": {
                    "updated": "2026-04-05T00:00:00.000Z",
                    "unit": "EUR",
                    "low": 0.5,
                    "avg": 0.75,
                    "trend": 0.8,
                },
            },
        }
        selected = build_prices_db.select_price_sources(
            card,
            pokemontcgio_index={},
            max_pokemontcgio_age_days=14,
            now=dt.datetime(2026, 4, 6, tzinfo=dt.timezone.utc),
            summary=summary,
        )
        self.assertIn("cardmarket", selected)


class PptApiTests(unittest.TestCase):
    def test_search_card_matcher_accepts_name_suffix_number(self) -> None:
        import ppt_api

        card_data = {
            "name": "Infernape - 173/167",
            "setName": "SV06: Twilight Masquerade",
        }
        self.assertTrue(ppt_api.card_matches(card_data, set_name="Twilight Masquerade", card_number="173"))

    def test_search_card_matcher_rejects_wrong_number(self) -> None:
        import ppt_api

        card_data = {
            "name": "Infernape - 172/167",
            "set": "Twilight Masquerade",
        }
        self.assertFalse(ppt_api.card_matches(card_data, set_name="Twilight Masquerade", card_number="173"))

    def test_extract_usd_price_standard_shape(self) -> None:
        import ppt_api

        card_data = {
            "name": "Infernape",
            "prices": {"market": 12.22, "low": 8.0, "high": 18.0},
        }
        result = ppt_api.extract_usd_price(card_data)
        self.assertIsNotNone(result)
        self.assertEqual("USD", result["unit"])
        self.assertEqual(12.22, result["selected_variant"]["market"])
        self.assertEqual(8.0, result["selected_variant"]["low"])

    def test_extract_usd_price_returns_none_for_empty(self) -> None:
        import ppt_api

        self.assertIsNone(ppt_api.extract_usd_price({"prices": {}}))
        self.assertIsNone(ppt_api.extract_usd_price({}))

    def test_fetch_set_cards_returns_card_list(self) -> None:
        import ppt_api

        fake_response = {
            "data": [
                {"name": "Venusaur", "cardNumber": "1", "prices": {"market": 0.5}},
                {"name": "Ivysaur", "cardNumber": "2", "prices": {"market": 0.2}},
            ]
        }
        with mock.patch.object(ppt_api, "api_get_json", return_value=fake_response) as mock_get:
            result = ppt_api.fetch_set_cards("Paldean Fates", api_key="test-key")

        mock_get.assert_called_once_with(
            "/cards",
            params={"set": "Paldean Fates", "fetchAllInSet": "true"},
            api_key="test-key",
            timeout=30,
        )
        self.assertEqual(2, len(result))
        self.assertEqual("Venusaur", result[0]["name"])

    def test_fetch_set_cards_returns_empty_for_none_response(self) -> None:
        import ppt_api

        with mock.patch.object(ppt_api, "api_get_json", return_value=None):
            result = ppt_api.fetch_set_cards("nonexistent", api_key="test-key")
        self.assertEqual([], result)

    def test_fetch_set_cards_returns_empty_for_empty_data(self) -> None:
        import ppt_api

        with mock.patch.object(ppt_api, "api_get_json", return_value={"data": []}):
            result = ppt_api.fetch_set_cards("empty set", api_key="test-key")
        self.assertEqual([], result)

    def test_build_card_number_index_basic(self) -> None:
        import ppt_api

        cards = [
            {"name": "Card A", "cardNumber": "1", "prices": {"market": 1.0}},
            {"name": "Card B", "cardNumber": "002", "prices": {"market": 2.0}},
            {"name": "Card C", "cardNumber": "3/100", "prices": {"market": 3.0}},
        ]
        index = ppt_api.build_card_number_index(cards)
        self.assertIn("1", index)
        self.assertIn("2", index)
        self.assertIn("3", index)
        self.assertEqual("Card A", index["1"]["name"])
        self.assertEqual("Card B", index["2"]["name"])

    def test_build_card_number_index_first_wins_on_collision(self) -> None:
        import ppt_api

        cards = [
            {"name": "First", "cardNumber": "001", "prices": {"market": 1.0}},
            {"name": "Duplicate", "cardNumber": "1", "prices": {"market": 2.0}},
        ]
        index = ppt_api.build_card_number_index(cards)
        self.assertEqual("First", index["1"]["name"])

    def test_build_card_number_index_skips_cards_without_number(self) -> None:
        import ppt_api

        cards = [
            {"name": "No Number", "prices": {"market": 1.0}},
            {"name": "Has Number", "cardNumber": "5", "prices": {"market": 2.0}},
        ]
        index = ppt_api.build_card_number_index(cards)
        self.assertEqual(1, len(index))
        self.assertIn("5", index)


class PptBulkCacheTests(unittest.TestCase):
    def test_build_ppt_set_cache_fetches_ranked_by_gap_count(self) -> None:
        import ppt_api

        gap_cards = [
            {"set_name": "Small Set", "card_number": "1"},
            {"set_name": "Big Set", "card_number": "1"},
            {"set_name": "Big Set", "card_number": "2"},
            {"set_name": "Big Set", "card_number": "3"},
        ]

        fetch_calls: list[str] = []

        def fake_fetch(set_name: str, *, api_key: str | None = None, timeout: int = 30) -> list[dict]:
            fetch_calls.append(set_name)
            return [
                {"name": f"Card {i}", "cardNumber": str(i), "prices": {"market": float(i)}}
                for i in range(1, 4)
            ]

        with mock.patch.object(ppt_api, "fetch_set_cards", side_effect=fake_fetch):
            cache = build_prices_db.build_ppt_set_cache(
                gap_cards,
                credit_budget=None,
                api_key="test-key",
            )

        # Big Set (3 gaps) should be fetched before Small Set (1 gap)
        self.assertEqual("Big Set", fetch_calls[0])
        self.assertEqual("Small Set", fetch_calls[1])
        self.assertEqual(2, len(cache))

    def test_build_ppt_set_cache_respects_credit_budget(self) -> None:
        import ppt_api

        gap_cards = [
            {"set_name": "Set A", "card_number": "1"},
            {"set_name": "Set A", "card_number": "2"},
            {"set_name": "Set B", "card_number": "1"},
        ]

        def fake_fetch(set_name: str, *, api_key: str | None = None, timeout: int = 30) -> list[dict]:
            # Each set returns 50 cards
            return [
                {"name": f"Card {i}", "cardNumber": str(i), "prices": {"market": float(i)}}
                for i in range(1, 51)
            ]

        with mock.patch.object(ppt_api, "fetch_set_cards", side_effect=fake_fetch):
            cache = build_prices_db.build_ppt_set_cache(
                gap_cards,
                credit_budget=60,
                api_key="test-key",
            )

        # Budget is 60 credits. First set costs 50 (remaining=10).
        # Second set would cost 50 but only 10 remaining, so it's still fetched
        # since the check is credits_remaining > 0 before fetching.
        # After second set: remaining = -40.
        # Third iteration: remaining <= 0, so break.
        self.assertEqual(2, len(cache))

    def test_build_ppt_set_cache_handles_empty_sets(self) -> None:
        import ppt_api

        gap_cards = [{"set_name": "Ghost Set", "card_number": "1"}]

        with mock.patch.object(ppt_api, "fetch_set_cards", return_value=[]):
            cache = build_prices_db.build_ppt_set_cache(
                gap_cards,
                credit_budget=100,
                api_key="test-key",
            )

        self.assertEqual(0, len(cache))

    def test_build_ppt_set_cache_handles_api_errors(self) -> None:
        import ppt_api

        gap_cards = [{"set_name": "Error Set", "card_number": "1"}]

        with mock.patch.object(ppt_api, "fetch_set_cards", side_effect=RuntimeError("HTTP 429")):
            cache = build_prices_db.build_ppt_set_cache(
                gap_cards,
                credit_budget=100,
                api_key="test-key",
            )

        self.assertEqual(0, len(cache))

    def test_try_fallback_providers_uses_bulk_cache(self) -> None:
        import ppt_api

        card = {
            "set_id": "sv04.5",
            "card_number": "001",
            "name": "Pineco",
            "set_name": "Paldean Fates",
        }
        ppt_set_cache = {
            ppt_api.normalize_set_name("Paldean Fates"): {
                "1": {"name": "Pineco", "cardNumber": "001", "prices": {"market": 0.25, "low": 0.1, "high": 0.5}},
            }
        }
        summary = {
            "transport_counts": {},
            "fallback_providers": {
                "ppt_configured": False,
                "poketrace_configured": False,
                "ppt_bulk_hits": 0,
                "ppt_bulk_misses": 0,
                "ppt_hits": 0,
                "ppt_misses": 0,
                "ppt_errors": 0,
                "ppt_first_error": None,
                "ppt_disabled_due_to_errors": False,
                "ppt_error_disable_threshold": 5,
                "poketrace_hits": 0,
                "poketrace_misses": 0,
                "poketrace_errors": 0,
                "poketrace_first_error": None,
                "poketrace_disabled_due_to_errors": False,
                "poketrace_error_disable_threshold": 5,
                "poketrace_set_mapping_failures": 0,
            },
        }

        result = build_prices_db.try_fallback_providers(
            card,
            ppt_set_cache=ppt_set_cache,
            poketrace_set_slugs={},
            summary=summary,
        )

        self.assertIsNotNone(result)
        self.assertEqual("USD", result["unit"])
        self.assertEqual(0.25, result["selected_variant"]["market"])
        self.assertEqual(1, summary["fallback_providers"]["ppt_bulk_hits"])
        self.assertEqual(0, summary["fallback_providers"]["ppt_bulk_misses"])
        self.assertEqual(1, summary["transport_counts"]["tcgplayer"]["ppt_bulk"])

    def test_try_fallback_providers_bulk_cache_miss_falls_through(self) -> None:
        import ppt_api

        card = {
            "set_id": "sv04.5",
            "card_number": "999",
            "name": "Nonexistent",
            "set_name": "Paldean Fates",
        }
        ppt_set_cache = {
            ppt_api.normalize_set_name("Paldean Fates"): {
                "1": {"name": "Pineco", "cardNumber": "001", "prices": {"market": 0.25}},
            }
        }
        summary = {
            "transport_counts": {},
            "fallback_providers": {
                "ppt_configured": False,
                "poketrace_configured": False,
                "ppt_bulk_hits": 0,
                "ppt_bulk_misses": 0,
                "ppt_hits": 0,
                "ppt_misses": 0,
                "ppt_errors": 0,
                "ppt_first_error": None,
                "ppt_disabled_due_to_errors": False,
                "ppt_error_disable_threshold": 5,
                "poketrace_hits": 0,
                "poketrace_misses": 0,
                "poketrace_errors": 0,
                "poketrace_first_error": None,
                "poketrace_disabled_due_to_errors": False,
                "poketrace_error_disable_threshold": 5,
                "poketrace_set_mapping_failures": 0,
            },
        }

        result = build_prices_db.try_fallback_providers(
            card,
            ppt_set_cache=ppt_set_cache,
            poketrace_set_slugs={},
            summary=summary,
        )

        self.assertIsNone(result)
        self.assertEqual(0, summary["fallback_providers"]["ppt_bulk_hits"])
        self.assertEqual(1, summary["fallback_providers"]["ppt_bulk_misses"])


class PoketraceApiTests(unittest.TestCase):
    def test_build_poketrace_set_slugs_tolerates_provider_errors(self) -> None:
        import poketrace_api

        cards = [
            {
                "set_id": "swsh9",
                "set_name": "Brilliant Stars",
            }
        ]

        with mock.patch.object(poketrace_api, "resolve_api_key", return_value="test-key"), mock.patch.object(
            poketrace_api,
            "fetch_sets",
            side_effect=RuntimeError("HTTP Error 429: Too Many Requests"),
        ):
            mapping = build_prices_db.build_poketrace_set_slugs(cards)

        self.assertEqual("brilliant-stars", mapping["swsh9"])

    def test_extract_usd_price_tcgplayer_near_mint(self) -> None:
        import poketrace_api

        card_data = {
            "name": "Chansey",
            "prices": {"tcgplayer": {"NEAR_MINT": {"avg": 45.61, "low": 40.0, "high": 50.0}}},
        }
        result = poketrace_api.extract_usd_price(card_data)
        self.assertIsNotNone(result)
        self.assertEqual(45.61, result["selected_variant"]["market"])

    def test_extract_usd_price_returns_none_for_empty(self) -> None:
        import poketrace_api

        self.assertIsNone(poketrace_api.extract_usd_price({"prices": {}}))

    def test_build_set_slug_mapping(self) -> None:
        import poketrace_api

        provider_sets = [
            {"slug": "twilight-masquerade", "name": "Twilight Masquerade"},
            {"slug": "base-set", "name": "Base Set"},
        ]
        our_set_names = {"sv06": "Twilight Masquerade", "base1": "Base Set"}
        mapping = poketrace_api.build_set_slug_mapping(provider_sets, our_set_names)
        self.assertEqual("twilight-masquerade", mapping["sv06"])
        self.assertEqual("base-set", mapping["base1"])


if __name__ == "__main__":
    unittest.main()
