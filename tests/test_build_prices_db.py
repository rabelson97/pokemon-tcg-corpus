from __future__ import annotations

import datetime as dt
import sys
import unittest
from pathlib import Path


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


class PoketraceApiTests(unittest.TestCase):
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
