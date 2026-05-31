from __future__ import annotations

import datetime as dt
import sqlite3
import sys
import tempfile
import urllib.error
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_prices_db  # noqa: E402


class BuildPricesDbTests(unittest.TestCase):
    def test_fetch_set_scoped_pokemontcgio_cards_avoids_global_catalog_query(self) -> None:
        english_cards = [
            {"set_id": "sv01", "card_number": "001"},
            {"set_id": "sv01", "card_number": "002"},
        ]
        payload = {
            "data": [
                {"id": "sv01-1", "number": "001", "set": {"id": "sv01"}},
                {"id": "sv01-2", "number": "002", "set": {"id": "sv01"}},
            ],
            "totalCount": 2,
        }

        with mock.patch.object(build_prices_db, "pokemontcgio_api_get_json", return_value=payload) as mock_get:
            cards = build_prices_db.fetch_set_scoped_pokemontcgio_cards(english_cards)

        self.assertEqual(["sv01-1", "sv01-2"], [card["id"] for card in cards])
        queries = [call.kwargs["params"]["q"] for call in mock_get.call_args_list]
        self.assertIn("set.id:sv01", queries)
        self.assertNotIn("set.id:*", queries)
        self.assertTrue(all(query.startswith("set.id:") for query in queries))

    def test_fetch_set_scoped_pokemontcgio_cards_skips_unknown_provider_sets(self) -> None:
        english_cards = [{"set_id": "unknown", "card_number": "001"}]
        not_found = urllib.error.HTTPError(
            url="https://api.pokemontcg.io/v2/cards",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,
        )

        with mock.patch.object(build_prices_db, "pokemontcgio_api_get_json", side_effect=not_found):
            cards = build_prices_db.fetch_set_scoped_pokemontcgio_cards(english_cards)

        self.assertEqual([], cards)

    def test_fetch_set_scoped_pokemontcgio_cards_fails_on_transient_provider_errors(self) -> None:
        english_cards = [{"set_id": "sv01", "card_number": "001"}]

        with mock.patch.object(build_prices_db, "pokemontcgio_api_get_json", side_effect=TimeoutError("timed out")):
            with self.assertRaises(RuntimeError):
                build_prices_db.fetch_set_scoped_pokemontcgio_cards(english_cards)

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

    def test_select_price_sources_prefers_tcgdex_tcgplayer_for_english_cards(self) -> None:
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
        self.assertEqual("2026-04-05T12:34:56.000Z", rows[0][7])
        self.assertEqual(1, summary["transport_counts"]["tcgplayer"]["tcgdex"])
        self.assertNotIn("cardmarket", selected)

    def test_select_price_sources_falls_back_to_pokemontcgio_when_tcgdex_stale(self) -> None:
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
                    "updated": "2026-03-01T00:00:00.000Z",
                    "unit": "USD",
                    "normal": {
                        "lowPrice": 999.0,
                        "marketPrice": 999.0,
                        "highPrice": 999.0,
                    },
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

    def test_select_price_sources_falls_back_to_pokemontcgio_when_tcgdex_missing(self) -> None:
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
            "pricing": {},
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

    def test_select_price_sources_uses_stale_pokemontcgio_prices_as_last_resort(self) -> None:
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

        self.assertIn("tcgplayer", selected)
        self.assertEqual(2.2, selected["tcgplayer"]["selected_variant"]["market"])
        self.assertNotIn("cardmarket", selected)
        self.assertEqual(1, summary["pokemontcgio"]["stale_tcgplayer_rows"])
        self.assertEqual(1, summary["pokemontcgio"]["stale_reasons"]["older_than_max_age"])

    def test_candidate_pokemontcgio_match_keys_cover_alias_set_and_number_formats(self) -> None:
        self.assertIn(
            ("swsh35", "71"),
            build_prices_db.candidate_pokemontcgio_match_keys("swsh3.5", "71"),
        )
        self.assertIn(
            ("zsv10pt5", "157"),
            build_prices_db.candidate_pokemontcgio_match_keys("sv10.5b", "157"),
        )
        self.assertIn(
            ("rsv10pt5", "157"),
            build_prices_db.candidate_pokemontcgio_match_keys("sv10.5w", "157"),
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

    def test_select_price_sources_maps_black_bolt_alias_under_original_cardhawk_id(self) -> None:
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
        card = {
            "id": "pokemon:en:sv10.5b:157",
            "locale": "en",
            "set_id": "sv10.5b",
            "set_name": "Black Bolt",
            "card_number": "157",
            "pricing": {},
        }
        pokemontcgio_index = {
            ("zsv10pt5", "157"): {
                "id": "zsv10pt5-157",
                "tcgplayer": {
                    "updatedAt": "2026/04/10",
                    "prices": {
                        "holofoil": {
                            "low": 10.0,
                            "market": 12.5,
                            "high": 20.0,
                        }
                    },
                },
            }
        }

        selected = build_prices_db.select_price_sources(
            card,
            pokemontcgio_index=pokemontcgio_index,
            max_pokemontcgio_age_days=14,
            now=dt.datetime(2026, 4, 10, tzinfo=dt.timezone.utc),
            summary=summary,
        )
        rows = build_prices_db.extract_price_rows_from_selected_sources(card["id"], selected)

        self.assertIn("tcgplayer", selected)
        self.assertEqual("pokemon:en:sv10.5b:157", rows[0][0])
        self.assertEqual(12.5, rows[0][5])
        self.assertEqual(1, summary["identity_audit"]["alias_required_matches"])

    def test_fetch_targeted_pokemontcgio_cards_tries_explicit_alias_provider_id(self) -> None:
        card = {
            "upstream_id": "sv10.5b-157",
            "set_id": "sv10.5b",
            "card_number": "157",
        }
        provider_card = {"id": "zsv10pt5-157", "set": {"id": "zsv10pt5"}, "number": "157"}

        def fake_fetch_card_by_id(card_id: str) -> dict | None:
            if card_id == "zsv10pt5-157":
                return provider_card
            return None

        with mock.patch.object(build_prices_db, "fetch_card_by_id", side_effect=fake_fetch_card_by_id) as fetch_mock, mock.patch.object(
            build_prices_db,
            "search_card_by_set_and_number",
            return_value=None,
        ) as search_mock:
            fetched = build_prices_db.fetch_targeted_pokemontcgio_cards([card])

        self.assertEqual([provider_card], fetched)
        self.assertIn(mock.call("zsv10pt5-157"), fetch_mock.mock_calls)
        self.assertEqual(1, search_mock.call_count)

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

    def test_build_prices_db_identity_audit_warns_with_deterministic_gap_sample(self) -> None:
        cards = [
            {
                "id": "pokemon:en:sv01:002",
                "locale": "en",
                "upstream_id": "sv01-002",
                "set_id": "sv01",
                "set_name": "Scarlet & Violet",
                "name": "Ivysaur",
                "card_number": "002",
                "pricing": {},
            },
            {
                "id": "pokemon:en:sv10.5b:157",
                "locale": "en",
                "upstream_id": "sv10.5b-157",
                "set_id": "sv10.5b",
                "set_name": "Black Bolt",
                "name": "Kyurem ex",
                "card_number": "157",
                "pricing": {},
            },
        ]
        targeted_result = [
            {
                "set": {"id": "zsv10pt5"},
                "number": "157",
                "tcgplayer": {"updatedAt": "2026/04/10", "prices": {"holofoil": {"low": 10.0, "market": 12.5, "high": 20.0}}},
            }
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "prices.db"
            with mock.patch.object(build_prices_db, "fetch_all_card_records", return_value=(cards, {"en": 2})), mock.patch.object(
                build_prices_db,
                "fetch_targeted_pokemontcgio_cards",
                return_value=targeted_result,
            ), mock.patch.object(
                build_prices_db,
                "build_pokemontcgio_index",
                return_value={("zsv10pt5", "157"): targeted_result[0]},
            ), mock.patch.object(
                build_prices_db,
                "build_poketrace_set_slugs",
                return_value={},
            ):
                summary = build_prices_db.build_prices_db(
                    output_path,
                    locales=["en"],
                    limit=2,
                    min_row_count=0,
                    max_pokemontcgio_age_days=90,
                    max_fallback_cards=0,
                )

        audit = summary["identity_audit"]
        self.assertEqual(1, audit["english_cards_without_usd"])
        self.assertEqual(1, audit["pokemontcgio_unmatched_by_set_number"])
        self.assertEqual(1, audit["alias_required_matches"])
        self.assertEqual(0, audit["price_rows_without_cardhawk_card_id"])
        self.assertEqual(["pokemon:en:sv01:002"], [sample["card_id"] for sample in audit["sample_gaps"]])
        self.assertEqual("no_pokemontcgio_match", audit["sample_gaps"][0]["reason"])
        self.assertIn("sv1-2", audit["sample_gaps"][0]["attempted_pokemontcgio_match_keys"])

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
                    max_pokemontcgio_age_days=90,
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


class PkmnggPriceSourceTests(unittest.TestCase):
    def test_normalize_number_matches_prefixed_zero_promo_numbers(self) -> None:
        self.assertEqual("BW4", build_prices_db.normalize_number("BW04"))
        self.assertEqual("BW4", build_prices_db.normalize_number("BW004"))
        self.assertEqual("SV31", build_prices_db.normalize_number("SV031"))

    def test_match_card_in_candidates_accepts_prefixed_zero_number_difference(self) -> None:
        card = {
            "id": "pokemon:en:bwp:BW04",
            "locale": "en",
            "set_id": "bwp",
            "name": "Reshiram",
            "card_number": "BW04",
            "pricing": {},
        }
        candidates = [
            {
                "name": "Reshiram",
                "number": "BW004",
                "numberDisplay": "BW004",
                "variantMap": {"holofoil": {"price": 4.22, "notMarket": False}},
            }
        ]

        match, reason = build_prices_db.match_card_in_candidates(card, candidates)

        self.assertIsNotNone(match)
        self.assertEqual("exact_match", reason)

    def test_candidate_pkmngg_set_paths_include_classic_collection(self) -> None:
        card = {
            "id": "pokemon:en:cel25:107A",
            "locale": "en",
            "set_id": "cel25",
            "set_name": "Celebrations",
            "name": "Donphan",
            "card_number": "107A",
            "pricing": {},
        }

        paths = build_prices_db.candidate_pkmngg_set_paths(card)

        self.assertIn(("sword-shield", "celebrations"), paths)
        self.assertIn(("sword-shield", "celebrations-classic-collection"), paths)

    def test_candidate_pkmngg_set_paths_infers_exact_sitemap_slug(self) -> None:
        card = {
            "id": "pokemon:en:dc1:1",
            "locale": "en",
            "set_id": "dc1",
            "set_name": "Double Crisis",
            "name": "Team Aqua's Spheal",
            "card_number": "1",
            "pricing": {},
        }

        paths = build_prices_db.candidate_pkmngg_set_paths(
            card,
            sitemap_paths=[("xy", "double-crisis"), ("xy", "generations")],
        )

        self.assertEqual([("xy", "double-crisis")], paths)

    def test_extract_pkmngg_usd_price_skips_non_market_variants(self) -> None:
        card_data = {
            "variantMap": {
                "stamp": {"price": 99.0, "notMarket": True, "priceDisplay": "$99.00"},
                "normal": {
                    "price": 1.5,
                    "notMarket": False,
                    "priceDisplay": "$1.50",
                    "tcgPlayerId": "12345",
                },
            }
        }

        result = build_prices_db.extract_pkmngg_usd_price(card_data, updated_at="2026/04/10 12:00:00")

        self.assertIsNotNone(result)
        self.assertEqual("USD", result["unit"])
        self.assertEqual(1.5, result["selected_variant"]["marketPrice"])
        self.assertEqual("normal", result["selected_variant"]["pkmnggVariantKey"])

    def test_select_price_sources_falls_back_to_pkmngg(self) -> None:
        summary = {
            "transport_counts": {},
            "fallback_providers": {},
            "pokemontcgio": {
                "english_cards_considered": 0,
                "english_counts": 0,
                "english_cards_with_match": 0,
                "english_cards_without_match": 0,
                "english_cards_with_tcgplayer": 0,
                "english_cards_without_tcgplayer": 0,
                "stale_tcgplayer_rows": 0,
                "stale_reasons": {},
            },
        }
        card = {
            "id": "pokemon:en:sve:001",
            "locale": "en",
            "set_id": "sve",
            "name": "Grass Energy",
            "card_number": "001",
            "pricing": {},
        }
        
        # Mock fetch_pkmngg_set_cards to return fake NEXT_DATA pageProps cardData
        fake_cards = [
            {
                "name": "Grass Energy",
                "number": "001",
                "variantMap": {
                    "normal": {
                        "price": 1.5,
                        "tcgPlayerId": 12345
                    }
                }
            }
        ]
        
        pkmngg_set_cache = {}
        
        with mock.patch("build_prices_db.fetch_pkmngg_set_cards", return_value=fake_cards) as mock_fetch:
            # First, check that select_price_sources returns empty (no pokemontcgio match)
            selected = build_prices_db.select_price_sources(
                card,
                pokemontcgio_index={},
                max_pokemontcgio_age_days=14,
                now=dt.datetime(2026, 4, 10, tzinfo=dt.timezone.utc),
                summary=summary,
            )
            
            self.assertEqual({}, selected)
            
            # Now, simulate the dynamic build loop fallback block logic:
            set_id = card["set_id"]
            self.assertIn(set_id, build_prices_db.EXPLICIT_SET_MAPPINGS)
            
            mapping = build_prices_db.EXPLICIT_SET_MAPPINGS[set_id]
            series, slug = mapping
            cache_key = f"{series}/{slug}"
            if cache_key not in pkmngg_set_cache:
                pkmngg_set_cache[cache_key] = build_prices_db.fetch_pkmngg_set_cards(series, slug)
                
            candidates = pkmngg_set_cache[cache_key]
            match, reason = build_prices_db.match_card_in_candidates(card, candidates)
            self.assertIsNotNone(match)
            self.assertEqual("exact_match", reason)
            
            result = build_prices_db.extract_pkmngg_usd_price(match, updated_at="2026/04/10 12:00:00")
            self.assertIsNotNone(result)
            
            # Construct the pricing row structure
            selected["pkmngg"] = result
            
            # Verify rows extraction
            rows = build_prices_db.extract_price_rows_from_selected_sources(card["id"], selected)
            self.assertEqual(1, len(rows))
            self.assertEqual(
                (
                    "pokemon:en:sve:001",
                    "US",
                    "USD",
                    "pkmngg",
                    None,
                    1.5,
                    None,
                    "2026/04/10 12:00:00",
                    1,
                ),
                rows[0]
            )

    def test_parse_pricecharting_used_price(self) -> None:
        page = """
        <tr id="used_price">
          <td>Ungraded</td>
          <td class="price numeric used_price"><span class="js-price">$610.00</span></td>
        </tr>
        """

        self.assertEqual(610.0, build_prices_db.parse_pricecharting_used_price(page))

    def test_parse_scrydex_set_prices(self) -> None:
        page = """
        <a class="card-grid-card" href="/pokemon/cards/treecko/wb1-1?variant=normal">
          <span>Treecko #1</span>
          <span>$399.39 (LP)</span>
        </a>
        <a class="card-grid-card" href="/pokemon/cards/pikachu/wb1-5?variant=normal">
          <span>Pikachu #5</span>
          <span>$165.95</span>
        </a>
        """

        prices = build_prices_db.parse_scrydex_set_prices(page)

        self.assertEqual({"1", "5"}, set(prices))
        self.assertEqual("Treecko", prices["1"]["name"])
        self.assertEqual(399.39, prices["1"]["price"])
        self.assertEqual("Pikachu", prices["5"]["name"])
        self.assertEqual(165.95, prices["5"]["price"])

    def test_static_scraped_price_fallback_uses_scrydex(self) -> None:
        summary = {
            "transport_counts": {},
            "fallback_providers": {
                "scrydex_hits": 0,
                "scrydex_misses": 0,
                "scrydex_errors": 0,
                "scrydex_first_error": None,
                "scrydex_sets_fetched": 0,
                "pricecharting_hits": 0,
                "pricecharting_misses": 0,
                "pricecharting_errors": 0,
                "pricecharting_first_error": None,
            },
        }
        card = {
            "id": "pokemon:en:ex5.5:5",
            "locale": "en",
            "set_id": "ex5.5",
            "set_name": "Poké Card Creator Pack",
            "name": "Pikachu",
            "card_number": "5",
        }
        page_cache = {
            build_prices_db.SCRYDEX_SET_URLS["ex5.5"]: """
            <a href="/pokemon/cards/pikachu/wb1-5?variant=normal">
              <span>Pikachu #5</span><span>$165.95</span>
            </a>
            """
        }

        result = build_prices_db.try_static_scraped_price_fallback(
            card,
            updated_at="2026/05/30 12:00:00",
            pricecharting_page_cache={},
            scrydex_page_cache=page_cache,
            scrydex_price_cache={},
            summary=summary,
        )

        self.assertIsNotNone(result)
        self.assertEqual("scrydex", result["source_name"])
        self.assertEqual(165.95, result["selected_variant"]["marketPrice"])
        self.assertEqual(1, summary["fallback_providers"]["scrydex_hits"])

    def test_static_scraped_price_fallback_uses_pricecharting(self) -> None:
        summary = {
            "transport_counts": {},
            "fallback_providers": {
                "scrydex_hits": 0,
                "scrydex_misses": 0,
                "scrydex_errors": 0,
                "scrydex_first_error": None,
                "scrydex_sets_fetched": 0,
                "pricecharting_hits": 0,
                "pricecharting_misses": 0,
                "pricecharting_errors": 0,
                "pricecharting_first_error": None,
            },
        }
        card = {
            "id": "pokemon:en:dpp:DP25",
            "locale": "en",
            "set_id": "dpp",
            "set_name": "DP Black Star Promos",
            "name": "Tropical Wind",
            "card_number": "DP25",
        }
        slug = build_prices_db.PRICECHARTING_FALLBACK_SLUGS[card["id"]]
        url = f"{build_prices_db.PRICECHARTING_BASE_URL}/{slug}"
        page_cache = {
            url: """
            <tr id="used_price">
              <td class="price numeric used_price"><span>$610.00</span></td>
            </tr>
            """
        }

        result = build_prices_db.try_static_scraped_price_fallback(
            card,
            updated_at="2026/05/30 12:00:00",
            pricecharting_page_cache=page_cache,
            scrydex_page_cache={},
            scrydex_price_cache={},
            summary=summary,
        )

        self.assertIsNotNone(result)
        self.assertEqual("pricecharting", result["source_name"])
        self.assertEqual(610.0, result["selected_variant"]["marketPrice"])
        self.assertEqual(1, summary["fallback_providers"]["pricecharting_hits"])

    def test_no_usd_market_fallback_writes_honest_terminal_row(self) -> None:
        summary = {
            "transport_counts": {},
            "fallback_providers": {"no_usd_market_rows": 0},
        }
        card = {
            "id": "pokemon:en:bwp:BW78",
            "locale": "en",
            "set_id": "bwp",
            "set_name": "BW Black Star Promos",
            "name": "Raichu",
            "card_number": "BW78",
        }

        result = build_prices_db.try_no_usd_market_fallback(
            card,
            updated_at="2026/05/30 12:00:00",
            summary=summary,
        )

        self.assertIsNotNone(result)
        rows = build_prices_db.extract_price_rows_from_selected_sources(card["id"], {"no_usd_market": result})
        self.assertEqual(1, len(rows))
        self.assertEqual("USD", rows[0][2])
        self.assertEqual("no_usd_market", rows[0][3])
        self.assertIsNone(rows[0][5])
        self.assertEqual(1, summary["fallback_providers"]["no_usd_market_rows"])


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
    def test_candidate_poketrace_card_numbers_cover_prefixed_and_suffix_forms(self) -> None:
        self.assertEqual(
            ["BW04", "BW4", "BW004", "4"],
            build_prices_db.candidate_poketrace_card_numbers("BW04"),
        )
        self.assertIn("107", build_prices_db.candidate_poketrace_card_numbers("107A"))

    def test_try_fallback_providers_retries_poketrace_number_variants_with_name_validation(self) -> None:
        import poketrace_api

        card = {
            "set_id": "bwp",
            "card_number": "BW04",
            "name": "Reshiram",
            "set_name": "BW Black Star Promos",
        }
        summary = {
            "transport_counts": {},
            "fallback_providers": {
                "ppt_configured": False,
                "poketrace_configured": True,
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

        def fake_lookup(slug: str, number: str, *, api_key: str | None = None) -> dict | None:
            if number == "BW4":
                return {"name": "Wrong Card", "prices": {"tcgplayer": {"NEAR_MINT": {"avg": 99.0}}}}
            if number == "BW004":
                return {"name": "Reshiram", "prices": {"tcgplayer": {"NEAR_MINT": {"avg": 4.22, "low": 3.0, "high": 5.0}}}}
            return None

        with mock.patch.object(poketrace_api, "resolve_api_key", return_value="test-key"), mock.patch.object(
            poketrace_api,
            "lookup_card",
            side_effect=fake_lookup,
        ) as lookup_mock:
            result = build_prices_db.try_fallback_providers(
                card,
                ppt_set_cache={},
                poketrace_set_slugs={"bwp": "bw-black-star-promos"},
                summary=summary,
            )

        self.assertIsNotNone(result)
        self.assertEqual("poketrace", result["source_name"])
        self.assertEqual(4.22, result["selected_variant"]["market"])
        self.assertEqual(1, summary["fallback_providers"]["poketrace_hits"])
        self.assertIn(mock.call("bw-black-star-promos", "BW004", api_key="test-key"), lookup_mock.mock_calls)

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
