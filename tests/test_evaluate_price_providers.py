from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import evaluate_price_providers as evp  # noqa: E402


class LoadSampleCsvTests(unittest.TestCase):
    def test_load_real_sample(self) -> None:
        sample_path = ROOT / "docs" / "english_price_gap_sample.csv"
        if not sample_path.exists():
            self.skipTest("sample CSV not present")
        cards = evp.load_sample_csv(sample_path)
        self.assertEqual(200, len(cards))
        self.assertEqual("pokemon:en:sv06:173", cards[0].card_id)


class SelectEvaluationCardsTests(unittest.TestCase):
    def _make_card(self, card_id: str, stratum: str = "recent:standard") -> evp.SampleCard:
        return evp.SampleCard(
            card_id=card_id,
            name="Test",
            set_id="sv01",
            set_name="Test Set",
            card_number="1",
            gap_group="eur_only",
            stratum=stratum,
        )

    def test_screenshot_examples_come_first(self) -> None:
        cards = [
            self._make_card("pokemon:en:base1:1"),
            self._make_card("pokemon:en:sv06:173"),
            self._make_card("pokemon:en:svp:047"),
            self._make_card("pokemon:en:base1:2"),
        ]
        selected = evp.select_evaluation_cards(cards, max_cards=3)
        self.assertEqual("pokemon:en:sv06:173", selected[0].card_id)
        self.assertEqual("pokemon:en:svp:047", selected[1].card_id)
        self.assertEqual(3, len(selected))

    def test_round_robin_across_strata(self) -> None:
        cards = [
            self._make_card("a", stratum="recent:standard"),
            self._make_card("b", stratum="recent:standard"),
            self._make_card("c", stratum="legacy:promo"),
            self._make_card("d", stratum="legacy:promo"),
        ]
        selected = evp.select_evaluation_cards(cards, max_cards=3)
        strata = [c.stratum for c in selected]
        self.assertIn("recent:standard", strata)
        self.assertIn("legacy:promo", strata)

    def test_max_cards_respected(self) -> None:
        cards = [self._make_card(f"card_{i}") for i in range(50)]
        selected = evp.select_evaluation_cards(cards, max_cards=10)
        self.assertEqual(10, len(selected))


class ParsePoketracePricesTests(unittest.TestCase):
    def test_parses_tcgplayer_near_mint(self) -> None:
        data = {
            "name": "Infernape",
            "prices": {
                "tcgplayer": {
                    "NEAR_MINT": {
                        "avg": 4.50,
                        "low": 2.00,
                        "high": 8.00,
                    }
                }
            },
        }
        low, market, high = evp.parse_poketrace_prices(data)
        self.assertEqual(2.00, low)
        self.assertEqual(4.50, market)
        self.assertEqual(8.00, high)

    def test_falls_back_to_ebay(self) -> None:
        data = {
            "prices": {
                "tcgplayer": {},
                "ebay": {
                    "NEAR_MINT": {
                        "avg": 3.00,
                        "low": 1.50,
                        "high": 5.00,
                    }
                },
            },
        }
        low, market, high = evp.parse_poketrace_prices(data)
        self.assertEqual(1.50, low)
        self.assertEqual(3.00, market)
        self.assertEqual(5.00, high)

    def test_empty_prices_returns_none(self) -> None:
        data = {"prices": {}}
        low, market, high = evp.parse_poketrace_prices(data)
        self.assertIsNone(low)
        self.assertIsNone(market)
        self.assertIsNone(high)


class ParsePptPricesTests(unittest.TestCase):
    def test_card_matcher_accepts_name_suffix_number(self) -> None:
        data = {"name": "N's Plan - 170/086", "setName": "SV: Black Bolt"}
        self.assertTrue(evp.ppt_card_matches(data, set_name="Black Bolt", card_number="170"))

    def test_card_matcher_rejects_wrong_number(self) -> None:
        data = {"name": "N's Plan - 171/086", "set": "Black Bolt"}
        self.assertFalse(evp.ppt_card_matches(data, set_name="Black Bolt", card_number="170"))

    def test_parses_standard_shape(self) -> None:
        data = {
            "name": "Charizard",
            "prices": {
                "market": 120.00,
                "low": 80.00,
                "mid": 110.00,
                "high": 150.00,
            },
        }
        low, market, high = evp.parse_ppt_prices(data)
        self.assertEqual(80.00, low)
        self.assertEqual(120.00, market)
        self.assertEqual(150.00, high)

    def test_falls_back_to_mid(self) -> None:
        data = {"prices": {"mid": 5.00, "low": 2.00}}
        low, market, high = evp.parse_ppt_prices(data)
        self.assertEqual(2.00, low)
        self.assertEqual(5.00, market)
        self.assertIsNone(high)

    def test_no_prices_returns_none(self) -> None:
        data = {"prices": {}}
        low, market, high = evp.parse_ppt_prices(data)
        self.assertIsNone(low)
        self.assertIsNone(market)
        self.assertIsNone(high)


class ParseScrydexPricesTests(unittest.TestCase):
    def test_parses_variant_prices(self) -> None:
        data = {
            "name": "Charizard",
            "variants": [
                {
                    "name": "holofoil",
                    "prices": [
                        {"currency": "USD", "low": 10.0, "market": 15.0, "high": 20.0},
                    ],
                }
            ],
        }
        low, market, high = evp.parse_scrydex_prices(data)
        self.assertEqual(10.0, low)
        self.assertEqual(15.0, market)
        self.assertEqual(20.0, high)

    def test_ignores_non_usd(self) -> None:
        data = {
            "variants": [
                {
                    "name": "normal",
                    "prices": [
                        {"currency": "EUR", "low": 1.0, "market": 2.0, "high": 3.0},
                    ],
                }
            ],
        }
        low, market, high = evp.parse_scrydex_prices(data)
        self.assertIsNone(low)
        self.assertIsNone(market)
        self.assertIsNone(high)

    def test_empty_variants(self) -> None:
        data = {"variants": []}
        low, market, high = evp.parse_scrydex_prices(data)
        self.assertIsNone(low)
        self.assertIsNone(market)
        self.assertIsNone(high)


class BuildPoketraceSetMappingTests(unittest.TestCase):
    def test_maps_by_name(self) -> None:
        provider_sets = [
            {"slug": "twilight-masquerade", "name": "Twilight Masquerade"},
            {"slug": "base-set", "name": "Base Set"},
        ]
        cards = [
            evp.SampleCard(
                card_id="pokemon:en:sv06:173",
                name="Infernape",
                set_id="sv06",
                set_name="Twilight Masquerade",
                card_number="173",
                gap_group="eur_only",
                stratum="recent:alt_art",
            ),
        ]
        mapping = evp.build_poketrace_set_mapping_from_cards(provider_sets, cards)
        self.assertEqual({"sv06": "twilight-masquerade"}, mapping)

    def test_no_match_returns_empty(self) -> None:
        provider_sets = [{"slug": "base-set", "name": "Base Set"}]
        cards = [
            evp.SampleCard(
                card_id="pokemon:en:sv06:173",
                name="Infernape",
                set_id="sv06",
                set_name="Twilight Masquerade",
                card_number="173",
                gap_group="eur_only",
                stratum="recent:alt_art",
            ),
        ]
        mapping = evp.build_poketrace_set_mapping_from_cards(provider_sets, cards)
        self.assertEqual({}, mapping)


class BuildProviderSummaryTests(unittest.TestCase):
    def test_summary_counts(self) -> None:
        results = [
            evp.LookupResult(card_id="a", provider="test", found=True, market_price=5.0),
            evp.LookupResult(card_id="b", provider="test", found=True, market_price=None, notes="found_no_price"),
            evp.LookupResult(card_id="c", provider="test", found=False),
        ]
        summary = evp.build_provider_summary(results)
        self.assertEqual(3, summary["test"]["cards_tested"])
        self.assertEqual(2, summary["test"]["cards_found"])
        self.assertEqual(1, summary["test"]["cards_with_price"])
        self.assertAlmostEqual(66.7, summary["test"]["coverage_rate"], places=1)
        self.assertAlmostEqual(33.3, summary["test"]["price_rate"], places=1)

    def test_screenshot_examples_tracked(self) -> None:
        results = [
            evp.LookupResult(
                card_id="pokemon:en:sv06:173",
                provider="test",
                found=True,
                market_price=4.50,
            ),
        ]
        summary = evp.build_provider_summary(results)
        self.assertIn("pokemon:en:sv06:173", summary["test"]["screenshot_examples"])
        self.assertEqual(4.50, summary["test"]["screenshot_examples"]["pokemon:en:sv06:173"]["market_price"])


if __name__ == "__main__":
    unittest.main()
