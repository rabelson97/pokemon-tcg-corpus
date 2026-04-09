from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_english_price_gap_sample as gap_sample  # noqa: E402


class BuildEnglishPriceGapSampleTests(unittest.TestCase):
    def test_classify_card_class_prefers_promo(self) -> None:
        self.assertEqual(
            "promo",
            gap_sample.classify_card_class(
                set_id="svp",
                set_name="SVP Black Star Promos",
                card_number="047",
                rarity="Promo",
            ),
        )

    def test_classify_card_class_detects_alt_art(self) -> None:
        self.assertEqual(
            "alt_art",
            gap_sample.classify_card_class(
                set_id="sv06",
                set_name="Twilight Masquerade",
                card_number="173",
                rarity="Illustration Rare",
            ),
        )

    def test_select_sample_keeps_screenshot_examples_first(self) -> None:
        cards = [
            gap_sample.GapCard(
                card_id="pokemon:en:base3:024",
                name="Kabutops",
                set_id="base3",
                set_name="Fossil",
                card_number="24",
                rarity="Rare Holo",
                gap_group="eur_only",
                source_names=("cardmarket",),
                release_date="1999-10-10",
                release_bucket="legacy",
                card_class="standard",
                stratum="legacy:standard",
                notes="",
            ),
            gap_sample.GapCard(
                card_id="pokemon:en:sv06:173",
                name="Infernape",
                set_id="sv06",
                set_name="Twilight Masquerade",
                card_number="173",
                rarity="Illustration Rare",
                gap_group="eur_only",
                source_names=("cardmarket",),
                release_date="2024-05-24",
                release_bucket="recent",
                card_class="alt_art",
                stratum="recent:alt_art",
                notes="screenshot-example",
            ),
        ]

        sample = gap_sample.select_sample(cards, sample_size=2)

        self.assertEqual("pokemon:en:sv06:173", sample[0].card_id)
        self.assertEqual(2, len(sample))


if __name__ == "__main__":
    unittest.main()
