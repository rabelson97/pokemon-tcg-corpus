from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_prices_db  # noqa: E402


class BuildPricesDbTests(unittest.TestCase):
    def test_extract_price_rows_prefers_current_tcgplayer_variant_fields(self) -> None:
        card = {
            "id": "pokemon:en:sv01:001",
            "locale": "en",
            "pricing": {
                "tcgplayer": {
                    "updated": "2026-04-05T12:34:56.000Z",
                    "unit": "USD",
                    "normal": {
                        "lowPrice": 1.25,
                        "midPrice": 2.5,
                        "highPrice": 4.0,
                        "marketPrice": 2.2,
                        "directLowPrice": 2.1,
                    },
                    "reverse": {
                        "lowPrice": 8.0,
                        "midPrice": 9.0,
                        "highPrice": 10.0,
                        "marketPrice": 8.5,
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

        rows = build_prices_db.extract_price_rows(card)

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
                "2026-04-05T12:34:56.000Z",
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


if __name__ == "__main__":
    unittest.main()
