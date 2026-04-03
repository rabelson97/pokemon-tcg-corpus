#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path
from typing import Any

from pokemontcg_api import fetch_all_cards


DB_USER_VERSION = 1


PRICE_VARIANT_PRIORITY = [
    "normal",
    "holofoil",
    "reverseHolofoil",
    "1stEditionHolofoil",
    "1stEditionNormal",
    "unlimitedHolofoil",
    "unlimitedNormal",
]


def extract_market_price(card: dict[str, Any]) -> float | None:
    tcgplayer = card.get("tcgplayer") or {}
    prices = tcgplayer.get("prices") or {}
    if not isinstance(prices, dict):
        return None

    direct_market = prices.get("market")
    if isinstance(direct_market, (int, float)):
        return float(direct_market)

    for key in PRICE_VARIANT_PRIORITY:
        variant = prices.get(key)
        if isinstance(variant, dict) and isinstance(variant.get("market"), (int, float)):
            return float(variant["market"])

    for variant in prices.values():
        if isinstance(variant, dict) and isinstance(variant.get("market"), (int, float)):
            return float(variant["market"])
    return None


def validate_prices_db(db_path: Path, *, min_row_count: int) -> int:
    with sqlite3.connect(db_path) as connection:
        integrity = connection.execute("PRAGMA integrity_check;").fetchone()
        if integrity is None or integrity[0] != "ok":
            raise RuntimeError(f"PRAGMA integrity_check failed: {integrity}")

        user_version = int(connection.execute("PRAGMA user_version;").fetchone()[0])
        if user_version != DB_USER_VERSION:
            raise RuntimeError(f"PRAGMA user_version expected {DB_USER_VERSION}, got {user_version}")

        row_count = int(connection.execute("SELECT COUNT(*) FROM prices;").fetchone()[0])
        if row_count < min_row_count:
            raise RuntimeError(f"prices row count {row_count} is below minimum {min_row_count}")
        return row_count


def build_prices_db(output_path: Path, *, api_key: str, limit: int | None, min_row_count: int) -> int:
    cards = fetch_all_cards(
        api_key=api_key,
        limit=limit,
        select_fields=["id", "tcgplayer"],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()

    with sqlite3.connect(temp_path) as connection:
        connection.execute("PRAGMA journal_mode=DELETE;")
        connection.execute("PRAGMA synchronous=NORMAL;")
        connection.execute(f"PRAGMA user_version={DB_USER_VERSION};")
        connection.execute("DROP TABLE IF EXISTS prices;")
        connection.execute(
            """
            CREATE TABLE prices (
              card_id TEXT PRIMARY KEY,
              market_price REAL,
              updated_at TEXT
            );
            """
        )

        rows = []
        priced_rows = 0
        for card in cards:
            market_price = extract_market_price(card)
            if market_price is not None:
                priced_rows += 1
            updated_at = (card.get("tcgplayer") or {}).get("updatedAt")
            rows.append((card["id"], market_price, updated_at))

        connection.executemany(
            "INSERT INTO prices (card_id, market_price, updated_at) VALUES (?, ?, ?);",
            rows,
        )
        connection.commit()

    row_count = validate_prices_db(temp_path, min_row_count=min_row_count)
    temp_path.replace(output_path)
    print(f"wrote {output_path} rows={row_count} priced_rows={priced_rows}")
    return row_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a SQLite price database from pokemontcg.io.")
    parser.add_argument("--output", default="prices.db", help="Output SQLite database path")
    parser.add_argument("--limit", type=int, help="Optional card limit for local verification")
    parser.add_argument("--min-row-count", type=int, default=1000)
    args = parser.parse_args()

    api_key = os.environ.get("POKEMONTCG_API_KEY")
    if not api_key:
        raise SystemExit("POKEMONTCG_API_KEY is required")

    build_prices_db(
        Path(args.output).resolve(),
        api_key=api_key,
        limit=args.limit,
        min_row_count=args.min_row_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
