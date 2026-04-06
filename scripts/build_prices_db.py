#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from tcgdex_api import fetch_all_card_records, parse_locales


DB_USER_VERSION = 2
TCGPLAYER_NUMERIC_KEYS = [
    "lowPrice",
    "midPrice",
    "highPrice",
    "marketPrice",
    "directLowPrice",
    # Backward-compatible fallbacks for older payloads.
    "low",
    "mid",
    "high",
    "market",
    "averageSellPrice",
]
TCGPLAYER_VARIANT_PREFERENCE = [
    "normal",
    "reverse",
    "holo",
    # Backward-compatible fallbacks for older payloads / historical docs.
    "holofoil",
    "reverseHolofoil",
    "1stEditionHolofoil",
    "1stEditionNormal",
    "unlimitedHolofoil",
    "unlimitedNormal",
]


def first_number(payload: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def has_any_numeric_price(payload: dict[str, Any], keys: list[str]) -> bool:
    return first_number(payload, keys) is not None


def score_tcgplayer_variant(payload: dict[str, Any]) -> tuple[int, float]:
    score = sum(1 for key in TCGPLAYER_NUMERIC_KEYS if first_number(payload, [key]) is not None)
    market_anchor = first_number(payload, ["marketPrice", "market", "midPrice", "mid", "averageSellPrice"]) or 0.0
    return score, market_anchor


def best_tcgplayer_variant(payload: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    direct_payload = {key: value for key, value in payload.items() if key not in TCGPLAYER_VARIANT_PREFERENCE}
    if has_any_numeric_price(direct_payload, TCGPLAYER_NUMERIC_KEYS):
        return None, payload

    for key in TCGPLAYER_VARIANT_PREFERENCE:
        variant = payload.get(key)
        if isinstance(variant, dict) and has_any_numeric_price(variant, TCGPLAYER_NUMERIC_KEYS):
            return key, variant

    fallback_variants: list[tuple[str, dict[str, Any]]] = []
    for key, variant in payload.items():
        if isinstance(variant, dict) and has_any_numeric_price(variant, TCGPLAYER_NUMERIC_KEYS):
            fallback_variants.append((str(key), variant))
    if not fallback_variants:
        return None, None
    fallback_variants.sort(key=lambda item: score_tcgplayer_variant(item[1]), reverse=True)
    return fallback_variants[0]


def create_locale_coverage_audit(locales: list[str]) -> dict[str, dict[str, int]]:
    return {
        locale: {
            "cards_total": 0,
            "cards_with_tcgplayer": 0,
            "cards_with_cardmarket": 0,
            "cards_with_both_sources": 0,
            "cards_with_tcgplayer_only": 0,
            "cards_with_cardmarket_only": 0,
            "cards_without_prices": 0,
            "cards_primary_tcgplayer": 0,
            "cards_primary_cardmarket": 0,
            "tcgplayer_rows": 0,
            "cardmarket_rows": 0,
        }
        for locale in locales
    }


def update_locale_coverage_audit(
    audit: dict[str, dict[str, int]],
    *,
    locale: str,
    extracted_rows: list[tuple[str, str, str, str, float | None, float | None, float | None, str | None, int]],
) -> None:
    locale_counts = audit.setdefault(
        locale,
        {
            "cards_total": 0,
            "cards_with_tcgplayer": 0,
            "cards_with_cardmarket": 0,
            "cards_with_both_sources": 0,
            "cards_with_tcgplayer_only": 0,
            "cards_with_cardmarket_only": 0,
            "cards_without_prices": 0,
            "cards_primary_tcgplayer": 0,
            "cards_primary_cardmarket": 0,
            "tcgplayer_rows": 0,
            "cardmarket_rows": 0,
        },
    )
    locale_counts["cards_total"] += 1
    has_tcgplayer = any(row[3] == "tcgplayer" for row in extracted_rows)
    has_cardmarket = any(row[3] == "cardmarket" for row in extracted_rows)
    if has_tcgplayer:
        locale_counts["cards_with_tcgplayer"] += 1
        locale_counts["tcgplayer_rows"] += 1
    if has_cardmarket:
        locale_counts["cards_with_cardmarket"] += 1
        locale_counts["cardmarket_rows"] += 1
    if has_tcgplayer and has_cardmarket:
        locale_counts["cards_with_both_sources"] += 1
    elif has_tcgplayer:
        locale_counts["cards_with_tcgplayer_only"] += 1
    elif has_cardmarket:
        locale_counts["cards_with_cardmarket_only"] += 1
    else:
        locale_counts["cards_without_prices"] += 1

    for row in extracted_rows:
        if row[-1] != 1:
            continue
        if row[3] == "tcgplayer":
            locale_counts["cards_primary_tcgplayer"] += 1
        elif row[3] == "cardmarket":
            locale_counts["cards_primary_cardmarket"] += 1


def content_hash_for_rows(
    rows: list[tuple[str, str, str, str, float | None, float | None, float | None, str | None, int]]
) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: (item[0], item[3])):
        digest.update(json.dumps(list(row), separators=(",", ":"), allow_nan=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def summarize_prices_db(db_path: Path) -> dict[str, Any]:
    with sqlite3.connect(db_path) as connection:
        rows = [
            (
                str(card_id),
                str(market_code),
                str(currency_code),
                str(source_name),
                float(low_price) if low_price is not None else None,
                float(market_price) if market_price is not None else None,
                float(high_price) if high_price is not None else None,
                str(updated_at) if updated_at is not None else None,
                int(is_primary),
            )
            for card_id, market_code, currency_code, source_name, low_price, market_price, high_price, updated_at, is_primary in connection.execute(
                """
                SELECT
                  card_id,
                  market_code,
                  currency_code,
                  source_name,
                  low_price,
                  market_price,
                  high_price,
                  updated_at,
                  is_primary
                FROM prices
                ORDER BY card_id, source_name;
                """
            )
        ]

    per_locale: dict[str, dict[str, int]] = {}
    current_card_id = None
    current_locale = None
    current_rows: list[tuple[str, str, str, str, float | None, float | None, float | None, str | None, int]] = []
    for row in rows:
        card_id = row[0]
        parts = card_id.split(":")
        locale = parts[1] if len(parts) > 2 else "unknown"
        if current_card_id is None:
            current_card_id = card_id
            current_locale = locale
        if card_id != current_card_id:
            assert current_locale is not None
            update_locale_coverage_audit(per_locale, locale=current_locale, extracted_rows=current_rows)
            current_card_id = card_id
            current_locale = locale
            current_rows = []
        current_rows.append(row)
    if current_card_id is not None and current_locale is not None:
        update_locale_coverage_audit(per_locale, locale=current_locale, extracted_rows=current_rows)

    return {
        "row_count": len(rows),
        "prices_content_sha256": content_hash_for_rows(rows),
        "per_locale_coverage": {locale: per_locale[locale] for locale in sorted(per_locale)},
    }


def write_summary_json(summary_json: Path, payload: dict[str, Any]) -> None:
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def extract_price_rows(card: dict[str, Any]) -> list[tuple[str, str, str, str, float | None, float | None, float | None, str | None, int]]:
    pricing = card.get("pricing") or {}
    rows: list[tuple[str, str, str, str, float | None, float | None, float | None, str | None, int]] = []

    tcgplayer = pricing.get("tcgplayer")
    if isinstance(tcgplayer, dict):
        _, variant = best_tcgplayer_variant(tcgplayer)
        if variant is not None:
            rows.append(
                (
                    card["id"],
                    "US",
                    str(tcgplayer.get("unit") or "USD"),
                    "tcgplayer",
                    first_number(variant, ["lowPrice", "low"]),
                    first_number(variant, ["marketPrice", "midPrice", "directLowPrice", "market", "mid", "averageSellPrice"]),
                    first_number(variant, ["highPrice", "high"]),
                    str(tcgplayer.get("updated") or variant.get("updated") or "").strip() or None,
                    1,
                )
            )

    cardmarket = pricing.get("cardmarket")
    if isinstance(cardmarket, dict) and any(
        first_number(cardmarket, [key]) is not None
        for key in ("low", "avg", "trend", "avg1", "avg7", "avg30")
    ):
        rows.append(
            (
                card["id"],
                "EU",
                str(cardmarket.get("unit") or "EUR"),
                "cardmarket",
                first_number(cardmarket, ["low"]),
                first_number(cardmarket, ["avg", "trend"]),
                first_number(cardmarket, ["trend", "avg1", "avg7", "avg30"]),
                str(cardmarket.get("updated") or "").strip() or None,
                0,
            )
        )

    if rows and not any(row[-1] == 1 for row in rows):
        first = rows[0]
        rows[0] = (*first[:-1], 1)
    return rows


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

        bad_primary = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM (
                  SELECT card_id, SUM(is_primary) AS primary_rows
                  FROM prices
                  GROUP BY card_id
                )
                WHERE primary_rows != 1;
                """
            ).fetchone()[0]
        )
        if bad_primary > 0:
            raise RuntimeError(f"Found {bad_primary} card_id groups without exactly one primary price row")

        duplicate_sources = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM (
                  SELECT card_id, source_name, COUNT(*) AS rows_per_source
                  FROM prices
                  GROUP BY card_id, source_name
                )
                WHERE rows_per_source != 1;
                """
            ).fetchone()[0]
        )
        if duplicate_sources > 0:
            raise RuntimeError(f"Found {duplicate_sources} duplicate card_id/source_name groups")
        return row_count


def build_prices_db(
    output_path: Path,
    *,
    locales: list[str],
    limit: int | None,
    min_row_count: int,
    summary_json: Path | None = None,
) -> dict[str, Any]:
    cards, _ = fetch_all_card_records(locales, limit=limit)

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

        rows: list[tuple[str, str, str, str, float | None, float | None, float | None, str | None, int]] = []
        priced_rows = 0
        cards_without_prices = 0
        cards_with_tcgplayer = 0
        cards_with_cardmarket = 0
        locale_coverage = create_locale_coverage_audit(locales)
        for card in cards:
            extracted = extract_price_rows(card)
            update_locale_coverage_audit(locale_coverage, locale=str(card["locale"]), extracted_rows=extracted)
            if not extracted:
                cards_without_prices += 1
            rows.extend(extracted)
            cards_with_tcgplayer += sum(1 for row in extracted if row[3] == "tcgplayer")
            cards_with_cardmarket += sum(1 for row in extracted if row[3] == "cardmarket")
            priced_rows += sum(1 for row in extracted if any(value is not None for value in row[4:7]))

        connection.executemany(
            """
            INSERT INTO prices (
              card_id,
              market_code,
              currency_code,
              source_name,
              low_price,
              market_price,
              high_price,
              updated_at,
              is_primary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            rows,
        )
        connection.commit()

    row_count = validate_prices_db(temp_path, min_row_count=min_row_count)
    summary = {
        "locales": locales,
        "row_count": row_count,
        "priced_rows": priced_rows,
        "cards_total": len(cards),
        "cards_without_prices": cards_without_prices,
        "tcgplayer_rows": cards_with_tcgplayer,
        "cardmarket_rows": cards_with_cardmarket,
        "per_locale_coverage": {locale: locale_coverage[locale] for locale in locales},
        "prices_content_sha256": content_hash_for_rows(rows),
    }
    if summary_json is not None:
        write_summary_json(summary_json, summary)
    temp_path.replace(output_path)
    print(
        f"wrote {output_path} rows={row_count} priced_rows={priced_rows} "
        f"cards_without_prices={cards_without_prices} tcgplayer_rows={cards_with_tcgplayer} "
        f"cardmarket_rows={cards_with_cardmarket}"
    )
    for locale in locales:
        coverage = locale_coverage[locale]
        print(
            "coverage "
            f"locale={locale} cards_total={coverage['cards_total']} "
            f"tcgplayer={coverage['cards_with_tcgplayer']} "
            f"cardmarket={coverage['cards_with_cardmarket']} "
            f"both={coverage['cards_with_both_sources']} "
            f"tcgplayer_only={coverage['cards_with_tcgplayer_only']} "
            f"cardmarket_only={coverage['cards_with_cardmarket_only']} "
            f"without_prices={coverage['cards_without_prices']} "
            f"primary_tcgplayer={coverage['cards_primary_tcgplayer']} "
            f"primary_cardmarket={coverage['cards_primary_cardmarket']}"
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a SQLite price database from TCGdex.")
    parser.add_argument("--output", default="prices.db", help="Output SQLite database path")
    parser.add_argument("--inspect-db", help="Inspect an existing SQLite prices DB instead of rebuilding it")
    parser.add_argument("--limit", type=int, help="Optional card limit for local verification")
    parser.add_argument("--locales", default="en,ja,fr,de,it,es", help="Comma-separated TCGdex locales")
    parser.add_argument("--min-row-count", type=int, default=1000)
    parser.add_argument("--summary-json", help="Optional path for a build or inspection summary JSON file")
    args = parser.parse_args()

    summary_json = Path(args.summary_json).resolve() if args.summary_json else None
    if args.inspect_db:
        db_path = Path(args.inspect_db).resolve()
        validate_prices_db(db_path, min_row_count=args.min_row_count)
        summary = summarize_prices_db(db_path)
        if summary_json is not None:
            write_summary_json(summary_json, summary)
        print(
            f"inspected {db_path} rows={summary['row_count']} "
            f"prices_content_sha256={summary['prices_content_sha256']}"
        )
        return 0

    build_prices_db(
        Path(args.output).resolve(),
        locales=parse_locales(args.locales),
        limit=args.limit,
        min_row_count=args.min_row_count,
        summary_json=summary_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
