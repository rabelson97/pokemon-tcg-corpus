#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from pokemontcgio_api import API_KEY_ENV_VARS, fetch_card_by_id, fetch_english_cards, search_card_by_set_and_number
from poketrace_api import POKETRACE_API_KEY_ENV_VAR
from ppt_api import PPT_API_KEY_ENV_VAR
from tcgdex_api import fetch_all_card_records, parse_locales


DB_USER_VERSION = 2
DEFAULT_MAX_POKEMONTCG_IO_AGE_DAYS = 14
MAX_IDENTITY_AUDIT_SAMPLE_GAPS = 10
TCGPLAYER_NUMERIC_KEYS = [
    "low",
    "mid",
    "high",
    "market",
    "directLow",
    "lowPrice",
    "midPrice",
    "highPrice",
    "marketPrice",
    "directLowPrice",
    # Backward-compatible fallbacks for older payloads.
    "averageSellPrice",
]
TCGPLAYER_VARIANT_PREFERENCE = [
    "normal",
    "reverse",
    "holo",
    "holofoil",
    "reverseHolofoil",
    # Backward-compatible fallbacks for older payloads / historical docs.
    "1stEditionHolofoil",
    "1stEditionNormal",
    "unlimitedHolofoil",
    "unlimitedNormal",
]
CARDMARKET_NUMERIC_KEYS = [
    "low",
    "avg",
    "trend",
    "avg1",
    "avg7",
    "avg30",
    "lowPrice",
    "averageSellPrice",
    "trendPrice",
]

POKEMONTCGIO_SET_ID_ALIASES: dict[str, list[str]] = {
    # PokemonTCG.io uses a compact set id for Champion's Path instead of the
    # TCGdex-style half-set id.
    "swsh3.5": ["swsh35"],
    # Black Bolt and White Flare are split English SV10.5 sets in TCGdex /
    # CardHawk, but PokemonTCG.io publishes them as separate provider sets.
    "sv10.5b": ["zsv10pt5"],
    "sv10.5w": ["rsv10pt5"],
}


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


def parse_price_updated_at(value: str | None) -> dt.datetime | None:
    clean_value = (value or "").strip()
    if not clean_value:
        return None
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y/%m/%d",
        "%Y/%m/%d %H:%M:%S",
    ):
        try:
            return dt.datetime.strptime(clean_value, fmt).replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Unsupported updated_at format: {clean_value}")


def is_price_payload_fresh(value: str | None, *, max_age_days: int, now: dt.datetime) -> bool:
    updated_at = parse_price_updated_at(value)
    if updated_at is None:
        return False
    return now - updated_at <= dt.timedelta(days=max_age_days)


def normalize_tcgplayer_payload(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    prices_payload = payload.get("prices")
    variant_source = prices_payload if isinstance(prices_payload, dict) else payload
    _, variant = best_tcgplayer_variant(variant_source)
    if variant is None:
        return None, None
    updated_at = str(payload.get("updated") or payload.get("updatedAt") or variant.get("updated") or variant.get("updatedAt") or "").strip() or None
    currency = str(payload.get("unit") or "USD")
    return (
        {
            "unit": currency,
            "updated": updated_at,
            "selected_variant": variant,
        },
        updated_at,
    )


def normalize_cardmarket_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    prices_payload = payload.get("prices")
    cardmarket_payload = prices_payload if isinstance(prices_payload, dict) else payload
    if not any(first_number(cardmarket_payload, [key]) is not None for key in CARDMARKET_NUMERIC_KEYS):
        return None
    updated_at = str(payload.get("updated") or payload.get("updatedAt") or "").strip() or None
    currency = str(payload.get("unit") or "EUR")
    return {
        "unit": currency,
        "updated": updated_at,
        "selected_variant": cardmarket_payload,
    }


def build_pokemontcgio_index(cards: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for card in cards:
        set_info = card.get("set")
        if not isinstance(set_info, dict):
            continue
        set_id = str(set_info.get("id") or "").strip()
        number = str(card.get("number") or "").strip()
        if not set_id or not number:
            continue
        key = (set_id, number)
        existing = index.get(key)
        if existing is None:
            index[key] = card
            continue
        existing_tcgplayer, _ = normalize_tcgplayer_payload(existing.get("tcgplayer") or {})
        current_tcgplayer, _ = normalize_tcgplayer_payload(card.get("tcgplayer") or {})
        if existing_tcgplayer is None and current_tcgplayer is not None:
            index[key] = card
    return index


def slugify_poketrace_set_name(set_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", set_name.strip().lower()).strip("-")
    return re.sub(r"-{2,}", "-", slug)


def load_poketrace_set_mapping_overrides() -> dict[str, str]:
    overrides_path = Path(__file__).resolve().parents[1] / "docs" / "provider_set_mapping.json"
    if not overrides_path.exists():
        return {}
    try:
        payload = json.loads(overrides_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    mapping: dict[str, str] = {}
    for set_id, slug in payload.items():
        clean_set_id = str(set_id).strip()
        clean_slug = str(slug).strip()
        if clean_set_id and clean_slug:
            mapping[clean_set_id] = clean_slug
    return mapping


def dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def compact_numeric_token(value: str) -> str:
    clean_value = value.strip()
    if clean_value.isdigit():
        return str(int(clean_value))
    match = re.fullmatch(r"([A-Za-z]+)0+([1-9][0-9]*)", clean_value)
    if match:
        return f"{match.group(1)}{match.group(2)}"
    return clean_value


def alias_set_ids_for_pokemontcgio(set_id: str, card_number: str) -> list[str]:
    clean_set_id = set_id.strip()
    clean_number = card_number.strip().upper()
    candidates = [clean_set_id]

    candidates.extend(POKEMONTCGIO_SET_ID_ALIASES.get(clean_set_id, []))

    compact_set_id = re.sub(r"^([A-Za-z]+)0+([1-9][0-9]*(?:\.5)?)$", r"\1\2", clean_set_id)
    candidates.append(compact_set_id)

    pt5_candidates = [candidate.replace(".5", "pt5") for candidate in candidates if ".5" in candidate]
    candidates.extend(pt5_candidates)

    gallery_suffix = ""
    if clean_number.startswith("TG"):
        gallery_suffix = "tg"
    elif clean_number.startswith("GG"):
        gallery_suffix = "gg"
    if gallery_suffix:
        candidates.extend(f"{candidate}{gallery_suffix}" for candidate in list(candidates))

    return dedupe_strings(candidates)


def candidate_pokemontcgio_match_keys(set_id: str, card_number: str) -> list[tuple[str, str]]:
    clean_number = card_number.strip()
    number_candidates = dedupe_strings([clean_number, compact_numeric_token(clean_number)])
    return [
        (candidate_set_id, candidate_number)
        for candidate_set_id in alias_set_ids_for_pokemontcgio(set_id, card_number)
        for candidate_number in number_candidates
    ]


def format_pokemontcgio_match_key(match_key: tuple[str, str]) -> str:
    return f"{match_key[0]}-{match_key[1]}"


def find_pokemontcgio_card_match(
    pokemontcgio_index: dict[tuple[str, str], dict[str, Any]],
    *,
    set_id: str,
    card_number: str,
) -> tuple[dict[str, Any] | None, tuple[str, str] | None, list[tuple[str, str]]]:
    match_keys = candidate_pokemontcgio_match_keys(set_id, card_number)
    for match_key in match_keys:
        matched_card = pokemontcgio_index.get(match_key)
        if matched_card is not None:
            return matched_card, match_key, match_keys
    return None, None, match_keys


def match_pokemontcgio_card(
    pokemontcgio_index: dict[tuple[str, str], dict[str, Any]],
    *,
    set_id: str,
    card_number: str,
) -> dict[str, Any] | None:
    matched_card, _, _ = find_pokemontcgio_card_match(
        pokemontcgio_index,
        set_id=set_id,
        card_number=card_number,
    )
    return matched_card


def increment_counter(mapping: dict[str, Any], key: str) -> None:
    mapping[key] = int(mapping.get(key, 0)) + 1


def ensure_identity_audit(summary: dict[str, Any]) -> dict[str, Any]:
    audit = summary.setdefault(
        "identity_audit",
        {
            "english_cards_without_usd": 0,
            "pokemontcgio_unmatched_by_set_number": 0,
            "alias_required_matches": 0,
            "stale_or_missing_tcgplayer": 0,
            "price_rows_without_cardhawk_card_id": 0,
            "sample_gaps": [],
        },
    )
    audit.setdefault("english_cards_without_usd", 0)
    audit.setdefault("pokemontcgio_unmatched_by_set_number", 0)
    audit.setdefault("alias_required_matches", 0)
    audit.setdefault("stale_or_missing_tcgplayer", 0)
    audit.setdefault("price_rows_without_cardhawk_card_id", 0)
    audit.setdefault("sample_gaps", [])
    return audit


def record_identity_resolution(
    summary: dict[str, Any],
    *,
    card: dict[str, Any],
    attempted_keys: list[tuple[str, str]],
    reason: str | None,
) -> None:
    card_id = str(card.get("id") or "")
    if not card_id:
        return
    work = summary.setdefault("_identity_audit_work", {})
    work[card_id] = {
        "attempted_pokemontcgio_match_keys": [format_pokemontcgio_match_key(key) for key in attempted_keys],
        "reason": reason,
    }


def build_identity_gap_sample(
    card: dict[str, Any],
    *,
    attempted_keys: list[str],
    reason: str,
) -> dict[str, Any]:
    return {
        "card_id": str(card.get("id") or ""),
        "card_name": str(card.get("name") or ""),
        "tcgdex_set_id": str(card.get("set_id") or ""),
        "tcgdex_set_name": str(card.get("set_name") or ""),
        "card_number": str(card.get("card_number") or ""),
        "attempted_pokemontcgio_match_keys": attempted_keys,
        "reason": reason,
    }


def record_identity_gap_sample(
    summary: dict[str, Any],
    *,
    card: dict[str, Any],
    reason: str,
) -> None:
    audit = ensure_identity_audit(summary)
    work = summary.get("_identity_audit_work") or {}
    card_work = work.get(str(card.get("id") or ""), {}) if isinstance(work, dict) else {}
    attempted_keys = card_work.get("attempted_pokemontcgio_match_keys")
    if not isinstance(attempted_keys, list):
        attempted_keys = [
            format_pokemontcgio_match_key(key)
            for key in candidate_pokemontcgio_match_keys(
                str(card.get("set_id") or ""),
                str(card.get("card_number") or ""),
            )
        ]
    sample = build_identity_gap_sample(card, attempted_keys=[str(key) for key in attempted_keys], reason=reason)
    audit["sample_gaps"].append(sample)


def fetch_targeted_pokemontcgio_cards(english_cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fetched: list[dict[str, Any]] = []
    seen_upstream_ids: set[str] = set()
    for card in english_cards:
        upstream_id = str(card.get("upstream_id") or "").strip()
        if not upstream_id or upstream_id in seen_upstream_ids:
            continue
        seen_upstream_ids.add(upstream_id)
        matched = fetch_card_by_id(upstream_id)
        if matched is None:
            for candidate_set_id, candidate_number in candidate_pokemontcgio_match_keys(
                str(card.get("set_id") or "").strip(),
                str(card.get("card_number") or "").strip(),
            ):
                matched = fetch_card_by_id(f"{candidate_set_id}-{candidate_number}")
                if matched is None:
                    matched = search_card_by_set_and_number(candidate_set_id, candidate_number)
                if matched is not None:
                    break
        if matched is not None:
            fetched.append(matched)
    return fetched


def select_price_sources(
    card: dict[str, Any],
    *,
    pokemontcgio_index: dict[tuple[str, str], dict[str, Any]],
    max_pokemontcgio_age_days: int,
    now: dt.datetime,
    summary: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    selected_sources: dict[str, dict[str, Any]] = {}
    pricing = card.get("pricing") or {}
    locale = str(card.get("locale") or "")

    if locale != "en":
        cardmarket = pricing.get("cardmarket")
        if isinstance(cardmarket, dict):
            normalized_cardmarket = normalize_cardmarket_payload(cardmarket)
            if normalized_cardmarket is not None:
                selected_sources["cardmarket"] = normalized_cardmarket
                increment_counter(summary["transport_counts"].setdefault("cardmarket", {}), "tcgdex")
        return selected_sources

    summary["pokemontcgio"]["english_cards_considered"] += 1
    identity_audit = ensure_identity_audit(summary)
    matched_card, matched_key, attempted_keys = find_pokemontcgio_card_match(
        pokemontcgio_index,
        set_id=str(card.get("set_id") or "").strip(),
        card_number=str(card.get("card_number") or "").strip(),
    )
    resolution_reason: str | None = None
    tcgplayer_resolved = False
    if matched_card is not None:
        summary["pokemontcgio"]["english_cards_with_match"] += 1
        if matched_key is not None and attempted_keys and matched_key != attempted_keys[0]:
            identity_audit["alias_required_matches"] += 1
        tcgplayer_payload = matched_card.get("tcgplayer")
        if isinstance(tcgplayer_payload, dict):
            normalized_tcgplayer, updated_at = normalize_tcgplayer_payload(tcgplayer_payload)
            if normalized_tcgplayer is not None and updated_at is not None:
                invalid_updated_at = False
                try:
                    is_fresh = is_price_payload_fresh(updated_at, max_age_days=max_pokemontcgio_age_days, now=now)
                except ValueError:
                    is_fresh = False
                    invalid_updated_at = True
                    summary["pokemontcgio"]["stale_tcgplayer_rows"] += 1
                    increment_counter(summary["pokemontcgio"].setdefault("stale_reasons", {}), "invalid_updated_at")
                    identity_audit["stale_or_missing_tcgplayer"] += 1
                    resolution_reason = "pokemontcgio_tcgplayer_invalid_updated_at"
                if is_fresh:
                    selected_sources["tcgplayer"] = normalized_tcgplayer
                    summary["pokemontcgio"]["english_cards_with_tcgplayer"] += 1
                    increment_counter(summary["transport_counts"].setdefault("tcgplayer", {}), "pokemontcgio")
                    tcgplayer_resolved = True
                elif not invalid_updated_at:
                    summary["pokemontcgio"]["stale_tcgplayer_rows"] += 1
                    increment_counter(summary["pokemontcgio"].setdefault("stale_reasons", {}), "older_than_max_age")
                    if resolution_reason is None:
                        identity_audit["stale_or_missing_tcgplayer"] += 1
                        resolution_reason = "pokemontcgio_tcgplayer_older_than_max_age"
            elif normalized_tcgplayer is None:
                summary["pokemontcgio"]["english_cards_without_tcgplayer"] += 1
                identity_audit["stale_or_missing_tcgplayer"] += 1
                resolution_reason = "pokemontcgio_tcgplayer_without_numeric_prices"
            elif updated_at is None:
                summary["pokemontcgio"]["stale_tcgplayer_rows"] += 1
                increment_counter(summary["pokemontcgio"].setdefault("stale_reasons", {}), "missing_updated_at")
                identity_audit["stale_or_missing_tcgplayer"] += 1
                resolution_reason = "pokemontcgio_tcgplayer_missing_updated_at"
        else:
            summary["pokemontcgio"]["english_cards_without_tcgplayer"] += 1
            identity_audit["stale_or_missing_tcgplayer"] += 1
            resolution_reason = "pokemontcgio_missing_tcgplayer_payload"
    else:
        summary["pokemontcgio"]["english_cards_without_match"] += 1
        identity_audit["pokemontcgio_unmatched_by_set_number"] += 1
        resolution_reason = "no_pokemontcgio_match"

    if not tcgplayer_resolved:
        record_identity_resolution(
            summary,
            card=card,
            attempted_keys=attempted_keys,
            reason=resolution_reason,
        )
    return selected_sources


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


def load_existing_prices_rows(
    db_path: Path,
) -> dict[str, list[tuple[str, str, str, str, float | None, float | None, float | None, str | None, int]]]:
    rows_by_card_id: dict[str, list[tuple[str, str, str, str, float | None, float | None, float | None, str | None, int]]] = {}
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
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
        ).fetchall()
    for row in rows:
        normalized_row = (
            str(row[0]),
            str(row[1]),
            str(row[2]),
            str(row[3]),
            float(row[4]) if row[4] is not None else None,
            float(row[5]) if row[5] is not None else None,
            float(row[6]) if row[6] is not None else None,
            str(row[7]) if row[7] is not None else None,
            int(row[8]),
        )
        rows_by_card_id.setdefault(normalized_row[0], []).append(normalized_row)
    return rows_by_card_id


def collect_reusable_existing_tcgplayer_rows(
    rows_by_card_id: dict[str, list[tuple[str, str, str, str, float | None, float | None, float | None, str | None, int]]],
    *,
    updated_date_prefix: str,
) -> dict[str, list[tuple[str, str, str, str, float | None, float | None, float | None, str | None, int]]]:
    reusable: dict[str, list[tuple[str, str, str, str, float | None, float | None, float | None, str | None, int]]] = {}
    for card_id, rows in rows_by_card_id.items():
        reusable_rows = [
            row
            for row in rows
            if row[3] == "tcgplayer" and isinstance(row[7], str) and row[7].startswith(updated_date_prefix)
        ]
        if reusable_rows:
            reusable[card_id] = reusable_rows
    return reusable


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


def extract_price_rows_from_selected_sources(
    card_id: str,
    selected_sources: dict[str, dict[str, Any]],
) -> list[tuple[str, str, str, str, float | None, float | None, float | None, str | None, int]]:
    rows: list[tuple[str, str, str, str, float | None, float | None, float | None, str | None, int]] = []

    tcgplayer = selected_sources.get("tcgplayer")
    if isinstance(tcgplayer, dict):
        variant = tcgplayer["selected_variant"]
        rows.append(
            (
                card_id,
                "US",
                str(tcgplayer.get("unit") or "USD"),
                "tcgplayer",
                first_number(variant, ["lowPrice", "low"]),
                first_number(variant, ["marketPrice", "market", "midPrice", "mid", "directLowPrice", "directLow", "averageSellPrice"]),
                first_number(variant, ["highPrice", "high"]),
                str(tcgplayer.get("updated") or "").strip() or None,
                1,
            )
        )

    cardmarket = selected_sources.get("cardmarket")
    if isinstance(cardmarket, dict):
        payload = cardmarket["selected_variant"]
        rows.append(
            (
                card_id,
                "EU",
                str(cardmarket.get("unit") or "EUR"),
                "cardmarket",
                first_number(payload, ["low", "lowPrice"]),
                first_number(payload, ["avg", "averageSellPrice", "trend", "trendPrice"]),
                first_number(payload, ["trend", "trendPrice", "avg1", "avg7", "avg30"]),
                str(cardmarket.get("updated") or "").strip() or None,
                0,
            )
        )

    if rows and not any(row[-1] == 1 for row in rows):
        first = rows[0]
        rows[0] = (*first[:-1], 1)
    return rows


def extract_price_rows(card: dict[str, Any]) -> list[tuple[str, str, str, str, float | None, float | None, float | None, str | None, int]]:
    pricing = card.get("pricing") or {}
    selected_sources: dict[str, dict[str, Any]] = {}
    tcgplayer = pricing.get("tcgplayer")
    if isinstance(tcgplayer, dict):
        normalized_tcgplayer, _ = normalize_tcgplayer_payload(tcgplayer)
        if normalized_tcgplayer is not None:
            selected_sources["tcgplayer"] = normalized_tcgplayer
    cardmarket = pricing.get("cardmarket")
    if isinstance(cardmarket, dict):
        normalized_cardmarket = normalize_cardmarket_payload(cardmarket)
        if normalized_cardmarket is not None:
            selected_sources["cardmarket"] = normalized_cardmarket
    return extract_price_rows_from_selected_sources(str(card["id"]), selected_sources)


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


def try_fallback_providers(
    card: dict[str, Any],
    *,
    ppt_set_cache: dict[str, dict[str, dict[str, Any]]],
    poketrace_set_slugs: dict[str, str],
    summary: dict[str, Any],
) -> dict[str, Any] | None:
    """Try PPT then PokeTrace for an English card missing tcgplayer USD.

    The PPT cache (populated by ``build_ppt_set_cache``) is checked first for
    a free O(1) lookup before falling back to per-card API calls.

    Returns a normalized tcgplayer-shaped source dict, or None.
    """
    import ppt_api
    import poketrace_api

    set_id = str(card.get("set_id") or "").strip()
    card_number = str(card.get("card_number") or "").strip()
    card_name = str(card.get("name") or "").strip()
    set_name = str(card.get("set_name") or "").strip()
    fallback_summary = summary["fallback_providers"]

    # --- PPT: try pre-fetched bulk cache first (zero API cost) ---
    if ppt_set_cache and card_number:
        normalized_set = ppt_api.normalize_set_name(set_name)
        set_index = ppt_set_cache.get(normalized_set)
        if set_index is not None:
            normalized_number = ppt_api.normalize_card_number(card_number)
            cached_card = set_index.get(normalized_number)
            if cached_card is not None:
                result = ppt_api.extract_usd_price(cached_card)
                if result is not None:
                    increment_counter(summary["transport_counts"].setdefault("tcgplayer", {}), "ppt_bulk")
                    fallback_summary["ppt_bulk_hits"] += 1
                    return result
            fallback_summary["ppt_bulk_misses"] += 1

    # --- PPT: per-card API fallback ---
    ppt_key = ppt_api.resolve_api_key()
    if ppt_key and not bool(fallback_summary.get("ppt_disabled_due_to_errors")):
        try:
            ppt_card = None
            if card_name:
                ppt_card = ppt_api.search_card(
                    card_name,
                    set_name,
                    card_number=card_number,
                    api_key=ppt_key,
                )
            if ppt_card is not None:
                result = ppt_api.extract_usd_price(ppt_card)
                if result is not None:
                    increment_counter(summary["transport_counts"].setdefault("tcgplayer", {}), "ppt")
                    fallback_summary["ppt_hits"] += 1
                    return result
            fallback_summary["ppt_misses"] += 1
        except Exception as error:
            fallback_summary["ppt_errors"] += 1
            if not fallback_summary.get("ppt_first_error"):
                fallback_summary["ppt_first_error"] = f"{type(error).__name__}: {error}"
            threshold = int(fallback_summary.get("ppt_error_disable_threshold") or 5)
            if fallback_summary["ppt_hits"] == 0 and fallback_summary["ppt_errors"] >= threshold:
                fallback_summary["ppt_disabled_due_to_errors"] = True
                print(
                    f"warning: disabling PPT fallback after {fallback_summary['ppt_errors']} errors; "
                    f"first_error={fallback_summary.get('ppt_first_error')}"
                )

    poketrace_key = poketrace_api.resolve_api_key()
    if poketrace_key and not bool(fallback_summary.get("poketrace_disabled_due_to_errors")):
        slug = poketrace_set_slugs.get(set_id)
        if slug:
            try:
                pt_card = poketrace_api.lookup_card(slug, card_number, api_key=poketrace_key)
                if pt_card is not None:
                    result = poketrace_api.extract_usd_price(pt_card)
                    if result is not None:
                        increment_counter(summary["transport_counts"].setdefault("tcgplayer", {}), "poketrace")
                        fallback_summary["poketrace_hits"] += 1
                        return result
                fallback_summary["poketrace_misses"] += 1
            except Exception as error:
                fallback_summary["poketrace_errors"] += 1
                if not fallback_summary.get("poketrace_first_error"):
                    fallback_summary["poketrace_first_error"] = f"{type(error).__name__}: {error}"
                threshold = int(fallback_summary.get("poketrace_error_disable_threshold") or 5)
                if fallback_summary["poketrace_hits"] == 0 and fallback_summary["poketrace_errors"] >= threshold:
                    fallback_summary["poketrace_disabled_due_to_errors"] = True
                    print(
                        "warning: disabling PokeTrace fallback after "
                        f"{fallback_summary['poketrace_errors']} errors; "
                        f"first_error={fallback_summary.get('poketrace_first_error')}"
                    )
        else:
            fallback_summary["poketrace_set_mapping_failures"] += 1

    return None


def build_poketrace_set_slugs(cards: list[dict[str, Any]]) -> dict[str, str]:
    """Build a set_id -> PokeTrace slug mapping for the given cards."""
    import poketrace_api

    api_key = poketrace_api.resolve_api_key()

    our_set_names: dict[str, str] = {}
    for card in cards:
        set_id = str(card.get("set_id") or "").strip()
        set_name = str(card.get("set_name") or "").strip()
        if set_id and set_name and set_id not in our_set_names:
            our_set_names[set_id] = set_name

    if not our_set_names:
        return {}

    mapping = {
        set_id: slug
        for set_id, slug in load_poketrace_set_mapping_overrides().items()
        if set_id in our_set_names
    }
    if mapping:
        print(f"loaded {len(mapping)} cached PokeTrace set slug overrides")

    if api_key:
        print(f"fetching PokeTrace sets for slug mapping ({len(our_set_names)} target sets)...")
        try:
            provider_sets = poketrace_api.fetch_sets(api_key=api_key)
        except Exception as error:
            print(f"warning: failed to fetch PokeTrace sets for slug mapping: {error}")
        else:
            print(f"fetched {len(provider_sets)} sets from PokeTrace")
            mapping.update(poketrace_api.build_set_slug_mapping(provider_sets, our_set_names))

    heuristic_mapped = 0
    for set_id, set_name in our_set_names.items():
        if set_id in mapping:
            continue
        slug = slugify_poketrace_set_name(set_name)
        if not slug:
            continue
        mapping[set_id] = slug
        heuristic_mapped += 1

    if heuristic_mapped:
        print(f"derived {heuristic_mapped} heuristic PokeTrace set slugs from set names")
    print(f"mapped {len(mapping)} of {len(our_set_names)} set IDs to PokeTrace slugs")
    return mapping


def build_ppt_set_cache(
    gap_cards: list[dict[str, Any]],
    *,
    credit_budget: int | None,
    api_key: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Bulk-fetch PPT cards for sets that contain gap cards.

    Uses ``fetchAllInSet=true`` to retrieve entire sets at once.  Sets are
    prioritised by how many gap cards they contain (most gaps first) so the
    limited credit budget is spent where it helps most.

    Returns a nested dict:  ``set_name -> normalized_card_number -> card_data``
    ready for O(1) lookup during the fallback loop.
    """
    import ppt_api

    set_gap_counts: dict[str, int] = defaultdict(int)
    for card in gap_cards:
        set_name = str(card.get("set_name") or "").strip()
        if set_name:
            set_gap_counts[set_name] += 1

    if not set_gap_counts:
        return {}

    ranked_sets = sorted(set_gap_counts.items(), key=lambda item: item[1], reverse=True)
    credits_remaining = credit_budget
    cache: dict[str, dict[str, dict[str, Any]]] = {}
    sets_fetched = 0
    sets_empty = 0
    total_cards_cached = 0

    print(
        f"PPT bulk: {len(ranked_sets)} sets with gap cards, "
        f"credit_budget={credits_remaining if credits_remaining is not None else 'unlimited'}"
    )

    for set_name, gap_count in ranked_sets:
        if credits_remaining is not None and credits_remaining <= 0:
            break
        try:
            cards = ppt_api.fetch_set_cards(set_name, api_key=api_key)
        except Exception as error:
            print(f"PPT bulk: error fetching set '{set_name}': {type(error).__name__}: {error}")
            continue

        if not cards:
            sets_empty += 1
            continue

        index = ppt_api.build_card_number_index(cards)
        cache[ppt_api.normalize_set_name(set_name)] = index
        sets_fetched += 1
        total_cards_cached += len(index)

        if credits_remaining is not None:
            credits_remaining -= len(cards)

        print(
            f"PPT bulk: fetched {len(cards)} cards from '{set_name}' "
            f"(gap={gap_count}, indexed={len(index)}, "
            f"credits_remaining={credits_remaining if credits_remaining is not None else 'unlimited'})"
        )

    print(
        f"PPT bulk: done — {sets_fetched} sets fetched, {sets_empty} empty, "
        f"{total_cards_cached} cards indexed"
    )
    return cache


def build_prices_db(
    output_path: Path,
    *,
    locales: list[str],
    limit: int | None,
    min_row_count: int,
    summary_json: Path | None = None,
    max_pokemontcgio_age_days: int = DEFAULT_MAX_POKEMONTCG_IO_AGE_DAYS,
    max_fallback_cards: int | None = None,
    seed_db_path: Path | None = None,
    reuse_existing_tcgplayer_date: str | None = None,
) -> dict[str, Any]:
    cards, _ = fetch_all_card_records(locales, limit=limit)
    existing_rows_by_card_id: dict[str, list[tuple[str, str, str, str, float | None, float | None, float | None, str | None, int]]] = {}
    reusable_existing_tcgplayer_rows: dict[str, list[tuple[str, str, str, str, float | None, float | None, float | None, str | None, int]]] = {}
    if seed_db_path is not None and seed_db_path.exists():
        existing_rows_by_card_id = load_existing_prices_rows(seed_db_path)
        if reuse_existing_tcgplayer_date:
            reusable_existing_tcgplayer_rows = collect_reusable_existing_tcgplayer_rows(
                existing_rows_by_card_id,
                updated_date_prefix=reuse_existing_tcgplayer_date,
            )
    pokemontcgio_cards: list[dict[str, Any]] = []
    pokemontcgio_index: dict[tuple[str, str], dict[str, Any]] = {}
    poketrace_set_slugs: dict[str, str] = {}
    ppt_set_cache: dict[str, dict[str, dict[str, Any]]] = {}
    allow_fallback_queries = max_fallback_cards is None or max_fallback_cards > 0
    if "en" in locales:
        english_cards = [card for card in cards if str(card.get("locale") or "") == "en"]
        english_cards_needing_primary_refresh = [
            card for card in english_cards if str(card.get("id") or "") not in reusable_existing_tcgplayer_rows
        ]
        if limit is not None and english_cards_needing_primary_refresh:
            pokemontcgio_cards = fetch_targeted_pokemontcgio_cards(english_cards_needing_primary_refresh)
        elif reuse_existing_tcgplayer_date and english_cards_needing_primary_refresh:
            pokemontcgio_cards = fetch_targeted_pokemontcgio_cards(english_cards_needing_primary_refresh)
        elif reuse_existing_tcgplayer_date:
            pokemontcgio_cards = []
        else:
            pokemontcgio_cards = fetch_english_cards()
        pokemontcgio_index = build_pokemontcgio_index(pokemontcgio_cards)
        if allow_fallback_queries:
            poketrace_set_slugs = build_poketrace_set_slugs(english_cards)

            # Pre-identify likely gap cards (English cards not covered by
            # pokemontcg.io and not reusable from the seed DB) so we can
            # bulk-fetch their sets from PPT up front.
            import ppt_api as _ppt_api

            _ppt_key = _ppt_api.resolve_api_key()
            if _ppt_key:
                likely_gap_cards = [
                    card
                    for card in english_cards
                    if (
                        str(card.get("id") or "") not in reusable_existing_tcgplayer_rows
                        and match_pokemontcgio_card(
                            pokemontcgio_index,
                            set_id=str(card.get("set_id") or "").strip(),
                            card_number=str(card.get("card_number") or "").strip(),
                        )
                        is None
                    )
                ]
                if likely_gap_cards:
                    ppt_set_cache = build_ppt_set_cache(
                        likely_gap_cards,
                        credit_budget=max_fallback_cards,
                        api_key=_ppt_key,
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
        now = dt.datetime.now(dt.timezone.utc)
        build_metadata: dict[str, Any] = {
            "transport_counts": defaultdict(lambda: defaultdict(int)),
            "pokemontcgio": {
                "api_key_env_vars": list(API_KEY_ENV_VARS),
                "api_key_configured": any(bool(os.environ.get(env_var, "").strip()) for env_var in API_KEY_ENV_VARS),
                "english_cards_fetched": len(pokemontcgio_cards),
                "english_unique_match_keys": len(pokemontcgio_index),
                "english_cards_considered": 0,
                "english_cards_with_match": 0,
                "english_cards_without_match": 0,
                "english_cards_with_tcgplayer": 0,
                "english_cards_without_tcgplayer": 0,
                "stale_tcgplayer_rows": 0,
                "stale_reasons": defaultdict(int),
                "max_age_days": max_pokemontcgio_age_days,
                "fetched_at": now.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            },
            "fallback_providers": {
                "ppt_configured": bool(os.environ.get(PPT_API_KEY_ENV_VAR, "").strip()),
                "poketrace_configured": bool(os.environ.get(POKETRACE_API_KEY_ENV_VAR, "").strip()),
                "poketrace_set_slugs_mapped": len(poketrace_set_slugs),
                "max_cards": max_fallback_cards,
                "english_cards_tried_fallback": 0,
                "english_cards_skipped_due_to_budget": 0,
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
                "ppt_bulk_sets_cached": len(ppt_set_cache),
                "ppt_bulk_hits": 0,
                "ppt_bulk_misses": 0,
            },
            "seed_reuse": {
                "seed_db_path": str(seed_db_path) if seed_db_path is not None else None,
                "reuse_existing_tcgplayer_date": reuse_existing_tcgplayer_date,
                "cards_with_reused_tcgplayer_rows": 0,
            },
            "identity_audit": {
                "english_cards_without_usd": 0,
                "pokemontcgio_unmatched_by_set_number": 0,
                "alias_required_matches": 0,
                "stale_or_missing_tcgplayer": 0,
                "price_rows_without_cardhawk_card_id": 0,
                "sample_gaps": [],
            },
            "_identity_audit_work": {},
        }
        has_fallback_providers = (
            allow_fallback_queries
            and (
                build_metadata["fallback_providers"]["ppt_configured"]
                or build_metadata["fallback_providers"]["poketrace_configured"]
                or bool(ppt_set_cache)
            )
        )
        fallback_attempts = 0
        for card in cards:
            locale = str(card.get("locale") or "")
            card_id = str(card["id"])
            reusable_rows = reusable_existing_tcgplayer_rows.get(card_id, []) if locale == "en" else []
            if reusable_rows:
                extracted = reusable_rows
                build_metadata["seed_reuse"]["cards_with_reused_tcgplayer_rows"] += 1
            else:
                selected_sources = select_price_sources(
                    card,
                    pokemontcgio_index=pokemontcgio_index,
                    max_pokemontcgio_age_days=max_pokemontcgio_age_days,
                    now=now,
                    summary=build_metadata,
                )
                if locale == "en" and "tcgplayer" not in selected_sources and has_fallback_providers:
                    if max_fallback_cards is not None and fallback_attempts >= max_fallback_cards:
                        build_metadata["fallback_providers"]["english_cards_skipped_due_to_budget"] += 1
                    else:
                        fallback_attempts += 1
                        build_metadata["fallback_providers"]["english_cards_tried_fallback"] += 1
                        if fallback_attempts % 100 == 1:
                            fb = build_metadata["fallback_providers"]
                            print(
                                f"fallback attempt={fallback_attempts} "
                                f"ppt_hits={fb['ppt_hits']} ppt_misses={fb['ppt_misses']} ppt_errors={fb['ppt_errors']} "
                                f"ppt_bulk_hits={fb['ppt_bulk_hits']} ppt_bulk_misses={fb['ppt_bulk_misses']} "
                                f"poketrace_hits={fb['poketrace_hits']} poketrace_misses={fb['poketrace_misses']} "
                                f"card_id={card_id}"
                            )
                        fallback_result = try_fallback_providers(
                            card,
                            ppt_set_cache=ppt_set_cache,
                            poketrace_set_slugs=poketrace_set_slugs,
                            summary=build_metadata,
                        )
                        if fallback_result is not None:
                            selected_sources["tcgplayer"] = fallback_result

                extracted = extract_price_rows_from_selected_sources(card_id, selected_sources)
            if locale == "en" and not any(row[3] == "tcgplayer" and row[2] == "USD" for row in extracted):
                identity_audit = ensure_identity_audit(build_metadata)
                identity_audit["english_cards_without_usd"] += 1
                work = build_metadata.get("_identity_audit_work") or {}
                card_work = work.get(card_id, {}) if isinstance(work, dict) else {}
                reason = str(card_work.get("reason") or "no_usd_provider_after_fallback")
                record_identity_gap_sample(build_metadata, card=card, reason=reason)
            update_locale_coverage_audit(locale_coverage, locale=str(card["locale"]), extracted_rows=extracted)
            if not extracted:
                cards_without_prices += 1
            rows.extend(extracted)
            cards_with_tcgplayer += sum(1 for row in extracted if row[3] == "tcgplayer")
            cards_with_cardmarket += sum(1 for row in extracted if row[3] == "cardmarket")
            priced_rows += sum(1 for row in extracted if any(value is not None for value in row[4:7]))

        fetched_card_ids = {str(card.get("id") or "") for card in cards}
        orphan_price_row_card_ids = sorted({row[0] for row in rows if row[0] not in fetched_card_ids})
        identity_audit = ensure_identity_audit(build_metadata)
        identity_audit["price_rows_without_cardhawk_card_id"] = len(orphan_price_row_card_ids)
        if orphan_price_row_card_ids:
            identity_audit["price_rows_without_cardhawk_card_id_samples"] = orphan_price_row_card_ids[:MAX_IDENTITY_AUDIT_SAMPLE_GAPS]

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
    identity_audit_summary = dict(build_metadata["identity_audit"])
    identity_audit_summary["sample_gaps"] = sorted(
        identity_audit_summary.get("sample_gaps", []),
        key=lambda sample: str(sample.get("card_id") or ""),
    )[:MAX_IDENTITY_AUDIT_SAMPLE_GAPS]

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
        "providers": {
            "tcgplayer": {
                "selected_transport": "pokemontcgio",
                "market_code": "US",
                "currency_code": "USD",
                "max_age_days": max_pokemontcgio_age_days,
            },
            "cardmarket": {
                "selected_transport": "tcgdex",
                "market_code": "EU",
                "currency_code": "EUR",
            },
        },
        "transport_counts": {
            source_name: {transport: count for transport, count in transports.items()}
            for source_name, transports in build_metadata["transport_counts"].items()
        },
        "pokemontcgio": {
            **{key: value for key, value in build_metadata["pokemontcgio"].items() if key != "stale_reasons"},
            "stale_reasons": dict(build_metadata["pokemontcgio"]["stale_reasons"]),
        },
        "fallback_providers": dict(build_metadata["fallback_providers"]),
        "seed_reuse": dict(build_metadata["seed_reuse"]),
        "identity_audit": identity_audit_summary,
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
    audit = summary["identity_audit"]
    if (
        audit["english_cards_without_usd"]
        or audit["pokemontcgio_unmatched_by_set_number"]
        or audit["stale_or_missing_tcgplayer"]
        or audit["price_rows_without_cardhawk_card_id"]
    ):
        print(
            "warning: identity audit "
            f"english_cards_without_usd={audit['english_cards_without_usd']} "
            f"pokemontcgio_unmatched_by_set_number={audit['pokemontcgio_unmatched_by_set_number']} "
            f"alias_required_matches={audit['alias_required_matches']} "
            f"stale_or_missing_tcgplayer={audit['stale_or_missing_tcgplayer']} "
            f"price_rows_without_cardhawk_card_id={audit['price_rows_without_cardhawk_card_id']}"
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a SQLite price database from locale-first pricing sources.")
    parser.add_argument("--output", default="prices.db", help="Output SQLite database path")
    parser.add_argument("--inspect-db", help="Inspect an existing SQLite prices DB instead of rebuilding it")
    parser.add_argument("--limit", type=int, help="Optional card limit for local verification")
    parser.add_argument("--locales", default="en,ja", help="Comma-separated TCGdex locales")
    parser.add_argument("--min-row-count", type=int, default=1000)
    parser.add_argument("--summary-json", help="Optional path for a build or inspection summary JSON file")
    parser.add_argument("--max-pokemontcgio-age-days", type=int, default=DEFAULT_MAX_POKEMONTCG_IO_AGE_DAYS)
    parser.add_argument("--max-fallback-cards", type=int, help="Optional cap on English cards that may query fallback USD providers")
    parser.add_argument("--seed-db", help="Optional existing prices.db to reuse rows from during a rebuild")
    parser.add_argument("--reuse-existing-tcgplayer-date", help="Reuse existing tcgplayer rows whose updated_at starts with this date prefix, e.g. 2026/04/10")
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
        max_pokemontcgio_age_days=args.max_pokemontcgio_age_days,
        max_fallback_cards=args.max_fallback_cards,
        seed_db_path=Path(args.seed_db).resolve() if args.seed_db else None,
        reuse_existing_tcgplayer_date=args.reuse_existing_tcgplayer_date,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
