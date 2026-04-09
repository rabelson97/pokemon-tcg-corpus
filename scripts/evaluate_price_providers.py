#!/usr/bin/env python3
"""Evaluate third-party price providers against the English price gap sample.

Providers are auto-detected via environment variables:
  POKETRACE_API_KEY          -> PokeTrace
  PPT_API_KEY                -> PokemonPriceTracker
  SCRYDEX_API_KEY + SCRYDEX_TEAM_ID -> Scrydex

Usage:
  python scripts/evaluate_price_providers.py \
    --sample-csv docs/english_price_gap_sample.csv \
    --output-csv docs/provider_evaluation_results.csv \
    --output-summary-json docs/provider_evaluation_summary.json \
    --max-cards 50
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


USER_AGENT = "pokemon-tcg-corpus-provider-eval/1.0"
SCREENSHOT_CARD_IDS = (
    "pokemon:en:sv06:173",
    "pokemon:en:sv10.5b:170",
    "pokemon:en:svp:047",
)
CARD_NUMBER_SUFFIX_PATTERN = re.compile(r"\b([A-Z0-9]+(?:/[A-Z0-9]+)?)\s*$")
SET_NAME_PREFIX_PATTERN = re.compile(r"^[A-Z0-9.]+:\s*")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SampleCard:
    card_id: str
    name: str
    set_id: str
    set_name: str
    card_number: str
    gap_group: str
    stratum: str


@dataclass
class LookupResult:
    card_id: str
    provider: str
    found: bool
    market_price: float | None = None
    low_price: float | None = None
    high_price: float | None = None
    provider_card_name: str = ""
    match_confidence: str = ""
    response_time_ms: int = 0
    notes: str = ""


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def http_get_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
    retries: int = 3,
) -> Any:
    all_headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    if headers:
        all_headers.update(headers)

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(url, headers=all_headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code == 404:
                return None
            if error.code == 401:
                raise
            if error.code not in {429, 500, 502, 503, 504} or attempt == retries:
                raise
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as error:
            last_error = error
            if attempt == retries:
                raise
        time.sleep(min(15, 2 ** (attempt - 1)))

    assert last_error is not None
    raise RuntimeError(f"Request failed after {retries} attempts: {last_error}")


# ---------------------------------------------------------------------------
# Sample loading and selection
# ---------------------------------------------------------------------------

def load_sample_csv(path: Path) -> list[SampleCard]:
    cards: list[SampleCard] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            cards.append(SampleCard(
                card_id=row["card_id"],
                name=row["name"],
                set_id=row["set_id"],
                set_name=row["set_name"],
                card_number=row["card_number"],
                gap_group=row["gap_group"],
                stratum=row["stratum"],
            ))
    return cards


def select_evaluation_cards(cards: list[SampleCard], max_cards: int) -> list[SampleCard]:
    screenshot_ids = set(SCREENSHOT_CARD_IDS)
    screenshots = [c for c in cards if c.card_id in screenshot_ids]
    others = [c for c in cards if c.card_id not in screenshot_ids]

    selected = list(screenshots)
    strata: dict[str, list[SampleCard]] = {}
    for card in others:
        strata.setdefault(card.stratum, []).append(card)

    stratum_keys = sorted(strata.keys())
    while len(selected) < max_cards:
        made_progress = False
        for key in stratum_keys:
            pool = strata.get(key, [])
            if not pool:
                continue
            selected.append(pool.pop(0))
            made_progress = True
            if len(selected) >= max_cards:
                break
        if not made_progress:
            break
    return selected


# ---------------------------------------------------------------------------
# Provider: PokeTrace
# ---------------------------------------------------------------------------

def poketrace_auth_headers(api_key: str) -> dict[str, str]:
    return {"X-API-Key": api_key}


def poketrace_fetch_sets(api_key: str) -> list[dict[str, Any]]:
    sets: list[dict[str, Any]] = []
    cursor: str | None = None
    headers = poketrace_auth_headers(api_key)
    while True:
        params: dict[str, str] = {"game": "pokemon", "limit": "20"}
        if cursor:
            params["cursor"] = cursor
        url = f"https://api.poketrace.com/v1/sets?{urllib.parse.urlencode(params)}"
        payload = http_get_json(url, headers=headers)
        if not isinstance(payload, dict):
            break
        batch = payload.get("data")
        if not isinstance(batch, list) or not batch:
            break
        sets.extend(batch)
        pagination = payload.get("pagination") or {}
        if not pagination.get("hasMore"):
            break
        cursor = pagination.get("nextCursor")
        if not cursor:
            break
    return sets


def build_poketrace_set_mapping(
    provider_sets: list[dict[str, Any]],
    target_set_ids: set[str],
) -> dict[str, str]:
    """Map our TCGdex set_ids to PokeTrace slugs by matching set names."""
    name_to_slug: dict[str, str] = {}
    for s in provider_sets:
        slug = str(s.get("slug") or "").strip()
        name = str(s.get("name") or "").strip().lower()
        if slug and name:
            name_to_slug[name] = slug

    mapping: dict[str, str] = {}
    for target_id in target_set_ids:
        if target_id in mapping:
            continue
        # We don't know the name for our set_ids here; the caller passes it.
    return mapping


def build_poketrace_set_mapping_from_cards(
    provider_sets: list[dict[str, Any]],
    cards: list[SampleCard],
) -> dict[str, str]:
    """Map TCGdex set_ids to PokeTrace slugs using set names from the sample cards."""
    name_to_slug: dict[str, str] = {}
    for s in provider_sets:
        slug = str(s.get("slug") or "").strip()
        name = str(s.get("name") or "").strip().lower()
        if slug and name:
            name_to_slug[name] = slug

    our_id_to_name: dict[str, str] = {}
    for card in cards:
        if card.set_id not in our_id_to_name:
            our_id_to_name[card.set_id] = card.set_name

    mapping: dict[str, str] = {}
    for set_id, set_name in our_id_to_name.items():
        lowered = set_name.lower()
        slug = name_to_slug.get(lowered)
        if slug:
            mapping[set_id] = slug
    return mapping


def poketrace_lookup_card(
    api_key: str,
    set_slug: str,
    card_number: str,
) -> dict[str, Any] | None:
    params = urllib.parse.urlencode({
        "set": set_slug,
        "card_number": card_number,
        "market": "US",
        "limit": "1",
    })
    url = f"https://api.poketrace.com/v1/cards?{params}"
    payload = http_get_json(url, headers=poketrace_auth_headers(api_key))
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return None
    return data[0]


def parse_poketrace_prices(card_data: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    prices = card_data.get("prices") or {}
    tcgplayer = prices.get("tcgplayer") or {}
    near_mint = tcgplayer.get("NEAR_MINT") or tcgplayer.get("near_mint") or {}
    if not near_mint:
        for condition_data in tcgplayer.values():
            if isinstance(condition_data, dict):
                near_mint = condition_data
                break
    if not near_mint:
        ebay = prices.get("ebay") or {}
        near_mint = ebay.get("NEAR_MINT") or ebay.get("near_mint") or {}
        if not near_mint:
            for condition_data in ebay.values():
                if isinstance(condition_data, dict):
                    near_mint = condition_data
                    break
    low = near_mint.get("low")
    avg = near_mint.get("avg") or near_mint.get("market")
    high = near_mint.get("high")
    return (
        float(low) if isinstance(low, (int, float)) else None,
        float(avg) if isinstance(avg, (int, float)) else None,
        float(high) if isinstance(high, (int, float)) else None,
    )


def normalize_card_number(value: Any) -> str:
    clean_value = str(value or "").strip().upper()
    if not clean_value:
        return ""
    base_value = clean_value.split("/", 1)[0]
    match = re.fullmatch(r"0*(\d+)([A-Z]*)", base_value)
    if match:
        digits = str(int(match.group(1))) if match.group(1) else ""
        return f"{digits}{match.group(2)}"
    return base_value


def extract_ppt_card_number(card_data: dict[str, Any]) -> str:
    for key in ("cardNumber", "number", "collectorNumber"):
        number = normalize_card_number(card_data.get(key))
        if number:
            return number

    name = str(card_data.get("name") or "").strip()
    if not name:
        return ""
    suffix = name.rsplit("-", 1)[-1].strip()
    match = CARD_NUMBER_SUFFIX_PATTERN.search(suffix)
    if not match:
        return ""
    return normalize_card_number(match.group(1))


def normalize_set_name(value: Any) -> str:
    clean_value = str(value or "").strip().lower()
    if not clean_value:
        return ""
    return SET_NAME_PREFIX_PATTERN.sub("", clean_value.upper()).lower()


def ppt_card_matches(card_data: dict[str, Any], *, set_name: str, card_number: str) -> bool:
    provider_set_name = str(
        card_data.get("set")
        or card_data.get("setName")
        or card_data.get("set_name")
        or "",
    ).strip()
    if provider_set_name and normalize_set_name(provider_set_name) != normalize_set_name(set_name):
        return False
    provider_card_number = extract_ppt_card_number(card_data)
    if provider_card_number and provider_card_number != normalize_card_number(card_number):
        return False
    return bool(provider_card_number)


def evaluate_poketrace(
    api_key: str,
    cards: list[SampleCard],
    set_mapping: dict[str, str],
) -> list[LookupResult]:
    results: list[LookupResult] = []
    for card in cards:
        slug = set_mapping.get(card.set_id)
        if slug is None:
            results.append(LookupResult(
                card_id=card.card_id,
                provider="poketrace",
                found=False,
                notes=f"no set mapping for {card.set_id}",
            ))
            continue
        start = time.monotonic_ns()
        try:
            data = poketrace_lookup_card(api_key, slug, card.card_number)
        except Exception as exc:
            results.append(LookupResult(
                card_id=card.card_id,
                provider="poketrace",
                found=False,
                notes=f"error: {exc}",
            ))
            continue
        elapsed_ms = int((time.monotonic_ns() - start) / 1_000_000)

        if data is None:
            results.append(LookupResult(
                card_id=card.card_id,
                provider="poketrace",
                found=False,
                response_time_ms=elapsed_ms,
            ))
            continue

        low, market, high = parse_poketrace_prices(data)
        results.append(LookupResult(
            card_id=card.card_id,
            provider="poketrace",
            found=True,
            market_price=market,
            low_price=low,
            high_price=high,
            provider_card_name=str(data.get("name") or ""),
            match_confidence="exact" if market is not None else "found_no_price",
            response_time_ms=elapsed_ms,
        ))
        time.sleep(2.1)  # free tier: 1 req per 2 seconds
    return results


# ---------------------------------------------------------------------------
# Provider: PokemonPriceTracker
# ---------------------------------------------------------------------------

def ppt_auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def ppt_lookup_card(
    api_key: str,
    tcgplayer_id: str,
) -> dict[str, Any] | None:
    params = urllib.parse.urlencode({"tcgPlayerId": tcgplayer_id})
    url = f"https://www.pokemonpricetracker.com/api/v2/cards?{params}"
    payload = http_get_json(url, headers=ppt_auth_headers(api_key))
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return None
    return data[0]


def ppt_search_card(
    api_key: str,
    name: str,
    set_name: str,
    card_number: str,
) -> dict[str, Any] | None:
    params = urllib.parse.urlencode({"search": name, "set": set_name, "limit": "10"})
    url = f"https://www.pokemonpricetracker.com/api/v2/cards?{params}"
    payload = http_get_json(url, headers=ppt_auth_headers(api_key))
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return None
    for candidate in data:
        if isinstance(candidate, dict) and ppt_card_matches(candidate, set_name=set_name, card_number=card_number):
            return candidate
    return data[0]


def parse_ppt_prices(card_data: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    prices = card_data.get("prices") or {}
    low = prices.get("low")
    market = prices.get("market") or prices.get("mid")
    high = prices.get("high")
    return (
        float(low) if isinstance(low, (int, float)) else None,
        float(market) if isinstance(market, (int, float)) else None,
        float(high) if isinstance(high, (int, float)) else None,
    )


def evaluate_ppt(
    api_key: str,
    cards: list[SampleCard],
) -> list[LookupResult]:
    results: list[LookupResult] = []
    for card in cards:
        start = time.monotonic_ns()
        try:
            data = ppt_search_card(api_key, card.name, card.set_name, card.card_number)
        except Exception as exc:
            results.append(LookupResult(
                card_id=card.card_id,
                provider="pokemonpricetracker",
                found=False,
                notes=f"error: {exc}",
            ))
            continue
        elapsed_ms = int((time.monotonic_ns() - start) / 1_000_000)

        if data is None:
            results.append(LookupResult(
                card_id=card.card_id,
                provider="pokemonpricetracker",
                found=False,
                response_time_ms=elapsed_ms,
            ))
            continue

        low, market, high = parse_ppt_prices(data)
        results.append(LookupResult(
            card_id=card.card_id,
            provider="pokemonpricetracker",
            found=True,
            market_price=market,
            low_price=low,
            high_price=high,
            provider_card_name=str(data.get("name") or ""),
            match_confidence="exact" if market is not None else "found_no_price",
            response_time_ms=elapsed_ms,
        ))
        time.sleep(1.1)  # be polite with rate limits
    return results


# ---------------------------------------------------------------------------
# Provider: Scrydex
# ---------------------------------------------------------------------------

def scrydex_auth_headers(api_key: str, team_id: str) -> dict[str, str]:
    return {"X-Api-Key": api_key, "X-Team-ID": team_id}


def scrydex_lookup_card(
    api_key: str,
    team_id: str,
    set_id: str,
    card_number: str,
) -> dict[str, Any] | None:
    card_ref = f"{set_id}-{card_number}"
    url = f"https://api.scrydex.com/pokemon/v1/cards/{urllib.parse.quote(card_ref, safe='')}?include=prices"
    payload = http_get_json(url, headers=scrydex_auth_headers(api_key, team_id))
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if isinstance(data, dict):
        return data
    if isinstance(data, list) and data:
        return data[0]
    return payload if "id" in payload else None


def parse_scrydex_prices(card_data: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    low: float | None = None
    market: float | None = None
    high: float | None = None

    variants = card_data.get("variants") or []
    for variant in variants:
        prices_list = variant.get("prices") or []
        for price_entry in prices_list:
            if not isinstance(price_entry, dict):
                continue
            currency = str(price_entry.get("currency") or "").upper()
            if currency != "USD":
                continue
            p_low = price_entry.get("low")
            p_market = price_entry.get("market")
            p_high = price_entry.get("high")
            if isinstance(p_market, (int, float)) and (market is None or p_market > market):
                market = float(p_market)
            if isinstance(p_low, (int, float)) and (low is None or p_low < low):
                low = float(p_low)
            if isinstance(p_high, (int, float)) and (high is None or p_high > high):
                high = float(p_high)

    if market is None:
        top_level_market = card_data.get("market")
        if isinstance(top_level_market, (int, float)):
            market = float(top_level_market)
        top_level_low = card_data.get("low")
        if isinstance(top_level_low, (int, float)):
            low = float(top_level_low)

    return low, market, high


def evaluate_scrydex(
    api_key: str,
    team_id: str,
    cards: list[SampleCard],
) -> list[LookupResult]:
    results: list[LookupResult] = []
    for card in cards:
        start = time.monotonic_ns()
        try:
            data = scrydex_lookup_card(api_key, team_id, card.set_id, card.card_number)
        except Exception as exc:
            results.append(LookupResult(
                card_id=card.card_id,
                provider="scrydex",
                found=False,
                notes=f"error: {exc}",
            ))
            continue
        elapsed_ms = int((time.monotonic_ns() - start) / 1_000_000)

        if data is None:
            results.append(LookupResult(
                card_id=card.card_id,
                provider="scrydex",
                found=False,
                response_time_ms=elapsed_ms,
            ))
            continue

        low, market, high = parse_scrydex_prices(data)
        results.append(LookupResult(
            card_id=card.card_id,
            provider="scrydex",
            found=True,
            market_price=market,
            low_price=low,
            high_price=high,
            provider_card_name=str(data.get("name") or ""),
            match_confidence="exact" if market is not None else "found_no_price",
            response_time_ms=elapsed_ms,
        ))
        time.sleep(0.5)
    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

RESULT_FIELDS = [
    "card_id", "provider", "found", "market_price", "low_price", "high_price",
    "provider_card_name", "match_confidence", "response_time_ms", "notes",
]


def write_results_csv(path: Path, results: list[LookupResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def build_provider_summary(results: list[LookupResult]) -> dict[str, Any]:
    providers: dict[str, dict[str, Any]] = {}
    for result in results:
        p = providers.setdefault(result.provider, {
            "cards_tested": 0,
            "cards_found": 0,
            "cards_with_price": 0,
            "coverage_rate": 0.0,
            "price_rate": 0.0,
            "set_mapping_failures": 0,
            "errors": 0,
            "screenshot_examples": {},
        })
        p["cards_tested"] += 1
        if result.found:
            p["cards_found"] += 1
        if result.market_price is not None:
            p["cards_with_price"] += 1
        if "no set mapping" in result.notes:
            p["set_mapping_failures"] += 1
        if "error:" in result.notes:
            p["errors"] += 1
        if result.card_id in SCREENSHOT_CARD_IDS:
            p["screenshot_examples"][result.card_id] = {
                "found": result.found,
                "market_price": result.market_price,
                "low_price": result.low_price,
                "high_price": result.high_price,
                "provider_card_name": result.provider_card_name,
                "notes": result.notes,
            }

    for p in providers.values():
        tested = p["cards_tested"]
        if tested > 0:
            p["coverage_rate"] = round(p["cards_found"] / tested * 100, 1)
            p["price_rate"] = round(p["cards_with_price"] / tested * 100, 1)

    return providers


def build_summary(
    results: list[LookupResult],
    cards_evaluated: int,
    set_mapping: dict[str, str] | None,
) -> dict[str, Any]:
    return {
        "cards_in_sample": cards_evaluated,
        "providers": build_provider_summary(results),
        "poketrace_set_mapping": set_mapping or {},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate price providers against the English gap sample.")
    parser.add_argument("--sample-csv", required=True, help="Path to english_price_gap_sample.csv")
    parser.add_argument("--output-csv", required=True, help="Output results CSV path")
    parser.add_argument("--output-summary-json", required=True, help="Output summary JSON path")
    parser.add_argument("--max-cards", type=int, default=50, help="Max cards to evaluate per provider")
    parser.add_argument("--set-mapping-cache", default=None, help="Path to cache PokeTrace set mapping JSON")
    args = parser.parse_args()

    sample_csv = Path(args.sample_csv).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_summary_json = Path(args.output_summary_json).resolve()

    all_cards = load_sample_csv(sample_csv)
    cards = select_evaluation_cards(all_cards, args.max_cards)
    print(f"selected {len(cards)} cards for evaluation (from {len(all_cards)} in sample)")

    poketrace_key = os.environ.get("POKETRACE_API_KEY", "").strip()
    ppt_key = os.environ.get("PPT_API_KEY", "").strip()
    scrydex_key = os.environ.get("SCRYDEX_API_KEY", "").strip()
    scrydex_team = os.environ.get("SCRYDEX_TEAM_ID", "").strip()

    configured = []
    if poketrace_key:
        configured.append("poketrace")
    if ppt_key:
        configured.append("pokemonpricetracker")
    if scrydex_key and scrydex_team:
        configured.append("scrydex")

    if not configured:
        print("no provider API keys configured. Set POKETRACE_API_KEY, PPT_API_KEY, or SCRYDEX_API_KEY+SCRYDEX_TEAM_ID")
        return 1

    print(f"configured providers: {', '.join(configured)}")

    all_results: list[LookupResult] = []
    poketrace_set_mapping: dict[str, str] | None = None

    if "poketrace" in configured:
        print("--- PokeTrace ---")
        cache_path = Path(args.set_mapping_cache).resolve() if args.set_mapping_cache else None
        if cache_path and cache_path.exists():
            poketrace_set_mapping = json.loads(cache_path.read_text(encoding="utf-8"))
            print(f"loaded cached set mapping ({len(poketrace_set_mapping)} sets)")
        else:
            print("fetching PokeTrace sets for slug mapping...")
            provider_sets = poketrace_fetch_sets(poketrace_key)
            print(f"fetched {len(provider_sets)} sets from PokeTrace")
            poketrace_set_mapping = build_poketrace_set_mapping_from_cards(provider_sets, cards)
            print(f"mapped {len(poketrace_set_mapping)} of {len({c.set_id for c in cards})} set IDs")
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(poketrace_set_mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                print(f"cached set mapping to {cache_path}")

        results = evaluate_poketrace(poketrace_key, cards, poketrace_set_mapping)
        all_results.extend(results)
        found = sum(1 for r in results if r.found)
        priced = sum(1 for r in results if r.market_price is not None)
        print(f"PokeTrace: {found}/{len(results)} found, {priced}/{len(results)} with price")

    if "pokemonpricetracker" in configured:
        print("--- PokemonPriceTracker ---")
        results = evaluate_ppt(ppt_key, cards)
        all_results.extend(results)
        found = sum(1 for r in results if r.found)
        priced = sum(1 for r in results if r.market_price is not None)
        print(f"PokemonPriceTracker: {found}/{len(results)} found, {priced}/{len(results)} with price")

    if "scrydex" in configured:
        print("--- Scrydex ---")
        results = evaluate_scrydex(scrydex_key, scrydex_team, cards)
        all_results.extend(results)
        found = sum(1 for r in results if r.found)
        priced = sum(1 for r in results if r.market_price is not None)
        print(f"Scrydex: {found}/{len(results)} found, {priced}/{len(results)} with price")

    write_results_csv(output_csv, all_results)
    summary = build_summary(all_results, len(cards), poketrace_set_mapping)
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"\nwrote results csv={output_csv} rows={len(all_results)}")
    print(f"wrote summary json={output_summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
