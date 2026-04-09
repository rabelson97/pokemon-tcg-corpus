"""PokemonPriceTracker API client for USD price lookups.

Env var: PPT_API_KEY
Docs: https://www.pokemonpricetracker.com/api-reference
"""
from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


API_BASE_URL = "https://www.pokemonpricetracker.com/api/v2"
USER_AGENT = "pokemon-tcg-corpus-db-builder/2.0"
PPT_API_KEY_ENV_VAR = "PPT_API_KEY"
CARD_NUMBER_SUFFIX_PATTERN = re.compile(r"\b([A-Z0-9]+(?:/[A-Z0-9]+)?)\s*$")
SET_NAME_PREFIX_PATTERN = re.compile(r"^[A-Z0-9.]+:\s*")


def resolve_api_key() -> str:
    return os.environ.get(PPT_API_KEY_ENV_VAR, "").strip()


def api_get_json(
    path: str,
    *,
    params: dict[str, str] | None = None,
    timeout: int = 30,
    retries: int = 5,
    api_key: str | None = None,
) -> Any:
    query = urllib.parse.urlencode(params or {})
    url = f"{API_BASE_URL}{path}"
    if query:
        url = f"{url}?{query}"
    clean_key = (api_key or resolve_api_key()).strip()
    headers: dict[str, str] = {
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }
    if clean_key:
        headers["Authorization"] = f"Bearer {clean_key}"

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(url, headers=headers)
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
            if error.code == 429:
                retry_after = error.headers.get("Retry-After") if error.headers else None
                if retry_after:
                    try:
                        wait = min(60, max(1, int(retry_after)))
                    except ValueError:
                        wait = min(30, 2 ** attempt)
                else:
                    wait = min(30, 2 ** attempt)
                time.sleep(wait)
                continue
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as error:
            last_error = error
            if attempt == retries:
                raise
        time.sleep(min(15, 2 ** (attempt - 1)))

    assert last_error is not None
    raise RuntimeError(f"PPT request failed after {retries} attempts: {last_error}")


def lookup_card(
    tcgplayer_id: str,
    *,
    api_key: str | None = None,
) -> dict[str, Any] | None:
    """Look up a card by PokemonPriceTracker's numeric ``tcgPlayerId``."""
    payload = api_get_json(
        "/cards",
        params={"tcgPlayerId": str(tcgplayer_id).strip()},
        api_key=api_key,
    )
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return None
    return data[0]


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


def extract_card_number(card_data: dict[str, Any]) -> str:
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


def card_matches(card_data: dict[str, Any], *, set_name: str, card_number: str) -> bool:
    provider_set_name = str(
        card_data.get("set")
        or card_data.get("setName")
        or card_data.get("set_name")
        or "",
    ).strip()
    if provider_set_name and normalize_set_name(provider_set_name) != normalize_set_name(set_name):
        return False
    provider_card_number = extract_card_number(card_data)
    if provider_card_number and provider_card_number != normalize_card_number(card_number):
        return False
    return bool(provider_card_number)


def search_card(
    name: str,
    set_name: str,
    *,
    card_number: str | None = None,
    limit: int = 10,
    api_key: str | None = None,
) -> dict[str, Any] | None:
    """Search by card name and set, optionally enforcing an exact card number match."""
    payload = api_get_json(
        "/cards",
        params={"search": name, "set": set_name, "limit": str(limit)},
        api_key=api_key,
    )
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return None
    if card_number:
        for candidate in data:
            if isinstance(candidate, dict) and card_matches(candidate, set_name=set_name, card_number=card_number):
                return candidate
    return data[0]


def extract_usd_price(card_data: dict[str, Any]) -> dict[str, Any] | None:
    """Extract a normalized tcgplayer-shaped price dict from a PPT card response.

    Returns a dict compatible with the ``selected_sources["tcgplayer"]`` shape
    used by ``extract_price_rows_from_selected_sources``, or None.
    """
    prices = card_data.get("prices")
    if not isinstance(prices, dict):
        return None
    market = prices.get("market") or prices.get("mid")
    low = prices.get("low")
    high = prices.get("high")
    if not isinstance(market, (int, float)) and not isinstance(low, (int, float)):
        return None
    variant: dict[str, Any] = {}
    if isinstance(low, (int, float)):
        variant["low"] = float(low)
    if isinstance(market, (int, float)):
        variant["market"] = float(market)
    if isinstance(high, (int, float)):
        variant["high"] = float(high)
    return {
        "unit": "USD",
        "updated": None,
        "selected_variant": variant,
    }
