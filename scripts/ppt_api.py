"""PokemonPriceTracker API client for USD price lookups.

Env var: PPT_API_KEY
Docs: https://www.pokemonpricetracker.com/api-reference
"""
from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


API_BASE_URL = "https://www.pokemonpricetracker.com/api/v2"
USER_AGENT = "pokemon-tcg-corpus-db-builder/2.0"
PPT_API_KEY_ENV_VAR = "PPT_API_KEY"


def resolve_api_key() -> str:
    return os.environ.get(PPT_API_KEY_ENV_VAR, "").strip()


def api_get_json(
    path: str,
    *,
    params: dict[str, str] | None = None,
    timeout: int = 30,
    retries: int = 3,
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
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as error:
            last_error = error
            if attempt == retries:
                raise
        time.sleep(min(15, 2 ** (attempt - 1)))

    assert last_error is not None
    raise RuntimeError(f"PPT request failed after {retries} attempts: {last_error}")


def lookup_card(
    set_id: str,
    card_number: str,
    *,
    api_key: str | None = None,
) -> dict[str, Any] | None:
    """Look up a card by set_id and card_number. Returns the card dict or None."""
    tcgplayer_id = f"{set_id}-{card_number}"
    payload = api_get_json(
        "/cards",
        params={"tcgPlayerId": tcgplayer_id},
        api_key=api_key,
    )
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return None
    return data[0]


def search_card(
    name: str,
    set_name: str,
    *,
    api_key: str | None = None,
) -> dict[str, Any] | None:
    """Fallback search by card name and set name."""
    payload = api_get_json(
        "/cards",
        params={"search": name, "set": set_name, "limit": "1"},
        api_key=api_key,
    )
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return None
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
