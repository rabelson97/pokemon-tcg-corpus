"""PokeTrace API client for USD price lookups.

Env var: POKETRACE_API_KEY
Docs: https://poketrace.com/docs
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


API_BASE_URL = "https://api.poketrace.com/v1"
USER_AGENT = "pokemon-tcg-corpus-db-builder/2.0"
POKETRACE_API_KEY_ENV_VAR = "POKETRACE_API_KEY"


def resolve_api_key() -> str:
    return os.environ.get(POKETRACE_API_KEY_ENV_VAR, "").strip()


def api_get_json(
    path: str,
    *,
    params: dict[str, str] | None = None,
    timeout: int = 10,
    retries: int = 2,
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
        headers["X-API-Key"] = clean_key

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
                    wait = min(5, 2 ** attempt)
                time.sleep(wait)
                continue
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as error:
            last_error = error
            if attempt == retries:
                raise
        time.sleep(min(5, 2 ** (attempt - 1)))

    assert last_error is not None
    raise RuntimeError(f"PokeTrace request failed after {retries} attempts: {last_error}")


def fetch_sets(*, api_key: str | None = None, page_delay: float = 0.5) -> list[dict[str, Any]]:
    """Fetch all Pokemon sets from PokeTrace."""
    sets: list[dict[str, Any]] = []
    cursor: str | None = None
    pages_fetched = 0
    while True:
        params: dict[str, str] = {"game": "pokemon", "limit": "100"}
        if cursor:
            params["cursor"] = cursor
        payload = api_get_json("/sets", params=params, api_key=api_key)
        pages_fetched += 1
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
        if page_delay > 0:
            time.sleep(page_delay)
    return sets


def build_set_slug_mapping(
    provider_sets: list[dict[str, Any]],
    our_set_names: dict[str, str],
) -> dict[str, str]:
    """Map our TCGdex set_ids to PokeTrace slugs using set name matching.

    Args:
        provider_sets: List of set dicts from PokeTrace ``GET /sets``.
        our_set_names: Mapping of our set_id -> set_name.

    Returns:
        Mapping of our set_id -> PokeTrace slug.
    """
    name_to_slug: dict[str, str] = {}
    for s in provider_sets:
        slug = str(s.get("slug") or "").strip()
        name = str(s.get("name") or "").strip().lower()
        if slug and name:
            name_to_slug[name] = slug

    mapping: dict[str, str] = {}
    for set_id, set_name in our_set_names.items():
        slug = name_to_slug.get(set_name.lower())
        if slug:
            mapping[set_id] = slug
    return mapping


def lookup_card(
    set_slug: str,
    card_number: str,
    *,
    api_key: str | None = None,
) -> dict[str, Any] | None:
    """Look up a card by PokeTrace set slug and card number."""
    payload = api_get_json(
        "/cards",
        params={
            "set": set_slug,
            "card_number": card_number,
            "market": "US",
            "limit": "1",
        },
        api_key=api_key,
    )
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return None
    return data[0]


def extract_usd_price(card_data: dict[str, Any]) -> dict[str, Any] | None:
    """Extract a normalized tcgplayer-shaped price dict from a PokeTrace card response.

    Returns a dict compatible with the ``selected_sources["tcgplayer"]`` shape
    used by ``extract_price_rows_from_selected_sources``, or None.
    """
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
    if not near_mint:
        return None

    avg = near_mint.get("avg") or near_mint.get("market")
    low = near_mint.get("low")
    high = near_mint.get("high")
    if not isinstance(avg, (int, float)) and not isinstance(low, (int, float)):
        return None

    variant: dict[str, Any] = {}
    if isinstance(low, (int, float)):
        variant["low"] = float(low)
    if isinstance(avg, (int, float)):
        variant["market"] = float(avg)
    if isinstance(high, (int, float)):
        variant["high"] = float(high)

    last_updated = str(card_data.get("lastUpdated") or "").strip() or None
    return {
        "unit": "USD",
        "updated": last_updated,
        "selected_variant": variant,
    }
