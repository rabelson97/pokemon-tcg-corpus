from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


API_BASE_URL = "https://api.pokemontcg.io/v2"
USER_AGENT = "pokemon-tcg-corpus-db-builder/2.0"
DEFAULT_PAGE_SIZE = 250
DEFAULT_RETRIES = 5
API_KEY_ENV_VARS = ("POKEMONTCG_API_KEY",)
API_KEY_ENV_VAR = API_KEY_ENV_VARS[0]


def resolve_api_key(explicit_api_key: str | None = None) -> str:
    if explicit_api_key and explicit_api_key.strip():
        return explicit_api_key.strip()
    for env_var in API_KEY_ENV_VARS:
        value = os.environ.get(env_var, "").strip()
        if value:
            return value
    return ""


def api_get_json(
    path: str,
    *,
    params: dict[str, str] | None = None,
    timeout: int = 60,
    retries: int = DEFAULT_RETRIES,
    api_key: str | None = None,
) -> Any:
    query = urllib.parse.urlencode(params or {})
    url = f"{API_BASE_URL}{path}"
    if query:
        url = f"{url}?{query}"
    headers = {
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }
    clean_api_key = resolve_api_key(api_key)
    if clean_api_key:
        headers["X-Api-Key"] = clean_api_key

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code not in {429, 500, 502, 503, 504} or attempt == retries:
                raise
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as error:
            last_error = error
            if attempt == retries:
                raise
        time.sleep(min(30, 2 ** (attempt - 1)))

    assert last_error is not None
    raise RuntimeError(f"Request failed after {retries} attempts: {last_error}")


def fetch_english_cards(
    *,
    api_key: str | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    page = 1
    total_count: int | None = None

    while True:
        payload = api_get_json(
            "/cards",
            params={
                "q": "set.id:*",
                "page": str(page),
                "pageSize": str(page_size),
                "select": "id,number,set,tcgplayer",
            },
            api_key=api_key,
        )
        if not isinstance(payload, dict):
            raise RuntimeError("Unexpected PokemonTCG.io cards payload: expected object")

        batch = payload.get("data")
        if not isinstance(batch, list):
            raise RuntimeError("Unexpected PokemonTCG.io cards payload: missing data list")

        if total_count is None:
            raw_total_count = payload.get("totalCount")
            if isinstance(raw_total_count, int):
                total_count = raw_total_count

        cards.extend(card for card in batch if isinstance(card, dict))
        print(
            f"pokemontcgio page={page} batch={len(batch)} total_so_far={len(cards)}"
            + (f" total_count={total_count}" if total_count is not None else "")
        )

        if not batch:
            return cards
        if total_count is not None and len(cards) >= total_count:
            return cards
        if len(batch) < page_size:
            return cards
        page += 1


def fetch_card_by_id(
    card_id: str,
    *,
    api_key: str | None = None,
) -> dict[str, Any] | None:
    try:
        payload = api_get_json(
            f"/cards/{urllib.parse.quote(card_id, safe='')}",
            params={"select": "id,number,set,tcgplayer"},
            api_key=api_key,
        )
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise
    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected PokemonTCG.io card payload: expected object")
    card = payload.get("data")
    if not isinstance(card, dict):
        return None
    return card


def search_card_by_set_and_number(
    set_id: str,
    number: str,
    *,
    api_key: str | None = None,
) -> dict[str, Any] | None:
    try:
        payload = api_get_json(
            "/cards",
            params={
                "q": f'set.id:"{set_id}" number:"{number}"',
                "page": "1",
                "pageSize": "1",
                "select": "id,number,set,tcgplayer",
            },
            api_key=api_key,
        )
    except urllib.error.HTTPError as error:
        if error.code in {400, 404}:
            return None
        raise
    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected PokemonTCG.io cards payload: expected object")
    cards = payload.get("data")
    if not isinstance(cards, list) or not cards:
        return None
    first = cards[0]
    if not isinstance(first, dict):
        return None
    return first
