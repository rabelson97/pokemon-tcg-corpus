from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


API_BASE_URL = "https://api.pokemontcg.io/v2/cards"
USER_AGENT = "pokemon-tcg-corpus-db-builder/1.0"


def api_get_json(
    url: str,
    *,
    api_key: str | None = None,
    timeout: int = 60,
    retries: int = 5,
) -> dict[str, Any]:
    headers = {
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }
    if api_key:
        headers["X-Api-Key"] = api_key

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
        except urllib.error.URLError as error:
            last_error = error
            if attempt == retries:
                raise

        time.sleep(min(30, 2 ** (attempt - 1)))

    assert last_error is not None
    raise RuntimeError(f"Request failed after {retries} attempts: {last_error}")


def fetch_all_cards(
    *,
    select_fields: list[str],
    api_key: str | None = None,
    limit: int | None = None,
    page_size: int = 250,
    order_by: str = "id",
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    page = 1
    total_count: int | None = None

    while True:
        params = {
            "page": str(page),
            "pageSize": str(page_size),
            "orderBy": order_by,
            "select": ",".join(select_fields),
        }
        url = f"{API_BASE_URL}?{urllib.parse.urlencode(params)}"
        payload = api_get_json(url, api_key=api_key)

        batch = payload.get("data", [])
        if not isinstance(batch, list):
            raise RuntimeError("Unexpected cards API response: missing list payload")

        cards.extend(batch)
        total_count = int(payload.get("totalCount", len(cards)))
        print(f"fetched page={page} batch={len(batch)} total_so_far={len(cards)} expected={total_count}")

        if limit is not None and len(cards) >= limit:
            return cards[:limit]
        if not batch or len(cards) >= total_count:
            return cards
        page += 1


def download_binary(
    url: str,
    destination: Path,
    *,
    timeout: int = 90,
    retries: int = 5,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": USER_AGENT}

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                destination.write_bytes(response.read())
            return
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code not in {429, 500, 502, 503, 504} or attempt == retries:
                raise
        except urllib.error.URLError as error:
            last_error = error
            if attempt == retries:
                raise
        time.sleep(min(30, 2 ** (attempt - 1)))

    assert last_error is not None
    raise RuntimeError(f"Download failed after {retries} attempts: {last_error}")


def sanitize_card_id(card_id: str) -> str:
    return card_id.replace("/", "_")
