from __future__ import annotations

import concurrent.futures
import json
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


API_BASE_URL = "https://api.tcgdex.net/v2"
DEFAULT_LOCALES = ("en", "ja", "fr", "de", "it", "es")
USER_AGENT = "pokemon-tcg-corpus-db-builder/2.0"


def api_get_json(
    url: str,
    *,
    timeout: int = 60,
    retries: int = 5,
) -> Any:
    headers = {
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }
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
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as error:
            last_error = error
            if attempt == retries:
                raise
        time.sleep(min(30, 2 ** (attempt - 1)))

    assert last_error is not None
    raise RuntimeError(f"Download failed after {retries} attempts: {last_error}")


def parse_locales(raw: str | None) -> list[str]:
    if raw is None or not raw.strip():
        return list(DEFAULT_LOCALES)
    locales = []
    seen: set[str] = set()
    for part in raw.split(","):
        locale = part.strip()
        if not locale or locale in seen:
            continue
        seen.add(locale)
        locales.append(locale)
    if not locales:
        raise ValueError("At least one locale is required")
    invalid = [locale for locale in locales if locale not in DEFAULT_LOCALES]
    if invalid:
        raise ValueError(f"Unsupported locales: {', '.join(invalid)}")
    return locales


def fetch_card_briefs(
    locale: str,
    *,
    limit: int | None = None,
    items_per_page: int = 100,
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    seen_upstream_ids: set[str] = set()
    duplicate_briefs = 0
    page = 1
    while True:
        params = {
            "pagination:page": str(page),
            "pagination:itemsPerPage": str(items_per_page),
        }
        url = f"{API_BASE_URL}/{locale}/cards?{urllib.parse.urlencode(params)}"
        payload = api_get_json(url)
        if not isinstance(payload, list):
            raise RuntimeError(f"Unexpected cards payload for locale={locale}: expected list")
        if not payload:
            if duplicate_briefs:
                print(f"deduped locale={locale} duplicate_briefs={duplicate_briefs}")
            return cards
        unique_payload: list[dict[str, Any]] = []
        for brief in payload:
            upstream_id = str(brief.get("id") or "").strip()
            if not upstream_id:
                unique_payload.append(brief)
                continue
            if upstream_id in seen_upstream_ids:
                duplicate_briefs += 1
                continue
            seen_upstream_ids.add(upstream_id)
            unique_payload.append(brief)
        cards.extend(unique_payload)
        print(
            f"listed locale={locale} page={page} batch={len(payload)} "
            f"unique_batch={len(unique_payload)} total_so_far={len(cards)}"
        )
        if limit is not None and len(cards) >= limit:
            if duplicate_briefs:
                print(f"deduped locale={locale} duplicate_briefs={duplicate_briefs}")
            return cards[:limit]
        if len(payload) < items_per_page:
            if duplicate_briefs:
                print(f"deduped locale={locale} duplicate_briefs={duplicate_briefs}")
            return cards
        page += 1


def fetch_card_detail(locale: str, upstream_id: str) -> dict[str, Any]:
    payload = api_get_json(f"{API_BASE_URL}/{locale}/cards/{urllib.parse.quote(upstream_id, safe='')}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected card payload for locale={locale} id={upstream_id}: expected object")
    return payload


def build_canonical_card_id(locale: str, set_id: str, local_id: str) -> str:
    clean_set_id = set_id.strip()
    clean_local_id = local_id.strip()
    if not clean_set_id or not clean_local_id:
        raise ValueError(f"Missing set/local id for canonical card identity: set_id={set_id!r} local_id={local_id!r}")
    return f"pokemon:{locale}:{clean_set_id}:{clean_local_id}"


def build_equivalence_key(upstream_id: str) -> str:
    clean_upstream_id = upstream_id.strip()
    if not clean_upstream_id:
        raise ValueError("upstream_id is required to build an equivalence key")
    return f"pokemon:tcgdex:{clean_upstream_id}"


def sanitize_card_id(card_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", card_id)


def normalize_image_url(image_url: str) -> str:
    clean_url = image_url.strip()
    if not clean_url:
        return ""
    if re.search(r"\.(avif|gif|jpe?g|png|webp)(\?.*)?$", clean_url, re.IGNORECASE):
        return clean_url
    return clean_url.rstrip("/") + "/high.webp"


def normalize_card_record(locale: str, card: dict[str, Any]) -> dict[str, Any]:
    set_info = card.get("set") or {}
    upstream_id = str(card.get("id") or "").strip()
    set_id = str(set_info.get("id") or "").strip()
    local_id = str(card.get("localId") or "").strip()
    image_url = normalize_image_url(str(card.get("image") or ""))
    if not upstream_id:
        raise ValueError(f"Card payload missing id for locale={locale}")
    if not set_id:
        raise ValueError(f"Card payload missing set.id for locale={locale} upstream_id={upstream_id}")
    if not local_id:
        raise ValueError(f"Card payload missing localId for locale={locale} upstream_id={upstream_id}")

    canonical_id = build_canonical_card_id(locale, set_id, local_id)
    return {
        "id": canonical_id,
        "locale": locale,
        "upstream_source": "tcgdex",
        "upstream_id": upstream_id,
        "set_id": set_id,
        "set_name": str(set_info.get("name") or "").strip(),
        "card_number": local_id,
        "name": str(card.get("name") or canonical_id).strip(),
        "rarity": str(card.get("rarity") or "Unknown").strip() or "Unknown",
        "image_url": image_url,
        "equivalence_key": build_equivalence_key(upstream_id),
        "pricing": card.get("pricing") or {},
    }


def fetch_all_card_records(
    locales: list[str],
    *,
    limit: int | None = None,
    detail_workers: int = 12,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    listed_counts: dict[str, int] = {}
    for locale in locales:
        briefs = fetch_card_briefs(locale, limit=limit)
        listed_counts[locale] = len(briefs)
        if not briefs:
            continue

        fetched: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=detail_workers) as executor:
            futures = [executor.submit(fetch_card_detail, locale, str(brief["id"])) for brief in briefs]
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                detail = future.result()
                fetched.append(normalize_card_record(locale, detail))
                completed += 1
                if completed % 250 == 0 or completed == len(futures):
                    print(f"detailed locale={locale} fetched={completed}/{len(futures)}")

        fetched.sort(key=lambda row: row["id"])
        records.extend(fetched)
    return records, listed_counts
