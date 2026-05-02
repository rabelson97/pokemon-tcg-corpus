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

# ---------------------------------------------------------------------------
# Local card-detail cache
# ---------------------------------------------------------------------------
# Stores raw API responses keyed by (locale, upstream_id) so that re-runs
# only hit the network for cards not yet seen.  The cache is a JSONL file
# where each line is {"locale": ..., "upstream_id": ..., "payload": {...}}.
# ---------------------------------------------------------------------------

_detail_cache: dict[tuple[str, str], dict[str, Any]] | None = None
_detail_cache_path: Path | None = None


def _load_detail_cache(cache_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    key = (entry["locale"], entry["upstream_id"])
                    cache[key] = entry["payload"]
                except Exception:
                    pass
    return cache


def set_detail_cache_path(path: Path) -> None:
    """Call before fetch_all_card_records to enable caching."""
    global _detail_cache, _detail_cache_path
    _detail_cache_path = path
    _detail_cache = _load_detail_cache(path)
    print(f"detail cache loaded: {len(_detail_cache)} entries from {path}")


def _cache_detail(locale: str, upstream_id: str, payload: dict[str, Any]) -> None:
    if _detail_cache is None or _detail_cache_path is None:
        return
    key = (locale, upstream_id)
    if key in _detail_cache:
        return  # already cached
    _detail_cache[key] = payload
    _detail_cache_path.parent.mkdir(parents=True, exist_ok=True)
    with _detail_cache_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"locale": locale, "upstream_id": upstream_id, "payload": payload}, ensure_ascii=False) + "\n")


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
            # Only retry on specific HTTP status codes (429, 5xx), re-raise others immediately
            if error.code not in {429, 500, 502, 503, 504}:
                raise
            # For retryable status codes, only re-raise if we've exhausted retries
            if attempt == retries:
                raise
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as error:
            # Network/timeout errors: always retry, only re-raise if we've exhausted retries
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
            # Only retry on specific HTTP status codes (429, 5xx), re-raise others immediately
            if error.code not in {429, 500, 502, 503, 504}:
                raise
            # For retryable status codes, only re-raise if we've exhausted retries
            if attempt == retries:
                raise
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as error:
            # Network/timeout errors: always retry, only re-raise if we've exhausted retries
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


def _briefs_from_detail_cache(locale: str) -> list[dict[str, Any]] | None:
    """Return minimal brief-style dicts from the detail cache for a locale.

    Returns None if there are no cached entries for this locale (so the caller
    falls back to the live listing API).  If the cache has ANY entries for the
    locale we assume it is complete and skip the slow paginated listing.
    """
    if _detail_cache is None:
        return None
    briefs = [{"id": uid} for (loc, uid) in _detail_cache if loc == locale]
    if not briefs:
        return None
    print(f"briefs from cache locale={locale} count={len(briefs)}")
    return briefs


def fetch_card_briefs(
    locale: str,
    *,
    limit: int | None = None,
    items_per_page: int = 100,
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    seen_upstream_ids: set[str] = set()
    duplicate_briefs = 0

    cached_briefs = _briefs_from_detail_cache(locale)
    if cached_briefs is not None:
        return cached_briefs[:limit] if limit is not None else cached_briefs

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
    if _detail_cache is not None:
        cached = _detail_cache.get((locale, upstream_id))
        if cached is not None:
            return cached
    payload = api_get_json(f"{API_BASE_URL}/{locale}/cards/{urllib.parse.quote(upstream_id, safe='')}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected card payload for locale={locale} id={upstream_id}: expected object")
    _cache_detail(locale, upstream_id, payload)
    return payload


def build_canonical_card_id(locale: str, set_id: str, local_id: str) -> str:
    clean_set_id = set_id.strip()
    clean_local_id = local_id.strip()
    if not clean_set_id or not clean_local_id:
        raise ValueError(f"Missing set/local id for canonical card identity: set_id={set_id!r} local_id={local_id!r}")
    return f"pokemon:{locale}:{clean_set_id}:{clean_local_id}"


def _normalize_damage(raw: Any) -> str:
    """Normalize attack damage to a canonical form.

    TCGdex inconsistently uses fullwidth operators in JA (e.g. ``'20＋'``)
    and halfwidth in EN (``'20+'``).  Normalize to halfwidth so the same
    attack produces an identical signature across locales.
    """
    text = str(raw or "").strip()
    if not text:
        return ""
    return (
        text
        .replace("\uff0b", "+")   # fullwidth plus → +
        .replace("\uff0d", "-")   # fullwidth minus → -
        .replace("\u2212", "-")   # minus sign → -
        .replace("\uff1d", "=")   # fullwidth equals → =
        .replace("\u00d7", "x")   # multiplication sign → x
        .replace("\uff38", "x")   # fullwidth X → x
    )


def _attack_signature(attacks: list[dict[str, Any]] | None) -> str:
    """Build a locale-invariant signature from a card's attacks.

    Only cost (energy types) and damage are used — attack names and effect
    text are localized and therefore excluded.
    """
    if not attacks:
        return ""
    parts: list[str] = []
    for attack in attacks:
        cost = ",".join(str(c) for c in (attack.get("cost") or []))
        damage = _normalize_damage(attack.get("damage"))
        parts.append(f"{cost}:{damage}")
    return "|".join(parts)


def build_equivalence_key(upstream_id: str) -> str:
    """Legacy upstream-id-based key.  Used as fallback for cards that lack
    enough locale-invariant fields (Trainer items, basic Energies, etc.)."""
    clean_upstream_id = upstream_id.strip()
    if not clean_upstream_id:
        raise ValueError("upstream_id is required to build an equivalence key")
    return f"pokemon:tcgdex:{clean_upstream_id}"


def build_cross_locale_equivalence_key(card: dict[str, Any]) -> str:
    """Build a composite equivalence key from locale-invariant game-mechanic
    fields.  Cards that represent the same physical design across EN / JA / FR
    etc. will produce the same key.

    Falls back to the upstream-id-based singleton key when the card lacks
    enough discriminating fields (e.g. Trainer items, basic Energy).
    """
    category = str(card.get("category") or "").strip()
    illustrator = str(card.get("illustrator") or "").strip()
    upstream_id = str(card.get("id") or "").strip()

    # Only Pokemon cards carry the rich game-mechanic payload needed for a
    # reliable composite key.  Trainer and Energy cards share too few
    # locale-invariant fields for heuristic matching.
    if category.lower() != "pokemon" or not illustrator:
        return build_equivalence_key(upstream_id)

    dex_ids = card.get("dexId") or []
    hp = str(card.get("hp") or "").strip()
    types = ",".join(str(t) for t in (card.get("types") or []))
    stage = str(card.get("stage") or "").strip()
    suffix = str(card.get("suffix") or "").strip()
    retreat = str(card.get("retreat") if card.get("retreat") is not None else "").strip()
    attacks_sig = _attack_signature(card.get("attacks"))
    abilities = card.get("abilities") or []
    ability_count = str(len(abilities))

    # dexId is the strongest discriminator but is missing for ~12% of SV-era
    # JA cards in TCGdex.  When present, include it; when absent, the
    # remaining fields still resolve most cards.
    dex_part = ",".join(str(d) for d in sorted(dex_ids)) if dex_ids else "_"

    composite = ":".join([
        "pokemon",
        dex_part,
        illustrator.lower(),
        hp,
        types.lower(),
        stage.lower(),
        suffix.lower(),
        retreat,
        ability_count,
        attacks_sig.lower(),
    ])
    return f"pokemon:xlocale:{composite}"


def sanitize_card_id(card_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", card_id)


def normalize_image_urls(image_value: Any) -> tuple[str, str | None]:
    if isinstance(image_value, dict):
        high_url = str(image_value.get("large") or image_value.get("high") or "").strip()
        low_url = str(image_value.get("small") or image_value.get("low") or "").strip() or None
        if high_url:
            return high_url, low_url
        image_value = str(image_value.get("url") or "")

    clean_url = str(image_value or "").strip()
    if not clean_url:
        return "", None
    if re.search(r"\.(avif|gif|jpe?g|png|webp)(\?.*)?$", clean_url, re.IGNORECASE):
        return clean_url, None

    asset_root = clean_url.rstrip("/")
    return asset_root + "/high.webp", asset_root + "/low.webp"


def normalize_image_url(image_url: str) -> str:
    return normalize_image_urls(image_url)[0]


def normalize_card_image_urls(card: dict[str, Any]) -> tuple[str, str | None]:
    image_url, image_url_low = normalize_image_urls(card.get("image"))
    if image_url:
        return image_url, image_url_low
    return normalize_image_urls(card.get("images"))


def normalize_card_record(locale: str, card: dict[str, Any]) -> dict[str, Any]:
    set_info = card.get("set") or {}
    upstream_id = str(card.get("id") or "").strip()
    set_id = str(set_info.get("id") or "").strip()
    local_id = str(card.get("localId") or "").strip()
    image_url, image_url_low = normalize_card_image_urls(card)
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
        "image_url_low": image_url_low,
        "equivalence_key": build_cross_locale_equivalence_key(card),
        "pricing": card.get("pricing") or {},
        "hp": str(card.get("hp") or "").strip() or None,
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
