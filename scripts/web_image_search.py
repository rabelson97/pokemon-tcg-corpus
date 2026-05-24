from __future__ import annotations

import json
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from PIL import Image


USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
SEARCH_BASE_URL = "https://duckduckgo.com/"
IMAGE_SEARCH_URL = "https://duckduckgo.com/i.js"
MIN_IMAGE_WIDTH = 240
MIN_IMAGE_HEIGHT = 320
MIN_CARD_ASPECT = 1.20
MAX_CARD_ASPECT = 1.85

# DuckDuckGo image search circuit breaker to prevent cascading hangs in CI
CONSECUTIVE_DDG_FAILURES = 0
MAX_CONSECUTIVE_DDG_FAILURES = 5
DDG_BLOCKED = False


@dataclass(frozen=True)
class WebImageCandidate:
    image_url: str
    source_url: str | None
    title: str | None
    width: int | None
    height: int | None


def build_card_image_query(card: dict[str, Any]) -> str:
    parts = [
        str(card.get("name") or "").strip(),
        str(card.get("set_name") or "").strip(),
        str(card.get("card_number") or "").strip(),
        "Pokemon card",
    ]
    query = " ".join(part for part in parts if part)
    # Remove any URL-encoded characters (like %3F)
    query = re.sub(r"%[0-9A-Fa-f]{2}", " ", query)
    # Remove other suspicious characters that might trigger WAF filters (like ?, !, etc.)
    query = re.sub(r"[?!#$@*()\"']", " ", query)
    # Normalize multiple spaces
    query = re.sub(r"\s+", " ", query).strip()
    return query


def resolve_web_image_fallback(card: dict[str, Any], *, max_candidates: int = 20) -> str | None:
    query = build_card_image_query(card)
    if not query:
        return None

    print(f"Attempting web image fallback for missing URL: {card['id']} ({query})")
    candidates = search_duckduckgo_images(query, max_candidates=max_candidates)
    for candidate in sorted(candidates, key=lambda item: score_candidate(card, item), reverse=True):
        if validate_candidate_image(card, candidate):
            print(f"  Web image fallback SUCCESS: {candidate.image_url}")
            if candidate.source_url:
                print(f"  Source page: {candidate.source_url}")
            return candidate.image_url
    print("  Web image fallback: no candidate passed validation.")
    return None


def search_duckduckgo_images(query: str, *, max_candidates: int = 20) -> list[WebImageCandidate]:
    global CONSECUTIVE_DDG_FAILURES, DDG_BLOCKED
    if DDG_BLOCKED:
        print("  DuckDuckGo image search is short-circuited due to consecutive blocks/timeouts.")
        return []

    # Rate limit requests to avoid DuckDuckGo bot/WAF blocks
    time.sleep(1.5)
    try:
        vqd = fetch_duckduckgo_vqd(query)
        # Sleep slightly before secondary search payload to mimic human behavior
        time.sleep(0.5)
        payload = http_json(
            IMAGE_SEARCH_URL,
            params={
                "l": "us-en",
                "o": "json",
                "q": query,
                "vqd": vqd,
                "f": ",,,",
                "p": "1",
            },
        )
        # Reset consecutive failure counter on success
        CONSECUTIVE_DDG_FAILURES = 0
    except Exception as exc:  # pragma: no cover - network fallback path
        CONSECUTIVE_DDG_FAILURES += 1
        print(f"  Web image fallback search failed (consecutive={CONSECUTIVE_DDG_FAILURES}): {exc}")
        if CONSECUTIVE_DDG_FAILURES >= MAX_CONSECUTIVE_DDG_FAILURES:
            DDG_BLOCKED = True
            print("  CRITICAL: DuckDuckGo is blocking us consecutively. Short-circuiting all future web search fallbacks!")
        return []

    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        return []

    candidates: list[WebImageCandidate] = []
    for item in results[:max_candidates]:
        if not isinstance(item, dict):
            continue
        image_url = str(item.get("image") or "").strip()
        if not image_url:
            continue
        candidates.append(
            WebImageCandidate(
                image_url=image_url,
                source_url=str(item.get("url") or "").strip() or None,
                title=str(item.get("title") or "").strip() or None,
                width=_int_or_none(item.get("width")),
                height=_int_or_none(item.get("height")),
            )
        )
    return candidates


def fetch_duckduckgo_vqd(query: str) -> str:
    html = http_text(SEARCH_BASE_URL, params={"q": query})
    match = re.search(r"vqd=['\"]?([^&'\"\\]+)", html)
    if match:
        return match.group(1)
    match = re.search(r"'vqd'\s*:\s*'([^']+)'", html)
    if match:
        return match.group(1)
    raise RuntimeError("DuckDuckGo image token not found")


def score_candidate(card: dict[str, Any], candidate: WebImageCandidate) -> int:
    haystack = " ".join(
        part.lower()
        for part in [candidate.title, candidate.source_url, candidate.image_url]
        if part
    )
    score = 0
    name = str(card.get("name") or "").lower()
    for token in re.findall(r"[a-z0-9]+", name):
        if len(token) >= 3 and token in haystack:
            score += 2
    for key in ("card_number", "set_id"):
        value = str(card.get(key) or "").lower().strip()
        if value and value in haystack:
            score += 3
    set_name = str(card.get("set_name") or "").lower()
    for token in re.findall(r"[a-z0-9]+", set_name):
        if len(token) >= 3 and token in haystack:
            score += 1
    if "pokemon" in haystack:
        score += 1
    if "card" in haystack:
        score += 1
    return score


def validate_candidate_image(card: dict[str, Any], candidate: WebImageCandidate) -> bool:
    if not looks_like_supported_image_url(candidate.image_url):
        return False
    if candidate.width is not None and candidate.width < MIN_IMAGE_WIDTH:
        return False
    if candidate.height is not None and candidate.height < MIN_IMAGE_HEIGHT:
        return False
    if candidate.width and candidate.height:
        aspect = candidate.height / candidate.width
        if aspect < MIN_CARD_ASPECT or aspect > MAX_CARD_ASPECT:
            return False
    if score_candidate(card, candidate) < 4:
        return False

    try:
        image_bytes = http_bytes(candidate.image_url, timeout=30, max_bytes=8 * 1024 * 1024)
        with Image.open(BytesIO(image_bytes)) as image:
            width, height = image.size
            if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                return False
            aspect = height / width
            return MIN_CARD_ASPECT <= aspect <= MAX_CARD_ASPECT
    except Exception as exc:  # pragma: no cover - network/image failure path
        print(f"  Web image candidate rejected after probe: {candidate.image_url} ({exc})")
        return False


def looks_like_supported_image_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    lower_path = parsed.path.lower()
    if lower_path.endswith((".svg", ".gif")):
        return False
    return True


def http_text(url: str, *, params: dict[str, str] | None = None, timeout: int = 30) -> str:
    data = http_bytes(url, params=params, timeout=timeout)
    return data.decode("utf-8", errors="replace")


def http_json(url: str, *, params: dict[str, str] | None = None, timeout: int = 30) -> Any:
    data = http_bytes(url, params=params, timeout=timeout)
    return json.loads(data.decode("utf-8"))


def http_bytes(
    url: str,
    *,
    params: dict[str, str] | None = None,
    timeout: int = 30,
    retries: int = 3,
    max_bytes: int | None = None,
) -> bytes:
    if params:
        separator = "&" if urllib.parse.urlparse(url).query else "?"
        url = f"{url}{separator}{urllib.parse.urlencode(params)}"
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Sec-Ch-Ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
        "User-Agent": USER_AGENT,
    }
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read(max_bytes + 1 if max_bytes is not None else -1)
            if max_bytes is not None and len(data) > max_bytes:
                raise RuntimeError(f"response exceeded {max_bytes} bytes")
            return data
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code not in {429, 500, 502, 503, 504} or attempt == retries:
                raise
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as error:
            last_error = error
            if attempt == retries:
                raise
        time.sleep(min(8, 2 ** (attempt - 1)))

    assert last_error is not None
    raise RuntimeError(f"Request failed after {retries} attempts: {last_error}")


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
