#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import re
import sqlite3
import sys
import time
import urllib.parse
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageEnhance, ImageFilter

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from embedder_contract import (
    EXPECTED_DIM,
    IMAGE_SIZE as EMBED_IMAGE_SIZE,
    MEAN,
    STD,
    prepare_base_image,
)
from tcgdex_api import download_binary, fetch_all_card_records, parse_locales, sanitize_card_id, set_detail_cache_path
from pokemontcgio_api import api_get_json as pio_api_get_json, resolve_api_key as pio_resolve_api_key

# Variants are generated at DB build time so the index has K vectors per card,
# capturing the kinds of degradation real screen captures introduce. Augments are
# intentionally lighter than the training profile: we want representative
# degradation, not training-style invariance forcing.
VARIANT_TAGS: tuple[str, ...] = ("clean", "blur_lo", "jpeg_lo", "glare_mild")
VARIANT_K = len(VARIANT_TAGS)

DB_USER_VERSION = 7
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "card_embedder.onnx"
MODEL_NAME = "cardhawk:card_embedder.onnx"
SCAN_EXCLUDED_SERIES_IDS = {"tcgp"}

POKEMONTCGIO_SET_ID_ALIASES: dict[str, list[str]] = {
    "sm3.5": ["sm35"],
    "sm7.5": ["sm75"],
    "swsh3.5": ["swsh35"],
    "swsh4.5": ["swsh45"],
    "swsh10.5": ["swsh12pt5"],
    "sv10.5b": ["zsv10pt5"],
    "sv10.5w": ["rsv10pt5"],
    "hgssp": ["hsp"],
    "fut2020": ["fut20"],
    "lc": ["base6"],
    "me02.5": ["me2pt5"],
    "2011bw": ["mcd11"],
    "2012bw": ["mcd12"],
    "2014xy": ["mcd14"],
    "2015xy": ["mcd15"],
    "2016xy": ["mcd16"],
    "2017sm": ["mcd17"],
    "2018sm": ["mcd18"],
    "2019sm": ["mcd19"],
    "2021swsh": ["mcd21"],
    "2022swsh": ["mcd22"],
    "cel25": ["cel25c"],
}


def _pio_set_id_to_ours(pio_set_id: str) -> str:
    reverse_aliases: dict[str, str] = {}
    for our_id, aliases in POKEMONTCGIO_SET_ID_ALIASES.items():
        for alias in aliases:
            reverse_aliases[alias] = our_id
    if pio_set_id in reverse_aliases:
        return reverse_aliases[pio_set_id]
    import re
    result = pio_set_id
    result = re.sub(r'pt5gg$', '.5gg', result)
    result = re.sub(r'pt5$', '.5', result)
    tg_match = re.match(r'^(sv|swsh|sm)(\d+)tg$', result)
    if tg_match:
        return f"{tg_match.group(1)}{tg_match.group(2).zfill(2)}"
    sv_match = re.match(r'^(swsh)(\d+)sv$', result)
    if sv_match:
        num = int(sv_match.group(2))
        if num in (45, 35):
            return f"swsh{num // 10}.{num % 10}"
        return f"{sv_match.group(1)}{sv_match.group(2).zfill(2)}"
    gg_match = re.match(r'^(sv|swsh|sm)([\d.]+)gg$', result)
    if gg_match:
        return f"{gg_match.group(1)}{gg_match.group(2)}"
    def _pad_single_digit(m):
        prefix = m.group(1)
        digits = m.group(2)
        if len(digits) == 1:
            return f"{prefix}0{digits}"
        return m.group(0)
    result = re.sub(r'^(sv|swsh|sm|bw|xy|ex|dp|hgss|base|me|ru|si|col|rs|pl|ec|g1|g2|dp|la|mt|fb|ma)(\d+)', _pad_single_digit, result)
    if result in reverse_aliases:
        return reverse_aliases[result]
    return result


def fetch_supplementary_pokemontcgio_cards(
    existing_cards: list[dict[str, Any]],
    *,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    clean_key = pio_resolve_api_key(api_key)
    if not clean_key:
        print("pokemontcgio supplementary fetch skipped: no API key")
        return []

    existing_ids: set[str] = set()
    for card in existing_cards:
        existing_ids.add(f"{card['set_id']}|{card['card_number']}")

    try:
        sets_payload = pio_api_get_json("/sets", params={"pageSize": "500"}, api_key=clean_key)
    except Exception as exc:
        print(f"pokemontcgio supplementary fetch failed: {exc}")
        return []
    if not isinstance(sets_payload, dict):
        return []
    pio_sets = [s for s in (sets_payload.get("data") or []) if isinstance(s, dict)]
    print(f"pokemontcgio listed {len(pio_sets)} sets")

    reverse_aliases: dict[str, str] = {}
    for our_id, aliases in POKEMONTCGIO_SET_ID_ALIASES.items():
        for alias in aliases:
            reverse_aliases[alias] = our_id

    existing_set_ids = {card["set_id"] for card in existing_cards}

    def _resolve_our_set_id(pio_set_id_raw: str) -> str | None:
        candidate = _pio_set_id_to_ours(pio_set_id_raw)
        if candidate in existing_set_ids:
            return candidate
        import re
        unpadded = re.sub(r'^(sv|swsh|sm|bw|xy|ex|dp|hgss|base)0+(\d)', r'\1\2', candidate)
        if unpadded in existing_set_ids:
            return unpadded
        return candidate

    sets_with_gaps: list[dict[str, Any]] = []
    for pio_set in pio_sets:
        pio_set_id = str(pio_set.get("id") or "").strip()
        if not pio_set_id:
            continue
        our_set_id = _resolve_our_set_id(pio_set_id)
        if our_set_id not in existing_set_ids:
            pio_count = int(pio_set.get("total") or 0)
            sets_with_gaps.append({
                "pio_set_id": pio_set_id,
                "our_set_id": our_set_id,
                "pio_count": pio_count,
                "our_count": 0,
                "gap": pio_count,
                "set_name": str(pio_set.get("name") or pio_set_id),
            })
            continue
        our_count = sum(1 for c in existing_cards if c["set_id"] == our_set_id)
        pio_count = int(pio_set.get("total") or 0)
        if pio_count > our_count:
            sets_with_gaps.append({
                "pio_set_id": pio_set_id,
                "our_set_id": our_set_id,
                "pio_count": pio_count,
                "our_count": our_count,
                "gap": pio_count - our_count,
                "set_name": str(pio_set.get("name") or pio_set_id),
            })

    sets_entirely_missing = [s for s in sets_with_gaps if s["our_count"] == 0]
    sets_partially_missing = [s for s in sets_with_gaps if s["our_count"] > 0]
    print(
        f"pokemontcgio gaps: {len(sets_entirely_missing)} entirely missing sets, "
        f"{len(sets_partially_missing)} partial sets with {sum(s['gap'] for s in sets_partially_missing)} estimated missing cards"
    )

    supplementary: list[dict[str, Any]] = []

    for gap_set in sets_entirely_missing + sets_partially_missing:
        pio_set_id = gap_set["pio_set_id"]
        our_set_id = gap_set["our_set_id"]
        page = 1
        while True:
            try:
                payload = pio_api_get_json(
                    "/cards",
                    params={
                        "q": f'set.id:"{pio_set_id}"',
                        "page": str(page),
                        "pageSize": "250",
                        "select": "id,number,name,set,hp,rarity,images,artist,supertype,subtypes,types",
                    },
                    api_key=clean_key,
                )
            except Exception as exc:
                print(f"pokemontcgio fetch failed for set {pio_set_id}: {exc}")
                break
            if not isinstance(payload, dict):
                break
            batch = payload.get("data")
            if not isinstance(batch, list) or not batch:
                break
            for pio_card in batch:
                if not isinstance(pio_card, dict):
                    continue
                pio_number = str(pio_card.get("number") or "").strip()
                if not pio_number:
                    continue
                combo = f"{our_set_id}|{pio_number}"
                if combo in existing_ids:
                    continue

                image_data = pio_card.get("images") or {}
                image_url = str((image_data.get("large") or "")).strip()
                image_url_low = str((image_data.get("small") or "")).strip() or None

                card_id = f"pokemon:en:{our_set_id}:{pio_number}"
                hp_raw = str(pio_card.get("hp") or "").strip()
                types_raw = pio_card.get("types")

                record: dict[str, Any] = {
                    "id": card_id,
                    "locale": "en",
                    "upstream_source": "pokemontcgio",
                    "upstream_id": pio_card.get("id", ""),
                    "set_id": our_set_id,
                    "set_name": gap_set["set_name"],
                    "card_number": pio_number,
                    "name": str(pio_card.get("name") or card_id).strip(),
                    "rarity": str(pio_card.get("rarity") or "Unknown").strip() or "Unknown",
                    "image_url": image_url,
                    "image_url_low": image_url_low,
                    "equivalence_key": f"pokemon:pokemontcgio:{pio_card.get('id', '')}",
                    "illustrator": str(pio_card.get("artist") or "").strip() or None,
                    "hp": hp_raw or None,
                    "types": types_raw,
                }
                supplementary.append(record)
                existing_ids.add(combo)

            if len(batch) < 250:
                break
            page += 1

    print(f"pokemontcgio supplementary: {len(supplementary)} new cards")
    return supplementary


@dataclass(frozen=True)
class DownloadedCard:
    card: dict[str, Any]
    image_path: Path


@dataclass(frozen=True)
class SkippedCard:
    card_id: str
    locale: str
    reason: str
    detail: str | None = None


@dataclass(frozen=True)
class ImageResolution:
    url: str
    source: str


def card_row(
    card: dict[str, Any],
) -> tuple[str, str, str, str, str, str, str, str, str | None, str | None, str, str | None, str | None]:
    types_list = card.get("types")
    types_str = ",".join(str(t) for t in types_list) if types_list else None
    return (
        card["id"],
        card["locale"],
        card["upstream_id"],
        card["set_id"],
        card["set_name"],
        card["card_number"],
        card["name"],
        card["rarity"],
        public_image_url_or_none(card.get("image_url")),
        public_image_url_or_none(card.get("image_url_low")),
        card["equivalence_key"],
        card.get("hp"),
        types_str,
    )


def public_image_url_or_none(value: Any) -> str | None:
    url = str(value or "").strip()
    if not url:
        return None
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        return None
    host = parsed.hostname or ""
    if host in {"localhost", "127.0.0.1", "0.0.0.0"} or host.endswith(".local"):
        return None
    return url


def image_url_for_card(card: dict[str, Any]) -> str:
    image_url = public_image_url_or_none(card.get("image_url"))
    if not image_url:
        raise RuntimeError(f"Card {card['id']} is missing public https image_url")
    return image_url


def has_cached_image(image_path: Path) -> bool:
    return image_path.exists() and image_path.stat().st_size > 0


def is_scan_eligible(card: dict[str, Any]) -> bool:
    series_id = str(card.get("set_series_id") or "").strip().lower()
    return series_id not in SCAN_EXCLUDED_SERIES_IDS


def base_pil_for_card(image_path: Path) -> Image.Image:
    """Load a card image and produce the canonical 224x224 PIL crop used for embedding."""
    with Image.open(image_path) as image:
        return prepare_base_image(image, image_size=EMBED_IMAGE_SIZE)


def normalize_pil_to_nchw(image: Image.Image) -> np.ndarray:
    """Normalize a 224x224 RGB PIL image to the (1,3,224,224) ImageNet tensor the model expects."""
    array = np.asarray(image, dtype=np.float32) / 255.0
    normalized = (array - MEAN) / STD
    chw = np.transpose(normalized, (2, 0, 1))
    return np.expand_dims(chw.astype(np.float32, copy=False), axis=0)


def _variant_blur_lo(image: Image.Image, rng: random.Random) -> Image.Image:
    blurred = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.6, 1.0)))
    width, height = blurred.size
    small = blurred.resize((max(1, int(width * 0.71)), max(1, int(height * 0.71))), Image.Resampling.BILINEAR)
    return small.resize((width, height), Image.Resampling.BICUBIC)


def _variant_jpeg_lo(image: Image.Image, rng: random.Random) -> Image.Image:
    quality = rng.randint(50, 60)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=False)
    buffer.seek(0)
    decoded = Image.open(buffer).convert("RGB")
    decoded = ImageEnhance.Brightness(decoded).enhance(rng.uniform(0.92, 1.08))
    decoded = ImageEnhance.Contrast(decoded).enhance(rng.uniform(0.92, 1.08))
    return decoded


def _variant_glare_mild(image: Image.Image, rng: random.Random) -> Image.Image:
    array = np.asarray(image, dtype=np.float32).copy()
    height, width = array.shape[:2]
    xs = np.linspace(0.0, 1.0, width, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    center_x = rng.uniform(0.3, 0.7)
    center_y = rng.uniform(0.25, 0.75)
    sigma_x = rng.uniform(0.10, 0.20)
    sigma_y = rng.uniform(0.08, 0.16)
    glare = np.exp(-0.5 * (((xx - center_x) / sigma_x) ** 2 + ((yy - center_y) / sigma_y) ** 2))
    intensity = rng.uniform(45.0, 80.0)
    array += glare[..., None] * intensity
    return Image.fromarray(np.clip(array, 0.0, 255.0).astype(np.uint8))


def render_variant(base_image: Image.Image, variant_idx: int, rng: random.Random) -> Image.Image:
    if variant_idx == 0:
        return base_image
    if variant_idx == 1:
        return _variant_blur_lo(base_image, rng)
    if variant_idx == 2:
        return _variant_jpeg_lo(base_image, rng)
    if variant_idx == 3:
        return _variant_glare_mild(base_image, rng)
    raise ValueError(f"Unknown variant_idx: {variant_idx}")


def card_variant_seed(card_id: str, variant_idx: int) -> int:
    digest = hashlib.sha256(f"{card_id}|{variant_idx}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def init_db(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=DELETE;")
    connection.execute("PRAGMA synchronous=NORMAL;")
    connection.execute("PRAGMA foreign_keys=ON;")
    connection.execute(f"PRAGMA user_version={DB_USER_VERSION};")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS cards (
          id TEXT PRIMARY KEY,
          locale TEXT NOT NULL,
          upstream_id TEXT NOT NULL,
          set_id TEXT NOT NULL,
          set_name TEXT NOT NULL,
          card_number TEXT NOT NULL,
          name TEXT NOT NULL,
          rarity TEXT NOT NULL,
          image_url TEXT,
          image_url_low TEXT,
          equivalence_key TEXT NOT NULL,
          hp TEXT,
          types TEXT
        );

        CREATE TABLE IF NOT EXISTS embeddings (
          card_id TEXT NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
          model_name TEXT NOT NULL,
          variant_idx INTEGER NOT NULL DEFAULT 0,
          variant_tag TEXT NOT NULL DEFAULT 'clean',
          dim INTEGER NOT NULL,
          vector_blob BLOB NOT NULL,
          PRIMARY KEY (card_id, model_name, variant_idx)
        );

        CREATE INDEX IF NOT EXISTS idx_embeddings_card ON embeddings(card_id);

        CREATE TABLE IF NOT EXISTS card_equivalents (
          card_id TEXT NOT NULL PRIMARY KEY REFERENCES cards(id) ON DELETE CASCADE,
          equivalence_key TEXT NOT NULL,
          upstream_source TEXT NOT NULL,
          upstream_id TEXT NOT NULL,
          locale TEXT NOT NULL,
          set_id TEXT NOT NULL,
          local_id TEXT NOT NULL
        );
        """
    )
    create_int8_table(connection)
    card_columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(cards);").fetchall()}
    if "image_url_low" not in card_columns:
        connection.execute("ALTER TABLE cards ADD COLUMN image_url_low TEXT;")
    if "types" not in card_columns:
        connection.execute("ALTER TABLE cards ADD COLUMN types TEXT;")


def create_int8_table(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS embeddings_int8 (
          card_id TEXT NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
          variant_idx INTEGER NOT NULL DEFAULT 0,
          variant_tag TEXT NOT NULL DEFAULT 'clean',
          dim INTEGER NOT NULL,
          vector_int8 BLOB NOT NULL,
          PRIMARY KEY (card_id, variant_idx)
        );

        CREATE INDEX IF NOT EXISTS idx_embeddings_int8_card ON embeddings_int8(card_id);
        """
    )


def recreate_int8_table(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        DROP INDEX IF EXISTS idx_embeddings_int8_card;
        DROP TABLE IF EXISTS embeddings_int8;
        """
    )
    create_int8_table(connection)
    connection.execute(f"PRAGMA user_version={DB_USER_VERSION};")
    connection.commit()


def load_seed_card_records(seed_db: Path, locales: list[str]) -> list[dict[str, Any]]:
    if not seed_db.exists():
        return []
    allowed_locales = set(locales)
    with sqlite3.connect(seed_db) as connection:
        card_columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(cards);").fetchall()}
        required = {"id", "locale", "upstream_id", "set_name", "card_number", "name", "rarity", "image_url", "equivalence_key"}
        if not required.issubset(card_columns):
            return []
        set_column = "set_id" if "set_id" in card_columns else "set_code" if "set_code" in card_columns else None
        if set_column is None:
            return []
        image_low_select = "image_url_low" if "image_url_low" in card_columns else "NULL AS image_url_low"
        hp_select = "hp" if "hp" in card_columns else "NULL AS hp"
        types_select = "types" if "types" in card_columns else "NULL AS types"
        rows = connection.execute(
            f"""
            SELECT id, locale, upstream_id, {set_column} AS set_id, set_name, card_number,
                   name, rarity, image_url, {image_low_select}, equivalence_key, hp, {types_select}
            FROM cards
            ORDER BY id;
            """
        ).fetchall()

    cards: list[dict[str, Any]] = []
    for row in rows:
        locale = str(row[1])
        if locale not in allowed_locales:
            continue
        types_value = row[12]
        cards.append(
            {
                "id": str(row[0]),
                "locale": locale,
                "upstream_source": "seed",
                "upstream_id": str(row[2]),
                "set_id": str(row[3]),
                "set_name": str(row[4]),
                "card_number": str(row[5]),
                "name": str(row[6]),
                "rarity": str(row[7]),
                "image_url": str(row[8]) if row[8] is not None else "",
                "image_url_low": str(row[9]) if row[9] is not None else None,
                "equivalence_key": str(row[10]),
                "pricing": {},
                "illustrator": None,
                "hp": str(row[11]) if row[11] is not None else None,
                "types": str(types_value).split(",") if types_value else [],
            }
        )
    return cards


def sample_embedding_diagnostics(
    db_path: Path,
    *,
    sample_size: int = 16,
    print_rows: int = 5,
) -> dict[str, Any]:
    with sqlite3.connect(db_path) as connection:
        # Sample only variant_idx=0 (clean) rows for the diagnostics so the
        # cross-pair cosine check sees distinct cards, not different variants
        # of the same card.
        rows = connection.execute(
            """
            SELECT e.card_id, c.name, e.dim, e.vector_blob, e.variant_tag
            FROM embeddings e
            JOIN cards c ON c.id = e.card_id
            WHERE e.variant_idx = 0
            ORDER BY e.card_id
            LIMIT ?;
            """,
            (sample_size,),
        ).fetchall()

        variant_breakdown_rows = connection.execute(
            "SELECT variant_tag, COUNT(*) FROM embeddings GROUP BY variant_tag ORDER BY variant_tag;"
        ).fetchall()

    diagnostics: list[dict[str, Any]] = []
    hashes: list[str] = []
    cosine_samples: list[dict[str, Any]] = []
    decoded_vectors: list[tuple[str, str, np.ndarray]] = []

    for card_id, name, dim, blob, variant_tag in rows:
        vector = np.frombuffer(blob, dtype="<f4")
        vector_hash = hashlib.sha256(blob).hexdigest()
        decoded_vectors.append((str(card_id), str(name), vector))
        hashes.append(vector_hash)
        diagnostics.append(
            {
                "card_id": str(card_id),
                "name": str(name),
                "variant_tag": str(variant_tag),
                "dim": int(dim),
                "blob_len": len(blob),
                "first8": vector[:8].astype(float).tolist(),
                "norm": float(np.linalg.norm(vector)),
                "min": float(np.min(vector)),
                "max": float(np.max(vector)),
                "has_nan": bool(np.isnan(vector).any()),
                "has_inf": bool(np.isinf(vector).any()),
                "sha256_16": vector_hash[:16],
            }
        )

    for left in range(min(4, len(decoded_vectors))):
        for right in range(left + 1, min(6, len(decoded_vectors))):
            left_card_id, left_name, left_vector = decoded_vectors[left]
            right_card_id, right_name, right_vector = decoded_vectors[right]
            cosine = float(np.dot(left_vector, right_vector))
            cosine_samples.append(
                {
                    "left_card_id": left_card_id,
                    "left_name": left_name,
                    "right_card_id": right_card_id,
                    "right_name": right_name,
                    "cosine": cosine,
                }
            )

    variant_breakdown = {str(tag): int(count) for tag, count in variant_breakdown_rows}
    summary = {
        "sample_count": len(rows),
        "distinct_hashes": len(set(hashes)),
        "variant_breakdown": variant_breakdown,
        "rows": diagnostics[:print_rows],
        "cosine_samples": cosine_samples[:8],
    }
    print(json.dumps({"embedding_diagnostics": summary}, indent=2))
    return summary


def validate_embeddings_db(
    db_path: Path,
    *,
    min_row_count: int,
    require_user_version: bool = True,
) -> tuple[int, int, dict[str, Any]]:
    with sqlite3.connect(db_path) as connection:
        integrity = connection.execute("PRAGMA integrity_check;").fetchone()
        if integrity is None or integrity[0] != "ok":
            raise RuntimeError(f"PRAGMA integrity_check failed: {integrity}")

        user_version = int(connection.execute("PRAGMA user_version;").fetchone()[0])
        if require_user_version and user_version != DB_USER_VERSION:
            raise RuntimeError(f"PRAGMA user_version expected {DB_USER_VERSION}, got {user_version}")

        card_count = int(connection.execute("SELECT COUNT(*) FROM cards;").fetchone()[0])
        embedding_count = int(connection.execute("SELECT COUNT(*) FROM embeddings;").fetchone()[0])
        if card_count < min_row_count:
            raise RuntimeError(f"cards row count {card_count} is below minimum {min_row_count}")
        expected_embeddings = card_count * VARIANT_K
        if embedding_count != expected_embeddings:
            raise RuntimeError(
                f"embeddings count {embedding_count} != cards ({card_count}) * variants ({VARIANT_K}) = {expected_embeddings}"
            )

        variants_per_card_min = int(
            connection.execute(
                "SELECT MIN(variant_count) FROM (SELECT card_id, COUNT(*) AS variant_count FROM embeddings GROUP BY card_id);"
            ).fetchone()[0]
        )
        variants_per_card_max = int(
            connection.execute(
                "SELECT MAX(variant_count) FROM (SELECT card_id, COUNT(*) AS variant_count FROM embeddings GROUP BY card_id);"
            ).fetchone()[0]
        )
        if variants_per_card_min != VARIANT_K or variants_per_card_max != VARIANT_K:
            raise RuntimeError(
                f"Each card must have exactly {VARIANT_K} variants, "
                f"got min={variants_per_card_min} max={variants_per_card_max}"
            )

        distinct_tags = {
            str(row[0])
            for row in connection.execute("SELECT DISTINCT variant_tag FROM embeddings;").fetchall()
        }
        if distinct_tags != set(VARIANT_TAGS):
            raise RuntimeError(f"Unexpected variant_tag set: {sorted(distinct_tags)} vs expected {list(VARIANT_TAGS)}")

        bad_dim = int(connection.execute("SELECT COUNT(*) FROM embeddings WHERE dim <= 0;").fetchone()[0])
        if bad_dim > 0:
            raise RuntimeError("Found embeddings rows with dim <= 0")

        bad_blob = int(
            connection.execute(
                "SELECT COUNT(*) FROM embeddings WHERE length(vector_blob) != dim * 4;"
            ).fetchone()[0]
        )
        if bad_blob > 0:
            raise RuntimeError("Found embeddings rows with invalid vector_blob lengths")

        local_url_count = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM cards
                WHERE image_url LIKE 'file:%'
                   OR image_url_low LIKE 'file:%'
                   OR image_url LIKE '%/home/runner/%'
                   OR image_url_low LIKE '%/home/runner/%';
                """
            ).fetchone()[0]
        )
        if local_url_count > 0:
            raise RuntimeError(f"Found {local_url_count} cards with local runner image URLs")

        blank_image_url_count = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM cards
                WHERE image_url IS NULL OR trim(image_url) = '';
                """
            ).fetchone()[0]
        )
        if blank_image_url_count > 0:
            raise RuntimeError(f"Found {blank_image_url_count} cards without a public image_url")

    diagnostics = sample_embedding_diagnostics(db_path)
    sample_count = int(diagnostics["sample_count"])
    distinct_hashes = int(diagnostics["distinct_hashes"])
    if sample_count > 1 and distinct_hashes < max(2, sample_count // 2):
        raise RuntimeError(
            f"Too few distinct sampled vectors: distinct_hashes={distinct_hashes} sample_count={sample_count}"
        )

    for row in diagnostics["rows"]:
        norm = float(row["norm"])
        if not np.isfinite(norm) or norm < 0.5:
            raise RuntimeError(f"Sampled vector has suspicious norm for {row['card_id']}: {norm}")
        if row["has_nan"] or row["has_inf"]:
            raise RuntimeError(f"Sampled vector has NaN/Inf for {row['card_id']}")
        if int(row["blob_len"]) != int(row["dim"]) * 4:
            raise RuntimeError(
                f"Sampled vector has invalid blob_len for {row['card_id']}: {row['blob_len']} vs {row['dim']}*4"
            )

    suspicious_cosines = [sample for sample in diagnostics["cosine_samples"] if abs(float(sample["cosine"])) > 0.9999]
    if len(suspicious_cosines) >= max(2, sample_count // 2):
        raise RuntimeError(f"Too many suspiciously identical sampled cosine similarities: {suspicious_cosines}")

    return card_count, embedding_count, diagnostics


def quantize_to_int8(vector: np.ndarray) -> bytes:
    scaled = np.clip(np.round(vector * 127.0), -128, 127).astype(np.int8)
    return scaled.tobytes()


def dequantize_int8(blob: bytes, dim: int) -> np.ndarray:
    raw = np.frombuffer(blob, dtype=np.int8).astype(np.float32)
    if raw.shape[0] != dim:
        raise RuntimeError(f"int8 blob length {raw.shape[0]} != expected dim {dim}")
    return raw / 127.0


def insert_int8_embeddings(connection: sqlite3.Connection) -> tuple[int, int]:
    rows = connection.execute(
        """
        SELECT e.card_id, e.variant_idx, e.variant_tag, e.dim, e.vector_blob
        FROM embeddings e
        ORDER BY e.card_id, e.variant_idx
        """
    ).fetchall()

    int8_rows: list[tuple[str, int, str, int, bytes]] = []
    for card_id, variant_idx, variant_tag, dim, blob in rows:
        vector = np.frombuffer(blob, dtype="<f4")
        if vector.shape[0] != dim:
            raise RuntimeError(f"vector dim mismatch for {card_id} variant={variant_idx}: {vector.shape[0]} vs {dim}")
        int8_blob = quantize_to_int8(vector)
        int8_rows.append((card_id, int(variant_idx), str(variant_tag), dim, int8_blob))

    connection.executemany(
        """
        INSERT INTO embeddings_int8 (card_id, variant_idx, variant_tag, dim, vector_int8)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(card_id, variant_idx) DO NOTHING;
        """,
        int8_rows,
    )
    connection.commit()

    total_bytes = sum(len(row[4]) for row in int8_rows)
    row_count = len(int8_rows)
    print(f"inserted {row_count} int8 variant vectors, total vector bytes = {total_bytes:,} ({total_bytes / 1024 / 1024:.1f} MB)")
    return row_count, total_bytes


def rebuild_int8_embeddings(
    db_path: Path,
    *,
    summary_json: Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys=ON;")
        card_count = int(connection.execute("SELECT COUNT(*) FROM cards;").fetchone()[0])
        embedding_count = int(connection.execute("SELECT COUNT(*) FROM embeddings;").fetchone()[0])
        if card_count <= 0 or embedding_count <= 0:
            raise RuntimeError(f"Cannot rebuild int8 rows from empty db: cards={card_count}, embeddings={embedding_count}")

        recreate_int8_table(connection)
        int8_row_count, int8_total_bytes = insert_int8_embeddings(connection)
        int8_validation = validate_int8_quantization(connection)
        variant_count = int(
            connection.execute("SELECT COUNT(DISTINCT variant_idx) FROM embeddings;").fetchone()[0]
        )

    summary = {
        "status": "int8_rebuilt",
        "output_db": str(db_path),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "cards_count": card_count,
        "embeddings_count": embedding_count,
        "variants_per_card": variant_count,
        "user_version": DB_USER_VERSION,
        "int8_row_count": int8_row_count,
        "int8_total_bytes": int8_total_bytes,
        "int8_validation": int8_validation,
    }
    if int8_row_count != embedding_count:
        raise RuntimeError(f"int8 row count mismatch: {int8_row_count} != embeddings count {embedding_count}")
    if summary_json is not None:
        write_json_file(summary_json, summary)
    print(json.dumps(summary, indent=2))
    return summary


def validate_int8_quantization(
    connection: sqlite3.Connection,
    sample_size: int = 500,
) -> dict[str, Any]:
    rows = connection.execute(
        """
        SELECT e.card_id, e.variant_idx, e.dim, e.vector_blob, i.vector_int8
        FROM embeddings e
        JOIN embeddings_int8 i ON i.card_id = e.card_id AND i.variant_idx = e.variant_idx
        ORDER BY RANDOM()
        LIMIT ?;
        """,
        (sample_size,),
    ).fetchall()

    if not rows:
        raise RuntimeError("No rows found for int8 validation")

    cosine_errors: list[float] = []
    l2_errors: list[float] = []
    for card_id, variant_idx, dim, f32_blob, i8_blob in rows:
        f32_vec = np.frombuffer(f32_blob, dtype="<f4").copy()
        if f32_vec.shape[0] != dim:
            continue
        f32_norm = np.linalg.norm(f32_vec)
        if f32_norm < 1e-9:
            continue
        f32_unit = f32_vec / f32_norm

        i8_vec = dequantize_int8(i8_blob, dim)
        i8_norm = np.linalg.norm(i8_vec)
        if i8_norm < 1e-9:
            continue
        i8_unit = i8_vec / i8_norm

        cosine = float(np.dot(f32_unit, i8_unit))
        cosine_error = 1.0 - cosine
        cosine_errors.append(cosine_error)

        l2 = float(np.linalg.norm(f32_unit - i8_unit))
        l2_errors.append(l2)

    if not cosine_errors:
        raise RuntimeError("No valid samples for int8 quantization validation")

    mean_cosine_error = float(np.mean(cosine_errors))
    max_cosine_error = float(np.max(cosine_errors))
    mean_l2_error = float(np.mean(l2_errors))
    max_l2_error = float(np.max(l2_errors))

    summary = {
        "sample_count": len(cosine_errors),
        "mean_cosine_error": mean_cosine_error,
        "max_cosine_error": max_cosine_error,
        "mean_l2_error": mean_l2_error,
        "max_l2_error": max_l2_error,
    }
    print(json.dumps({"int8_quantization_validation": summary}, indent=2))

    if mean_cosine_error > 0.01:
        raise RuntimeError(
            f"int8 quantization mean cosine error {mean_cosine_error:.6f} exceeds threshold 0.01"
        )

    return summary


def fallback_manifest_path(cache_dir: Path) -> Path:
    # Save the fallbacks file in the repository root so it can be committed to git,
    # except when running unit tests where a temporary directory is used for isolation.
    try:
        import tempfile
        if cache_dir.is_relative_to(Path(tempfile.gettempdir())):
            return cache_dir / "image-fallbacks.json"
    except Exception:
        pass
    return REPO_ROOT / "image-fallbacks.json"


def load_fallback_manifest(cache_dir: Path) -> dict[str, dict[str, str]]:
    path = fallback_manifest_path(cache_dir)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    manifest: dict[str, dict[str, str]] = {}
    for card_id, value in payload.items():
        if not isinstance(card_id, str) or not isinstance(value, dict):
            continue
        url = public_image_url_or_none(value.get("url"))
        source = str(value.get("source") or "").strip()
        if url and source:
            manifest[card_id] = {"url": url, "source": source}
    return manifest


def write_fallback_manifest(cache_dir: Path, manifest: dict[str, dict[str, str]]) -> None:
    path = fallback_manifest_path(cache_dir)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        clean_value = value.strip()
        if not clean_value or clean_value in seen:
            continue
        seen.add(clean_value)
        deduped.append(clean_value)
    return deduped


def compact_numeric_token(value: str) -> str:
    clean_value = value.strip()
    if clean_value.isdigit():
        return str(int(clean_value))
    match = re.fullmatch(r"([A-Za-z]+)0+([1-9][0-9]*)", clean_value)
    if match:
        return f"{match.group(1)}{int(match.group(2))}"
    return clean_value


def candidate_pokemontcgio_set_ids(set_id: str, card_number: str) -> list[str]:
    clean_set_id = set_id.strip()
    candidates = [clean_set_id]
    candidates.extend(POKEMONTCGIO_SET_ID_ALIASES.get(clean_set_id, []))
    candidates.append(re.sub(r"^([A-Za-z]+)0+([1-9][0-9]*(?:\.5)?)$", r"\1\2", clean_set_id))
    if ".5" in clean_set_id:
        candidates.append(clean_set_id.replace(".5", ""))
        candidates.append(clean_set_id.replace(".5", "pt5"))
    clean_number = card_number.strip().upper()
    if clean_number.startswith("TG"):
        candidates.extend(f"{candidate}tg" for candidate in list(candidates))
    elif clean_number.startswith("GG"):
        candidates.extend(f"{candidate}gg" for candidate in list(candidates))
    return dedupe_strings(candidates)


def candidate_pokemontcgio_numbers(card_number: str) -> list[str]:
    clean_number = card_number.strip()
    candidates = [clean_number, compact_numeric_token(clean_number)]
    classic_match = re.fullmatch(r"([0-9]+)A.*", clean_number.upper())
    if classic_match:
        candidates.append(str(int(classic_match.group(1))))
    return dedupe_strings(candidates)


def normalize_card_name_for_match(value: Any) -> str:
    text = str(value or "").lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"\b(gx|ex|vmax|vstar)\b", r"\1", text)
    text = text.replace("v-union", "vunion")
    return re.sub(r"[^a-z0-9]+", "", text)


def candidate_matches_card(card: dict[str, Any], candidate: dict[str, Any]) -> bool:
    if normalize_card_name_for_match(candidate.get("name")) != normalize_card_name_for_match(card.get("name")):
        return False
    cand_hp = str(candidate.get("hp") or "").strip()
    card_hp = str(card.get("hp") or "").strip()
    if cand_hp and card_hp and cand_hp != card_hp:
        return False
    cand_artist = str(candidate.get("artist") or "").lower().strip()
    card_artist = str(card.get("illustrator") or "").lower().strip()
    return not cand_artist or not card_artist or cand_artist == card_artist


def resolve_pokemontcgio_image_by_identity(card: dict[str, Any]) -> ImageResolution | None:
    try:
        from scripts.pokemontcgio_api import search_card_by_set_and_number
    except ImportError:
        try:
            from pokemontcgio_api import search_card_by_set_and_number
        except ImportError:
            return None

    set_id = str(card.get("set_id") or "").strip()
    card_number = str(card.get("card_number") or "").strip()
    if not set_id or not card_number:
        return None
    for candidate_set_id in candidate_pokemontcgio_set_ids(set_id, card_number):
        for candidate_number in candidate_pokemontcgio_numbers(card_number):
            candidate = search_card_by_set_and_number(candidate_set_id, candidate_number)
            if not candidate or not candidate_matches_card(card, candidate):
                continue
            large_image = public_image_url_or_none((candidate.get("images") or {}).get("large"))
            if large_image:
                print(
                    f"  Fallback SUCCESS! Mapped TCGdex {card['id']} "
                    f"to pokemontcg.io {candidate.get('id')} by set/number"
                )
                return ImageResolution(url=large_image, source="pokemontcgio_set_number_name")
    return None


def resolve_fallback_image(card: dict[str, Any], *, allow_web_image_fallback: bool) -> ImageResolution | None:
    identity_match = resolve_pokemontcgio_image_by_identity(card)
    if identity_match:
        return identity_match

    try:
        from scripts.pokemontcgio_api import search_cards_by_name
    except ImportError:
        try:
            from pokemontcgio_api import search_cards_by_name
        except ImportError:
            search_cards_by_name = None

    name = card.get("name")
    if not name:
        return None

    print(f"Attempting image fallback for missing URL: {card['id']} ({name})")

    if search_cards_by_name is not None:
        # Try query pokemontcg.io
        card_number = str(card.get("card_number") or "").strip().lstrip("0")
        candidates = search_cards_by_name(name, number=card_number)
        if not candidates:
            print(f"  Fallback: No matches found for name '{name}' and number '{card_number}'")

        for cand in candidates:
            cand_artist = str(cand.get("artist") or "").lower().strip()
            card_artist = str(card.get("illustrator") or "").lower().strip()

            cand_hp = str(cand.get("hp") or "").strip()
            card_hp = str(card.get("hp") or "").strip()
            cand_number = str(cand.get("number") or "").strip().lstrip("0")
            card_number = str(card.get("card_number") or "").strip().lstrip("0")

            # Heuristically verify matching illustrator and HP to ensure same card design/artwork.
            # If illustrator or HP is missing in either dataset, we allow the match based on name and number.
            artist_match = (
                not cand_artist
                or not card_artist
                or (cand_artist == card_artist)
                or (card_artist in cand_artist)
                or (cand_artist in card_artist)
            )
            hp_match = (not cand_hp or not card_hp or cand_hp == card_hp)
            number_match = bool(card_number) and cand_number == card_number

            if artist_match and hp_match and number_match:
                large_image = public_image_url_or_none(cand.get("images", {}).get("large"))
                if large_image:
                    print(f"  Fallback SUCCESS! Mapped TCGDex {card['id']} to pokemontcg.io {cand['id']}")
                    print(f"  Selected Large Image: {large_image}")
                    return ImageResolution(url=large_image, source="pokemontcgio_name_artist_hp")

        if candidates:
            print(f"  Fallback: Matches found but none passed artist/HP alignment check.")
    if allow_web_image_fallback:
        try:
            from scripts.web_image_search import resolve_web_image_fallback
        except ImportError:
            try:
                from web_image_search import resolve_web_image_fallback
            except ImportError:
                return None
        web_image = resolve_web_image_fallback(card)
        if public_image_url_or_none(web_image):
            return ImageResolution(url=web_image, source="web_search_unverified")
    return None


def skipped_cards_by_locale_and_reason(skipped_cards: list[SkippedCard]) -> tuple[dict[str, int], dict[str, int]]:
    skipped_by_locale: dict[str, int] = {}
    skipped_reasons: dict[str, int] = {}
    for skipped in skipped_cards:
        skipped_by_locale[skipped.locale] = skipped_by_locale.get(skipped.locale, 0) + 1
        skipped_reasons[skipped.reason] = skipped_reasons.get(skipped.reason, 0) + 1
    return skipped_by_locale, skipped_reasons


def build_missing_images_report(
    skipped_cards: list[SkippedCard],
    cards: list[dict[str, Any]],
    *,
    image_cache_dir: Path,
    excluded_cards: list[dict[str, Any]],
) -> dict[str, Any]:
    card_by_id = {str(card["id"]): card for card in cards}
    skipped_by_locale, skipped_reasons = skipped_cards_by_locale_and_reason(skipped_cards)
    entries: list[dict[str, Any]] = []
    for skipped in skipped_cards:
        card = card_by_id.get(skipped.card_id, {})
        image_path = image_cache_dir / f"{sanitize_card_id(skipped.card_id)}.img"
        entry: dict[str, Any] = {
            "card_id": skipped.card_id,
            "locale": skipped.locale,
            "reason": skipped.reason,
            "name": card.get("name"),
            "set_id": card.get("set_id"),
            "set_name": card.get("set_name"),
            "set_series_id": card.get("set_series_id"),
            "set_series_name": card.get("set_series_name"),
            "card_number": card.get("card_number"),
            "upstream_id": card.get("upstream_id"),
            "illustrator": card.get("illustrator"),
            "hp": card.get("hp"),
            "types": card.get("types"),
            "expected_cache_file": str(image_path),
            "cache_file_exists": has_cached_image(image_path),
        }
        if skipped.detail:
            entry["detail"] = skipped.detail
        entries.append(entry)
    return {
        "status": "blocked_missing_scan_images",
        "message": "No embeddings release was published. Seed or resolve these real card images, then rerun the workflow.",
        "missing_count": len(skipped_cards),
        "skipped_by_locale": skipped_by_locale,
        "skipped_reasons": skipped_reasons,
        "excluded_non_scan_cards": len(excluded_cards),
        "excluded_series_ids": sorted(SCAN_EXCLUDED_SERIES_IDS),
        "recovery_steps": [
            "For each entry, add a verified real card image to the image cache using expected_cache_file naming or add a verified HTTPS fallback URL to image-fallbacks.json.",
            "Do not add placeholder/card-back images for scan-eligible cards; embeddings must be generated from real card art.",
            "Rerun Build Embeddings DB. The app keeps using the previous embeddings-latest release until a complete build succeeds.",
        ],
        "cards": entries,
    }


def write_json_file(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ensure_images(
    cards: list[dict[str, Any]],
    cache_dir: Path,
    *,
    download_workers: int,
    allow_web_image_fallback: bool,
) -> tuple[list[DownloadedCard], list[SkippedCard], float, dict[str, str]]:
    started = time.perf_counter()
    cache_dir.mkdir(parents=True, exist_ok=True)
    fallback_manifest = load_fallback_manifest(cache_dir)
    image_paths = [cache_dir / f"{sanitize_card_id(card['id'])}.img" for card in cards]
    skipped: list[SkippedCard] = []
    image_sources: dict[str, str] = {}

    pending: list[tuple[dict[str, Any], Path]] = []
    for card, image_path in zip(cards, image_paths, strict=True):
        card_id = card["id"]
        image_url = public_image_url_or_none(card.get("image_url"))
        card["image_url"] = image_url
        card["image_url_low"] = public_image_url_or_none(card.get("image_url_low"))
        cached_image_available = has_cached_image(image_path)

        # Ensure downloads only use public HTTPS URLs. If missing, look up from fallback manifest first.
        if not image_url:
            cached_fallback = fallback_manifest.get(card_id)
            if cached_fallback:
                card["image_url"] = cached_fallback["url"]
                image_sources[card_id] = cached_fallback["source"] + ":cached"
            else:
                fallback = resolve_fallback_image(card, allow_web_image_fallback=allow_web_image_fallback)
                if fallback:
                    card["image_url"] = fallback.url
                    image_sources[card_id] = fallback.source
                    fallback_manifest[card_id] = {"url": fallback.url, "source": fallback.source}
                else:
                    skipped.append(SkippedCard(card_id=card_id, locale=card["locale"], reason="missing_image_url"))
                    continue
        else:
            image_sources[card_id] = "upstream"

        # Check if the image file is already cached locally and has content
        if has_cached_image(image_path):
            continue

        pending.append((card, image_path))

    if pending:
        print(f"downloading {len(pending)} new images to {cache_dir}")

        pending_by_id = {card["id"]: (card, image_path) for card, image_path in pending}

        def task(item: tuple[dict[str, Any], Path]) -> tuple[str, str | None]:
            card, image_path = item
            try:
                download_binary(image_url_for_card(card), image_path)
                return card["id"], None
            except Exception as exc:
                card_id = card["id"]
                fallback_entry = fallback_manifest.get(card_id)
                if fallback_entry:
                    fallback_url = fallback_entry.get("url")
                    if fallback_url:
                        try:
                            download_binary(fallback_url, image_path)
                            return card_id, None
                        except Exception:
                            pass
                return card_id, str(exc)

        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=download_workers) as executor:
            futures = [executor.submit(task, item) for item in pending]
            for future in concurrent.futures.as_completed(futures):
                card_id, error = future.result()
                completed += 1
                if error is not None:
                    locale = card_id.split(":")[1] if card_id.startswith("pokemon:") else "unknown"
                    skipped.append(SkippedCard(card_id=card_id, locale=locale, reason="download_failed", detail=error))
                if completed % 250 == 0 or completed == len(pending):
                    print(f"downloaded {completed}/{len(pending)} pending images")
    else:
        print(f"using cached images from {cache_dir}")

    if fallback_manifest:
        write_fallback_manifest(cache_dir, fallback_manifest)

    ready: list[DownloadedCard] = []
    skipped_ids = {item.card_id for item in skipped}
    for card, image_path in zip(cards, image_paths, strict=True):
        if card["id"] in skipped_ids:
            continue
        if not image_path.exists() or image_path.stat().st_size == 0:
            skipped.append(
                SkippedCard(
                    card_id=card["id"],
                    locale=card["locale"],
                    reason="image_missing_after_download",
                )
            )
            skipped_ids.add(card["id"])
            continue
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as exc:
            skipped.append(
                SkippedCard(
                    card_id=card["id"],
                    locale=card["locale"],
                    reason="invalid_image",
                    detail=str(exc),
                )
            )
            skipped_ids.add(card["id"])
            continue
        ready.append(DownloadedCard(card=card, image_path=image_path))

    return ready, skipped, time.perf_counter() - started, image_sources


def preflight_image_urls(
    cards: list[dict[str, Any]],
    cache_dir: Path,
    *,
    allow_web_image_fallback: bool,
) -> tuple[list[SkippedCard], dict[str, str]]:
    """Resolve public display image URLs before seed embedding reuse.

    A cached local image is useful for avoiding re-downloads, but it is not a
    valid published display URL. The release DB must carry a public HTTPS URL
    for every scan-eligible card, including cards whose embeddings are reused
    from a seed DB.
    """
    fallback_manifest = load_fallback_manifest(cache_dir)
    skipped: list[SkippedCard] = []
    image_sources: dict[str, str] = {}

    for card in cards:
        card_id = str(card["id"])
        image_url = public_image_url_or_none(card.get("image_url"))
        card["image_url"] = image_url
        card["image_url_low"] = public_image_url_or_none(card.get("image_url_low"))

        if image_url:
            image_sources[card_id] = "upstream"
            continue

        cached_fallback = fallback_manifest.get(card_id)
        if cached_fallback:
            card["image_url"] = cached_fallback["url"]
            image_sources[card_id] = cached_fallback["source"] + ":cached"
            continue

        fallback = resolve_fallback_image(card, allow_web_image_fallback=allow_web_image_fallback)
        if fallback:
            card["image_url"] = fallback.url
            image_sources[card_id] = fallback.source
            fallback_manifest[card_id] = {"url": fallback.url, "source": fallback.source}
            continue

        skipped.append(SkippedCard(card_id=card_id, locale=card["locale"], reason="missing_image_url"))

    if fallback_manifest:
        write_fallback_manifest(cache_dir, fallback_manifest)
    return skipped, image_sources


def inspect_model_contract(connection: sqlite3.Connection) -> list[tuple[str, int, int]]:
    rows = connection.execute(
        "SELECT model_name, dim, COUNT(*) FROM embeddings GROUP BY model_name, dim ORDER BY model_name, dim;"
    ).fetchall()
    return [(str(model_name), int(dim), int(count)) for model_name, dim, count in rows]


def locale_row_counts(connection: sqlite3.Connection) -> dict[str, int]:
    rows = connection.execute(
        "SELECT locale, COUNT(*) FROM cards GROUP BY locale ORDER BY locale;"
    ).fetchall()
    return {str(locale): int(count) for locale, count in rows}


def insert_card_metadata(connection: sqlite3.Connection, cards: list[dict[str, Any]]) -> None:
    card_rows = [card_row(card) for card in cards]
    connection.executemany(
        """
        INSERT INTO cards (
          id,
          locale,
          upstream_id,
          set_id,
          set_name,
          card_number,
          name,
          rarity,
          image_url,
          image_url_low,
          equivalence_key,
          hp,
          types
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          locale = excluded.locale,
          upstream_id = excluded.upstream_id,
          set_id = excluded.set_id,
          set_name = excluded.set_name,
          card_number = excluded.card_number,
          name = excluded.name,
          rarity = excluded.rarity,
          image_url = excluded.image_url,
          image_url_low = excluded.image_url_low,
          equivalence_key = excluded.equivalence_key,
          hp = excluded.hp,
          types = excluded.types;
        """,
        card_rows,
    )
    connection.executemany(
        """
        INSERT INTO card_equivalents (
          card_id,
          equivalence_key,
          upstream_source,
          upstream_id,
          locale,
          set_id,
          local_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(card_id) DO UPDATE SET
          equivalence_key = excluded.equivalence_key,
          upstream_source = excluded.upstream_source,
          upstream_id = excluded.upstream_id,
          locale = excluded.locale,
          set_id = excluded.set_id,
          local_id = excluded.local_id;
        """,
        [
            (
                row[0],
                row[10],
                card.get("upstream_source", "tcgdex"),
                row[2],
                row[1],
                row[3],
                row[5],
            )
            for row, card in zip(card_rows, cards, strict=True)
        ],
    )


def copy_seed_embeddings(output_db: Path, seed_db: Path, cards: list[dict[str, Any]]) -> set[str]:
    if not seed_db.exists():
        raise RuntimeError(f"Seed embeddings db not found: {seed_db}")
    if seed_db.resolve() == output_db.resolve():
        raise RuntimeError("Seed embeddings db must be different from output db")

    card_ids = [str(card["id"]) for card in cards]
    if not card_ids:
        return set()

    with sqlite3.connect(output_db) as connection:
        connection.execute("PRAGMA foreign_keys=ON;")
        connection.execute("ATTACH DATABASE ? AS seed;", (str(seed_db),))
        try:
            reused_ids: set[str] = set()
            for start in range(0, len(card_ids), 500):
                chunk = card_ids[start : start + 500]
                placeholders = ",".join("?" for _ in chunk)
                rows = connection.execute(
                    f"""
                    SELECT card_id
                    FROM seed.embeddings
                    WHERE card_id IN ({placeholders})
                      AND model_name = ?
                      AND dim = ?
                    GROUP BY card_id
                    HAVING COUNT(*) = ?
                       AND COUNT(DISTINCT variant_idx) = ?;
                    """,
                    [*chunk, MODEL_NAME, EXPECTED_DIM, VARIANT_K, VARIANT_K],
                ).fetchall()
                reused_ids.update(str(row[0]) for row in rows)

            reused_cards = [card for card in cards if str(card["id"]) in reused_ids]
            insert_card_metadata(connection, reused_cards)
            for start in range(0, len(reused_ids), 500):
                chunk = list(sorted(reused_ids))[start : start + 500]
                placeholders = ",".join("?" for _ in chunk)
                connection.execute(
                    f"""
                    INSERT INTO embeddings (card_id, model_name, variant_idx, variant_tag, dim, vector_blob)
                    SELECT card_id, model_name, variant_idx, variant_tag, dim, vector_blob
                    FROM seed.embeddings
                    WHERE card_id IN ({placeholders})
                      AND model_name = ?
                      AND dim = ?;
                    """,
                    [*chunk, MODEL_NAME, EXPECTED_DIM],
                )
            connection.commit()
            print(f"reused {len(reused_ids)} cards × {VARIANT_K} variants from seed db")
            return reused_ids
        finally:
            connection.execute("DETACH DATABASE seed;")


def load_onnx_session(model_path: Path) -> tuple[ort.InferenceSession, str, int, float]:
    started = time.perf_counter()
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    output_shape = session.get_outputs()[0].shape
    if len(output_shape) < 2 or int(output_shape[-1]) != EXPECTED_DIM:
        raise RuntimeError(f"Unexpected ONNX output shape: {output_shape}")
    return session, input_name, EXPECTED_DIM, time.perf_counter() - started


def insert_new_embeddings(
    output_db: Path,
    records: list[DownloadedCard],
    *,
    model_path: Path,
) -> tuple[int, float, float]:
    if not records:
        return 0, 0.0, 0.0

    session, input_name, output_dim, model_load_seconds = load_onnx_session(model_path)
    started = time.perf_counter()

    with sqlite3.connect(output_db) as connection:
        init_db(connection)
        embedding_rows: list[tuple[str, str, int, str, int, bytes]] = []
        for record in records:
            card_id: str = record.card["id"]
            try:
                base_image = base_pil_for_card(record.image_path)
            except Exception as exc:
                raise RuntimeError(f"Failed to load base image for {card_id}: {exc}") from exc

            for variant_idx, variant_tag in enumerate(VARIANT_TAGS):
                rng = random.Random(card_variant_seed(card_id, variant_idx))
                variant_image = render_variant(base_image, variant_idx, rng)
                input_tensor = normalize_pil_to_nchw(variant_image)
                outputs = session.run(None, {input_name: input_tensor})
                vector = np.asarray(outputs[0][0], dtype=np.float32)
                if vector.ndim != 1 or vector.shape[0] != output_dim:
                    raise RuntimeError(
                        f"Unexpected embedding vector shape for {card_id} variant={variant_tag}: {vector.shape}"
                    )
                if not np.isfinite(vector).all():
                    raise RuntimeError(f"Non-finite embedding values for {card_id} variant={variant_tag}")
                vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
                embedding_rows.append(
                    (
                        card_id,
                        MODEL_NAME,
                        variant_idx,
                        variant_tag,
                        output_dim,
                        np.asarray(vector, dtype="<f4").tobytes(),
                    )
                )
        insert_card_metadata(connection, [record.card for record in records])
        connection.executemany(
            """
            INSERT INTO embeddings (card_id, model_name, variant_idx, variant_tag, dim, vector_blob)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(card_id, model_name, variant_idx) DO NOTHING;
            """,
            embedding_rows,
        )
        connection.commit()
    inserted_cards = len(records)
    inserted_vectors = len(embedding_rows)
    print(f"embedded {inserted_cards} cards × {VARIANT_K} variants = {inserted_vectors} vectors")
    return inserted_cards, model_load_seconds, time.perf_counter() - started


def build_embeddings_db(
    output_db: Path,
    *,
    model_path: Path,
    locales: list[str],
    image_cache_dir: Path,
    download_workers: int,
    allow_web_image_fallback: bool,
    allow_missing_images: bool,
    limit: int | None,
    min_row_count: int,
    seed_db: Path | None,
    pokemontcgio_api_key: str | None,
    summary_json: Path | None,
    missing_images_json: Path | None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "model_name": MODEL_NAME,
        "model_path": str(model_path),
        "locales": locales,
    }
    started = time.perf_counter()

    cards, listed_counts = fetch_all_card_records(locales, limit=limit)
    if not cards:
        raise RuntimeError("No cards were returned from the API")

    supplementary_cards = fetch_supplementary_pokemontcgio_cards(
        cards, api_key=pokemontcgio_api_key,
    )
    cards.extend(supplementary_cards)
    remote_card_count = len(cards)

    seed_carried_forward_cards = 0
    if seed_db is not None:
        known_card_ids = {str(card["id"]) for card in cards}
        for seed_card in load_seed_card_records(seed_db, locales):
            if str(seed_card["id"]) in known_card_ids:
                continue
            cards.append(seed_card)
            known_card_ids.add(str(seed_card["id"]))
            seed_carried_forward_cards += 1

    summary["total_remote_cards"] = remote_card_count
    summary["total_candidate_cards"] = len(cards)
    summary["listed_counts"] = listed_counts
    summary["supplementary_pokemontcgio_cards"] = len(supplementary_cards)
    summary["seed_carried_forward_cards"] = seed_carried_forward_cards

    detailed_counts: dict[str, int] = {}
    for card in cards:
        detailed_counts[card["locale"]] = detailed_counts.get(card["locale"], 0) + 1
    summary["detailed_counts"] = detailed_counts

    eligible_cards = [card for card in cards if is_scan_eligible(card)]
    excluded_cards = [card for card in cards if not is_scan_eligible(card)]
    excluded_by_series: dict[str, int] = {}
    for card in excluded_cards:
        series_id = str(card.get("set_series_id") or "unknown")
        excluded_by_series[series_id] = excluded_by_series.get(series_id, 0) + 1
    eligible_counts: dict[str, int] = {}
    for card in eligible_cards:
        eligible_counts[card["locale"]] = eligible_counts.get(card["locale"], 0) + 1
    summary["scan_eligible_cards"] = len(eligible_cards)
    summary["excluded_cards"] = len(excluded_cards)
    summary["excluded_by_series"] = excluded_by_series

    preflight_skipped_cards, preflight_image_sources = preflight_image_urls(
        eligible_cards,
        image_cache_dir,
        allow_web_image_fallback=allow_web_image_fallback,
    )
    if preflight_skipped_cards and not allow_missing_images:
        skipped_by_locale, skipped_reasons = skipped_cards_by_locale_and_reason(preflight_skipped_cards)
        missing_report = build_missing_images_report(
            preflight_skipped_cards,
            eligible_cards,
            image_cache_dir=image_cache_dir,
            excluded_cards=excluded_cards,
        )
        write_json_file(missing_images_json, missing_report)
        summary.update(
            {
                "status": "blocked_missing_public_image_urls",
                "duration_seconds": round(time.perf_counter() - started, 3),
                "processed_cards": 0,
                "reused_seed_cards": 0,
                "skipped_cards": len(preflight_skipped_cards),
                "skipped_by_locale": skipped_by_locale,
                "skipped_reasons": skipped_reasons,
                "missing_images_report": str(missing_images_json) if missing_images_json is not None else None,
                "allow_web_image_fallback": allow_web_image_fallback,
                "allow_missing_images": allow_missing_images,
                "output_db": str(output_db),
            }
        )
        write_json_file(summary_json, summary)
        raise RuntimeError(
            "Refusing to build a partial scan corpus: "
            f"{len(preflight_skipped_cards)} scan-eligible cards are missing public HTTPS image URLs. "
            "Run the image-gap resolver, commit/update image-fallbacks.json, then rerun. "
            + json.dumps(missing_report["cards"][:20], indent=2)
        )

    output_db.parent.mkdir(parents=True, exist_ok=True)
    if output_db.exists():
        output_db.unlink()
    with sqlite3.connect(output_db) as connection:
        init_db(connection)
        connection.commit()
    print("initialized fresh embeddings db")

    reused_card_ids: set[str] = set()
    if seed_db is not None:
        reused_card_ids = copy_seed_embeddings(output_db, seed_db, eligible_cards)

    cards_to_embed = [card for card in eligible_cards if str(card["id"]) not in reused_card_ids]

    ready_records, skipped_cards, download_seconds, image_sources = ensure_images(
        cards_to_embed,
        image_cache_dir,
        download_workers=download_workers,
        allow_web_image_fallback=allow_web_image_fallback,
    )
    if skipped_cards and not allow_missing_images:
        skipped_by_locale, skipped_reasons = skipped_cards_by_locale_and_reason(skipped_cards)
        missing_report = build_missing_images_report(
            skipped_cards,
            eligible_cards,
            image_cache_dir=image_cache_dir,
            excluded_cards=excluded_cards,
        )
        write_json_file(missing_images_json, missing_report)
        summary.update(
            {
                "status": "blocked_missing_scan_images",
                "download_seconds": round(download_seconds, 3),
                "duration_seconds": round(time.perf_counter() - started, 3),
                "processed_cards": len(ready_records),
                "reused_seed_cards": len(reused_card_ids),
                "skipped_cards": len(skipped_cards),
                "skipped_by_locale": skipped_by_locale,
                "skipped_reasons": skipped_reasons,
                "missing_images_report": str(missing_images_json) if missing_images_json is not None else None,
                "allow_web_image_fallback": allow_web_image_fallback,
                "allow_missing_images": allow_missing_images,
                "output_db": str(output_db),
            }
        )
        write_json_file(summary_json, summary)
        raise RuntimeError(
            "Refusing to build a partial scan corpus: "
            f"{len(skipped_cards)} scan-eligible cards are missing verified image bytes. "
            "No release was published; the app will keep using the previous embeddings-latest release. "
            "Use the missing-images report to seed or resolve real card images, then rerun. "
            + json.dumps(missing_report["cards"][:20], indent=2)
        )
    inserted_count, model_load_seconds, inference_seconds = insert_new_embeddings(
        output_db,
        ready_records,
        model_path=model_path,
    )

    counts = validate_embeddings_db(output_db, min_row_count=min_row_count)
    with sqlite3.connect(output_db) as connection:
        model_groups = inspect_model_contract(connection)
        embedded_counts = locale_row_counts(connection)
    if model_groups != [(MODEL_NAME, EXPECTED_DIM, counts[1])]:
        raise RuntimeError(f"Unexpected model groups in embeddings db: {model_groups}")

    int8_row_count, int8_total_bytes = insert_int8_embeddings(sqlite3.connect(output_db))
    int8_validation = validate_int8_quantization(sqlite3.connect(output_db))

    skipped_by_locale: dict[str, int] = {}
    skipped_reasons: dict[str, int] = {}
    skipped_reason_examples: list[dict[str, str]] = []
    for skipped in skipped_cards:
        skipped_by_locale[skipped.locale] = skipped_by_locale.get(skipped.locale, 0) + 1
        skipped_reasons[skipped.reason] = skipped_reasons.get(skipped.reason, 0) + 1
        if len(skipped_reason_examples) < 20:
            example = {
                "card_id": skipped.card_id,
                "locale": skipped.locale,
                "reason": skipped.reason,
            }
            if skipped.detail:
                example["detail"] = skipped.detail
            skipped_reason_examples.append(example)

    image_source_counts: dict[str, int] = {}
    image_source_examples: list[dict[str, str]] = []
    all_image_sources = dict(preflight_image_sources)
    all_image_sources.update(image_sources)
    for card_id, source in sorted(all_image_sources.items()):
        image_source_counts[source] = image_source_counts.get(source, 0) + 1
        if len(image_source_examples) < 20:
            image_source_examples.append({"card_id": card_id, "source": source})

    summary.update(
        {
            "download_seconds": round(download_seconds, 3),
            "model_load_seconds": round(model_load_seconds, 3),
            "inference_and_sqlite_seconds": round(inference_seconds, 3),
            "duration_seconds": round(time.perf_counter() - started, 3),
            "cards_count": counts[0],
            "embeddings_count": counts[1],
            "variants_per_card": VARIANT_K,
            "variant_tags": list(VARIANT_TAGS),
            "user_version": DB_USER_VERSION,
            "expected_dim": EXPECTED_DIM,
            "processed_cards": len(ready_records),
            "inserted_embeddings": inserted_count,
            "reused_seed_cards": len(reused_card_ids),
            "per_locale_embedded_count": embedded_counts,
            "per_locale_skipped_count": skipped_by_locale,
            "skipped_cards": len(skipped_cards),
            "skipped_by_locale": skipped_by_locale,
            "skipped_reasons": skipped_reasons,
            "skipped_examples": skipped_reason_examples,
            "image_source_counts": image_source_counts,
            "image_source_examples": image_source_examples,
            "allow_web_image_fallback": allow_web_image_fallback,
            "allow_missing_images": allow_missing_images,
            "output_db": str(output_db),
            "model_groups": model_groups,
            "embedding_diagnostics": counts[2],
            "int8_row_count": int8_row_count,
            "int8_total_bytes": int8_total_bytes,
            "int8_validation": int8_validation,
        }
    )

    per_locale_listed_count = {locale: int(listed_counts.get(locale, 0)) for locale in locales}
    per_locale_detailed_count = {locale: int(detailed_counts.get(locale, 0)) for locale in locales}
    per_locale_scan_eligible_count = {locale: int(eligible_counts.get(locale, 0)) for locale in locales}
    per_locale_embedded_count = {locale: int(embedded_counts.get(locale, 0)) for locale in locales}
    per_locale_skipped_count = {locale: int(skipped_by_locale.get(locale, 0)) for locale in locales}
    if any(
        per_locale_embedded_count[locale] + per_locale_skipped_count[locale] != per_locale_scan_eligible_count[locale]
        for locale in locales
    ):
        raise RuntimeError(
            "Per-locale embedded/skipped totals do not reconcile with scan-eligible counts: "
            + json.dumps(
                {
                    "detailed": per_locale_detailed_count,
                    "scan_eligible": per_locale_scan_eligible_count,
                    "embedded": per_locale_embedded_count,
                    "skipped": per_locale_skipped_count,
                },
                indent=2,
            )
        )
    summary["per_locale_listed_count"] = per_locale_listed_count
    summary["per_locale_detailed_count"] = per_locale_detailed_count
    summary["per_locale_scan_eligible_count"] = per_locale_scan_eligible_count
    summary["per_locale_embedded_count"] = per_locale_embedded_count
    summary["per_locale_skipped_count"] = per_locale_skipped_count

    if summary_json is not None:
        write_json_file(summary_json, summary)

    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a locale-first SQLite embedding database from TCGdex card art.")
    parser.add_argument("--output-db", "--output", dest="output_db", default="embeddings.db")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Path to the ONNX card embedder model")
    parser.add_argument("--image-cache-dir", required=True, help="Persistent cache directory for downloaded card art")
    parser.add_argument("--summary-json", help="Optional JSON build summary output path")
    parser.add_argument("--missing-images-json", help="Optional missing scan-image report output path")
    parser.add_argument(
        "--rebuild-int8-only",
        action="store_true",
        help="Recreate only embeddings_int8 from existing embeddings rows; skips API, image downloads, and ONNX inference.",
    )
    parser.add_argument("--locales", default="en", help="Comma-separated TCGdex locales")
    parser.add_argument("--download-workers", type=int, default=16)
    parser.add_argument(
        "--allow-web-image-fallback",
        action="store_true",
        help="Use best-effort public web image search as a last-resort source for missing card art.",
    )
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Allow scan-eligible cards without verified image bytes to be skipped. Local debugging only.",
    )
    parser.add_argument("--limit", type=int, help="Optional card limit for local verification")
    parser.add_argument("--min-row-count", type=int, default=1000)
    parser.add_argument(
        "--seed-db",
        help="Optional existing embeddings.db to reuse float embeddings for unchanged cards; only missing cards are embedded.",
    )
    parser.add_argument(
        "--pokemontcgio-api-key",
        default=os.environ.get("POKEMONTCG_API_KEY", ""),
        help="PokemonTCG.io API key for supplementary card fetch. Defaults to POKEMONTCG_API_KEY env var.",
    )
    parser.add_argument(
        "--detail-cache",
        default="build/tcgdex-detail-cache.jsonl",
        help="Path to local card-detail response cache (avoids re-fetching on reruns)",
    )
    args = parser.parse_args()

    if args.rebuild_int8_only:
        rebuild_int8_embeddings(
            Path(args.output_db).resolve(),
            summary_json=Path(args.summary_json).resolve() if args.summary_json else None,
        )
        return 0

    model_path = Path(args.model_path).resolve()
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    if args.detail_cache:
        set_detail_cache_path(Path(args.detail_cache).resolve())

    build_embeddings_db(
        Path(args.output_db).resolve(),
        model_path=model_path,
        locales=parse_locales(args.locales),
        image_cache_dir=Path(args.image_cache_dir).resolve(),
        download_workers=args.download_workers,
        allow_web_image_fallback=args.allow_web_image_fallback,
        allow_missing_images=args.allow_missing_images,
        limit=args.limit,
        min_row_count=args.min_row_count,
        seed_db=Path(args.seed_db).resolve() if args.seed_db else None,
        pokemontcgio_api_key=args.pokemontcgio_api_key or None,
        summary_json=Path(args.summary_json).resolve() if args.summary_json else None,
        missing_images_json=Path(args.missing_images_json).resolve() if args.missing_images_json else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
