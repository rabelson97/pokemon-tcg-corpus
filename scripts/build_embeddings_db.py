#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image

from pokemontcg_api import download_binary, fetch_all_cards, sanitize_card_id


DB_USER_VERSION = 1
EXPECTED_DIM = 256
EMBED_IMAGE_SIZE = 224
EMBED_CROP_INSET_RATIO = 0.08
EMBED_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
EMBED_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "card_embedder.onnx"
MODEL_NAME = "cardhawk:card_embedder.onnx"


@dataclass(frozen=True)
class DownloadedCard:
    card: dict[str, Any]
    image_path: Path


def card_row(card: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    set_info = card.get("set") or {}
    set_code = set_info.get("ptcgoCode") or set_info.get("id") or ""
    set_name = set_info.get("name") or ""
    number = card.get("number") or ""
    rarity = card.get("rarity") or "Unknown"
    return (
        card["id"],
        card["name"],
        set_code,
        set_name,
        number,
        rarity,
    )


def image_url_for_card(card: dict[str, Any]) -> str:
    images = card.get("images") or {}
    image_url = images.get("large") or images.get("small")
    if not image_url:
        raise RuntimeError(f"Card {card['id']} is missing both images.large and images.small")
    return image_url


def crop_inset_for_embedder(image: Image.Image) -> Image.Image:
    width, height = image.size
    inset_x = int(width * EMBED_CROP_INSET_RATIO)
    inset_y = int(height * EMBED_CROP_INSET_RATIO)
    left = min(inset_x, max(0, width - 1))
    top = min(inset_y, max(0, height - 1))
    right = max(left + 1, width - inset_x)
    bottom = max(top + 1, height - inset_y)
    return image.crop((left, top, right, bottom))


def preprocess_for_embedder(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = crop_inset_for_embedder(image)
    image = image.resize((EMBED_IMAGE_SIZE, EMBED_IMAGE_SIZE), resample=Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    normalized = (array - EMBED_MEAN) / EMBED_STD
    chw = np.transpose(normalized, (2, 0, 1))
    return np.expand_dims(chw.astype(np.float32, copy=False), axis=0)


def init_db(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=DELETE;")
    connection.execute("PRAGMA synchronous=NORMAL;")
    connection.execute("PRAGMA foreign_keys=ON;")
    connection.execute(f"PRAGMA user_version={DB_USER_VERSION};")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS cards (
          id TEXT PRIMARY KEY,
          name TEXT NOT NULL,
          set_code TEXT NOT NULL,
          set_name TEXT NOT NULL,
          card_number TEXT NOT NULL,
          rarity TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS embeddings (
          card_id TEXT PRIMARY KEY REFERENCES cards(id) ON DELETE CASCADE,
          model_name TEXT NOT NULL,
          dim INTEGER NOT NULL,
          vector_blob BLOB NOT NULL
        );
        """
    )


def sample_embedding_diagnostics(
    db_path: Path,
    *,
    sample_size: int = 16,
    print_rows: int = 5,
) -> dict[str, Any]:
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT e.card_id, c.name, e.dim, e.vector_blob
            FROM embeddings e
            JOIN cards c ON c.id = e.card_id
            ORDER BY e.card_id
            LIMIT ?;
            """,
            (sample_size,),
        ).fetchall()

    diagnostics: list[dict[str, Any]] = []
    hashes: list[str] = []
    cosine_samples: list[dict[str, Any]] = []
    decoded_vectors: list[tuple[str, str, np.ndarray]] = []

    for card_id, name, dim, blob in rows:
        vector = np.frombuffer(blob, dtype="<f4")
        vector_hash = hashlib.sha256(blob).hexdigest()
        decoded_vectors.append((str(card_id), str(name), vector))
        hashes.append(vector_hash)
        diagnostics.append(
            {
                "card_id": str(card_id),
                "name": str(name),
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

    summary = {
        "sample_count": len(rows),
        "distinct_hashes": len(set(hashes)),
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
        if card_count != embedding_count:
            raise RuntimeError(f"cards ({card_count}) and embeddings ({embedding_count}) counts do not match")

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


def load_existing_card_ids(db_path: Path) -> set[str]:
    with sqlite3.connect(db_path) as connection:
        init_db(connection)
        rows = connection.execute("SELECT id FROM cards;").fetchall()
    return {str(row[0]) for row in rows}


def copy_base_db(base_db: Path, output_db: Path) -> None:
    output_db.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_db, output_db)


def ensure_images(
    cards: list[dict[str, Any]],
    cache_dir: Path,
    *,
    download_workers: int,
) -> tuple[list[DownloadedCard], list[str], float]:
    started = time.perf_counter()
    cache_dir.mkdir(parents=True, exist_ok=True)
    image_paths = [cache_dir / f"{sanitize_card_id(card['id'])}.img" for card in cards]
    failures: list[str] = []

    pending: list[tuple[dict[str, Any], Path]] = []
    for card, image_path in zip(cards, image_paths, strict=True):
        if image_path.exists() and image_path.stat().st_size > 0:
            continue
        pending.append((card, image_path))

    if pending:
        print(f"downloading {len(pending)} new images to {cache_dir}")

        def task(item: tuple[dict[str, Any], Path]) -> tuple[str, str | None]:
            card, image_path = item
            try:
                download_binary(image_url_for_card(card), image_path)
                return card["id"], None
            except Exception as exc:  # pragma: no cover - network failure path
                return card["id"], str(exc)

        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=download_workers) as executor:
            futures = [executor.submit(task, item) for item in pending]
            for future in concurrent.futures.as_completed(futures):
                card_id, error = future.result()
                completed += 1
                if error is not None:
                    failures.append(f"{card_id}: download failed: {error}")
                if completed % 250 == 0 or completed == len(pending):
                    print(f"downloaded {completed}/{len(pending)} pending images")
    else:
        print(f"using cached images from {cache_dir}")

    ready: list[DownloadedCard] = []
    for card, image_path in zip(cards, image_paths, strict=True):
        if not image_path.exists() or image_path.stat().st_size == 0:
            failures.append(f"{card['id']}: image missing after download")
            continue
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as exc:
            failures.append(f"{card['id']}: invalid image: {exc}")
            continue
        ready.append(DownloadedCard(card=card, image_path=image_path))

    return ready, failures, time.perf_counter() - started


def inspect_model_contract(connection: sqlite3.Connection) -> list[tuple[str, int, int]]:
    rows = connection.execute(
        "SELECT model_name, dim, COUNT(*) FROM embeddings GROUP BY model_name, dim ORDER BY model_name, dim;"
    ).fetchall()
    return [(str(model_name), int(dim), int(count)) for model_name, dim, count in rows]


def base_db_is_compatible(base_db: Path) -> bool:
    with sqlite3.connect(base_db) as connection:
        init_db(connection)
        model_groups = inspect_model_contract(connection)
    return model_groups == [(MODEL_NAME, EXPECTED_DIM, model_groups[0][2])] if model_groups else True


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
    inserted = 0

    with sqlite3.connect(output_db) as connection:
        init_db(connection)
        rows = []
        for record in records:
            input_tensor = preprocess_for_embedder(record.image_path)
            outputs = session.run(None, {input_name: input_tensor})
            vector = np.asarray(outputs[0][0], dtype=np.float32)
            if vector.ndim != 1 or vector.shape[0] != output_dim:
                raise RuntimeError(f"Unexpected embedding vector shape for {record.card['id']}: {vector.shape}")
            if not np.isfinite(vector).all():
                raise RuntimeError(f"Non-finite embedding values for {record.card['id']}")
            vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
            rows.append(
                (
                    *card_row(record.card),
                    MODEL_NAME,
                    output_dim,
                    np.asarray(vector, dtype="<f4").tobytes(),
                )
            )

        connection.executemany(
            """
            INSERT INTO cards (id, name, set_code, set_name, card_number, rarity)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO NOTHING;
            """,
            [row[:6] for row in rows],
        )
        connection.executemany(
            """
            INSERT INTO embeddings (card_id, model_name, dim, vector_blob)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(card_id) DO NOTHING;
            """,
            [(row[0], row[6], row[7], row[8]) for row in rows],
        )
        connection.commit()
        inserted = len(rows)
    print(f"embedded {inserted}/{len(records)} new cards")
    return inserted, model_load_seconds, time.perf_counter() - started


def build_embeddings_db(
    output_db: Path,
    *,
    model_path: Path,
    api_key: str | None,
    base_db: Path | None,
    force_rebuild: bool,
    image_cache_dir: Path | None,
    download_workers: int,
    limit: int | None,
    min_row_count: int,
    summary_json: Path | None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "force_rebuild": force_rebuild,
        "model_name": MODEL_NAME,
        "model_path": str(model_path),
    }
    started = time.perf_counter()

    cards = fetch_all_cards(
        api_key=api_key,
        limit=limit,
        select_fields=["id", "name", "number", "rarity", "set", "images"],
    )
    if not cards:
        raise RuntimeError("No cards were returned from the API")
    summary["total_remote_cards"] = len(cards)

    output_db.parent.mkdir(parents=True, exist_ok=True)
    use_base = bool(base_db and base_db.exists() and not force_rebuild)
    existing_ids: set[str] = set()

    if use_base:
        assert base_db is not None
        validate_embeddings_db(base_db, min_row_count=0, require_user_version=False)
        if base_db_is_compatible(base_db):
            copy_base_db(base_db, output_db)
            with sqlite3.connect(output_db) as connection:
                init_db(connection)
                connection.commit()
            existing_ids = load_existing_card_ids(output_db)
            print(f"loaded compatible base db {base_db} with existing_cards={len(existing_ids)}")
        else:
            print(f"base db {base_db} is incompatible with {MODEL_NAME}; rebuilding from scratch")
            use_base = False
            if output_db.exists():
                output_db.unlink()
            with sqlite3.connect(output_db) as connection:
                init_db(connection)
                connection.commit()
    else:
        if output_db.exists():
            output_db.unlink()
        with sqlite3.connect(output_db) as connection:
            init_db(connection)
            connection.commit()
        print("initialized fresh embeddings db")

    summary["existing_cards"] = len(existing_ids)
    missing_cards = [card for card in cards if card["id"] not in existing_ids]
    summary["new_cards"] = len(missing_cards)

    own_cache_dir = False
    if image_cache_dir is None:
        image_cache_dir = Path(tempfile.mkdtemp(prefix="pokemon-tcg-images-"))
        own_cache_dir = True

    download_seconds = 0.0
    model_load_seconds = 0.0
    inference_seconds = 0.0
    failures: list[str] = []
    inserted_count = 0
    ready_records: list[DownloadedCard] = []

    try:
        if missing_cards:
            ready_records, failures, download_seconds = ensure_images(
                missing_cards,
                image_cache_dir,
                download_workers=download_workers,
            )
            inserted_count, model_load_seconds, inference_seconds = insert_new_embeddings(
                output_db,
                ready_records,
                model_path=model_path,
            )
        else:
            print("no new card ids found; skipping embedding generation")

        counts = validate_embeddings_db(output_db, min_row_count=min_row_count)
        with sqlite3.connect(output_db) as connection:
            model_groups = inspect_model_contract(connection)
        if model_groups != [(MODEL_NAME, EXPECTED_DIM, counts[1])]:
            raise RuntimeError(f"Unexpected model groups in embeddings db: {model_groups}")
        summary.update(
            {
                "download_seconds": round(download_seconds, 3),
                "model_load_seconds": round(model_load_seconds, 3),
                "inference_and_sqlite_seconds": round(inference_seconds, 3),
                "duration_seconds": round(time.perf_counter() - started, 3),
                "cards_count": counts[0],
                "embeddings_count": counts[1],
                "user_version": DB_USER_VERSION,
                "expected_dim": EXPECTED_DIM,
                "processed_cards": len(ready_records),
                "inserted_embeddings": inserted_count,
                "skipped_cards": len(failures),
                "skipped_examples": failures[:20],
                "used_base_db": use_base,
                "base_db": str(base_db) if base_db else None,
                "output_db": str(output_db),
                "model_groups": model_groups,
                "embedding_diagnostics": counts[2],
            }
        )
    finally:
        if own_cache_dir and image_cache_dir is not None:
            shutil.rmtree(image_cache_dir, ignore_errors=True)

    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a SQLite embedding database from Pokemon card art.")
    parser.add_argument("--base-db", help="Optional existing embeddings.db used as the incremental starting point")
    parser.add_argument("--output-db", "--output", dest="output_db", default="embeddings.db")
    parser.add_argument("--force-rebuild", action="store_true", help="Ignore --base-db and rebuild from scratch")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Path to the ONNX card embedder model")
    parser.add_argument("--image-cache-dir", help="Persistent cache directory for downloaded card art")
    parser.add_argument("--summary-json", help="Optional JSON build summary output path")
    parser.add_argument("--download-workers", type=int, default=16)
    parser.add_argument("--limit", type=int, help="Optional card limit for local verification")
    parser.add_argument("--min-row-count", type=int, default=1000)
    args = parser.parse_args()

    model_path = Path(args.model_path).resolve()
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    build_embeddings_db(
        Path(args.output_db).resolve(),
        model_path=model_path,
        api_key=os.environ.get("POKEMONTCG_API_KEY"),
        base_db=Path(args.base_db).resolve() if args.base_db else None,
        force_rebuild=args.force_rebuild,
        image_cache_dir=Path(args.image_cache_dir).resolve() if args.image_cache_dir else None,
        download_workers=args.download_workers,
        limit=args.limit,
        min_row_count=args.min_row_count,
        summary_json=Path(args.summary_json).resolve() if args.summary_json else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
