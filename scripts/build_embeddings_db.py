#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from embedder_contract import EXPECTED_DIM, IMAGE_SIZE as EMBED_IMAGE_SIZE, preprocess_image_path
from tcgdex_api import download_binary, fetch_all_card_records, parse_locales, sanitize_card_id


DB_USER_VERSION = 2
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "card_embedder.onnx"
MODEL_NAME = "cardhawk:card_embedder.onnx"


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


def card_row(card: dict[str, Any]) -> tuple[str, str, str, str, str, str, str, str, str, str]:
    return (
        card["id"],
        card["locale"],
        card["upstream_id"],
        card["set_id"],
        card["set_name"],
        card["card_number"],
        card["name"],
        card["rarity"],
        card["image_url"],
        card["equivalence_key"],
    )


def image_url_for_card(card: dict[str, Any]) -> str:
    image_url = str(card.get("image_url") or "").strip()
    if not image_url:
        raise RuntimeError(f"Card {card['id']} is missing image_url")
    return image_url


def preprocess_for_embedder(image_path: Path) -> np.ndarray:
    return preprocess_image_path(image_path, image_size=EMBED_IMAGE_SIZE)


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
          image_url TEXT NOT NULL,
          equivalence_key TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS embeddings (
          card_id TEXT PRIMARY KEY REFERENCES cards(id) ON DELETE CASCADE,
          model_name TEXT NOT NULL,
          dim INTEGER NOT NULL,
          vector_blob BLOB NOT NULL
        );

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


def ensure_images(
    cards: list[dict[str, Any]],
    cache_dir: Path,
    *,
    download_workers: int,
) -> tuple[list[DownloadedCard], list[SkippedCard], float]:
    started = time.perf_counter()
    cache_dir.mkdir(parents=True, exist_ok=True)
    image_paths = [cache_dir / f"{sanitize_card_id(card['id'])}.img" for card in cards]
    skipped: list[SkippedCard] = []

    pending: list[tuple[dict[str, Any], Path]] = []
    for card, image_path in zip(cards, image_paths, strict=True):
        if not str(card.get("image_url") or "").strip():
            skipped.append(SkippedCard(card_id=card["id"], locale=card["locale"], reason="missing_image_url"))
            continue
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
                    locale = card_id.split(":")[1] if card_id.startswith("pokemon:") else "unknown"
                    skipped.append(SkippedCard(card_id=card_id, locale=locale, reason="download_failed", detail=error))
                if completed % 250 == 0 or completed == len(pending):
                    print(f"downloaded {completed}/{len(pending)} pending images")
    else:
        print(f"using cached images from {cache_dir}")

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

    return ready, skipped, time.perf_counter() - started


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
              equivalence_key
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO NOTHING;
            """,
            [row[:10] for row in rows],
        )
        connection.executemany(
            """
            INSERT INTO embeddings (card_id, model_name, dim, vector_blob)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(card_id) DO NOTHING;
            """,
            [(row[0], row[10], row[11], row[12]) for row in rows],
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
                    row[9],
                    record.card["upstream_source"],
                    row[2],
                    row[1],
                    row[3],
                    row[5],
                )
                for row, record in zip(rows, records, strict=True)
            ],
        )
        connection.commit()
    inserted = len(rows)
    print(f"embedded {inserted}/{len(records)} cards")
    return inserted, model_load_seconds, time.perf_counter() - started


def build_embeddings_db(
    output_db: Path,
    *,
    model_path: Path,
    locales: list[str],
    image_cache_dir: Path,
    download_workers: int,
    limit: int | None,
    min_row_count: int,
    summary_json: Path | None,
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
    summary["total_remote_cards"] = len(cards)
    summary["listed_counts"] = listed_counts

    detailed_counts: dict[str, int] = {}
    for card in cards:
        detailed_counts[card["locale"]] = detailed_counts.get(card["locale"], 0) + 1
    summary["detailed_counts"] = detailed_counts

    output_db.parent.mkdir(parents=True, exist_ok=True)
    if output_db.exists():
        output_db.unlink()
    with sqlite3.connect(output_db) as connection:
        init_db(connection)
        connection.commit()
    print("initialized fresh embeddings db")

    ready_records, skipped_cards, download_seconds = ensure_images(
        cards,
        image_cache_dir,
        download_workers=download_workers,
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
            "per_locale_embedded_count": embedded_counts,
            "per_locale_skipped_count": skipped_by_locale,
            "skipped_cards": len(skipped_cards),
            "skipped_by_locale": skipped_by_locale,
            "skipped_reasons": skipped_reasons,
            "skipped_examples": skipped_reason_examples,
            "output_db": str(output_db),
            "model_groups": model_groups,
            "embedding_diagnostics": counts[2],
        }
    )

    per_locale_listed_count = {locale: int(listed_counts.get(locale, 0)) for locale in locales}
    per_locale_detailed_count = {locale: int(detailed_counts.get(locale, 0)) for locale in locales}
    per_locale_embedded_count = {locale: int(embedded_counts.get(locale, 0)) for locale in locales}
    per_locale_skipped_count = {locale: int(skipped_by_locale.get(locale, 0)) for locale in locales}
    if any(
        per_locale_embedded_count[locale] + per_locale_skipped_count[locale] != per_locale_detailed_count[locale]
        for locale in locales
    ):
        raise RuntimeError(
            "Per-locale embedded/skipped totals do not reconcile with detailed counts: "
            + json.dumps(
                {
                    "detailed": per_locale_detailed_count,
                    "embedded": per_locale_embedded_count,
                    "skipped": per_locale_skipped_count,
                },
                indent=2,
            )
        )
    summary["per_locale_listed_count"] = per_locale_listed_count
    summary["per_locale_detailed_count"] = per_locale_detailed_count
    summary["per_locale_embedded_count"] = per_locale_embedded_count
    summary["per_locale_skipped_count"] = per_locale_skipped_count

    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a locale-first SQLite embedding database from TCGdex card art.")
    parser.add_argument("--output-db", "--output", dest="output_db", default="embeddings.db")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Path to the ONNX card embedder model")
    parser.add_argument("--image-cache-dir", required=True, help="Persistent cache directory for downloaded card art")
    parser.add_argument("--summary-json", help="Optional JSON build summary output path")
    parser.add_argument("--locales", default="en,ja,fr,de,it,es", help="Comma-separated TCGdex locales")
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
        locales=parse_locales(args.locales),
        image_cache_dir=Path(args.image_cache_dir).resolve(),
        download_workers=args.download_workers,
        limit=args.limit,
        min_row_count=args.min_row_count,
        summary_json=Path(args.summary_json).resolve() if args.summary_json else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
