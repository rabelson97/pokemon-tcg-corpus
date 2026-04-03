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
import open_clip
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from pokemontcg_api import download_binary, fetch_all_cards, sanitize_card_id


DEFAULT_MODEL_NAME = "ViT-B-32"
DEFAULT_PRETRAINED = "laion2b_s34b_b79k"
DB_USER_VERSION = 1


@dataclass(frozen=True)
class DownloadedCard:
    card: dict[str, Any]
    image_path: Path


class LetterboxSquare:
    def __init__(self, image_size: int, fill: tuple[int, int, int] = (255, 255, 255)) -> None:
        self.image_size = image_size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        scale = min(self.image_size / width, self.image_size / height)
        resized = image.resize(
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            resample=Image.Resampling.BICUBIC,
        )
        canvas = Image.new("RGB", (self.image_size, self.image_size), color=self.fill)
        offset_x = (self.image_size - resized.size[0]) // 2
        offset_y = (self.image_size - resized.size[1]) // 2
        canvas.paste(resized, (offset_x, offset_y))
        return canvas


class CardImageDataset(Dataset):
    def __init__(self, records: list[DownloadedCard], image_size: int, preprocess: Any) -> None:
        self.records = records
        self.preprocess = preprocess
        self.letterbox = LetterboxSquare(image_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = Image.open(self.records[index].image_path).convert("RGB")
        image = self.letterbox(image)
        return self.preprocess(image), index


def resolve_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_model_image_size(model: torch.nn.Module) -> int:
    image_size = getattr(model.visual, "image_size", None)
    if isinstance(image_size, tuple):
        return int(image_size[0])
    if isinstance(image_size, int):
        return image_size
    raise RuntimeError("Could not determine image size from open_clip model")


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

    suspicious_cosines = [
        sample for sample in diagnostics["cosine_samples"] if abs(float(sample["cosine"])) > 0.9999
    ]
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


def load_model(model_name: str, pretrained: str, device: torch.device) -> tuple[torch.nn.Module, Any, int, float]:
    started = time.perf_counter()
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )
    model.eval()
    return model, preprocess, get_model_image_size(model), time.perf_counter() - started


def insert_new_embeddings(
    output_db: Path,
    records: list[DownloadedCard],
    *,
    model_name: str,
    pretrained: str,
    batch_size: int,
    num_workers: int,
    device_name: str,
) -> tuple[int, float, float]:
    if not records:
        return 0, 0.0, 0.0

    device = resolve_device(device_name)
    print(f"using device={device}")
    model, preprocess, image_size, model_load_seconds = load_model(model_name, pretrained, device)

    dataset = CardImageDataset(records, image_size, preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    started = time.perf_counter()
    inserted = 0
    db_model_name = f"open_clip:{model_name}:{pretrained}"

    with sqlite3.connect(output_db) as connection:
        init_db(connection)
        with torch.no_grad():
            for images, indexes in dataloader:
                images = images.to(device)
                vectors = model.encode_image(images)
                vectors = vectors.detach().float().cpu().numpy().astype(np.float32, copy=False)
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                vectors = vectors / np.clip(norms, 1e-12, None)
                dim = int(vectors.shape[1])

                rows = []
                for vector, record_index in zip(vectors, indexes.tolist(), strict=True):
                    record = records[record_index]
                    rows.append(
                        (
                            *card_row(record.card),
                            db_model_name,
                            dim,
                            vector.astype("<f4", copy=False).tobytes(),
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
                inserted += len(rows)
                if inserted % 1000 == 0 or inserted == len(records):
                    print(f"embedded {inserted}/{len(records)} new cards")
        connection.commit()

    return inserted, model_load_seconds, time.perf_counter() - started


def build_embeddings_db(
    output_db: Path,
    *,
    model_name: str,
    pretrained: str,
    api_key: str | None,
    base_db: Path | None,
    force_rebuild: bool,
    image_cache_dir: Path | None,
    batch_size: int,
    num_workers: int,
    download_workers: int,
    device_name: str,
    limit: int | None,
    min_row_count: int,
    summary_json: Path | None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "force_rebuild": force_rebuild,
        "model_name": model_name,
        "pretrained": pretrained,
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
        copy_base_db(base_db, output_db)
        with sqlite3.connect(output_db) as connection:
            init_db(connection)
            connection.commit()
        existing_ids = load_existing_card_ids(output_db)
        print(f"loaded base db {base_db} with existing_cards={len(existing_ids)}")
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
                model_name=model_name,
                pretrained=pretrained,
                batch_size=batch_size,
                num_workers=num_workers,
                device_name=device_name,
            )
        else:
            print("no new card ids found; skipping embedding generation")

        counts = validate_embeddings_db(output_db, min_row_count=min_row_count)
        summary.update(
            {
                "download_seconds": round(download_seconds, 3),
                "model_load_seconds": round(model_load_seconds, 3),
                "inference_and_sqlite_seconds": round(inference_seconds, 3),
                "duration_seconds": round(time.perf_counter() - started, 3),
                "cards_count": counts[0],
                "embeddings_count": counts[1],
                "user_version": DB_USER_VERSION,
                "processed_cards": len(ready_records),
                "inserted_embeddings": inserted_count,
                "skipped_cards": len(failures),
                "skipped_examples": failures[:20],
                "used_base_db": use_base,
                "base_db": str(base_db) if base_db else None,
                "output_db": str(output_db),
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
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="open_clip model name")
    parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED, help="open_clip pretrained weights tag")
    parser.add_argument("--image-cache-dir", help="Persistent cache directory for downloaded card art")
    parser.add_argument("--summary-json", help="Optional JSON build summary output path")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--download-workers", type=int, default=16)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--limit", type=int, help="Optional card limit for local verification")
    parser.add_argument("--min-row-count", type=int, default=1000)
    args = parser.parse_args()

    build_embeddings_db(
        Path(args.output_db).resolve(),
        model_name=args.model_name,
        pretrained=args.pretrained,
        api_key=os.environ.get("POKEMONTCG_API_KEY"),
        base_db=Path(args.base_db).resolve() if args.base_db else None,
        force_rebuild=args.force_rebuild,
        image_cache_dir=Path(args.image_cache_dir).resolve() if args.image_cache_dir else None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download_workers=args.download_workers,
        device_name=args.device,
        limit=args.limit,
        min_row_count=args.min_row_count,
        summary_json=Path(args.summary_json).resolve() if args.summary_json else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
