#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from tcgdex_api import download_binary, fetch_all_card_records, parse_locales, sanitize_card_id, set_detail_cache_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a multilingual TCGdex training manifest and image cache.")
    parser.add_argument("--output", default="training/data/full/manifest.jsonl")
    parser.add_argument("--image-dir", default="training/data/full/images")
    parser.add_argument("--locales", default="en,ja,fr,de,it,es", help="Comma-separated TCGdex locales")
    parser.add_argument("--download-workers", type=int, default=16)
    parser.add_argument("--limit", type=int, help="Optional combined card limit for local verification")
    parser.add_argument("--summary-json", help="Optional JSON summary output path")
    parser.add_argument(
        "--detail-cache",
        default="build/tcgdex-detail-cache.jsonl",
        help="Path to local card-detail response cache (avoids re-fetching on reruns)",
    )
    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    image_dir = Path(args.image_dir).resolve()
    image_dir.mkdir(parents=True, exist_ok=True)

    if args.detail_cache:
        set_detail_cache_path(Path(args.detail_cache).resolve())

    locales = parse_locales(args.locales)
    cards, listed_counts = fetch_all_card_records(locales, limit=args.limit)
    if not cards:
        raise SystemExit("No cards returned from TCGdex")

    rows: list[dict[str, str]] = []
    detailed_counts = Counter(card["locale"] for card in cards)
    written_counts: Counter[str] = Counter()
    skipped_counts: Counter[str] = Counter()
    skipped_reasons: Counter[str] = Counter()
    skipped_examples: list[dict[str, str]] = []
    pending_downloads: list[tuple[dict[str, str], Path, str]] = []

    for card in cards:
        image_url = str(card.get("image_url") or "").strip()
        if not image_url:
            skipped_counts[card["locale"]] += 1
            skipped_reasons["missing_image_url"] += 1
            continue
        image_path = image_dir / f"{sanitize_card_id(card['id'])}.img"
        if not image_path.exists() or image_path.stat().st_size == 0:
            pending_downloads.append((card, image_path, image_url))

    failed_downloads: set[str] = set()
    if pending_downloads:
        print(f"downloading {len(pending_downloads)} new manifest images to {image_dir}")
        card_by_id = {str(card["id"]): card for card in cards}

        def task(item: tuple[dict[str, str], Path, str]) -> tuple[str, str | None]:
            card, image_path, image_url = item
            try:
                download_binary(image_url, image_path)
                return str(card["id"]), None
            except Exception as exc:  # pragma: no cover - network failure path
                return str(card["id"]), str(exc)

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.download_workers) as executor:
            futures = [executor.submit(task, item) for item in pending_downloads]
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                card_id, error = future.result()
                completed += 1
                if error is not None:
                    failed_downloads.add(card_id)
                    card = card_by_id[card_id]
                    skipped_counts[card["locale"]] += 1
                    skipped_reasons["download_failed"] += 1
                    if len(skipped_examples) < 20:
                        skipped_examples.append(
                            {
                                "card_id": card_id,
                                "locale": card["locale"],
                                "reason": "download_failed",
                                "detail": error,
                            }
                        )
                if completed % 250 == 0 or completed == len(futures):
                    print(f"downloaded {completed}/{len(futures)} manifest images")
    else:
        print(f"using cached manifest images from {image_dir}")

    for card in cards:
        image_url = str(card.get("image_url") or "").strip()
        if not image_url or str(card["id"]) in failed_downloads:
            continue
        image_path = image_dir / f"{sanitize_card_id(card['id'])}.img"
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as exc:
            skipped_counts[card["locale"]] += 1
            skipped_reasons["invalid_image"] += 1
            if len(skipped_examples) < 20:
                skipped_examples.append(
                    {
                        "card_id": card["id"],
                        "locale": card["locale"],
                        "reason": "invalid_image",
                        "detail": str(exc),
                    }
                )
            continue
        rows.append(
            {
                "card_id": card["id"],
                "name": card["name"],
                "set_name": card["set_name"],
                "number": card["card_number"],
                "image_path": str(image_path),
                "image_url": image_url,
                "locale": card["locale"],
                "upstream_id": card["upstream_id"],
                "set_id": card["set_id"],
                "equivalence_key": card["equivalence_key"],
                "hp": card.get("hp"),
            }
        )
        written_counts[card["locale"]] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "output": str(output_path),
        "records": len(rows),
        "locales": locales,
        "per_locale_listed_count": {locale: int(listed_counts.get(locale, 0)) for locale in locales},
        "per_locale_detailed_count": {locale: int(detailed_counts.get(locale, 0)) for locale in locales},
        "per_locale_written_count": {locale: int(written_counts.get(locale, 0)) for locale in locales},
        "per_locale_skipped_count": {locale: int(skipped_counts.get(locale, 0)) for locale in locales},
        "skipped_reasons": dict(skipped_reasons),
        "skipped_examples": skipped_examples,
    }
    if args.summary_json:
        summary_path = Path(args.summary_json).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
