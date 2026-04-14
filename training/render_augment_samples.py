#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image

from common import load_manifest, make_stream_artifact_image, resolve_stream_augment_profile


def save_preview(image: Image.Image, destination: Path) -> None:
    image.convert("RGB").save(destination, format="JPEG", quality=90, optimize=True)


def save_contact_sheet(images: list[Image.Image], destination: Path) -> None:
    thumb_size = (220, 302)
    gap = 12
    canvas_width = (thumb_size[0] * len(images)) + (gap * (len(images) + 1))
    canvas_height = thumb_size[1] + (gap * 2)
    canvas = Image.new("RGB", (canvas_width, canvas_height), (245, 245, 245))
    for index, image in enumerate(images):
        thumb = image.copy()
        thumb.thumbnail(thumb_size, Image.Resampling.BICUBIC)
        x = gap + index * (thumb_size[0] + gap) + (thumb_size[0] - thumb.width) // 2
        y = gap + (thumb_size[1] - thumb.height) // 2
        canvas.paste(thumb, (x, y))
    save_preview(canvas, destination)


def write_gallery(output_dir: Path, cards: list[tuple[str, Path]]) -> None:
    rows = []
    for label, sheet_path in cards:
        rel_path = sheet_path.relative_to(output_dir).as_posix()
        rows.append(f'<section><h2>{label}</h2><img src="{rel_path}" alt="{label}"></section>')
    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>Augmentation Samples</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; background: #fafafa; color: #111; }}
    section {{ margin-bottom: 28px; }}
    h2 {{ font-size: 16px; margin-bottom: 10px; }}
    img {{ max-width: 100%; border: 1px solid #ddd; background: white; }}
  </style>
</head>
<body>
  <h1>Augmentation Samples</h1>
  {'\n  '.join(rows)}
</body>
</html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render visual samples for a retrieval augmentation profile.")
    parser.add_argument("--manifest", default="training/data/full/manifest.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--augment-profile", default="baseline")
    parser.add_argument("--sample-count", type=int, default=6)
    parser.add_argument("--variants-per-card", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    records = load_manifest(args.manifest)
    if not records:
        raise SystemExit("Manifest is empty")

    profile = resolve_stream_augment_profile(args.augment_profile)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = records[: args.sample_count]
    gallery_cards: list[tuple[str, Path]] = []
    for card_index, record in enumerate(selected, start=1):
        card_dir = output_dir / f"{card_index:02d}_{record.card_id.replace(':', '_')}"
        card_dir.mkdir(parents=True, exist_ok=True)
        original = Image.open(record.image_path).convert("RGB")
        save_preview(original, card_dir / "original.jpg")
        contact_images = [original]
        for variant_index in range(1, args.variants_per_card + 1):
            variant_rng = random.Random(rng.randint(0, 1_000_000_000))
            augmented = make_stream_artifact_image(original.copy(), variant_rng, profile)
            contact_images.append(augmented)
            save_preview(augmented, card_dir / f"variant_{variant_index:02d}.jpg")
        sheet_path = card_dir / "sheet.jpg"
        save_contact_sheet(contact_images, sheet_path)
        gallery_cards.append((record.card_id, sheet_path))

    write_gallery(output_dir, gallery_cards)

    print(f"rendered profile={profile.name} samples to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
