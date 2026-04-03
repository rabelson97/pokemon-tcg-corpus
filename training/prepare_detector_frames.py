from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class FrameRecord:
    source_video: str
    extracted_frame: str
    output_image: str
    timestamp_seconds: float
    average_hash: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract detector-labeling frames from a stream recording."
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to a local video recording.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write extracted frames and manifest into.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=0.5,
        help="Frame sampling rate in frames per second. Default: 0.5 (one frame every 2s).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional hard cap on the number of kept frames. Default: 0 (no cap).",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        help="JPEG quality for saved frames. Default: 85.",
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=1440,
        help="Resize images so the longest edge is at most this many pixels. Default: 1440.",
    )
    parser.add_argument(
        "--dedupe-distance",
        type=int,
        default=6,
        help="Maximum average-hash Hamming distance to treat frames as duplicates. Default: 6.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep raw ffmpeg frame outputs for inspection.",
    )
    return parser.parse_args()


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required but was not found on PATH")


def average_hash(image: Image.Image, size: int = 8) -> str:
    grayscale = image.convert("L").resize((size, size))
    pixels = list(grayscale.getdata())
    avg = sum(pixels) / len(pixels)
    bits = ["1" if pixel >= avg else "0" for pixel in pixels]
    return "".join(bits)


def hamming_distance(first: str, second: str) -> int:
    return sum(a != b for a, b in zip(first, second))


def resize_for_output(image: Image.Image, max_dimension: int) -> Image.Image:
    width, height = image.size
    longest_edge = max(width, height)
    if longest_edge <= max_dimension:
        return image
    scale = max_dimension / float(longest_edge)
    resized = image.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)
    return resized


def ffmpeg_extract_frames(video_path: Path, output_dir: Path, fps: float) -> None:
    output_pattern = output_dir / "frame_%06d.jpg"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
        str(output_pattern),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    ensure_ffmpeg()

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "frames_manifest.jsonl"

    with tempfile.TemporaryDirectory(prefix="detector_frames_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        ffmpeg_extract_frames(video_path, temp_dir, args.fps)

        extracted_frames = sorted(temp_dir.glob("frame_*.jpg"))
        if not extracted_frames:
            raise SystemExit("ffmpeg did not extract any frames")

        kept_hashes: list[str] = []
        kept_records: list[FrameRecord] = []

        for index, frame_path in enumerate(extracted_frames, start=1):
            image = Image.open(frame_path).convert("RGB")
            resized = resize_for_output(image, args.max_dimension)
            frame_hash = average_hash(resized)

            is_duplicate = any(
                hamming_distance(frame_hash, existing_hash) <= args.dedupe_distance
                for existing_hash in kept_hashes
            )
            if is_duplicate:
                continue

            output_name = f"{video_path.stem}_{len(kept_records) + 1:05d}.jpg"
            output_path = images_dir / output_name
            resized.save(output_path, format="JPEG", quality=args.jpeg_quality, optimize=True)

            record = FrameRecord(
                source_video=str(video_path),
                extracted_frame=str(frame_path),
                output_image=str(output_path),
                timestamp_seconds=(index - 1) / args.fps,
                average_hash=frame_hash,
            )
            kept_records.append(record)
            kept_hashes.append(frame_hash)

            if args.max_frames > 0 and len(kept_records) >= args.max_frames:
                break

        with manifest_path.open("w", encoding="utf-8") as handle:
            for record in kept_records:
                handle.write(json.dumps(asdict(record)) + "\n")

        if args.keep_temp:
            preserved_dir = output_dir / "raw_frames"
            preserved_dir.mkdir(parents=True, exist_ok=True)
            for frame_path in extracted_frames:
                shutil.copy2(frame_path, preserved_dir / frame_path.name)

    print(
        json.dumps(
            {
                "video": str(video_path),
                "output_dir": str(output_dir),
                "kept_frames": len(kept_records),
                "images_dir": str(images_dir),
                "labels_dir": str(labels_dir),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
