#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "build" / "local-embeddings"
DEFAULT_CACHE_DIR = REPO_ROOT / ".cache" / "card-images"
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "card_embedder.onnx"
DEFAULT_MODEL_MANIFEST = REPO_ROOT / "models" / "card_embedder.manifest.json"


def run_python(script: Path, *args: str) -> None:
    command = [sys.executable, str(script), *args]
    print("$", " ".join(command))
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def ensure_build_dependencies() -> None:
    required_modules = {
        "numpy": "numpy",
        "PIL": "pillow",
        "onnxruntime": "onnxruntime",
    }
    missing = [
        package_name
        for module_name, package_name in required_modules.items()
        if importlib.util.find_spec(module_name) is None
    ]
    if not missing:
        return
    install_command = (
        f"{sys.executable} -m pip install -r requirements.txt"
        if Path(REPO_ROOT / "requirements.txt").exists()
        else f"{sys.executable} -m pip install " + " ".join(sorted(set(missing)))
    )
    missing_text = ", ".join(sorted(set(missing)))
    raise SystemExit(
        "Missing corpus build dependencies: "
        f"{missing_text}\n"
        "Run this first:\n"
        f"  cd {REPO_ROOT}\n"
        f"  {install_command}"
    )


def zip_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(source, arcname=source.name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Local one-command rebuild for embeddings.db using the currently promoted production embedder."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--locales", default="en,ja")
    parser.add_argument("--min-row-count", type=int, default=10000)
    parser.add_argument("--download-workers", type=int, default=16)
    parser.add_argument("--limit", type=int, help="Optional small-card local smoke test.")
    parser.add_argument("--skip-zip", action="store_true")
    parser.add_argument(
        "--detail-cache",
        default=str(REPO_ROOT / "build" / "tcgdex-detail-cache.jsonl"),
        help="Path to local card-detail response cache (shared with build_training_manifest.py)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ensure_build_dependencies()

    db_path = output_dir / "embeddings.db"
    summary_path = output_dir / "embeddings-build-summary.json"
    zip_path = output_dir / "embeddings.db.zip"

    run_python(
        REPO_ROOT / "training" / "validate_production_embedder.py",
        "--model",
        str(DEFAULT_MODEL_PATH),
        "--manifest",
        str(DEFAULT_MODEL_MANIFEST),
    )

    build_args = [
        "--output-db",
        str(db_path),
        "--model-path",
        str(DEFAULT_MODEL_PATH),
        "--image-cache-dir",
        str(cache_dir),
        "--summary-json",
        str(summary_path),
        "--locales",
        args.locales,
        "--min-row-count",
        str(args.min_row_count),
        "--download-workers",
        str(args.download_workers),
        "--detail-cache",
        str(Path(args.detail_cache).resolve()),
    ]
    if args.limit is not None:
        build_args.extend(["--limit", str(args.limit)])

    run_python(REPO_ROOT / "scripts" / "build_embeddings_db.py", *build_args)

    if not args.skip_zip:
        zip_file(db_path, zip_path)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print()
    print("Local embeddings rebuild complete.")
    print(f"DB: {db_path}")
    if not args.skip_zip:
        print(f"ZIP: {zip_path}")
    print(f"Summary: {summary_path}")
    print(
        "Rows:",
        summary.get("cards_count"),
        "cards /",
        summary.get("embeddings_count"),
        "embeddings",
    )
    print("Locales:", ", ".join(summary.get("locales", [])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
