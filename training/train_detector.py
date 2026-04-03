from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLO detector on a Roboflow-exported Pokemon card dataset."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to data.yaml in the YOLO dataset export.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Base YOLO model checkpoint to fine-tune. Default: yolov8n.pt",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs. Default: 30",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Training image size. Default: 960",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size. Default: 16",
    )
    parser.add_argument(
        "--device",
        default="mps",
        help="Training device, e.g. cpu, 0, mps. Default: mps",
    )
    parser.add_argument(
        "--project",
        default="training/detector_runs",
        help="Project directory for YOLO outputs. Default: training/detector_runs",
    )
    parser.add_argument(
        "--name",
        default="pokemon_card_detector",
        help="Run name. Default: pokemon_card_detector",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - import guard
        raise SystemExit(
            "ultralytics is required. Install it with `pip install ultralytics`."
        ) from exc

    data_path = Path(args.data).expanduser().resolve()
    if not data_path.exists():
        raise SystemExit(f"Dataset yaml not found: {data_path}")

    model = YOLO(args.model)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        pretrained=True,
        patience=8,
        degrees=4.0,
        translate=0.06,
        scale=0.10,
        shear=2.0,
        perspective=0.0005,
        fliplr=0.0,
        mosaic=0.15,
        mixup=0.05,
    )

    run_dir = Path(results.save_dir)
    summary = {
        "data": str(data_path),
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "run_dir": str(run_dir),
        "best_weights": str(run_dir / "weights" / "best.pt"),
        "last_weights": str(run_dir / "weights" / "last.pt"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
