from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a trained YOLO detector checkpoint to ONNX."
    )
    parser.add_argument("--weights", required=True, help="Path to YOLO .pt weights.")
    parser.add_argument("--imgsz", type=int, default=960, help="Export image size. Default: 960")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset. Default: 17")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Export device. Default: cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "ultralytics is required. Install it with `pip install ultralytics`."
        ) from exc

    weights = Path(args.weights).expanduser().resolve()
    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}")

    model = YOLO(str(weights))
    output_path = Path(
        model.export(
            format="onnx",
            imgsz=args.imgsz,
            opset=args.opset,
            device=args.device,
            simplify=True,
            dynamic=False,
            nms=False,
        )
    ).resolve()

    print(
        json.dumps(
            {
                "weights": str(weights),
                "onnx": str(output_path),
                "imgsz": args.imgsz,
                "opset": args.opset,
                "device": args.device,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
