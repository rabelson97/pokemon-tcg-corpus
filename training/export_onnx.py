#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(SCRIPT_DIR))

from common import CardEmbeddingModel  # noqa: E402


class EmbeddingOnly(torch.nn.Module):
    def __init__(self, model: CardEmbeddingModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        embeddings, _ = self.model(images)
        return embeddings


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the trained card retrieval model to ONNX.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = CardEmbeddingModel(
        embedding_dim=checkpoint["embedding_dim"],
        num_classes=len(checkpoint["label_map"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    export_model = EmbeddingOnly(model)
    dummy = torch.randn(1, 3, checkpoint["image_size"], checkpoint["image_size"], dtype=torch.float32)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        export_model,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"exported ONNX model to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
