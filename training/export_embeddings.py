#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    CardEmbeddingModel,
    CardInferenceDataset,
    build_eval_transform,
    load_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export corpus embeddings from a trained checkpoint.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    records = load_manifest(args.manifest)
    model = CardEmbeddingModel(
        embedding_dim=checkpoint["embedding_dim"],
        num_classes=len(checkpoint["label_map"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    print(f"using device={device}")

    dataloader = DataLoader(
        CardInferenceDataset(records, build_eval_transform(checkpoint["image_size"])),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )

    all_embeddings: list[np.ndarray] = []
    all_card_ids: list[str] = []

    with torch.no_grad():
        for images, card_ids in dataloader:
            images = images.to(device)
            embeddings, _ = model(images)
            all_embeddings.append(embeddings.cpu().numpy().astype(np.float32))
            all_card_ids.extend(card_ids)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        card_ids=np.array(all_card_ids),
        embeddings=np.concatenate(all_embeddings, axis=0),
    )

    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "manifest": str(Path(args.manifest).resolve()),
                "card_count": len(all_card_ids),
                "embedding_dim": checkpoint["embedding_dim"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved embeddings to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
