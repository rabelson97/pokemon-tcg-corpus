#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    CardImageDataset,
    CardInferenceDataset,
    CardEmbeddingModel,
    build_eval_transform,
    build_train_transform,
    create_label_map,
    load_manifest,
    split_records,
)


def evaluate(
    model: CardEmbeddingModel,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            _, logits = model(images)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.numel()
    return correct / max(1, total)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a first-pass Pokemon card embedding model.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True, help="Checkpoint path (.pt)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    records = load_manifest(args.manifest)
    train_records, val_records = split_records(records, val_fraction=args.val_fraction, seed=args.seed)
    label_map = create_label_map(records)

    train_loader = DataLoader(
        CardImageDataset(train_records, label_map, build_train_transform(args.image_size)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        CardImageDataset(val_records, label_map, build_eval_transform(args.image_size)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    model = CardEmbeddingModel(
        embedding_dim=args.embedding_dim,
        num_classes=len(label_map),
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val = 0.0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            _, logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_acc = evaluate(model, val_loader, device)
        avg_loss = running_loss / max(1, len(train_loader))
        print(f"epoch={epoch} loss={avg_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc >= best_val:
            best_val = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "embedding_dim": args.embedding_dim,
                    "image_size": args.image_size,
                    "label_map": label_map,
                    "best_val_acc": best_val,
                },
                output_path,
            )

    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "best_val_acc": best_val,
                "num_cards": len(records),
                "num_classes": len(label_map),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved checkpoint to {output_path}")
    print(f"saved metrics to {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
