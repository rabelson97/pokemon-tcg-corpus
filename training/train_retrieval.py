#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common import (
    CardInferenceDataset,
    CardEmbeddingModel,
    TwoViewCardDataset,
    build_eval_transform,
    build_stream_train_transform,
    count_records_by_locale,
    create_label_map,
    load_manifest,
    split_records,
)


def nt_xent_loss(embeddings: torch.Tensor, temperature: float) -> torch.Tensor:
    batch_size = embeddings.shape[0] // 2
    similarity = torch.matmul(embeddings, embeddings.T) / temperature
    mask = torch.eye(similarity.size(0), device=similarity.device, dtype=torch.bool)
    similarity = similarity.masked_fill(mask, float("-inf"))

    targets = torch.arange(similarity.size(0), device=similarity.device)
    targets = (targets + batch_size) % (2 * batch_size)
    return torch.nn.functional.cross_entropy(similarity, targets)


def embed_dataset(model: CardEmbeddingModel, dataloader: DataLoader, device: torch.device) -> tuple[torch.Tensor, list[str]]:
    model.eval()
    embeddings: list[torch.Tensor] = []
    card_ids: list[str] = []
    with torch.no_grad():
        for images, batch_ids in dataloader:
            images = images.to(device)
            batch_embeddings, _ = model(images)
            embeddings.append(batch_embeddings.cpu())
            card_ids.extend(batch_ids)

    return torch.cat(embeddings, dim=0), card_ids


def retrieval_at_1(
    model: CardEmbeddingModel,
    query_dataloader: DataLoader,
    reference_dataloader: DataLoader,
    device: torch.device,
) -> float:
    query_matrix, query_ids = embed_dataset(model, query_dataloader, device)
    reference_matrix, reference_ids = embed_dataset(model, reference_dataloader, device)
    similarity = query_matrix @ reference_matrix.T
    nearest = similarity.argmax(dim=1)
    correct = sum(1 for idx, match_idx in enumerate(nearest.tolist()) if query_ids[idx] == reference_ids[match_idx])
    return correct / max(1, len(query_ids))


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a contrastive retrieval model for Pokemon card embeddings.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.12)
    parser.add_argument("--classification-weight", type=float, default=0.25)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    torch.manual_seed(args.seed)
    print(f"using device={device}")

    records = load_manifest(args.manifest)
    train_records, val_records = split_records(records, val_fraction=args.val_fraction, seed=args.seed)
    label_map = create_label_map(records)
    print(
        json.dumps(
            {
                "manifest_counts": count_records_by_locale(records),
                "train_counts": count_records_by_locale(train_records),
                "validation_counts": count_records_by_locale(val_records),
                "num_cards": len(records),
                "num_train": len(train_records),
                "num_validation": len(val_records),
            },
            indent=2,
        )
    )

    train_loader = DataLoader(
        TwoViewCardDataset(train_records, build_stream_train_transform(args.image_size)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_query_loader = DataLoader(
        CardInferenceDataset(val_records, build_stream_train_transform(args.image_size)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_reference_loader = DataLoader(
        CardInferenceDataset(val_records, build_eval_transform(args.image_size)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = CardEmbeddingModel(
        embedding_dim=args.embedding_dim,
        num_classes=len(label_map),
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    classification_loss = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

    best_recall = 0.0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for view_a, view_b, card_ids in train_loader:
            images = torch.cat([view_a, view_b], dim=0).to(device)
            labels = torch.tensor([label_map[card_id] for card_id in card_ids], device=device)
            labels = torch.cat([labels, labels], dim=0)

            optimizer.zero_grad(set_to_none=True)
            embeddings, logits = model(images)
            loss = nt_xent_loss(embeddings, args.temperature)
            if args.classification_weight > 0:
                loss = loss + args.classification_weight * classification_loss(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_recall = retrieval_at_1(model, val_query_loader, val_reference_loader, device)
        avg_loss = running_loss / max(1, len(train_loader))
        print(f"epoch={epoch} loss={avg_loss:.4f} val_stream_recall_at_1={val_recall:.4f}")

        if val_recall >= best_recall:
            best_recall = val_recall
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "embedding_dim": args.embedding_dim,
                    "image_size": args.image_size,
                    "label_map": label_map,
                    "best_val_recall_at_1": best_recall,
                    "best_val_stream_recall_at_1": best_recall,
                    "training_mode": "contrastive",
                },
                output_path,
            )

    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "best_val_recall_at_1": best_recall,
                "best_val_stream_recall_at_1": best_recall,
                "num_cards": len(records),
                "num_classes": len(label_map),
                "training_mode": "contrastive",
                "manifest_counts": count_records_by_locale(records),
                "train_counts": count_records_by_locale(train_records),
                "validation_counts": count_records_by_locale(val_records),
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
