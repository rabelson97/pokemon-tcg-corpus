#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from common import load_manifest, same_art_key


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a sequential synthetic benchmark manifest from cached card images.")
    parser.add_argument("--manifest", default="training/data/full/manifest.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--max-clips", type=int, default=512)
    parser.add_argument("--frames-per-clip", type=int, default=6)
    parser.add_argument("--frame-step-ms", type=int, default=100)
    parser.add_argument("--same-art-clip-ratio", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    records = load_manifest(args.manifest)
    if not records:
        raise SystemExit("No manifest records found")

    by_group: dict[str, list] = {}
    for record in records:
        by_group.setdefault(same_art_key(record), []).append(record)

    hard_groups = [group for group in by_group.values() if len(group) > 1]
    singletons = [group[0] for group in by_group.values() if len(group) == 1]
    rng.shuffle(hard_groups)
    rng.shuffle(singletons)

    hard_target = min(len(hard_groups), int(args.max_clips * args.same_art_clip_ratio))
    selected_records = []
    for group in hard_groups[:hard_target]:
        selected_records.append(rng.choice(group))
    remaining = max(0, args.max_clips - len(selected_records))
    selected_records.extend(singletons[:remaining])
    if len(selected_records) < args.max_clips:
        fallback = [record for group in hard_groups[hard_target:] for record in group]
        rng.shuffle(fallback)
        selected_records.extend(fallback[: args.max_clips - len(selected_records)])
    selected_records = selected_records[: args.max_clips]

    output_rows = []
    clip_counts = Counter()
    same_art_counts = Counter()
    for clip_index, record in enumerate(selected_records):
        clip_id = f"clip_{clip_index:05d}"
        clip_group = same_art_key(record)
        group_size = len(by_group.get(clip_group, []))
        for frame_index in range(args.frames_per_clip):
            output_rows.append(
                {
                    "clip_id": clip_id,
                    "frame_index": frame_index,
                    "timestamp_millis": frame_index * args.frame_step_ms,
                    "card_id": record.card_id,
                    "name": record.name,
                    "locale": record.locale,
                    "set_id": record.set_id,
                    "set_name": record.set_name,
                    "number": record.number,
                    "image_path": record.image_path,
                    "same_art_key": clip_group,
                    "same_art_group_size": group_size,
                    "perturbation_seed": rng.randint(0, 2**31 - 1),
                }
            )
        clip_counts[record.locale or "unknown"] += 1
        same_art_counts["same_art"] += 1 if group_size > 1 else 0
        same_art_counts["singleton"] += 1 if group_size <= 1 else 0

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "manifest": str(Path(args.manifest).resolve()),
        "output": str(output_path),
        "clips": len(selected_records),
        "frames": len(output_rows),
        "frames_per_clip": args.frames_per_clip,
        "frame_step_millis": args.frame_step_ms,
        "same_art_clip_count": same_art_counts["same_art"],
        "singleton_clip_count": same_art_counts["singleton"],
        "per_locale_clip_count": dict(sorted(clip_counts.items())),
        "seed": args.seed,
    }
    if args.summary_json:
        summary_path = Path(args.summary_json).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
