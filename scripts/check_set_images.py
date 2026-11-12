"""Verify every TCGdex card in an English set against its full-card image."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from resolve_missing_images import EXACT_SET_MAPPINGS, resolve_card_image, validate_image_url
from tcgdex_api import fetch_set_detail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("set_id")
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--checklist-json", type=Path, help="Use a downloaded TCGdex set response")
    args = parser.parse_args()
    if args.checklist_json:
        detail = json.loads(args.checklist_json.read_text(encoding="utf-8"))
    else:
        detail = fetch_set_detail("en", args.set_id)
    if detail.get("id") != args.set_id:
        parser.error(f"Checklist does not belong to {args.set_id}")
    cards = detail["cards"]
    numbers = {card["localId"] for card in cards}
    if len(cards) != args.expected_count or len(numbers) != args.expected_count:
        parser.error(f"Expected {args.expected_count} unique card numbers, found {len(numbers)} in {len(cards)} rows")

    cache = {}
    candidates = []
    missing = []
    for brief in cards:
        card = {"set_id": args.set_id, "card_number": brief["localId"], "name": brief["name"]}
        if args.set_id in EXACT_SET_MAPPINGS:
            url, source = resolve_card_image(card, cache)
        else:
            base = brief.get("image")
            url, source = (f"{base}/high.webp" if base else None), "tcgdex"
        if url:
            candidates.append((brief["id"], url, source))
        else:
            missing.append(brief["id"])

    with ThreadPoolExecutor(max_workers=6) as executor:
        for card_id, _, _, valid in executor.map(validate_image_url, candidates):
            if not valid:
                missing.append(card_id)
    for card_id in missing:
        print(f"Unverified: {card_id}")
    print(f"{args.set_id}: verified {len(cards) - len(missing)}/{len(cards)} full-card images")
    return int(bool(missing))


if __name__ == "__main__":
    raise SystemExit(main())
