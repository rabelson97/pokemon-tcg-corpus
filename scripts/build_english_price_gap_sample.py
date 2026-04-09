#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from tcgdex_api import api_get_json


SCREENSHOT_CARD_IDS = (
    "pokemon:en:sv06:173",
    "pokemon:en:sv10.5b:170",
    "pokemon:en:svp:047",
)
PROMO_SET_IDS = {"basep", "bwp", "dpp", "g1", "hsp", "np", "pl", "pop", "smp", "svp", "swshp", "xyp"}
ALT_RARITY_KEYWORDS = (
    "illustration rare",
    "special illustration rare",
    "alt art",
    "alternate art",
    "hyper rare",
    "ultra rare",
    "secret rare",
)
RECENT_CUTOFF = date(2024, 1, 1)
MODERN_CUTOFF = date(2020, 1, 1)


@dataclass(frozen=True)
class GapCard:
    card_id: str
    name: str
    set_id: str
    set_name: str
    card_number: str
    rarity: str
    gap_group: str
    source_names: tuple[str, ...]
    release_date: str | None
    release_bucket: str
    card_class: str
    stratum: str
    notes: str


def parse_release_date(raw: str | None) -> date | None:
    if raw is None or not raw.strip():
        return None
    year, month, day = raw.strip().split("-")
    return date(int(year), int(month), int(day))


def classify_release_bucket(release_at: date | None) -> str:
    if release_at is None:
        return "unknown"
    if release_at >= RECENT_CUTOFF:
        return "recent"
    if release_at >= MODERN_CUTOFF:
        return "modern"
    return "legacy"


def is_promo(set_id: str, set_name: str, card_number: str) -> bool:
    lowered_set_name = set_name.lower()
    return (
        set_id.lower() in PROMO_SET_IDS
        or "promo" in lowered_set_name
        or card_number.upper().startswith(("SVP", "SWSH", "SM", "BW"))
        and "promo" in lowered_set_name
    )


def is_alt_art(rarity: str) -> bool:
    lowered_rarity = rarity.lower()
    return any(keyword in lowered_rarity for keyword in ALT_RARITY_KEYWORDS)


def classify_card_class(set_id: str, set_name: str, card_number: str, rarity: str) -> str:
    if is_promo(set_id, set_name, card_number):
        return "promo"
    if is_alt_art(rarity):
        return "alt_art"
    return "standard"


def sort_key(card: GapCard) -> tuple[Any, ...]:
    release_date_key = card.release_date or ""
    screenshot_rank = 0 if card.card_id in SCREENSHOT_CARD_IDS else 1
    return (
        screenshot_rank,
        0 if card.release_bucket == "recent" else 1 if card.release_bucket == "modern" else 2 if card.release_bucket == "legacy" else 3,
        release_date_key,
        card.set_id,
        card.card_number,
        card.card_id,
    )


def fetch_release_dates(set_ids: set[str]) -> dict[str, str | None]:
    release_dates: dict[str, str | None] = {}
    for set_id in sorted(set_ids):
        payload = api_get_json(f"https://api.tcgdex.net/v2/en/sets/{set_id}")
        if isinstance(payload, dict):
            release_dates[set_id] = str(payload.get("releaseDate") or "").strip() or None
        else:
            release_dates[set_id] = None
    return release_dates


def load_gap_population(prices_db: Path, embeddings_db: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with sqlite3.connect(embeddings_db) as embeddings_connection, sqlite3.connect(prices_db) as prices_connection:
        english_cards = embeddings_connection.execute(
            """
            SELECT
              id,
              name,
              set_id,
              set_name,
              card_number,
              rarity
            FROM cards
            WHERE locale = 'en'
            ORDER BY id;
            """
        ).fetchall()

        price_rows = prices_connection.execute(
            """
            SELECT
              card_id,
              source_name
            FROM prices
            WHERE card_id LIKE 'pokemon:en:%'
            ORDER BY card_id, source_name;
            """
        ).fetchall()

    price_index: dict[str, list[str]] = defaultdict(list)
    for card_id, source_name in price_rows:
        price_index[str(card_id)].append(str(source_name))

    eur_only: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for card_id, name, set_id, set_name, card_number, rarity in english_cards:
        sources = tuple(sorted(price_index.get(str(card_id), [])))
        row = {
            "card_id": str(card_id),
            "name": str(name),
            "set_id": str(set_id),
            "set_name": str(set_name),
            "card_number": str(card_number),
            "rarity": str(rarity),
            "source_names": sources,
        }
        if not sources:
            missing.append(row)
        elif sources == ("cardmarket",):
            eur_only.append(row)
    return eur_only, missing


def enrich_cards(rows: list[dict[str, Any]], gap_group: str, release_dates: dict[str, str | None]) -> list[GapCard]:
    cards: list[GapCard] = []
    for row in rows:
        release_date_value = release_dates.get(row["set_id"])
        release_bucket = classify_release_bucket(parse_release_date(release_date_value))
        card_class = classify_card_class(
            set_id=row["set_id"],
            set_name=row["set_name"],
            card_number=row["card_number"],
            rarity=row["rarity"],
        )
        notes = "screenshot-example" if row["card_id"] in SCREENSHOT_CARD_IDS else ""
        cards.append(
            GapCard(
                card_id=row["card_id"],
                name=row["name"],
                set_id=row["set_id"],
                set_name=row["set_name"],
                card_number=row["card_number"],
                rarity=row["rarity"],
                gap_group=gap_group,
                source_names=row["source_names"],
                release_date=release_date_value,
                release_bucket=release_bucket,
                card_class=card_class,
                stratum=f"{release_bucket}:{card_class}",
                notes=notes,
            )
        )
    return cards


def select_sample(cards: list[GapCard], sample_size: int) -> list[GapCard]:
    ordered_cards = sorted(cards, key=sort_key)
    selected: list[GapCard] = []
    selected_ids: set[str] = set()

    for card_id in SCREENSHOT_CARD_IDS:
        match = next((card for card in ordered_cards if card.card_id == card_id), None)
        if match is not None and match.card_id not in selected_ids:
            selected.append(match)
            selected_ids.add(match.card_id)
            if len(selected) >= sample_size:
                return selected

    pools: dict[str, list[GapCard]] = defaultdict(list)
    for card in ordered_cards:
        if card.card_id in selected_ids:
            continue
        pools[card.stratum].append(card)

    stratum_order = [
        "recent:promo",
        "recent:alt_art",
        "recent:standard",
        "modern:promo",
        "modern:alt_art",
        "modern:standard",
        "legacy:promo",
        "legacy:alt_art",
        "legacy:standard",
        "unknown:promo",
        "unknown:alt_art",
        "unknown:standard",
    ]

    while len(selected) < sample_size:
        made_progress = False
        for stratum in stratum_order:
            pool = pools.get(stratum, [])
            if not pool:
                continue
            card = pool.pop(0)
            selected.append(card)
            selected_ids.add(card.card_id)
            made_progress = True
            if len(selected) >= sample_size:
                break
        if not made_progress:
            break
    return selected


def write_sample_csv(path: Path, cards: list[GapCard]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_index",
                "gap_group",
                "card_id",
                "name",
                "set_id",
                "set_name",
                "card_number",
                "rarity",
                "source_names",
                "release_date",
                "release_bucket",
                "card_class",
                "stratum",
                "notes",
            ],
        )
        writer.writeheader()
        for index, card in enumerate(cards, start=1):
            writer.writerow(
                {
                    "sample_index": index,
                    **asdict(card),
                    "source_names": ",".join(card.source_names),
                }
            )


def summarise_population(cards: list[GapCard]) -> dict[str, Any]:
    return {
        "count": len(cards),
        "by_release_bucket": dict(Counter(card.release_bucket for card in cards)),
        "by_card_class": dict(Counter(card.card_class for card in cards)),
        "by_stratum": dict(Counter(card.stratum for card in cards)),
        "screenshot_examples_present": sorted(card.card_id for card in cards if card.card_id in SCREENSHOT_CARD_IDS),
    }


def build_summary(populations: dict[str, list[GapCard]], samples: dict[str, list[GapCard]]) -> dict[str, Any]:
    return {
        "populations": {name: summarise_population(cards) for name, cards in populations.items()},
        "samples": {
            name: {
                **summarise_population(cards),
                "card_ids": [card.card_id for card in cards],
            }
            for name, cards in samples.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a deterministic sample of English Pokemon cards missing USD prices.")
    parser.add_argument("--prices-db", required=True, help="Path to prices.db")
    parser.add_argument("--embeddings-db", required=True, help="Path to embeddings.db")
    parser.add_argument("--output-csv", required=True, help="Output CSV path")
    parser.add_argument("--output-summary-json", required=True, help="Output summary JSON path")
    parser.add_argument("--sample-size-per-group", type=int, default=100, help="Cards to sample for each gap group")
    args = parser.parse_args()

    prices_db = Path(args.prices_db).resolve()
    embeddings_db = Path(args.embeddings_db).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_summary_json = Path(args.output_summary_json).resolve()

    eur_only_rows, missing_rows = load_gap_population(prices_db=prices_db, embeddings_db=embeddings_db)
    release_dates = fetch_release_dates({row["set_id"] for row in eur_only_rows + missing_rows})

    populations = {
        "eur_only": enrich_cards(eur_only_rows, gap_group="eur_only", release_dates=release_dates),
        "missing": enrich_cards(missing_rows, gap_group="missing", release_dates=release_dates),
    }
    samples = {
        name: select_sample(cards, args.sample_size_per_group)
        for name, cards in populations.items()
    }
    combined_sample = samples["eur_only"] + samples["missing"]

    write_sample_csv(output_csv, combined_sample)
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json.write_text(json.dumps(build_summary(populations, samples), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        f"wrote sample csv={output_csv} rows={len(combined_sample)} "
        f"eur_only_population={len(populations['eur_only'])} missing_population={len(populations['missing'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
