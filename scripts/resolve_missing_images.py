#!/usr/bin/env python3
"""Phase 1 Standalone Missing Image Resolver.

Fetches and resolves missing scan card images from PokemonTCG.io or pkmn.gg
HTML NEXT_DATA page payloads using a curated, explicit mapping table.
Performs high-speed concurrent parallel image validation downloads.
Writes verified HTTPS fallbacks into the committed image-fallbacks.json registry.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = REPO_ROOT / "build" / "current-release" / "embeddings.db"
FALLBACKS_PATH = REPO_ROOT / "image-fallbacks.json"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# Curated, explicit set mappings for missing cards on pkmn.gg
EXPLICIT_SET_MAPPINGS: dict[str, Any] = {
    "mep": ("mega-evolution", "me-black-star-promos"),
    "miscp": ("other", "miscellaneous"),
    "smp": ("sun-moon", "sm-black-star-promos"),
    "svp": ("scarlet-violet", "scarlet-violet-black-star-promos"),
    "cel25": ("sword-shield", "celebrations"),
    "swshp": ("sword-shield", "swsh-black-star-promos"),
    "sve": ("scarlet-violet", "scarlet-violet-energy-2023"),
    "bog": ("other", "best-of-game"),
    "2023sv": ("other", "mcdonalds-collection-2023"),
    "2024sv": ("other", "mcdonalds-collection-2025"),
    "2022swsh": ("other", "mcdonalds-collection-2022"),
    "sm7.5": ("sun-moon", "dragon-majesty"),
    "sm3.5": ("sun-moon", "shining-legends"),
    "sm6": ("sun-moon", "forbidden-light"),
    "tk-xy-w": ("xy", "xy-trainer-kit-wigglytuff"),
    "tk-xy-sy": ("xy", "xy-trainer-kit-sylveon"),
    "tk-bw-z": ("black-white", "black-white-trainer-kit-zoroark"),
    "tk-xy-b": ("xy", "xy-trainer-kit-bisharp"),
    "tk-xy-latia": ("xy", "xy-trainer-kit-latias"),
    "tk-xy-latio": ("xy", "xy-trainer-kit-latios"),
    "tk-xy-n": ("xy", "xy-trainer-kit-noivern"),
    "tk-xy-p": ("xy", "xy-trainer-kit-pikachu-libre"),
    "tk-xy-su": ("xy", "xy-trainer-kit-suicune"),
    "tk-bw-e": ("black-white", "black-white-trainer-kit-excadrill"),
    "tk-sm-r": ("sun-moon", "sun-moon-trainer-kit-alolan-raichu"),
    "tk-sm-l": ("sun-moon", "sun-moon-trainer-kit-lycanroc"),
    "tk-dp-m": ("diamond-pearl", "dp-trainer-kit-manaphy"),
    "tk-dp-l": ("diamond-pearl", "dp-trainer-kit-lucario"),
    "ecard2": ("e-card", "aquapolis"),
    "ecard3": ("e-card", "skyridge"),
    
    # mfb ("My First Battle") maps to multiple decks
    "mfb": [
        ("other", "my-first-battle-bulbasaur"),
        ("other", "my-first-battle-charmander"),
        ("other", "my-first-battle-pikachu"),
        ("other", "my-first-battle-squirtle")
    ]
}

# Import pokemontcgio API helper if available
sys.path.insert(0, str(REPO_ROOT))
try:
    from scripts.pokemontcgio_api import search_card_by_set_and_number, search_cards_by_name
except ImportError:
    try:
        from pokemontcgio_api import search_card_by_set_and_number, search_cards_by_name
    except ImportError:
        search_card_by_set_and_number = None
        search_cards_by_name = None


def normalize_name(name: str) -> str:
    text = str(name or "").lower()
    text = text.replace("’", "'").replace("‘", "'").replace("–", "-").replace("—", "-")
    return re.sub(r"[^a-z0-9]", "", text)


def normalize_number(num: str) -> str:
    text = str(num or "").strip().upper()
    text = text.lstrip("0")
    if not text:
        return "0"
    return text


def fetch_pkmngg_set_cards(series: str, slug: str) -> list[dict[str, Any]]:
    url = f"https://www.pkmn.gg/series/{series}/{slug}"
    headers = {"User-Agent": USER_AGENT}
    req = urllib.request.Request(url, headers=headers)
    try:
        time.sleep(0.5)  # rate limit safety
        with urllib.request.urlopen(req, timeout=30) as res:
            html = res.read().decode("utf-8", errors="replace")
        
        match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html)
        if not match:
            print(f"    [pkmngg] Warning: __NEXT_DATA__ script block not found on {url}")
            return []
            
        data = json.loads(match.group(1))
        cards = data.get("props", {}).get("pageProps", {}).get("cardData", [])
        if isinstance(cards, list):
            return [c for c in cards if isinstance(c, dict)]
        return []
    except Exception as exc:
        print(f"    [pkmngg] Error fetching {url}: {exc}")
        return []


def validate_image_url(item: tuple[str, str, str]) -> tuple[str, str, str, bool]:
    card_id, url, source = item
    if not url or not url.startswith("https://"):
        return card_id, url, source, False
    try:
        headers = {"User-Agent": USER_AGENT}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=20) as res:
            img_bytes = res.read()
        
        with Image.open(BytesIO(img_bytes)) as img:
            img.verify()
            width, height = img.size
            if width < 200 or height < 200:
                return card_id, url, source, False
            aspect = height / width
            if aspect < 1.1 or aspect > 1.9:
                return card_id, url, source, False
        return card_id, url, source, True
    except Exception as exc:
        # Avoid flooding logs with concurrent traceback lines
        return card_id, url, source, False


def match_card_in_candidates(local_card: dict[str, Any], candidates: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    loc_name_norm = normalize_name(local_card["name"])
    loc_num_norm = normalize_number(local_card["card_number"])
    
    matches = []
    for cand in candidates:
        cand_name_norm = normalize_name(cand.get("name") or "")
        cand_num_norm = normalize_number(cand.get("number") or "")
        cand_num_disp_norm = normalize_number(cand.get("numberDisplay") or "")
        
        if cand_name_norm != loc_name_norm:
            continue
            
        # Match number display/number exactly
        if loc_num_norm == cand_num_norm or loc_num_norm == cand_num_disp_norm:
            matches.append(cand)
            
    if len(matches) == 1:
        return matches[0], "exact_match"
        
    if len(matches) > 1:
        # Resolve common energy/trainer duplicates safely
        name_lower = local_card["name"].lower()
        if "energy" in name_lower or name_lower in ("potion", "switch"):
            return matches[0], "duplicate_identical_allowed"
        return None, f"ambiguous_matches: {len(matches)}"
        
    # Attempt name-only fallback (e.g. for Promos where numbering schemes differ)
    name_only_matches = [
        c for c in candidates
        if normalize_name(c.get("name") or "") == loc_name_norm
    ]
    if len(name_only_matches) == 1:
        return name_only_matches[0], "name_only_match"
        
    if len(name_only_matches) > 1:
        # Resolve duplicates of the same name card in mfb or energy/trainers
        name_lower = local_card["name"].lower()
        if local_card["set_id"] == "mfb" or "energy" in name_lower or name_lower in ("potion", "switch", "charmander", "pikachu", "squirtle", "bulbasaur"):
            return name_only_matches[0], "name_only_duplicate_allowed"
        
    return None, "no_match"


def resolve_card_image(local_card: dict[str, Any], cache_candidates: dict[str, list[dict[str, Any]]], *, use_tcgio: bool = False) -> tuple[str | None, str]:
    set_id = local_card["set_id"]
    card_name = local_card["name"]
    card_number = local_card["card_number"]
    
    # 1. Try pkmn.gg NEXT_DATA payload FIRST (highly optimized set fetch)
    if set_id in EXPLICIT_SET_MAPPINGS:
        mapping = EXPLICIT_SET_MAPPINGS[set_id]
        candidates = []
        
        # Load pkmn.gg cards (using cache if already fetched in this run)
        if isinstance(mapping, list):
            # Combined decks (like My First Battle)
            for series, slug in mapping:
                cache_key = f"{series}/{slug}"
                if cache_key not in cache_candidates:
                    cache_candidates[cache_key] = fetch_pkmngg_set_cards(series, slug)
                candidates.extend(cache_candidates[cache_key])
        else:
            series, slug = mapping
            cache_key = f"{series}/{slug}"
            if cache_key not in cache_candidates:
                cache_candidates[cache_key] = fetch_pkmngg_set_cards(series, slug)
            candidates = cache_candidates[cache_key]
            
        # Match against our candidate pool
        match, reason = match_card_in_candidates(local_card, candidates)
        if match:
            large_image = match.get("largeImageUrl")
            if large_image:
                return large_image, f"pkmngg_{reason}"
            
    # 2. Try PokemonTCG.io exact lookup SECOND (only if explicitly enabled with --use-tcgio)
    if use_tcgio and search_card_by_set_and_number is not None:
        try:
            # We map local set ID to pokemontcg.io set aliases if supported
            from scripts.build_embeddings_db import candidate_pokemontcgio_set_ids, candidate_pokemontcgio_numbers, candidate_matches_card
            pio_sets = candidate_pokemontcgio_set_ids(set_id, card_number)
            pio_nums = candidate_pokemontcgio_numbers(card_number)
            
            for pio_set in pio_sets:
                for pio_num in pio_nums:
                    candidate = search_card_by_set_and_number(pio_set, pio_num)
                    if candidate and candidate_matches_card(local_card, candidate):
                        large_image = (candidate.get("images") or {}).get("large")
                        if large_image:
                            return large_image, "pokemontcgio_exact"
        except Exception as exc:
            # Silently fallback if anything fails
            pass
            
    return None, "unresolved"


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve missing database card images using exact fallback sources.")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB_PATH), help="Path to embeddings SQLite database")
    parser.add_argument("--registry", type=str, default=str(FALLBACKS_PATH), help="Path to image-fallbacks.json")
    parser.add_argument("--dry-run", action="store_true", help="Preview matches without writing fallback registry")
    parser.add_argument("--write", action="store_true", help="Perform concurrent image validation and write directly to fallback registry")
    parser.add_argument("--workers", type=int, default=20, help="Number of concurrent workers for image validation downloads")
    parser.add_argument("--use-tcgio", action="store_true", help="Enable PokemonTCG.io exact lookup fallback (requires internet/API key, can rate limit)")
    args = parser.parse_args()
    
    db_path = Path(args.db)
    registry_path = Path(args.registry)
    
    if not db_path.exists():
        print(f"Error: Database file does not exist at {db_path}")
        return
        
    print(f"Loading missing image cards from: {db_path}")
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute("""
            SELECT id, set_id, set_name, card_number, name, hp, types, upstream_id, locale
            FROM cards
            WHERE image_url IS NULL OR trim(image_url) = ''
            ORDER BY set_id, card_number, id
        """).fetchall()
    except Exception as exc:
        print(f"Error loading cards table: {exc}")
        return
    finally:
        connection.close()
        
    print(f"Found {len(rows)} cards currently missing image_url in database.")
    
    # Load current fallbacks manifest
    manifest: dict[str, dict[str, str]] = {}
    if registry_path.exists():
        try:
            payload = json.loads(registry_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                for k, v in payload.items():
                    if isinstance(v, dict) and "url" in v and "source" in v:
                        manifest[k] = {"url": v["url"], "source": v["source"]}
        except Exception as exc:
            print(f"Warning: Failed to parse fallback manifest: {exc}")
            
    print(f"Loaded existing registry with {len(manifest)} mapped fallback entries.")
    
    already_covered = 0
    uncovered_cards = []
    for r in rows:
        card_id = r["id"]
        if card_id in manifest:
            already_covered += 1
        else:
            uncovered_cards.append(dict(r))
            
    print(f"Coverage Audit: Already covered by registry = {already_covered}, Uncovered = {len(uncovered_cards)}")
    
    if not uncovered_cards:
        print("Success! Zero uncovered missing images left to resolve.")
        return
        
    print(f"\nProcessing {len(uncovered_cards)} uncovered missing cards...")
    
    cache_candidates: dict[str, list[dict[str, Any]]] = {}
    resolved_candidates: list[tuple[str, str, str]] = []
    unresolved_count = 0
    ambiguous_count = 0
    
    # Phase A: Resolution pass (local cached set matching - super fast)
    for idx, card in enumerate(uncovered_cards, 1):
        card_id = card["id"]
        set_id = card["set_id"]
        name = card["name"]
        number = card["card_number"]
        
        url, reason = resolve_card_image(card, cache_candidates, use_tcgio=args.use_tcgio)
        if url:
            resolved_candidates.append((card_id, url, reason))
        elif "ambiguous" in reason:
            ambiguous_count += 1
            print(f"  [AMBIGUOUS] {card_id} ({name} #{number} in {set_id}): {reason}")
        else:
            unresolved_count += 1
            print(f"  [UNRESOLVED] {card_id} ({name} #{number} in {set_id})")
            
    resolved_candidates_count = len(resolved_candidates)
    print(f"\nLocal resolution pass complete: resolved {resolved_candidates_count} cards.")
    
    verified_resolved_count = 0
    newly_resolved_manifest: dict[str, dict[str, str]] = {}
    
    # Phase B: High-speed concurrent validation pass
    if resolved_candidates_count > 0:
        if args.write:
            print(f"\nStarting concurrent image validation downloads using {args.workers} workers...")
            completed = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(validate_image_url, item): item for item in resolved_candidates}
                for future in concurrent.futures.as_completed(futures):
                    card_id, url, source, is_valid = future.result()
                    completed += 1
                    if is_valid:
                        newly_resolved_manifest[card_id] = {"url": url, "source": source}
                        verified_resolved_count += 1
                    else:
                        unresolved_count += 1
                        print(f"  [FAILED] Image validation failed for {card_id}: {url}")
                    if completed % 50 == 0 or completed == resolved_candidates_count:
                        print(f"  Validated {completed}/{resolved_candidates_count} resolved cards...")
        else:
            # Dry run counts resolved candidates directly
            verified_resolved_count = resolved_candidates_count
            for card_id, url, source in resolved_candidates:
                newly_resolved_manifest[card_id] = {"url": url, "source": source}
                
    # Print clean diagnostics report
    print("\n" + "=" * 50)
    print("PHASE 1 DIAGNOSTICS REPORT")
    print("=" * 50)
    print(f"Total Missing in DB:      {len(rows)}")
    print(f"Already Covered:          {already_covered}")
    print(f"Newly Resolved:           {verified_resolved_count}")
    print(f"Unresolved / Failed:      {unresolved_count}")
    print(f"Ambiguous / Discarded:    {ambiguous_count}")
    print(f"Remaining Uncovered:      {len(uncovered_cards) - verified_resolved_count}")
    print("=" * 50)
    
    if verified_resolved_count > 0 and args.write:
        # Merge and write updated fallbacks manifest
        manifest.update(newly_resolved_manifest)
        try:
            registry_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"\nSuccessfully wrote {verified_resolved_count} new entries to {registry_path}!")
        except Exception as exc:
            print(f"Error writing fallback registry: {exc}")
    elif verified_resolved_count > 0 and args.dry_run:
        print(f"\n[DRY-RUN] Previewed {verified_resolved_count} resolved candidates. No files were written.")
    else:
        print("\nNo fallback updates were registered.")


if __name__ == "__main__":
    main()
