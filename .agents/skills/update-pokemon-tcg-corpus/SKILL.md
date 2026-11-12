---
name: update-pokemon-tcg-corpus
description: Update the English Pokemon card list and scan images in pokemon-tcg-corpus when a new set is missing, an embeddings run fails, or a release needs verification.
---

# Update the Pokemon TCG Corpus

Run from the repository root with Python, dependencies from `requirements.txt`, GitHub CLI and SQLite. Any coding agent can read this file directly.

1. Confirm the English physical release and the complete card count (including secret rares) using the publisher and a full checklist. Find the actual set ID in `https://api.tcgdex.net/v2/en/sets` and check `/v2/en/sets/<id>` for card rows and image roots. Check PokemonTCG.io if the set is absent. Image mappings cannot add a set absent from both feeds. In that case, report the source gap and implement a verified metadata source only if needed.
2. Inspect `gh run list -w build-embeddings-db.yml --limit 5`. For a failed run, use `gh run view RUN_ID --log-failed` and `gh run download RUN_ID -n embeddings-build-summary -D "$(mktemp -d)"`. Read the summary and `missing-images-report.json` in the printed or chosen output directory. A successful run can be a no-op. Look for its actual `publish_release` decision.
3. For a TCGdex set with missing images, locate its English pkmn.gg set page at `/series/<series>/<slug>` and add the verified slug pair to `EXACT_SET_MAPPINGS` in `scripts/resolve_missing_images.py`. Run `python scripts/check_set_images.py SET_ID --expected-count COUNT`. It verifies that the checklist count matches and that each card resolves by name and number to a decodable full-card image. Spot-check the printed set code, collector number and art, especially for repeated names and secret rares. Preserve existing IDs. Investigate unmatched records and fix only verified source errors. Keep every released English card in scope and keep the strict image gate enabled.
4. Run `python -m pytest -q`. Ship through the repository's normal merge flow. On main, dispatch `gh workflow run build-embeddings-db.yml --ref main` when an immediate rebuild is needed. Check the completed run and its summary for zero skipped cards and `publish_release=true`.
5. Download and unpack `embeddings-latest` into a temp directory with `gh release download embeddings-latest -p embeddings.db.zip -D DIRECTORY` and `unzip`. Query `cards` for the requested `set_id`, compare the unique collector numbers with the checklist, and join `embeddings` to verify every card has all four variants. Report the published release, run link, card count, and any unresolved IDs. For pricing, inspect the separate Build Prices DB workflow.
