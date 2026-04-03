#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
from typing import Any


API_BASE_URL = "https://api.github.com"


def github_request(
    path: str,
    *,
    method: str = "GET",
    token: str,
) -> Any:
    request = urllib.request.Request(
        f"{API_BASE_URL}{path}",
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "pokemon-tcg-corpus-release-pruner/1.0",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.length == 0 or response.status == 204:
            return None
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser(description="Delete old embeddings-v* releases and tags from GitHub.")
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--keep", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.repo:
        raise SystemExit("Missing --repo and GITHUB_REPOSITORY")

    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("GH_TOKEN or GITHUB_TOKEN is required")

    releases = github_request(f"/repos/{args.repo}/releases?per_page=100", token=token)
    versioned = [
        release
        for release in releases
        if isinstance(release, dict) and str(release.get("tag_name", "")).startswith("embeddings-v")
    ]
    versioned.sort(
        key=lambda release: (
            release.get("published_at") or "",
            release.get("created_at") or "",
            release.get("tag_name") or "",
        ),
        reverse=True,
    )

    stale = versioned[args.keep :]
    if not stale:
        print("no old versioned embeddings releases to prune")
        return 0

    for release in stale:
        tag_name = str(release["tag_name"])
        release_id = int(release["id"])
        if args.dry_run:
            print(f"would delete release_id={release_id} tag={tag_name}")
            continue

        github_request(f"/repos/{args.repo}/releases/{release_id}", method="DELETE", token=token)
        encoded_tag = urllib.parse.quote(f"tags/{tag_name}", safe="")
        github_request(f"/repos/{args.repo}/git/refs/{encoded_tag}", method="DELETE", token=token)
        print(f"deleted release_id={release_id} tag={tag_name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
