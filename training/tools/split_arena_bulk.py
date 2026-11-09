#!/usr/bin/env python3
"""Split a Physical Arena bulk export into original per-episode JSON.gz files."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


TASK = "libero_object__pick_up_the_alphabet_soup_and_place_it_in_the_basket"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with gzip.open(args.source, "rt", encoding="utf-8") as stream:
        for line in stream:
            episode = json.loads(line)
            eid = episode.get("id")
            if (episode.get("schema") != "wemine.physical-arena.episode.v1"
                    or episode.get("task_id") != TASK
                    or episode.get("source") != "human_browser_unverified"
                    or not isinstance(eid, str) or eid in seen):
                raise ValueError("Invalid, nonhuman, or duplicate episode in bulk export")
            seen.add(eid)
            path = args.output_dir / f"{eid}.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as output:
                json.dump(episode, output, separators=(",", ":"), ensure_ascii=False)
            print(f"{eid} {bool(episode.get('result', {}).get('native_success'))} {path}")


if __name__ == "__main__":
    main()
