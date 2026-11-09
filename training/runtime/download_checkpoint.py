#!/usr/bin/env python3
"""Download the exact shared P checkpoint; never resolve an unpinned main branch."""
import argparse
import json
from pathlib import Path

MODEL_ID = 'lerobot/smolvla_libero'
MODEL_REVISION = 'a24f1f380461b665158d52acd140152c79e8b36d'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    args.out.mkdir(parents=True, exist_ok=True)
    existing = args.out / 'WERESCUE_PIN.json'
    pin = {'repo_id': MODEL_ID, 'revision': MODEL_REVISION}
    if existing.exists() and json.loads(existing.read_text()) != pin:
        parser.error('output directory identifies a different checkpoint')
    snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=str(args.out))
    existing.write_text(json.dumps(pin, indent=2))
    print(str(args.out.resolve()))


if __name__ == '__main__':
    main()
