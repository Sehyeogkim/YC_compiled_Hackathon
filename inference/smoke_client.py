#!/usr/bin/env python3
"""Verify a real server/model/native scene path; never substitutes test actions."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import urllib.request
import urllib.error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    def request(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(args.url + path, data=data,
            headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=300) as reply:
                return json.load(reply)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f'{path}: HTTP {exc.code}: {exc.read().decode()}') from exc

    health = request('/health')
    if not health['ready']:
        raise RuntimeError(f'Model is not ready: {health}')
    reset = request('/session/reset', {'initial_state_id': 0, 'seed': 123, 'control_source': 'policy'})
    session_id = reset['session_id']
    scene = request('/scene?session_id=' + session_id)
    initial = reset['frame']
    if len(initial['qpos']) != scene['nq'] or len(initial['qvel']) != scene['nv']:
        raise RuntimeError('Native frame and native model dimensions differ')
    replies = []
    for index in range(args.steps):
        result = request('/session/step', {'session_id': session_id, 'source': 'policy', 'steps': 1})
        for action in result['executed_actions']:
            if len(action) != 7 or not all(math.isfinite(v) for v in action):
                raise RuntimeError('Policy did not return a finite native 7D action')
        replies.append({'step': result['frame']['step'], 'eef': result['frame']['eef'],
            'sim_time': result['frame']['sim_time'], 'executed_actions': result['executed_actions'],
            'inference_ms': result['inference_ms'], 'success': result['frame']['success'],
            'done': result['frame']['done']})
        print(json.dumps(replies[-1]), flush=True)
        if result['frame']['done']:
            break
    final_health = request('/health')
    if not final_health['inference_verified']:
        raise RuntimeError('Server did not verify actual model execution')
    proof = {'schema': 'werescue.inference_smoke.v1',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'health': final_health, 'session_id': session_id,
        'scene': {key: scene[key] for key in ['nq', 'nv', 'scene_sha256', 'control']},
        'asset_count': len(scene['assets']),
        'asset_base64_bytes': sum(len(a['data']) for a in scene['assets']),
        'initial_eef': initial['eef'], 'steps': replies,
        'claim_scope': 'actual pinned policy inference and native simulator plumbing; not a benchmark success rate'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(proof, indent=2))
    print(str(args.out.resolve()), flush=True)


if __name__ == '__main__':
    main()
