#!/usr/bin/env python3
"""Explicit real-policy speed smoke over the same persistent HTTP tunnel as UI."""
import argparse
from datetime import datetime, timezone
import http.client
import json
import math
from pathlib import Path
import time
from urllib.parse import urlsplit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reset', action='store_true', required=True,
                        help='Explicit authorization to start a new isolated test session')
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 10 or args.steps < 1:
        parser.error('batch-size must be 1..10; steps must be positive')
    url = urlsplit(args.url)
    if url.scheme != 'http':
        parser.error('Use the local HTTP SSH tunnel')
    conn = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=180)

    def request(path, body=None):
        encoded = None if body is None else json.dumps(body)
        conn.request('GET' if body is None else 'POST', url.path.rstrip('/') + path,
                     body=encoded, headers={'Content-Type': 'application/json', 'Connection': 'keep-alive'})
        response = conn.getresponse()
        payload = response.read()
        if response.status != 200:
            raise RuntimeError(f'{path}: HTTP {response.status}: {payload.decode()[:1000]}')
        return json.loads(payload)

    health = request('/health')
    if not health['ready']:
        raise RuntimeError(f'Model not ready: {health}')
    reset = request('/session/reset', {'initial_state_id': 0, 'seed': 123, 'control_source': 'policy'})
    sid = reset['session_id']
    print('TEST_SESSION ' + sid, flush=True)
    # These are integration/performance actions, never human demonstrations.
    exclusions = Path(__file__).with_name('test_sessions.json')
    excluded = json.loads(exclusions.read_text())
    if sid not in excluded['excluded_session_ids']:
        excluded['excluded_session_ids'].append(sid)
        exclusions.write_text(json.dumps(excluded, indent=2) + '\n')
    replies = []
    begin = time.perf_counter()
    while sum(len(r['executed_actions']) for r in replies) < args.steps:
        remaining = args.steps - sum(len(r['executed_actions']) for r in replies)
        started = time.perf_counter()
        result = request('/session/step', {'session_id': sid, 'source': 'policy',
                                          'steps': min(args.batch_size, remaining)})
        actions = result['executed_actions']
        if not actions or any(len(a) != 7 or not all(math.isfinite(v) for v in a) for a in actions):
            raise RuntimeError('Missing or invalid actual model actions')
        row = {'step': result['frame']['step'], 'sim_time': result['frame']['sim_time'],
               'request_ms': (time.perf_counter() - started) * 1000,
               'inference_ms': result['inference_ms'], 'policy_call_ms': result['policy_call_ms'],
               'fresh_predictions': result['fresh_predictions'], 'eef': result['frame']['eef'],
               'executed_actions': actions, 'success': result['frame']['success'], 'done': result['frame']['done']}
        replies.append(row)
        print(json.dumps(row), flush=True)
        if row['done']:
            break
    duration = time.perf_counter() - begin
    count = sum(len(r['executed_actions']) for r in replies)
    proof = {'schema': 'werescue.policy_speed_smoke.v1', 'created_at': datetime.now(timezone.utc).isoformat(),
             'session_id': sid, 'test_excluded_from_training': True, 'initial_state_id': 0, 'seed': 123,
             'batch_size': args.batch_size, 'steps': count, 'client_wall_seconds': duration,
             'native_seconds': count * .05, 'observed_hz': count / duration,
             'wall_slowdown': duration / (count * .05), 'scene_sha256': reset['scene_sha256'],
             'health': request('/health'), 'requests': replies,
             'claim_scope': 'Real inference/native plumbing throughput smoke, not task success or a controlled benchmark'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(proof, indent=2))
    print(json.dumps({k: proof[k] for k in ('session_id', 'steps', 'client_wall_seconds', 'observed_hz', 'wall_slowdown')}), flush=True)
    conn.close()


if __name__ == '__main__':
    main()
