#!/usr/bin/env python3
"""Validate and aggregate partitioned HFVLA can development evaluation workers.

Does not modify worker outputs. Missing trials remain missing. Partial summaries
never claim a completed50-run result. Run repeatedly while detached workers run.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from protocol import digest
from run_baseline import atomic_json
from run_hfvla50 import (MODEL_ID, MODEL_REVISION, build_protocol, completed_trials,
                         selected_trials)


def pooled_interval(wins, count):
    if not count:
        return None
    z, p = 1.959963984540054, wins/count
    denominator = 1+z*z/count
    center = (p+z*z/(2*count))/denominator
    half = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/denominator
    return [max(0, center-half), min(1, center+half)]


def aggregate(workers):
    expected_protocol = build_protocol('object', [0], list(range(25)), [1000, 1001],
                                       camera_size=360, horizon=280, position_jitter=0.04)
    expected = {trial['trial_id']: trial for trial in selected_trials(expected_protocol)}
    common_protocol = {key: value for key, value in expected_protocol.items()
                       if key not in ('initial_states', 'protocol_sha256')}
    rows, planned, trial_sources, statuses = {}, {}, {}, []
    runtime_reference = None
    runtime_workers_checked = 0
    numerical_initial_hashes, settled_positions = {}, {}
    unique_workers = [Path(worker).resolve() for worker in workers]
    if len(set(unique_workers)) != len(unique_workers):
        raise ValueError('duplicate worker directories')
    if not unique_workers:
        raise ValueError('at least one worker directory required')
    for worker in unique_workers:
        path = worker / 'protocol.json'
        if not path.exists():
            statuses.append({'worker': str(worker), 'status': 'protocol_not_written', 'completed_trials': 0})
            continue
        protocol = json.loads(path.read_text())
        claimed_sha = protocol.get('protocol_sha256')
        if digest({key: value for key, value in protocol.items() if key != 'protocol_sha256'}) != claimed_sha:
            raise ValueError(f'{worker}: protocol digest is invalid')
        candidate = {key: value for key, value in protocol.items()
                     if key not in ('initial_states', 'protocol_sha256')}
        if candidate != common_protocol:
            raise ValueError(f'{worker}: shared worker protocol differs from frozen can-dev50 configuration')
        for trial in selected_trials(protocol):
            key = trial['trial_id']
            if key not in expected or trial != expected[key]:
                raise ValueError(f'{worker}: unexpected trial {key}')
            if key in planned:
                raise ValueError(f'duplicate planned trial {key} in {worker} and {planned[key]}')
            planned[key] = str(worker)
        runtime_path = worker / 'runtime.json'
        if not runtime_path.exists():
            if any(worker.glob('*/done.json')):
                raise ValueError(f'{worker}: completed trial exists without runtime provenance')
            statuses.append({'worker': str(worker), 'status': 'runtime_loading', 'completed_trials': 0,
                             'planned_trials': len(selected_trials(protocol))})
            continue
        runtime = json.loads(runtime_path.read_text())
        if runtime_reference is None:
            runtime_reference = runtime
        elif runtime_reference != runtime:
            raise ValueError(f'{worker}: source/model/runtime differs from other workers')
        runtime_sha = digest(runtime)
        runtime_workers_checked += 1
        completed = completed_trials(worker, protocol, runtime_sha)
        for row in completed:
            key = row['trial_id']
            if key in rows:
                raise ValueError(f'duplicate measured trial {key}')
            if not isinstance(row.get('initial_native_state_sha256'), str) or len(row['initial_native_state_sha256']) != 64:
                raise ValueError(f'{worker}: missing full native initial-state digest')
            state = row['initial_state_id']
            state_hash = row['initial_native_state_sha256']
            if state in numerical_initial_hashes and numerical_initial_hashes[state] != state_hash:
                raise ValueError(f'policy-seed pair for initial state{state} used different full native start states')
            numerical_initial_hashes[state] = state_hash
            position_setup = row.get('position_setup')
            if not isinstance(position_setup, dict):
                raise ValueError(f'{worker}: jitter trial missing measured position setup')
            position = position_setup.get('actual_settled_position')
            if not isinstance(position, list) or len(position) != 3 or not all(type(v) in (int, float) and math.isfinite(v) for v in position):
                raise ValueError(f'{worker}: invalid actual settled target position')
            if position_setup.get('layout_rng_seed') != 50000+state:
                raise ValueError(f'{worker}: target layout seed differs from protocol')
            if state in settled_positions and settled_positions[state] != position:
                raise ValueError(f'policy-seed pair for initial state{state} used different actual can positions')
            settled_positions[state] = position
            rows[key] = row
            trial_sources[key] = {'worker': str(worker), 'done_ref': str(worker / key / 'done.json'),
                                  'trace_ref': str(worker / key / 'trace.jsonl.gz'),
                                  'initial_state_ref': str(worker / key / 'initial_state.npz'),
                                  'video_ref': str(worker / key / row['video_ref']) if row.get('video_ref') else None}
        statuses.append({'worker': str(worker), 'status': 'complete' if len(completed) == len(selected_trials(protocol)) else 'running_or_stopped',
                         'completed_trials': len(completed), 'planned_trials': len(selected_trials(protocol)),
                         'protocol_sha256': claimed_sha, 'runtime_sha256': runtime_sha})
    ordered = [rows[key] for key in expected if key in rows]
    count, wins = len(ordered), sum(row['success'] for row in ordered)
    missing = [key for key in expected if key not in rows]
    span = [max(p[i] for p in settled_positions.values())-min(p[i] for p in settled_positions.values())
            for i in (0, 1)] if settled_positions else [0.0, 0.0]
    if len(settled_positions) > 1 and max(span) <= 1e-4:
        raise ValueError('completed trials do not demonstrate actual target-position variation')
    seed_sets = {}
    for row in ordered:
        seed_sets.setdefault(row['initial_state_id'], set()).add(row['policy_rng_seed'])
    paired_ids = [state for state, seeds in seed_sets.items() if seeds == {1000, 1001}]
    return {
        'schema_version': 'werescue.hfvla_parallel_summary.v1',
        'policy': {'repo_id': MODEL_ID, 'revision': MODEL_REVISION},
        'protocol': expected_protocol, 'expected_trials': 50, 'planned_trials': len(planned),
        'completed_trials': count, 'complete': not missing,
        'status': 'complete' if not missing else ('running_or_partial' if count else 'waiting_for_measured_trials'),
        'successes': wins, 'success_rate': wins/count if count else None,
        'wilson_95': pooled_interval(wins, count), 'missing_trial_ids': missing,
        'unassigned_trial_ids': [key for key in expected if key not in planned],
        'workers': statuses, 'shared_runtime_verified': runtime_workers_checked == len(unique_workers),
        'runtime_workers_checked': runtime_workers_checked,
        'runtime_sha256': digest(runtime_reference) if runtime_reference else None,
        'source_and_evidence_validation': 'worker protocols, shared runtime, per-trial metadata and trace/state/XML digests verified',
        'paired_native_initial_states_verified': bool(count) and len(paired_ids) == len(seed_sets),
        'paired_initial_state_pairs_checked': len(paired_ids),
        'unpaired_initial_state_ids': [state for state in seed_sets if state not in paired_ids],
        'actual_position_variation': {'verified': len(settled_positions)>1,
                                       'unique_initial_states_recorded': len(settled_positions),
                                       'actual_settled_xy_span_m': span},
        'claim_scope': 'development validation of50 planned can trials with target jitter; partial counts are not the final50-run result; not an exact reproduction of cited75.5%',
        'uncertainty_note': 'Descriptive pooled Wilson interval; policy seeds share layouts. Parallel GPU workers also compete for throughput.',
        'trials': [{**row, 'worker_evidence': trial_sources[row['trial_id']]} for row in ordered],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True, help='Master output directory; workers remain unchanged.')
    args = parser.parse_args()
    try:
        result = aggregate(args.workers)
    except (ValueError, KeyError, json.JSONDecodeError) as error:
        parser.exit(2, f'aggregation validation failed: {error}\n')
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out / 'summary.json', result)
    atomic_json(args.out / 'progress.json', {key: result[key] for key in
                ('status', 'complete', 'expected_trials', 'planned_trials', 'completed_trials',
                 'successes', 'success_rate', 'workers', 'missing_trial_ids')})
    print(json.dumps({key: result[key] for key in
          ('status', 'complete', 'expected_trials', 'planned_trials', 'completed_trials', 'successes', 'success_rate')}), flush=True)


if __name__ == '__main__':
    main()
