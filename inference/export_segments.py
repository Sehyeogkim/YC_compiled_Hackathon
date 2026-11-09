#!/usr/bin/env python3
"""Export native raw operator segments; preserve annotation qualification explicitly."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

CAMERAS = ('agentview', 'wrist')


def read_json(path):
    return json.loads(path.read_text())


def export_session(raw: Path, output: Path, *, accept_annotations=False,
                   operator_id=None, recovery_evidence=None):
    meta = read_json(raw / 'metadata.json')
    event_file = raw / 'events.jsonl'
    events = [json.loads(line) for line in event_file.read_text().splitlines() if line.strip()]
    sid = meta['session_id']
    result = {'session_id': sid, 'raw_root': str(raw.resolve()), 'qualification': 'unqualified',
              'training_eligible': False, 'recovery_predicate_verified': False, 'reasons': [],
              'source_metadata': meta, 'operator_id': operator_id,
              'events_sha256': hashlib.sha256(event_file.read_bytes()).hexdigest(),
              'image_orientation': 'upright_RGB_already_flipped_H_and_W; DO_NOT_flip_again',
              'camera_order': list(CAMERAS), 'png_refs_relative_to': 'raw_root',
              'timestamp_clock': 'UTC action_requested minus metadata.created_at; NOT observation capture or monotonic clock',
              'schema_version': 'werescue.native_segment_export.v1',
              'training_ready': False}
    markers = [e for e in events if e['kind'] == 'recovery_complete_annotation']
    if len(markers) != 1:
        result['reasons'].append('Exactly one recovery boundary is required')
        return result
    marker = markers[0]
    boundary = marker['step']
    executed = [e for e in events if e['kind'] == 'executed_action']
    operator = [e for e in executed if e['control_source'] == 'operator']
    if not operator:
        result['reasons'].append('No executed human actions')
        return result
    start, end = operator[0]['observation_step'], operator[-1]['observation_step'] + 1
    result.update(start_step=start, recovery_boundary_step=boundary, end_step=end)
    selected = [e for e in executed if start <= e['observation_step'] < end]
    if [e['observation_step'] for e in selected] != list(range(start, end)):
        result['reasons'].append('Action steps must be unique and contiguous')
        return result
    if any(e['control_source'] != 'operator' for e in selected):
        result['reasons'].append('Policy actions inside human intervention')
        return result
    if not start < boundary < end:
        result['reasons'].append('Both R_prime and R must be nonempty')
        return result
    if any(e['segment'] != ('recovery' if e['observation_step'] < boundary else 'task') for e in selected):
        result['reasons'].append('Source phase and boundary disagree')
        return result
    terminal = [e for e in events if e['kind'] == 'episode_terminal' and e['step'] == end]
    success = bool(terminal and terminal[-1].get('success') is True and
                   terminal[-1].get('predicate') == 'native_LIBERO_check_success')
    result['native_task_success'] = success
    if not success:
        result['reasons'].append('Native task SUCCESS not recorded')
    if any(e['kind'] == 'session_reset_requested' and not e.get('was_terminal') for e in events):
        result['reasons'].append('Reset aborted the active segment')
    if not operator_id:
        result['reasons'].append('Explicit --operator-id required for training eligibility')
    # Held-out and video starts are excluded from training selection by default.
    if meta.get('initial_state_id') not in range(20):
        result['reasons'].append('Initial state is reserved for validation/test/video, not training')
    verified = False
    if recovery_evidence:
        evidence = read_json(recovery_evidence)
        verified = (evidence.get('session_id') == sid and evidence.get('boundary_step') == boundary
                    and evidence.get('passed') is True and evidence.get('independently_verified') is True)
        result['recovery_evidence'] = {'path': str(recovery_evidence.resolve()), 'content': evidence}
    result['recovery_predicate_verified'] = verified
    if not verified and not accept_annotations:
        result['reasons'].append('Recovery is operator-annotated; opt in with --accept-operator-annotations')
    result['qualification'] = ('qualified' if verified else 'operator_annotated') if not result['reasons'] else 'unqualified'
    result['training_eligible'] = not result['reasons']
    requested = [e for e in events if e['kind'] == 'action_requested' and
                 start <= e['observation_step'] < end]
    if [e['observation_step'] for e in requested] != list(range(start, end)):
        result['reasons'].append('Ambiguous or unexecuted action retries inside human segment')
        result.update(qualification='unqualified', training_eligible=False)
        return result
    requests = {e['observation_step']: e for e in requested}
    rows = []
    created = datetime.fromisoformat(meta['created_at']).timestamp()
    for e in selected:
        i = e['observation_step']
        before_path, after_path = raw / f'{i:06d}.state.npz', raw / f'{i+1:06d}.state.npz'
        refs = [f'{i:06d}.{c}.png' for c in CAMERAS]
        next_refs = [f'{i+1:06d}.{c}.png' for c in CAMERAS]
        if not before_path.is_file() or not after_path.is_file() or not all((raw / f).is_file() for f in refs + next_refs):
            result['reasons'].append(f'Missing exact observation or next observation at native step {i}; no synthesis')
            result.update(qualification='unqualified', training_eligible=False)
            return result
        with np.load(before_path, allow_pickle=False) as b, np.load(after_path, allow_pickle=False) as a:
            state, next_state = b['observation_state'], a['observation_state']
            action = np.asarray(e['action'], dtype=np.float32)
            if state.shape != (8,) or next_state.shape != (8,) or action.shape != (7,) or not all(np.isfinite(x).all() for x in (state, next_state, action)):
                raise ValueError(f'{sid}: invalid state/action tensor at step {i}')
            dispatch = datetime.fromisoformat(requests[i]['utc']).timestamp() - created
            rows.append({'observation_state': state.copy(), 'action': action,
                'next_observation_state': next_state.copy(),
                'sim_state_before': b['flattened_state'].copy(), 'sim_state_after': a['flattened_state'].copy(),
                'actuator_ctrl_after': a['ctrl'].copy(), 'native_step_index': i,
                'frame_index': i - start, 'timestamp': dispatch, 'action_dispatch_timestamp': dispatch,
                'next_timestamp': datetime.fromisoformat(e['utc']).timestamp() - created,
                'sim_timestamp': float(b['sim_time']), 'next_sim_timestamp': float(a['sim_time']),
                'phase': 0 if i < boundary else 1, 'task_success': bool(success and i == end - 1),
                'png_refs': refs, 'next_png_refs': next_refs, 'control_source': 'operator'})
    destination = output / sid
    if not result['training_eligible']:
        destination = destination / 'candidate'
    destination.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    for name, lo, hi in (('R_prime', 0, boundary - start), ('R', boundary - start, end - start)):
        path = destination / f'{name}.npz'
        np.savez_compressed(path, **{key: value[lo:hi] for key, value in arrays.items()})
        result[name] = str(path.resolve())
    result['boundary_alignment'] = 'last R_prime next_observation_state/next_png_refs == first R observation_state/png_refs'
    result['contains_reset'] = False
    result['raw_preserved'] = True
    (destination / 'export.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw_root', type=Path, help='Directory containing native session UUID directories')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--session-id', action='append', help='Select explicit UUID(s); otherwise scan metadata.json')
    parser.add_argument('--operator-id')
    parser.add_argument('--accept-operator-annotations', action='store_true', help='Allow annotated recovery boundaries; never claims independent verification')
    parser.add_argument('--recovery-evidence-root', type=Path, help='Optional <session-id>.json independent recovery evidence')
    parser.add_argument('--exclude-sessions', type=Path, default=Path(__file__).with_name('test_sessions.json'))
    args = parser.parse_args()
    excluded = set(read_json(args.exclude_sessions)['excluded_session_ids'])
    sessions = args.session_id or sorted(p.parent.name for p in args.raw_root.glob('*/metadata.json'))
    results = []
    for sid in sessions:
        if sid in excluded:
            results.append({'session_id': sid, 'qualification': 'excluded_test', 'training_eligible': False})
            continue
        if Path(sid).name != sid:
            raise ValueError('Session IDs must be directory names')
        evidence = args.recovery_evidence_root / f'{sid}.json' if args.recovery_evidence_root else None
        if evidence is not None and not evidence.is_file():
            evidence = None
        try:
            result = export_session(args.raw_root / sid, args.output,
                accept_annotations=args.accept_operator_annotations, operator_id=args.operator_id,
                recovery_evidence=evidence)
        except (ValueError, KeyError, OSError) as exc:
            result = {'session_id': sid, 'qualification': 'unqualified', 'training_eligible': False, 'reasons': [str(exc)]}
        results.append(result)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {'schema_version': 'werescue.native_segment_selection.v1', 'sessions': results,
                'training_selection': [r for r in results if r['training_eligible']],
                'accept_operator_annotations': args.accept_operator_annotations,
                'warning': 'Raw segment export; pinned LeRobot dataset conversion still required. Independent recovery verification is not implied.'}
    path = args.output / 'manifest.json'
    path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'manifest': str(path.resolve()), 'eligible': len(manifest['training_selection']), 'sessions': len(results)}))


if __name__ == '__main__':
    main()
