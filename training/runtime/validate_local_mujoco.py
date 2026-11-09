"""Validate browser-MuJoCo human action exports without claiming VLA trainability.

Usage: python training/runtime/validate_local_mujoco.py episode.jsonl.gz [more...] --output manifest.json

These raw files contain 20 Hz actions and simulator states. They do not contain
the two per-step native LIBERO policy cameras or the 8D LeRobot observation, so
this tool creates an audit/replay manifest, never a LeRobot training dataset.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
from pathlib import Path

RG_TASK = 'libero_object__pick_up_the_alphabet_soup_and_place_it_in_the_basket'


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite_vector(value, *, length: int | None = None) -> bool:
    return (isinstance(value, list) and (length is None or len(value) == length)
            and len(value) > 0 and all(isinstance(x, (float, int)) and not isinstance(x, bool)
                                      and math.isfinite(x) for x in value))


def inspect_rg_episode(path: Path, digest: str, meta: dict, stream) -> dict:
    require(meta.get('control_hz') == 20, f'{path}: expected 20 Hz control')
    require(meta.get('source') == 'human_browser_unverified', f'{path}: source is not human browser')
    require(meta.get('action_dim') == 7, f'{path}: expected 7D actions')
    require(meta.get('scene_sha256') and len(meta['scene_sha256']) == 64,
            f'{path}: scene SHA256 missing')
    require(meta.get('controller') == 'local_mujoco_reimplementation',
            f'{path}: controller provenance missing')
    randomization = meta.get('robot_joint_randomization')
    require(isinstance(randomization, dict) and isinstance(randomization.get('seed'), (int, float))
            and math.isfinite(randomization['seed'])
            and finite_vector(randomization.get('robotJointQposBefore'))
            and finite_vector(randomization.get('robotJointQposAfter')),
            f'{path}: robot joint randomization provenance missing')
    require(finite_vector(meta.get('initial_robot_qpos')), f'{path}: initial robot joints missing')
    boundary = meta.get('recovery_complete_step')
    require(boundary is None or (isinstance(boundary, int) and boundary > 0),
            f'{path}: invalid recovery boundary')
    count = 0
    first_time = None
    action_hash = hashlib.sha256()
    terminal = None
    boundary_event = None
    for raw in stream:
        if not raw.strip():
            continue
        row = json.loads(raw)
        if row.get('type') == 'recovery_complete':
            require(boundary_event is None, f'{path}: duplicate recovery boundary')
            boundary_event = row
            continue
        if row.get('type') == 'terminal':
            require(terminal is None, f'{path}: duplicate terminal event')
            terminal = row
            continue
        require(row.get('type') == 'step' and row.get('step') == count,
                f'{path}: noncontiguous action step {count}')
        t = row.get('sim_time_before')
        require(isinstance(t, (int, float)) and math.isfinite(t), f'{path}: missing simulator time')
        if first_time is None:
            first_time = t
        require(math.isclose(t, first_time + count / 20, abs_tol=.002),
                f'{path}: noncontiguous 20 Hz time at step {count}')
        action = row.get('executed_action')
        require(finite_vector(action, length=7) and all(-1.00001 <= x <= 1.00001 for x in action),
                f'{path}: missing bounded 7D executed action at {count}')
        state = row.get('pre_action_state')
        require(isinstance(state, dict) and finite_vector(state.get('qpos'))
                and finite_vector(state.get('qvel')), f'{path}: missing pre-action qpos/qvel at {count}')
        require(row.get('segment') == ('recovery' if boundary is None or count < boundary else 'task'),
                f'{path}: incorrect R_prime/R split at {count}')
        action_hash.update(json.dumps(action, separators=(',', ':'), allow_nan=False).encode() + b'\n')
        count += 1
    require(count > 0 and count == meta.get('steps'), f'{path}: step count mismatch')
    require(terminal is not None and terminal.get('step') == count,
            f'{path}: terminal event missing or wrong step')
    require(boundary_event is None if boundary is None else boundary_event.get('step') == boundary,
            f'{path}: recovery boundary metadata/event mismatch')
    require(boundary is None or boundary <= count, f'{path}: boundary past final action')
    return {
        'path': str(path.resolve()), 'file_sha256': digest,
        'episode_id': meta.get('episode_id'), 'task_id': meta.get('task_id'),
        'scene_sha256': meta['scene_sha256'], 'model_sha256': meta.get('model_sha256'),
        'source': meta['source'], 'control_hz': 20, 'steps': count,
        'recovery_steps': boundary if boundary is not None else count,
        'task_steps': count - boundary if boundary is not None else 0,
        'sim_seconds': round(count * .05, 6), 'success_reported': terminal.get('success'),
        'randomization': randomization, 'executed_actions_sha256': action_hash.hexdigest(),
        'native_replay_required': True, 'ready_for_lerobot_training': False,
        'missing_for_lerobot_training': ['paired native agentview/wrist RGB per pre-action tick',
                                        'verified 8D LeRobot observation.state per pre-action tick',
                                        'native controller/action replay parity and success predicate'],
    }


def inspect_episode(path: Path) -> dict:
    require(path.is_file(), f'{path}: file missing')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        first = stream.readline()
        require(first.strip(), f'{path}: empty export')
        meta = json.loads(first)
        require(meta.get('type') == 'metadata', f'{path}: first row must be metadata')
        if meta.get('schema') == 'rg.local_mujoco_human_episode.v1':
            return inspect_rg_episode(path, digest, meta, stream)
        require(meta.get('schema') == 'wemine.physical-arena.episode.v1',
                f'{path}: unsupported export schema')
        require(meta.get('control_hz') == 20, f'{path}: expected 20 Hz control')
        require(meta.get('game_mode') == 'rg_local_collection' and meta.get('task_id') == RG_TASK,
                f'{path}: expected RG local soup-can collection task')
        start = meta.get('start_selection')
        require(isinstance(start, dict) and start.get('method') == 'random_official_start_without_immediate_repeat'
                and isinstance(start.get('selected_index'), int)
                and start.get('selected_state_sha256') == meta.get('start_state_sha256'),
                f'{path}: randomized official start provenance missing')
        require(meta.get('source') in ('human_browser_unverified', 'human_browser', 'human_local_mujoco'),
                f'{path}: source is not a human browser run')
        require(meta.get('assistance', {}).get('autopilot') is False,
                f'{path}: autopilot run cannot be labeled human')
        require(isinstance(meta.get('manifest_sha256'), str) and len(meta['manifest_sha256']) == 64,
                f'{path}: scene manifest hash missing')
        require(isinstance(meta.get('model_sha256'), str) and len(meta['model_sha256']) == 64,
                f'{path}: MuJoCo model hash missing')
        require(isinstance(meta.get('initial_state'), dict), f'{path}: initial state missing')
        require(finite_vector(meta['initial_state'].get('qpos')),
                f'{path}: initial qpos missing')
        count = 0
        last_t = None
        action_hash = hashlib.sha256()
        for raw in stream:
            if not raw.strip():
                continue
            row = json.loads(raw)
            require(row.get('type') == 'step', f'{path}: unexpected row type')
            require(row.get('step') == count, f'{path}: noncontiguous step at {count}')
            t = row.get('t')
            require(isinstance(t, (int, float)) and math.isfinite(t)
                    and math.isclose(t, .05 * count, abs_tol=1e-5),
                    f'{path}: step {count} is not on 20 Hz timeline')
            require(last_t is None or t > last_t, f'{path}: nonmonotonic time')
            action = row.get('action')
            require(finite_vector(action, length=7) and all(-1.00001 <= x <= 1.00001 for x in action),
                    f'{path}: step {count} lacks bounded 7D executed action')
            state = row.get('state')
            require(isinstance(state, dict) and finite_vector(state.get('qpos'))
                    and finite_vector(state.get('qvel')),
                    f'{path}: step {count} lacks native pre-action simulator state')
            action_hash.update(json.dumps(action, separators=(',', ':'), allow_nan=False).encode() + b'\n')
            last_t = t
            count += 1
    require(count > 0, f'{path}: no action steps')
    result = meta.get('result')
    require(isinstance(result, dict), f'{path}: incomplete run; finish/export before validation')
    require(result.get('steps') == count, f'{path}: result step count differs from rows')
    return {
        'path': str(path.resolve()), 'file_sha256': digest,
        'episode_id': meta.get('id'), 'task_id': meta.get('task_id'),
        'variant': meta.get('variant'), 'scene_manifest_sha256': meta['manifest_sha256'],
        'start_selection': start,
        'mujoco_model_sha256': meta['model_sha256'],
        'source': meta['source'], 'control_hz': 20, 'steps': count,
        'sim_seconds': round(count * .05, 6), 'success_reported': result.get('success'),
        'executed_actions_sha256': action_hash.hexdigest(),
        'source_training_eligible': meta.get('training_eligible') is True,
        'native_replay_required': True, 'ready_for_lerobot_training': False,
        'missing_for_lerobot_training': ['paired 256x256 native agentview RGB per pre-action tick',
                                        'paired 256x256 native wrist RGB per pre-action tick',
                                        'verified 8D LeRobot observation.state per pre-action tick',
                                        'native action replay alignment and success predicate verification'],
    }


def inspect_export(path: Path) -> list[dict]:
    """Accept one episode or the browser's Download all collection bundle."""
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        first = stream.readline()
        require(first.strip(), f'{path}: empty export')
        header = json.loads(first)
        if header.get('type') != 'collection':
            return [inspect_episode(path)]
        require(header.get('schema') == 'rg.local_mujoco_collection.v1',
                f'{path}: unsupported collection schema')
        chunks: list[list[str]] = []
        for line in stream:
            if not line.strip():
                continue
            kind = json.loads(line).get('type')
            if kind == 'metadata':
                chunks.append([])
            require(bool(chunks), f'{path}: episode row before metadata')
            chunks[-1].append(line)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = []
        for chunk in chunks:
            source = io.StringIO(''.join(chunk))
            meta = json.loads(source.readline())
            require(meta.get('schema') == 'rg.local_mujoco_human_episode.v1',
                    f'{path}: collection contains a non-RG episode')
            rows.append(inspect_rg_episode(path, digest, meta, source))
        require(len(rows) == header.get('count')
                and [row['episode_id'] for row in rows] == header.get('episode_ids'),
                f'{path}: collection index differs from episode content')
        return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episodes', type=Path, nargs='+')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    rows = [row for path in args.episodes for row in inspect_export(path)]
    ids = [row['episode_id'] for row in rows]
    require(all(ids) and len(set(ids)) == len(ids), 'duplicate or missing episode IDs')
    manifest = {'schema_version': 'rg.local_mujoco_replay_manifest.v1',
                'purpose': 'human action/state audit for native replay',
                'training_eligible': False, 'episodes': rows}
    body = json.dumps(manifest, indent=2, ensure_ascii=False) + '\n'
    if args.output:
        require(not args.output.exists(), f'{args.output}: choose a new output path')
        args.output.write_text(body)
    print(body, end='')


if __name__ == '__main__':
    main()
