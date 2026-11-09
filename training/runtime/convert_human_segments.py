#!/usr/bin/env python3
"""Convert up to ten qualified UI operator sessions into separate R and R-prime datasets.

Input is inference/export_segments.py's native_segment_selection.v1 manifest,
its R/R_prime NPZ files, and untouched native PNG/event/state files. This
converter does not infer a missing frame, claim independent verification for
an operator annotation, or accept practice/policy actions as human data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

CAMERAS = ('agentview', 'wrist')
SEGMENTS = (('R_prime', 0, 'recovery'), ('R', 1, 'task'))
IMAGE_SHAPE = (256, 256, 3)  # actual inference/runtime.py UI native render size


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def safe_child(root: Path, name: str) -> Path:
    require(isinstance(name, str) and name and Path(name).name == name,
            f'unsafe session or file name: {name!r}')
    return root / name


def image_file(raw: Path, ref: str) -> Path:
    require(isinstance(ref, str) and Path(ref).name == ref and ref.endswith('.png'),
            f'unsafe image reference: {ref!r}')
    path = raw / ref
    require(path.is_file(), f'missing native image: {path}')
    return path


def policy_rgb(path: Path) -> np.ndarray:
    """Undo UI's `_rgb` 180-degree preview transform exactly once.

    Native policy execution uses `self.observation['pixels']` via `_batch()`;
    inference/runtime.py saves PNGs from `_rgb()` after H and W reversal.
    """
    with Image.open(path) as picture:
        require(picture.mode == 'RGB' and picture.size == IMAGE_SHAPE[:2],
                f'expected 256x256 UI RGB PNG: {path}')
        return np.ascontiguousarray(np.asarray(picture)[::-1, ::-1])


def load_arrays(path: Path, phase: int) -> dict:
    require(path.is_file(), f'missing exported segment: {path}')
    with np.load(path, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    needed = ('observation_state', 'action', 'next_observation_state', 'sim_state_before',
              'sim_state_after', 'native_step_index', 'frame_index', 'sim_timestamp',
              'next_sim_timestamp', 'phase', 'task_success', 'png_refs', 'next_png_refs',
              'control_source')
    require(all(key in data for key in needed), f'{path}: missing native alignment fields')
    count = len(data['action'])
    require(count >= 1 and all(len(data[key]) == count for key in needed),
            f'{path}: inconsistent or empty segment length')
    for name, shape in (('observation_state', (count, 8)), ('next_observation_state', (count, 8)),
                        ('action', (count, 7)), ('png_refs', (count, 2)),
                        ('next_png_refs', (count, 2))):
        require(data[name].shape == shape, f'{path}: {name} expected {shape}, got {data[name].shape}')
    for name in ('observation_state', 'next_observation_state', 'action', 'sim_state_before',
                 'sim_state_after', 'sim_timestamp', 'next_sim_timestamp'):
        require(np.isfinite(data[name]).all(), f'{path}: nonfinite {name}')
    require((np.abs(data['action']) <= 1).all(), f'{path}: native action outside [-1,1]')
    require(np.all(data['phase'] == phase), f'{path}: wrong phase marker')
    require(np.all(data['control_source'] == 'operator'), f'{path}: policy/action contamination')
    require(np.all(np.diff(data['native_step_index']) == 1), f'{path}: native steps are not contiguous')
    require(np.all(np.diff(data['frame_index']) == 1), f'{path}: frame indices are not contiguous')
    dt = data['next_sim_timestamp'] - data['sim_timestamp']
    require(np.allclose(dt, .05, atol=.001) and np.all(np.diff(data['sim_timestamp']) > 0),
            f'{path}: not one 20 Hz native control tick per action')
    for index in range(count - 1):
        require(np.allclose(data['next_observation_state'][index], data['observation_state'][index + 1], atol=1e-5),
                f'{path}: state discontinuity at frame {index}')
        require(np.array_equal(data['next_png_refs'][index], data['png_refs'][index + 1]),
                f'{path}: image reference discontinuity at frame {index}')
        require(np.allclose(data['sim_state_after'][index], data['sim_state_before'][index + 1], atol=1e-6),
                f'{path}: simulator-state discontinuity at frame {index}')
    return data


def validate_sources(manifest_path: Path, raw_root: Path, exports_root: Path) -> tuple[list[dict], dict]:
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get('schema_version') == 'werescue.native_segment_selection.v1',
            'input must be UI native_segment_selection.v1')
    selected = manifest.get('training_selection')
    require(isinstance(selected, list) and 1 <= len(selected) <= 10,
            'between one and ten qualified source interventions are required')
    require(not (raw_root / 'WERESCUE_PRACTICE.json').exists()
            and not (exports_root / 'WERESCUE_PRACTICE.json').exists(),
            'practice self-imitation data cannot enter human conversion')
    listed = {row.get('session_id') for row in manifest.get('sessions', [])}
    ids, initial_splits, specs = set(), {}, []
    language = None
    for row in selected:
        sid = row.get('session_id')
        require(isinstance(sid, str) and sid not in ids and sid in listed, 'duplicate or unlisted session')
        ids.add(sid)
        raw = safe_child(raw_root, sid)
        exported = safe_child(exports_root, sid)
        require(raw.is_dir() and exported.is_dir(), f'{sid}: raw/export directory missing')
        saved = json.loads((exported / 'export.json').read_text())
        require(saved == row, f'{sid}: selected manifest entry differs from export.json')
        require(saved.get('schema_version') == 'werescue.native_segment_export.v1'
                and saved.get('training_eligible') is True
                and saved.get('qualification') in ('qualified', 'operator_annotated')
                and saved.get('native_task_success') is True
                and saved.get('contains_reset') is False
                and saved.get('raw_preserved') is True,
                f'{sid}: export is not qualified human intervention')
        require(saved.get('operator_id') and not saved.get('reasons'), f'{sid}: missing operator or export rejection')
        if saved['qualification'] == 'operator_annotated':
            require(manifest.get('accept_operator_annotations') is True
                    and saved.get('recovery_predicate_verified') is False,
                    f'{sid}: annotation status must remain explicit')
        else:
            evidence = saved.get('recovery_evidence', {}).get('content', {})
            require(saved.get('recovery_predicate_verified') is True
                    and evidence.get('session_id') == sid
                    and evidence.get('boundary_step') == saved.get('recovery_boundary_step')
                    and evidence.get('passed') is True
                    and evidence.get('independently_verified') is True,
                    f'{sid}: independent recovery evidence invalid')
        meta_path, events_path = raw / 'metadata.json', raw / 'events.jsonl'
        require(meta_path.is_file() and events_path.is_file(), f'{sid}: native metadata/events missing')
        meta = json.loads(meta_path.read_text())
        require(meta == saved.get('source_metadata') and meta.get('schema') == 'werescue.native_session.v1',
                f'{sid}: source metadata changed or wrong schema')
        require(meta.get('session_id') == sid and meta.get('suite') == 'libero_object'
                and meta.get('task_id') == 0 and meta.get('control_hz') == 20
                and meta.get('control_mode') == 'relative'
                and meta.get('image_transform') == 'raw native image flipped on H and W, matching LeRobot LIBERO processor'
                and saved.get('image_orientation') == 'upright_RGB_already_flipped_H_and_W; DO_NOT_flip_again',
                f'{sid}: native task, image, action or clock contract differs')
        require(meta.get('initial_state_id') in range(20), f'{sid}: reserved initial state in training')
        initial_splits.setdefault(meta['initial_state_id'], 'train')
        require(saved.get('events_sha256') == file_sha256(events_path), f'{sid}: event log changed')
        if language is None:
            language = meta.get('task')
            require(isinstance(language, str) and 'alphabet soup' in language and 'basket' in language,
                    'expected soup can into basket task')
        require(meta.get('task') == language, f'{sid}: task language differs')
        events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
        start, boundary, end = (saved.get(name) for name in ('start_step', 'recovery_boundary_step', 'end_step'))
        require(all(type(v) is int for v in (start, boundary, end)) and start < boundary < end,
                f'{sid}: invalid R-prime/R boundary')
        executed = [e for e in events if e.get('kind') == 'executed_action' and start <= e.get('observation_step', -1) < end]
        require([e.get('observation_step') for e in executed] == list(range(start, end))
                and all(e.get('control_source') == 'operator' for e in executed),
                f'{sid}: native human action event sequence differs')
        terminal = [e for e in events if e.get('kind') == 'episode_terminal' and e.get('step') == end]
        require(bool(terminal) and terminal[-1].get('success') is True
                and terminal[-1].get('predicate') == 'native_LIBERO_check_success',
                f'{sid}: native task success not independently present in event log')
        require(not any(e.get('kind') == 'session_reset_requested' and not e.get('was_terminal') for e in events),
                f'{sid}: reset during human segment')
        arrays = {}
        for name, phase, _ in SEGMENTS:
            arrays[name] = load_arrays(exported / f'{name}.npz', phase)
        a, b = arrays['R_prime'], arrays['R']
        require(len(a['action']) == boundary - start and len(b['action']) == end - boundary,
                f'{sid}: segment lengths disagree with annotation boundary')
        require(a['native_step_index'][0] == start and b['native_step_index'][0] == boundary,
                f'{sid}: segment starts differ from native event indices')
        require(np.allclose(a['next_observation_state'][-1], b['observation_state'][0], atol=1e-5)
                and np.array_equal(a['next_png_refs'][-1], b['png_refs'][0])
                and np.allclose(a['sim_state_after'][-1], b['sim_state_before'][0], atol=1e-6),
                f'{sid}: recovery-to-task boundary is discontinuous')
        require(not np.any(a['task_success']) and not np.any(b['task_success'][:-1])
                and bool(b['task_success'][-1]), f'{sid}: native success marker placement differs')
        combined = list(a['action']) + list(b['action'])
        for index, event in enumerate(executed):
            require(np.allclose(combined[index], event['action'], atol=1e-6),
                    f'{sid}: NPZ action differs from executed event at step {start + index}')
        image_digests = {}
        for name, _, _ in SEGMENTS:
            arr = arrays[name]
            h = hashlib.sha256()
            for refs in list(arr['png_refs']) + [arr['next_png_refs'][-1]]:
                for camera, ref in zip(CAMERAS, refs):
                    require(f'.{camera}.png' in str(ref), f'{sid}: camera order differs')
                    path = image_file(raw, str(ref))
                    with Image.open(path) as picture:
                        require(picture.mode == 'RGB' and picture.size == IMAGE_SHAPE[:2],
                                f'{sid}: expected 256x256 UI-native RGB PNG: {path}')
                    h.update(str(ref).encode() + b'\0' + bytes.fromhex(file_sha256(path)))
            image_digests[name] = h.hexdigest()
        specs.append({'session_id': sid, 'raw': raw, 'exported': exported, 'meta': meta,
                      'export': saved, 'arrays': arrays, 'image_digests': image_digests,
                      'events_sha256': file_sha256(events_path), 'metadata_sha256': file_sha256(meta_path)})
    return specs, manifest


def segment_manifest(specs: list[dict], input_manifest: Path) -> dict:
    rows = []
    for spec in specs:
        sid, meta, source = spec['session_id'], spec['meta'], spec['export']
        for name, _, _ in SEGMENTS:
            arr = spec['arrays'][name]
            rows.append({
                'segment_id': f'{sid}/{name}', 'dataset': name, 'episode_id': sid,
                'parent_episode_id': sid, 'initial_state_id': str(meta['initial_state_id']),
                'source_failure_policy': meta.get('model'),
                'split': 'train', 'task_id': '0', 'environment_id': 'libero_object',
                'robot_id': 'Franka Panda', 'operator_id': source['operator_id'],
                'action_schema': 'native_relative_OSC_POSE_7D', 'fps': 20,
                'start_frame': int(arr['native_step_index'][0]),
                'end_frame': int(arr['native_step_index'][-1]) + 1,
                'control_source': 'operator', 'contains_reset': False,
                'success_predicate': ('independent_recovery_pose_scene_predicate' if name == 'R_prime' and source['recovery_predicate_verified']
                                      else 'operator_recovery_annotation' if name == 'R_prime'
                                      else 'native_LIBERO_check_success'),
                'predicate_verified': bool(source['recovery_predicate_verified']) if name == 'R_prime' else True,
                'qualification': source['qualification'], 'source_manifest_sha256': file_sha256(input_manifest),
                'source_metadata_sha256': spec['metadata_sha256'],
                'source_events_sha256': spec['events_sha256'],
                'source_npz_sha256': file_sha256(spec['exported'] / f'{name}.npz'),
                'source_images_sha256': spec['image_digests'][name],
                'frames': len(arr['action']),
                'source_image_orientation': 'UI upright RGB, H and W already flipped from native',
                'dataset_image_orientation': 'native policy RGB after one H and W reversal of source PNG',
                'image_transform': 'reverse_UI_preview_flip_once',
                'timestamp_rule': 'LeRobot frame i/20 from native sim ticks; source UTC request time retained only in NPZ',
            })
    result = {'schema_version': 'werescue.segments.v1',
              'source_schema_version': 'werescue.native_segment_selection.v1',
              'source_manifest_sha256': file_sha256(input_manifest),
              'segments': rows,
              'validation_scope': 'native UI operator actions and source files verified; exact simulator replay and real robot transfer not verified'}
    result['manifest_sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return result


def write_dataset(specs: list[dict], name: str, root: Path, repo_id: str, language: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    features = {
        'observation.state': {'dtype': 'float32', 'shape': (8,), 'names': None},
        'action': {'dtype': 'float32', 'shape': (7,), 'names': None},
        'observation.images.image': {'dtype': 'video', 'shape': IMAGE_SHAPE,
                                     'names': ['height', 'width', 'channel']},
        'observation.images.image2': {'dtype': 'video', 'shape': IMAGE_SHAPE,
                                      'names': ['height', 'width', 'channel']},
    }
    dataset = LeRobotDataset.create(repo_id, fps=20, features=features, root=root,
                                    robot_type='franka', use_videos=True, video_backend='pyav')
    for spec in specs:
        arr = spec['arrays'][name]
        for i in range(len(arr['action'])):
            images = []
            for ref in arr['png_refs'][i]:
                images.append(policy_rgb(image_file(spec['raw'], str(ref))))
            dataset.add_frame({'observation.state': np.asarray(arr['observation_state'][i], dtype=np.float32),
                               'action': np.asarray(arr['action'][i], dtype=np.float32),
                               'observation.images.image': images[0],
                               'observation.images.image2': images[1], 'task': language})
        dataset.save_episode(parallel_encoding=False)
    dataset.finalize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True, help='UI export manifest.json')
    parser.add_argument('--raw-root', type=Path, required=True, help='Untouched UI UUID raw session dirs')
    parser.add_argument('--exports-root', type=Path, required=True, help='UI export UUID dirs')
    parser.add_argument('--output-root', type=Path, required=True, help='Fresh R and R_prime parent')
    parser.add_argument('--repo-prefix', default='werescue/human_native')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if not args.validate_only and args.output_root.exists():
        parser.error('output-root already exists; choose a fresh destination')
    specs, _ = validate_sources(args.manifest, args.raw_root, args.exports_root)
    derived = segment_manifest(specs, args.manifest)
    print(json.dumps({'source_interventions': len(specs), 'segments': len(derived['segments']),
                      'independent_recovery_evidence': sum(s['export']['recovery_predicate_verified'] for s in specs),
                      'operator_annotated_recovery': sum(not s['export']['recovery_predicate_verified'] for s in specs),
                      'validate_only': args.validate_only}, indent=2), flush=True)
    if args.validate_only:
        return
    language = specs[0]['meta']['task']
    args.output_root.mkdir(parents=True)
    manifest_bytes = json.dumps(derived, indent=2, sort_keys=True).encode() + b'\n'
    for name, arm in (('R_prime', 'recovery'), ('R', 'task')):
        root = args.output_root / name
        write_dataset(specs, name, root, f'{args.repo_prefix}_{name}', language)
        (root / 'WERESCUE_SEGMENTS.json').write_bytes(manifest_bytes)
        provenance = {
            'schema_version': 'werescue.human_lerobot_dataset.v1',
            'kind': 'qualified_operator_segments', 'arm': arm, 'dataset': name,
            'control_source': 'operator', 'source_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
            'segment_ids': [row['segment_id'] for row in derived['segments'] if row['dataset'] == name],
            'source_interventions': len(specs),
            'recovery_evidence_scope': 'independent only when source segment predicate_verified=true',
            'practice_data_excluded': True,
            'image_orientation': 'policy_view_raw_native_restored_from_upright_png_by_H_W_flip',
        }
        (root / 'WERESCUE_HUMAN_DATASET.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'R': str((args.output_root / 'R').resolve()),
                      'R_prime': str((args.output_root / 'R_prime').resolve()),
                      'human_source_episodes': len(specs), 'training_claim': 'none; conversion only'}), flush=True)


if __name__ == '__main__':
    main()
