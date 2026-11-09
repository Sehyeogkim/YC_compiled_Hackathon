#!/usr/bin/env python3
"""Validate operator episode metadata; produce auditable R and R-prime indexes.

This validates metadata and local asset existence, not tensor semantics or replay.
Frame intervals are zero-based and half-open: [start_frame, end_frame).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


class ValidationError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def text(value: object, name: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"{name}: nonempty string required")
    return value  # type: ignore[return-value]


def local_ref(value: object, root: Path, name: str) -> dict:
    raw = text(value, name)
    path = (root / raw).resolve()
    require(path.is_file(), f"{name}: missing local file {path}")
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def validate_collection(data: dict, root: Path) -> dict:
    require(isinstance(data, dict), 'collection must be a JSON object')
    require(data.get('schema_version') == 'werescue.collection.v1', 'unsupported collection schema_version')
    episodes = data.get('episodes')
    require(isinstance(episodes, list) and bool(episodes), 'episodes: nonempty list required')
    seen: set[str] = set()
    lineage_splits: dict[tuple[str, str], str] = {}
    segments: list[dict] = []
    for episode in episodes:
        require(isinstance(episode, dict), 'episode must be an object')
        eid = text(episode.get('episode_id'), 'episode_id')
        require(eid not in seen, f'duplicate episode_id: {eid}')
        seen.add(eid)
        split = episode.get('split')
        require(split in {'train', 'val', 'test'}, f'{eid}: split must be train, val or test')
        for field in ('parent_episode_id', 'initial_state_id'):
            value = text(episode.get(field), f'{eid}.{field}')
            key = (field, value)
            require(key not in lineage_splits or lineage_splits[key] == split,
                    f'{eid}: {field} {value!r} leaks across splits')
            lineage_splits[key] = split
        for field in ('task_id', 'environment_id', 'robot_id', 'operator_id', 'action_schema'):
            text(episode.get(field), f'{eid}.{field}')
        require(episode.get('actions_are') == 'executed', f'{eid}: executed actions required, not requested commands')
        count = episode.get('frame_count')
        require(type(count) is int and count > 1, f'{eid}: frame_count must be an integer > 1')
        fps = episode.get('fps')
        require(type(fps) in (int, float) and math.isfinite(fps) and fps > 0, f'{eid}: positive finite fps required')
        camera_refs = episode.get('camera_refs')
        require(isinstance(camera_refs, dict) and bool(camera_refs), f'{eid}: camera_refs required')
        assets = {"cameras": {text(k, 'camera name'): local_ref(v, root, f'{eid}.camera.{k}')
                              for k, v in sorted(camera_refs.items())},
                  "state": local_ref(episode.get('state_ref'), root, f'{eid}.state_ref'),
                  "action": local_ref(episode.get('action_ref'), root, f'{eid}.action_ref'),
                  "timestamps": local_ref(episode.get('timestamps_ref'), root, f'{eid}.timestamps_ref')}
        previous_end = None
        for name, dataset in (('recovery', 'R_prime'), ('task', 'R')):
            phase = episode.get(name)
            require(isinstance(phase, dict), f'{eid}.{name}: phase required')
            start, end = phase.get('start_frame'), phase.get('end_frame')
            require(type(start) is int and type(end) is int and 0 <= start < end <= count,
                    f'{eid}.{name}: invalid half-open frame range')
            if previous_end is not None:
                require(start == previous_end, f'{eid}: recovery end must equal task start at explicit handoff')
            previous_end = end
            require(phase.get('control_source') == 'operator', f'{eid}.{name}: operator demonstrations required')
            require(phase.get('success') is True, f'{eid}.{name}: failed segments are not successful BC targets')
            predicate = text(phase.get('success_predicate'), f'{eid}.{name}.success_predicate')
            require(phase.get('predicate_verified') is True, f'{eid}.{name}: independent predicate verification required')
            evidence = local_ref(phase.get('evidence_ref'), root, f'{eid}.{name}.evidence_ref')
            require(phase.get('contains_reset') is False, f'{eid}.{name}: contains_reset must explicitly be false')
            segments.append({"segment_id": f'{eid}/{dataset}', "dataset": dataset,
                             "episode_id": eid, "parent_episode_id": episode['parent_episode_id'],
                             "initial_state_id": episode['initial_state_id'], "split": split,
                             "task_id": episode['task_id'], "environment_id": episode['environment_id'],
                             "robot_id": episode['robot_id'], "operator_id": episode['operator_id'],
                             "action_schema": episode['action_schema'], "fps": fps,
                             "start_frame": start, "end_frame": end,
                             "control_source": "operator", "success_predicate": predicate,
                             "evidence": evidence, "assets": assets})
    segments.sort(key=lambda row: (row['dataset'], row['split'], row['episode_id']))
    payload = {"schema_version": "werescue.segments.v1", "segments": segments,
               "validation_scope": "metadata_and_local_file_hashes_only; tensor_alignment_and_native_replay_not_verified"}
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    payload['manifest_sha256'] = hashlib.sha256(canonical).hexdigest()
    return payload


def validate_recovery_video(data: dict, root: Path) -> dict:
    require(isinstance(data, dict), 'video manifest must be a JSON object')
    require(data.get('schema_version') == 'werescue.recovery_video.v1', 'unsupported recovery video schema_version')
    for key in ('checkpoint', 'failure_state_id', 'recovery_state_id', 'task_id', 'recovery_predicate'):
        text(data.get(key), key)
    require(data.get('control_source') == 'policy', 'video must record policy execution, not operator or scripted control')
    require(data.get('contains_reset') is False, 'recovery video must not contain simulation resets')
    require(data.get('recovery_predicate_passed') is True, 'recovery predicate must pass')
    require(data.get('split') in {'demo', 'test'}, 'split must distinguish demo from held-out test')
    start, end = data.get('start_seconds'), data.get('end_seconds')
    require(type(start) in (int, float) and type(end) in (int, float)
            and math.isfinite(start) and math.isfinite(end) and 0 <= start < end,
            'invalid video interval')
    result = dict(data)
    for key in ('video_ref', 'execution_trace_ref', 'predicate_evidence_ref'):
        result[key] = local_ref(data.get(key), root, key)
    result['claim_scope'] = 'one_recorded_recovery; does_not_establish_success_rate'
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['collection', 'recovery-video'])
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    validator = validate_collection if args.kind == 'collection' else validate_recovery_video
    try:
        result = validator(data, args.input.resolve().parent)
    except ValidationError as error:
        parser.exit(2, f'validation failed: {error}\n')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(f'Validated {args.kind}: {args.output}')


if __name__ == '__main__':
    main()
