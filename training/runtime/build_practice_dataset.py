#!/usr/bin/env python3
"""Turn one recorded P rollout into a clearly labeled *practice* LeRobot dataset.

This is self imitation for testing the training pipeline. It is neither human R
nor human R-prime and must never enter the product comparison or final test.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


TASK = 'pick up the alphabet soup and place it in the basket'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_steps(path: Path) -> list[dict]:
    with gzip.open(path, 'rt') as stream:
        rows = [json.loads(line) for line in stream]
    steps = [row for row in rows if isinstance(row.get('step'), int) and 'executed_action' in row]
    if len(steps) < 2 or [row['step'] for row in steps] != list(range(len(steps))):
        raise ValueError('trace must contain sequential executed policy steps')
    for row in steps:
        if row.get('control_source') != 'policy':
            raise ValueError('practice source must contain policy actions only')
        if len(row['observation_state_before_action']) != 8 or len(row['executed_action']) != 7:
            raise ValueError('expected native state8/action7')
    return steps


def frames(path: Path, expected: int):
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.average_rate and round(float(stream.average_rate)) != 20:
            raise ValueError('source video must be 20 fps')
        count = 0
        for frame in container.decode(video=0):
            if count == expected:
                break  # evaluator appends one terminal observation after all actions
            rgb = frame.to_ndarray(format='rgb24')
            h, w, c = rgb.shape
            if c != 3 or w != 2 * h:
                raise ValueError(f'expected two square RGB cameras side by side, got {rgb.shape}')
            # Evaluator rotates each native observation by 180 degrees for MP4.
            left = np.ascontiguousarray(rgb[:, :h][::-1, ::-1])
            right = np.ascontiguousarray(rgb[:, h:][::-1, ::-1])
            yield left, right
            count += 1
        if count != expected:
            raise ValueError(f'video has {count} action frames, trace has {expected}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo-id', default='werescue/practice_p_self_imitation')
    args = parser.parse_args()
    done_path = args.trial / 'done.json'
    trace_path = args.trial / 'trace.jsonl.gz'
    video_path = args.trial / 'rollout.mp4'
    for p in (done_path, trace_path, video_path):
        if not p.is_file():
            parser.error(f'missing {p}')
    if args.output.exists():
        parser.error('output exists; use a fresh practice dataset path')
    done = json.loads(done_path.read_text())
    if (done.get('control_source') != 'policy' or done.get('contains_human_intervention')
            or done.get('language') != TASK or done.get('success_predicate') != 'native_LIBERO_check_success'):
        parser.error('trial must be a policy-only native soup-can rollout')
    steps = read_steps(trace_path)
    if len(steps) != done.get('executed_steps'):
        parser.error('trace and done.json executed-step counts differ')
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    image_shape = [steps[0].get('camera_height', 360), steps[0].get('camera_height', 360), 3]
    features = {
        'observation.state': {'dtype': 'float32', 'shape': (8,), 'names': None},
        'action': {'dtype': 'float32', 'shape': (7,), 'names': None},
        'observation.images.image': {'dtype': 'video', 'shape': tuple(image_shape),
                                     'names': ['height', 'width', 'channel']},
        'observation.images.image2': {'dtype': 'video', 'shape': tuple(image_shape),
                                      'names': ['height', 'width', 'channel']},
    }
    # Source trace and MP4 were written at the same 20-Hz action loop. The
    # evaluator writes each observation frame before executing that action.
    dataset = None
    for index, (left, right) in enumerate(frames(video_path, len(steps))):
        if dataset is None:
            if left.shape != tuple(image_shape):
                features['observation.images.image']['shape'] = left.shape
                features['observation.images.image2']['shape'] = right.shape
            dataset = LeRobotDataset.create(args.repo_id, fps=20, features=features,
                                            root=args.output, robot_type='franka',
                                            use_videos=True, video_backend='pyav')
        row = steps[index]
        dataset.add_frame({
            'observation.state': np.asarray(row['observation_state_before_action'], dtype=np.float32),
            'action': np.asarray(row['executed_action'], dtype=np.float32),
            'observation.images.image': left,
            'observation.images.image2': right,
            'task': TASK,
        })
    if dataset is None:
        raise RuntimeError('no decoded video frames')
    dataset.save_episode(parallel_encoding=False)
    dataset.finalize()
    provenance = {
        'schema_version': 'werescue.practice_dataset.v1',
        'kind': 'practice_only_self_imitation_from_P_rollout',
        'not_human_R_or_R_prime': True,
        'excluded_from_product_comparison': True,
        'source_trial_id': done['trial_id'],
        'source_policy': done['policy'],
        'source_trial_success': done['success'],
        'source_trace_sha256': sha256(trace_path),
        'source_video_sha256': sha256(video_path),
        'frames': len(steps), 'fps': 20,
        'timestamp_rule': 'frame_i=pre-action observation_i; action_i is executed at i/20 seconds',
        'source_video_rotation_reversed': True,
    }
    (args.output / 'WERESCUE_PRACTICE.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'dataset': str(args.output.resolve()), **provenance}, indent=2), flush=True)


if __name__ == '__main__':
    main()
