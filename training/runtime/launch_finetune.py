#!/usr/bin/env python3
"""Prepare or execute one independent P→task/recovery finetune; no training by default."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

from download_checkpoint import MODEL_ID, MODEL_REVISION


def checked_human_provenance(root: Path, arm: str, episode_count: int) -> dict:
    if (root / 'WERESCUE_PRACTICE.json').exists():
        raise ValueError('practice self-imitation data cannot enter human R/R-prime training')
    path = root / 'WERESCUE_HUMAN_DATASET.json'
    manifest_path = root / 'WERESCUE_SEGMENTS.json'
    if not path.is_file() or not manifest_path.is_file():
        raise ValueError('human dataset requires WERESCUE_HUMAN_DATASET.json and WERESCUE_SEGMENTS.json')
    data = json.loads(path.read_text())
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    expected_dataset = 'R' if arm == 'task' else 'R_prime'
    if (data.get('schema_version') != 'werescue.human_lerobot_dataset.v1'
            or data.get('kind') != 'qualified_operator_segments'
            or data.get('arm') != arm or data.get('dataset') != expected_dataset
            or data.get('control_source') != 'operator'
            or data.get('practice_data_excluded') is not True):
        raise ValueError('human dataset provenance does not match selected arm')
    if data.get('source_manifest_sha256') != hashlib.sha256(manifest_bytes).hexdigest():
        raise ValueError('human source manifest hash mismatch')
    if manifest.get('schema_version') != 'werescue.segments.v1':
        raise ValueError('expected validated werescue.segments.v1 manifest')
    body = {key: value for key, value in manifest.items() if key != 'manifest_sha256'}
    canonical = json.dumps(body, sort_keys=True, separators=(',', ':')).encode()
    if manifest.get('manifest_sha256') != hashlib.sha256(canonical).hexdigest():
        raise ValueError('source segment manifest validation digest mismatch')
    ids = data.get('segment_ids')
    if not isinstance(ids, list) or len(ids) != episode_count or len(set(ids)) != episode_count:
        raise ValueError('one distinct source segment required for each LeRobot episode')
    by_id = {row.get('segment_id'): row for row in manifest.get('segments', [])}
    selected = [by_id.get(segment_id) for segment_id in ids]
    if any(row is None or row.get('dataset') != expected_dataset or row.get('split') != 'train'
           or row.get('control_source') != 'operator' or row.get('fps') != 20
           or row.get('environment_id') != 'libero_object' or row.get('task_id') != '0'
           or row.get('image_transform') != 'reverse_UI_preview_flip_once'
           or row.get('dataset_image_orientation') != 'native policy RGB after one H and W reversal of source PNG'
           for row in selected):
        raise ValueError('segments must be operator train segments with restored policy-view RGB for LIBERO-Object task0')
    source_ids = [row['episode_id'] for row in selected]
    if len(set(source_ids)) != episode_count:
        raise ValueError('each training segment must have a distinct source intervention')
    source_policies = [row.get('source_failure_policy') for row in selected]
    if any(not isinstance(policy, dict) or not policy.get('repo_id') or not policy.get('revision')
           for policy in source_policies):
        raise ValueError('each human segment must identify the policy that produced its failure')
    distinct_policies = list({json.dumps(policy, sort_keys=True): policy for policy in source_policies}.values())
    return {**data, 'verified_source_episode_ids': source_ids,
            'source_failure_policies': distinct_policies,
            'cross_policy_failure_source': any(policy != {'repo_id': MODEL_ID, 'revision': MODEL_REVISION}
                                               for policy in distinct_policies)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True, choices=['task', 'recovery'])
    parser.add_argument('--P', dest='pretrained', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--dataset-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error('steps and batch size must be positive')
    pin_file = args.pretrained / 'WERESCUE_PIN.json'
    if not pin_file.exists():
        parser.error('P must contain WERESCUE_PIN.json from download_checkpoint.py')
    if json.loads(pin_file.read_text()) != {'repo_id': MODEL_ID, 'revision': MODEL_REVISION}:
        parser.error('P identifies a different checkpoint')
    config_path = args.pretrained / 'config.json'
    if not config_path.is_file():
        parser.error('P must include its own config.json')
    config = json.loads(config_path.read_text())
    if (config.get('n_action_steps') != 50 or config.get('num_steps') != 10
            or config.get('chunk_size') != 50
            or config.get('input_features', {}).get('observation.state', {}).get('shape') != [8]
            or config.get('output_features', {}).get('action', {}).get('shape') != [7]
            or not all(key in config.get('input_features', {})
                       for key in ('observation.images.camera1', 'observation.images.camera2'))):
        parser.error('P config does not match pinned lerobot LIBERO state8/action7/camera1+camera2/default-NAS50/NS10')
    info_file = args.dataset_root / 'meta/info.json'
    if not info_file.is_file():
        parser.error('expected a converted LeRobot dataset with meta/info.json; raw browser JSON is insufficient')
    info = json.loads(info_file.read_text())
    if type(info.get('fps')) not in (int, float) or info['fps'] != 20:
        parser.error('dataset fps must be exactly 20 to match native LIBERO control')
    if type(info.get('total_episodes')) is not int or not 1 <= info['total_episodes'] <= 10:
        parser.error('dataset must contain 1..10 verified human segments for this arm')
    features = info.get('features', {})
    for key, shape in [('observation.state', [8]), ('action', [7])]:
        if features.get(key, {}).get('shape') != shape:
            parser.error(f'{key} must have shape {shape}')
    camera_map = {'observation.images.image': 'observation.images.camera1',
                  'observation.images.image2': 'observation.images.camera2'}
    for key, policy_key in camera_map.items():
        if key not in features:
            parser.error(f'missing native LIBERO camera {key}')
        expected_chw = config['input_features'][policy_key].get('shape')
        if not (isinstance(expected_chw, list) and len(expected_chw) == 3 and expected_chw[0] == 3):
            parser.error(f'P image feature shape is invalid for {key}')
        expected_hwc = [expected_chw[1], expected_chw[2], 3]
        if features[key].get('shape') != expected_hwc:
            parser.error(f'{key} must match P RGB feature {expected_hwc}')
    try:
        provenance = checked_human_provenance(args.dataset_root, args.arm, info['total_episodes'])
    except (ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    if args.execute and args.output.exists():
        parser.error('output exists; choose a fresh path instead of overwriting/resuming another branch')
    command = [sys.executable, str(Path(__file__).with_name('train_preserve_stats.py')),
               f'--policy.path={args.pretrained.resolve()}',
               f'--dataset.repo_id={args.dataset_id}', f'--dataset.root={args.dataset_root.resolve()}',
               f'--output_dir={args.output.resolve()}', f'--job_name=werescue_{args.arm}',
               f'--steps={args.steps}', f'--batch_size={args.batch_size}', f'--seed={args.seed}',
               '--num_workers=0', '--dataset.video_backend=pyav',
               '--policy.device=cuda', '--policy.n_action_steps=10',
               '--policy.freeze_vision_encoder=true', '--policy.train_expert_only=true',
               '--policy.train_state_proj=true', '--policy.optimizer_lr=0.00003',
               f'--policy.scheduler_warmup_steps={min(30, args.steps // 10)}',
               f'--policy.scheduler_decay_steps={args.steps}',
               '--policy.scheduler_decay_lr=0.000003', '--policy.push_to_hub=false',
               '--wandb.enable=false', f'--save_freq={args.steps}',
               f'--log_freq={min(10, args.steps)}',
               '--rename_map={"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}']
    print(json.dumps({'arm': 'P+R' if args.arm == 'task' else 'P+R_prime',
                      'initialization': {'repo_id': MODEL_ID, 'revision': MODEL_REVISION},
                      'normalization': 'retain_P_statistics', 'dataset': str(args.dataset_root.resolve()),
                      'reported_episodes': info.get('total_episodes'),
                      'human_source_segments': len(provenance['segment_ids']),
                      'human_source_intervention_ids': provenance['verified_source_episode_ids'],
                      'source_failure_policies': provenance['source_failure_policies'],
                      'cross_policy_failure_source': provenance['cross_policy_failure_source'],
                      'execute': args.execute,
                      'command': command}, indent=2))
    print(shlex.join(command), flush=True)
    if args.execute:
        subprocess.run(command, check=True)
        checkpoint = args.output / 'checkpoints' / f'{args.steps:06d}' / 'pretrained_model'
        if not (checkpoint / 'model.safetensors').is_file():
            raise RuntimeError(f'trainer exited without the expected checkpoint: {checkpoint}')
        record = {'schema_version': 'werescue.human_finetune.v1',
                  'arm': args.arm, 'base_policy': {'repo_id': MODEL_ID, 'revision': MODEL_REVISION},
                  'dataset': str(args.dataset_root.resolve()),
                  'human_dataset_provenance': provenance,
                  'steps': args.steps, 'batch_size': args.batch_size, 'seed': args.seed,
                  'checkpoint': str(checkpoint.resolve()),
                  'normalization': 'retain_P_statistics'}
        payload = json.dumps(record, indent=2) + '\n'
        (args.output / 'WERESCUE_HUMAN_FINETUNE.json').write_text(payload)
        (checkpoint / 'WERESCUE_HUMAN_FINETUNE.json').write_text(payload)


if __name__ == '__main__':
    main()
