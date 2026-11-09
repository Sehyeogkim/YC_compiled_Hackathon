#!/usr/bin/env python3
"""Short pinned lerobot/smolvla_libero self-imitation smoke; never label it R/R-prime."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

OLDP = {'repo_id': 'lerobot/smolvla_libero',
        'revision': 'a24f1f380461b665158d52acd140152c79e8b36d'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--P', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--dataset-id', default='werescue/practice_p_self_imitation')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.steps <= 50 or not 1 <= args.batch_size <= 8:
        parser.error('practice limits: 1..50 steps, batch size 1..8')
    pin_path = args.P / 'WERESCUE_PIN.json'
    if not pin_path.is_file() or json.loads(pin_path.read_text()) != OLDP:
        parser.error('P must be the pinned lerobot/smolvla_libero checkpoint')
    config = json.loads((args.P / 'config.json').read_text())
    if (config.get('input_features', {}).get('observation.state', {}).get('shape') != [8]
            or config.get('output_features', {}).get('action', {}).get('shape') != [7]
            or not all(key in config.get('input_features', {}) for key in
                       ('observation.images.camera1', 'observation.images.camera2'))):
        parser.error('P must keep old LIBERO state8/action7/camera1+camera2 contract')
    practice_path = args.dataset_root / 'WERESCUE_PRACTICE.json'
    info_path = args.dataset_root / 'meta/info.json'
    if not practice_path.is_file() or not info_path.is_file():
        parser.error('dataset must be built by build_practice_dataset.py')
    practice = json.loads(practice_path.read_text())
    info = json.loads(info_path.read_text())
    if (practice.get('kind') != 'practice_only_self_imitation_from_P_rollout'
            or practice.get('not_human_R_or_R_prime') is not True
            or practice.get('source_policy') != OLDP):
        parser.error('invalid practice provenance')
    if info.get('fps') != 20 or info.get('total_episodes') != 1:
        parser.error('practice dataset must have one 20-Hz episode')
    if info.get('features', {}).get('observation.state', {}).get('shape') != [8]:
        parser.error('expected state8')
    if info.get('features', {}).get('action', {}).get('shape') != [7]:
        parser.error('expected action7')
    for key in ('observation.images.image', 'observation.images.image2'):
        if key not in info['features']:
            parser.error(f'missing {key}')
        if info['features'][key].get('shape') != [256, 256, 3]:
            parser.error(f'{key} must be 256x256 RGB for old LIBERO P')
    if args.output.exists():
        parser.error('output exists; do not overwrite any training run')
    warmup = min(2, max(0, args.steps // 5))
    command = [sys.executable, str(Path(__file__).with_name('train_preserve_stats.py')),
               f'--policy.path={args.P.resolve()}',
               f'--dataset.repo_id={args.dataset_id}', f'--dataset.root={args.dataset_root.resolve()}',
               f'--output_dir={args.output.resolve()}', '--job_name=werescue_practice_only',
               f'--steps={args.steps}', f'--batch_size={args.batch_size}', f'--seed={args.seed}',
               '--num_workers=0', '--dataset.video_backend=pyav',
               '--policy.device=cuda', '--policy.n_action_steps=10',
               '--policy.freeze_vision_encoder=true', '--policy.train_expert_only=true',
               '--policy.train_state_proj=true', '--policy.optimizer_lr=0.00003',
               f'--policy.scheduler_warmup_steps={warmup}',
               f'--policy.scheduler_decay_steps={args.steps}',
               '--policy.scheduler_decay_lr=0.000003',
               '--policy.push_to_hub=false', '--wandb.enable=false',
               f'--save_freq={args.steps}', '--log_freq=1',
               '--rename_map={"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}']
    print(json.dumps({'kind': 'practice_only_self_imitation_from_P_rollout',
                      'not_human_R_or_R_prime': True, 'initialization': OLDP,
                      'normalization': 'retain_P_statistics',
                      'episodes': info['total_episodes'], 'steps': args.steps,
                      'execute': args.execute, 'command': command}, indent=2), flush=True)
    print(shlex.join(command), flush=True)
    if args.execute:
        subprocess.run(command, check=True)
        checkpoint = args.output / 'checkpoints' / f'{args.steps:06d}' / 'pretrained_model'
        if not (checkpoint / 'model.safetensors').is_file():
            raise RuntimeError(f'trainer exited without the expected checkpoint: {checkpoint}')
        record = {'schema_version': 'werescue.practice_training.v1',
                  'kind': 'practice_only_self_imitation_from_P_rollout',
                  'not_human_R_or_R_prime': True,
                  'excluded_from_product_comparison': True,
                  'base_policy': OLDP, 'dataset_provenance': practice,
                  'steps': args.steps, 'batch_size': args.batch_size,
                  'seed': args.seed, 'checkpoint': str(checkpoint.resolve()),
                  'normalization': 'retain_P_statistics'}
        payload = json.dumps(record, indent=2) + '\n'
        (args.output / 'WERESCUE_PRACTICE_TRAINING.json').write_text(payload)
        (checkpoint / 'WERESCUE_PRACTICE_TRAINING.json').write_text(payload)


if __name__ == '__main__':
    main()
