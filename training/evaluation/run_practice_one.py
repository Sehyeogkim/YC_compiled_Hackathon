#!/usr/bin/env python3
"""One native LIBERO can-task rollout for a local SmolVLA practice checkpoint.

This is an end-to-end plumbing check. A one-episode self-imitation practice
checkpoint is neither a human R/R' model nor evidence of improvement.
"""
from __future__ import annotations

import argparse
import gzip
import importlib
import json
import os
from pathlib import Path
import sys
import time

from protocol import digest
from run_baseline import atomic_json, file_hash, initial_objects, native_state_hash, observation_batch, prepare_assets, verify_pin
from run_hfvla50 import MODEL_ID, MODEL_REVISION, build_protocol, selected_trials, validate_done


def practice_protocol(policy_dir: Path, checkpoint: dict, initial_state: int, seed: int, camera_size: int, jitter: float) -> dict:
    protocol = build_protocol('object', [0], [initial_state], [seed], camera_size=camera_size,
                              position_jitter=jitter)
    protocol['policy'] = {
        'kind': 'local_self_imitation_practice',
        'base_repo_id': MODEL_ID,
        'base_revision': MODEL_REVISION,
        'checkpoint_path': str(policy_dir.resolve()),
        'checkpoint_config_sha256': file_hash(policy_dir / 'config.json'),
    }
    protocol['n_action_steps'] = checkpoint['n_action_steps']
    protocol['num_steps'] = checkpoint['num_steps']
    protocol['phase'] = 'practice_smoke_evaluation'
    protocol['claim_scope'] = 'one measured native rollout of a self-imitation practice checkpoint; no R/R_prime or improvement claim'
    protocol['protocol_sha256'] = digest({key: value for key, value in protocol.items() if key != 'protocol_sha256'})
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True, help='Fine-tuned pretrained_model directory')
    parser.add_argument('--base-policy', type=Path, required=True, help='Pinned HuggingFaceVLA P directory')
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--initial-state', type=int, default=24)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--camera-size', type=int, default=360)
    parser.add_argument('--position-jitter', type=float, default=0.04)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('output must be a fresh empty directory; no practice overwrite')
    verify_pin(args.base_policy, MODEL_ID, MODEL_REVISION)
    for name in ('config.json', 'policy_preprocessor.json', 'policy_postprocessor.json'):
        if not (args.policy / name).is_file():
            parser.error(f'fine-tuned checkpoint missing {name}')
    if not list(args.policy.glob('*.safetensors')):
        parser.error('fine-tuned checkpoint missing safetensors weights')
    checkpoint = json.loads((args.policy / 'config.json').read_text())
    if checkpoint['input_features']['observation.state']['shape'] != [8] or checkpoint['output_features']['action']['shape'] != [7]:
        parser.error('fine-tuned checkpoint must keep native state8/action7 contract')
    if not all(k in checkpoint['input_features'] for k in ('observation.images.image', 'observation.images.image2')):
        parser.error('fine-tuned checkpoint must keep both native cameras')
    protocol = practice_protocol(args.policy, checkpoint, args.initial_state, args.seed,
                                 args.camera_size, args.position_jitter)
    trial = selected_trials(protocol)[0]
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out / 'protocol.json', protocol)
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    assets_pin = prepare_assets(args.assets, args.out)
    import numpy as np
    import torch
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.factory import make_env
    from lerobot.envs.utils import preprocess_observation
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
    from native_smoke import VideoWriter, load_policy_stack, package_versions, snapshot_state

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU fallback for timed smoke evaluation')
    config = LiberoEnvConfig(task='libero_object', task_ids=[0],
                             observation_height=args.camera_size, observation_width=args.camera_size,
                             control_mode='relative', fps=20, episode_length=protocol['horizon'],
                             init_states=True, hard_reset=True)
    policy, pre, post, env_pre, env_post = load_policy_stack(str(args.policy), config,
                                                              checkpoint['n_action_steps'], args.device)
    runtime = {
        'packages': package_versions(), 'assets': assets_pin,
        'gpu': torch.cuda.get_device_name(0) if args.device.startswith('cuda') else args.device,
        'policy_config_sha256': file_hash(args.policy / 'config.json'),
        'own_preprocessor_sha256': file_hash(args.policy / 'policy_preprocessor.json'),
        'own_postprocessor_sha256': file_hash(args.policy / 'policy_postprocessor.json'),
        'weight_files': {p.name: file_hash(p) for p in sorted(args.policy.glob('*.safetensors'))},
        'evaluator_source_sha256': file_hash(Path(__file__)),
        'source_hashes': {name: file_hash(Path(importlib.import_module(name).__file__)) for name in
                          ('lerobot.envs.libero', 'lerobot.processor.env_processor',
                           'lerobot.policies.smolvla.modeling_smolvla', 'lerobot.policies.factory')},
        'renderer': os.environ['MUJOCO_GL'],
    }
    if args.position_jitter:
        runtime['position_setup_source_sha256'] = file_hash(Path(__file__).with_name('run_position_pilot.py'))
        runtime['position_protocol_source_sha256'] = file_hash(Path(__file__).with_name('position_protocol.py'))
    runtime_sha = digest(runtime)
    atomic_json(args.out / 'runtime.json', runtime)
    vec = make_env(config, n_envs=1)['libero_object'][0]
    try:
        env = vec.envs[0]
        if env._init_states is None or args.initial_state >= len(env._init_states):
            raise RuntimeError('requested native initial state absent')
        env.init_state_id = args.initial_state
        observation, _ = vec.reset(seed=[trial['env_seed']])
        position_setup = None
        if args.position_jitter:
            from run_position_pilot import perturb_target_position
            observation, position_setup = perturb_target_position(env, trial, args.position_jitter)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        policy.reset()
        problem = env._env.env
        directory = args.out / trial['trial_id']
        directory.mkdir()
        (directory / 'scene.xml').write_text(problem.sim.model.get_xml())
        initial = snapshot_state(problem)
        initial_hash = native_state_hash(initial)
        np.savez_compressed(directory / 'initial_state.npz', **initial)
        objects = initial_objects(problem)
        trace_path = directory / 'trace.jsonl.gz'
        writer = VideoWriter(directory / 'rollout.mp4', args.camera_size * 2, args.camera_size, 20)
        success = terminal = False
        steps = 0
        started = time.monotonic()
        try:
            with gzip.open(trace_path, 'wt') as trace:
                for step in range(protocol['horizon']):
                    batch = observation_batch(observation, vec, env_pre, preprocess_observation)
                    pixels = observation['pixels']
                    writer.write(np.concatenate([pixels['image'][0][::-1, ::-1], pixels['image2'][0][::-1, ::-1]], axis=1).copy())
                    before = batch['observation.state'].detach().cpu().numpy()[0].tolist()
                    with torch.inference_mode():
                        action = policy.select_action(pre(batch))
                    requested = env_post({'action': post(action)})['action'].detach().cpu().numpy()
                    if requested.shape != (1, 7) or not np.isfinite(requested).all():
                        raise RuntimeError('invalid native policy action')
                    applied = np.clip(requested, -1, 1)
                    observation, _, terminated, truncated, _ = vec.step(applied)
                    steps += 1
                    success = bool(env._env.check_success())
                    terminal = bool(terminated[0] or truncated[0])
                    trace.write(json.dumps({'step': step, 'observation_state_before_action': before,
                                            'postprocessed_policy_action_requested': requested[0].tolist(),
                                            'executed_action': applied[0].tolist(), 'control_source': 'policy',
                                            'native_success_after_action': success,
                                            'sim_time_after_action': float(problem.sim.data.time)}) + '\n')
                    if steps == 1 or steps % 10 == 0:
                        atomic_json(args.out / 'progress.json', {'status': 'running', 'trial_step': steps,
                                                                  'horizon': protocol['horizon'],
                                                                  'elapsed_seconds': time.monotonic() - started})
                    if success or terminal:
                        break
                pixels = observation['pixels']
                writer.write(np.concatenate([pixels['image'][0][::-1, ::-1], pixels['image2'][0][::-1, ::-1]], axis=1).copy())
        finally:
            writer.close()
        np.savez_compressed(directory / 'terminal_state.npz', **snapshot_state(problem))
        row = {'schema_version': 'werescue.hfvla_check_trial.v1', **trial,
               'protocol_sha256': protocol['protocol_sha256'], 'runtime_sha256': runtime_sha,
               'policy': protocol['policy'], 'suite': protocol['suite'], 'language': env.task_description,
               'success': success, 'success_predicate_verified': True,
               'success_predicate': 'native_LIBERO_check_success', 'executed_steps': steps,
               'seconds': time.monotonic() - started,
               'termination': 'native_success' if success else ('environment_terminal' if terminal else 'horizon'),
               'control_source': 'policy', 'contains_human_intervention': False,
               'contains_reset_during_rollout': False, 'initial_objects': objects,
               'initial_native_state_sha256': initial_hash,
               'initial_state_sha256': file_hash(directory / 'initial_state.npz'),
               'initial_xml_sha256': file_hash(directory / 'scene.xml'),
               'trace_sha256': file_hash(trace_path), 'video_ref': 'rollout.mp4',
               'position_setup': position_setup}
        validate_done(row, protocol, trial, directory, runtime_sha)
        atomic_json(directory / 'done.json', row)
        summary = {'claim_scope': protocol['claim_scope'], 'complete': True,
                   'attempts': 1, 'successes': int(success), 'native_success': success,
                   'trial_id': trial['trial_id'], 'steps': steps, 'seconds': row['seconds'],
                   'trial_evidence': str(directory), 'practice_checkpoint': str(args.policy.resolve())}
        atomic_json(args.out / 'summary.json', summary)
        atomic_json(args.out / 'progress.json', {'status': 'complete', **summary})
        print(json.dumps(summary), flush=True)
    finally:
        vec.close()


if __name__ == '__main__':
    main()
