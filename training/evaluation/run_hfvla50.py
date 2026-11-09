#!/usr/bin/env python3
"""Check pinned HuggingFaceVLA SmolVLA-LIBERO with its own NAS1/NS10 processors.

Default: Object task0 × development states0..24 × policy seeds1000/1001 =50 trials,
with actual target XY jitter±4cm. Final states25..49 remain reserved.
This is a local checkpoint check, not an exact reproduction of the external200-run
75.5% result. Runtime/source/renderer differences are retained in provenance.
"""
from __future__ import annotations

import argparse
import gzip
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time

from protocol import digest, parse_selection
from run_baseline import (atomic_json, file_hash, initial_objects, native_state_hash,
                          observation_batch, prepare_assets, verify_pin)

MODEL_ID = 'HuggingFaceVLA/smolvla_libero'
MODEL_REVISION = '6721902bc4d61e50a3bfdb11dfb4cb626f05d102'


def build_protocol(suite, task_ids, initial_states, seeds, camera_size=360,
                   horizon=None, position_jitter=0.0):
    if suite not in ('spatial', 'object'):
        raise ValueError('suite must be spatial or object')
    for name, values, maximum in [('task IDs', task_ids, 9), ('initial states', initial_states, 49)]:
        if not values or len(values) != len(set(values)) or min(values) < 0 or max(values) > maximum:
            raise ValueError(f'{name} must be nonempty, unique and inside0..{maximum}')
    if not seeds or len(seeds) != len(set(seeds)) or min(seeds) < 0:
        raise ValueError('policy RNG seeds must be nonempty, unique and nonnegative')
    if camera_size < 16 or camera_size % 2:
        raise ValueError('camera size must be an even positive dimension>=16')
    horizon = horizon if horizon is not None else (220 if suite == 'spatial' else 280)
    if horizon <= 0:
        raise ValueError('horizon must be positive')
    if not math.isfinite(position_jitter) or not 0 <= position_jitter <= 0.10:
        raise ValueError('position jitter must be between0 and0.10m per axis')
    if position_jitter and (suite != 'object' or task_ids != [0]):
        raise ValueError('target position jitter is only supported for Object task0')
    if suite == 'object' and task_ids == [0] and max(initial_states) >= 25:
        raise ValueError('can development check must keep final initial states25..49 reserved')
    result = {
        'schema_version': 'werescue.hfvla_check_protocol.v1',
        'policy': {'repo_id': MODEL_ID, 'revision': MODEL_REVISION},
        'suite': 'libero_' + suite, 'task_ids': sorted(task_ids),
        'initial_states': sorted(initial_states), 'policy_rng_seeds': sorted(seeds),
        'env_seed_rule': '10000+100*task_id+initial_state_id',
        'camera_hw': [camera_size, camera_size], 'control_mode': 'relative', 'control_hz': 20,
        'horizon': horizon, 'chunk_size': 50, 'n_action_steps': 1, 'num_steps': 10,
        'processors': 'own pinned HuggingFaceVLA checkpoint; no lerobot checkpoint stats or weights',
        'init_states': True, 'hard_reset': True,
        'position_jitter_max_abs_xy_m': position_jitter,
        'condition': 'explicit target XY jitter before rollout; nonstandard condition' if position_jitter else
                     'official native initial states; IDs do not guarantee different object positions',
        'layout_rng_seed_rule': '50000+initial_state_id' if position_jitter else None,
        'action_rule': 'clip postprocessed native action to[-1,1]; trace both request and applied action',
        'success_predicate': 'native_LIBERO_check_success',
        'contains_human_intervention': False, 'resets_inside_rollout': False,
        'claim_scope': 'local50-trial checkpoint check; not exact reproduction of cited200-trial75.5% result',
        'phase': 'development_validation',
        'final_can_initial_states_reserved': list(range(25, 50)),
        'reference_differences': {
            'cited_lerobot_revision_prefix': '8515d', 'local_expected_lerobot_revision_prefix': 'b9cb121',
            'cited_mujoco': '3.3.2', 'local_expected_mujoco': '3.3.7',
            'cited_renderer': 'OSMesa', 'local_renderer': 'EGL',
            'sample_count': 'default local50 versus cited200',
            'camera360': 'local rendering size matches the cited setup; checkpoint feature metadata remains256',
        },
    }
    result['protocol_sha256'] = digest(result)
    return result


def selected_trials(protocol):
    return [{'trial_id': f'task-{task:02d}_init-{state:02d}_rng-{seed}',
             'task_id': task, 'initial_state_id': state, 'policy_rng_seed': seed,
             'env_seed': 10000 + 100*task + state, 'layout_rng_seed': 50000 + state}
            for task in protocol['task_ids'] for state in protocol['initial_states']
            for seed in protocol['policy_rng_seeds']]


def validate_done(row, protocol, trial, directory, runtime_sha=None):
    if row.get('schema_version') != 'werescue.hfvla_check_trial.v1':
        raise ValueError('invalid completed trial schema')
    if row.get('protocol_sha256') != protocol['protocol_sha256'] or row.get('policy') != protocol['policy']:
        raise ValueError('completed trial uses a different protocol/checkpoint')
    if any(row.get(key) != value for key, value in trial.items()):
        raise ValueError('completed trial metadata differs')
    if runtime_sha is not None and row.get('runtime_sha256') != runtime_sha:
        raise ValueError('runtime changed; do not mix outputs')
    if type(row.get('success')) is not bool or row.get('success_predicate_verified') is not True:
        raise ValueError('trial needs a measured native success result')
    if row.get('control_source') != 'policy' or row.get('contains_human_intervention') is not False or row.get('contains_reset_during_rollout') is not False:
        raise ValueError('trial is not autonomous reset-free policy execution')
    if row.get('success_predicate') != protocol['success_predicate']:
        raise ValueError('native success predicate changed')
    if type(row.get('executed_steps')) is not int or not 1 <= row['executed_steps'] <= protocol['horizon']:
        raise ValueError('invalid executed-step count')
    for name, key in [('trace.jsonl.gz', 'trace_sha256'), ('initial_state.npz', 'initial_state_sha256'),
                      ('scene.xml', 'initial_xml_sha256')]:
        if not (directory / name).is_file() or file_hash(directory / name) != row.get(key):
            raise ValueError(f'missing or changed trial evidence: {name}')
    if row.get('video_ref') and not (directory / row['video_ref']).is_file():
        raise ValueError('missing retained representative video')


def completed_trials(out, protocol, runtime_sha=None):
    rows = []
    for trial in selected_trials(protocol):
        directory = out / trial['trial_id']
        if (directory / 'done.json').is_file():
            row = json.loads((directory / 'done.json').read_text())
            validate_done(row, protocol, trial, directory, runtime_sha)
            rows.append(row)
    return rows


def summary(rows, protocol):
    count, wins = len(rows), sum(row['success'] for row in rows)
    interval = None
    if count:
        z, p = 1.959963984540054, wins/count
        denominator = 1+z*z/count
        center = (p+z*z/(2*count))/denominator
        half = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/denominator
        interval = [max(0, center-half), min(1, center+half)]
    result = {'schema_version': 'werescue.hfvla_check_summary.v1',
              'protocol_sha256': protocol['protocol_sha256'], 'policy': protocol['policy'],
              'expected_trials': len(selected_trials(protocol)), 'completed_trials': count,
              'complete': count == len(selected_trials(protocol)), 'successes': wins,
              'success_rate': wins/count if count else None, 'wilson_95': interval,
              'claim_scope': protocol['claim_scope'], 'condition': protocol['condition'],
              'reference_differences': protocol['reference_differences'],
              'uncertainty_note': 'Descriptive pooled Wilson interval; repeated layouts/tasks are correlated.',
              'trials': rows}
    if protocol['position_jitter_max_abs_xy_m'] and rows:
        positions = {row['initial_state_id']: row['position_setup']['actual_settled_position'][:2] for row in rows}
        span = [max(p[i] for p in positions.values()) - min(p[i] for p in positions.values()) for i in (0, 1)]
        if len(positions) > 1 and max(span) <= 1e-4:
            raise ValueError('target-position jitter did not create actual settled position variation')
        result['actual_settled_target_xy_span_m'] = span
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['spatial', 'object'], default='object')
    parser.add_argument('--task-ids', default='0')
    parser.add_argument('--initial-states', default='0-24')
    parser.add_argument('--seeds', default='1000,1001')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--camera-size', type=int, default=360)
    parser.add_argument('--horizon', type=int)
    parser.add_argument('--position-jitter', type=float, default=0.04)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--stop-after', type=int, default=0, help='Stop after N newly completed trials;0 runs all. Resume with same output.')
    args = parser.parse_args()
    try:
        protocol = build_protocol(args.suite, parse_selection(args.task_ids), parse_selection(args.initial_states),
                                  parse_selection(args.seeds), args.camera_size, args.horizon, args.position_jitter)
        verify_pin(args.policy, MODEL_ID, MODEL_REVISION)
        checkpoint = json.loads((args.policy / 'config.json').read_text())
        if checkpoint['n_action_steps'] != 1 or checkpoint['num_steps'] != 10 or checkpoint['chunk_size'] != 50:
            raise ValueError('checkpoint no longer exposes the expected NAS1/NS10/chunk50 contract')
        if checkpoint['input_features']['observation.state']['shape'] != [8] or checkpoint['output_features']['action']['shape'] != [7]:
            raise ValueError('checkpoint state/action contract differs')
        if not all(key in checkpoint['input_features'] for key in ('observation.images.image', 'observation.images.image2')):
            raise ValueError('expected the own HFVLA image/image2 camera contract')
        if args.stop_after < 0:
            raise ValueError('stop-after must be nonnegative')
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        parser.error('output contains a different frozen protocol; choose a fresh directory')
    atomic_json(path, protocol)
    rows = completed_trials(args.out, protocol)
    atomic_json(args.out / 'summary.json', summary(rows, protocol))
    pending = [row for row in selected_trials(protocol) if row['trial_id'] not in {done['trial_id'] for done in rows}]
    if not pending:
        print(json.dumps(summary(rows, protocol)), flush=True)
        return
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

    def env_config(task):
        return LiberoEnvConfig(task=protocol['suite'], task_ids=[task],
                              observation_height=args.camera_size, observation_width=args.camera_size,
                              control_mode='relative', fps=20, episode_length=protocol['horizon'],
                              init_states=True, hard_reset=True)

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no scripted/CPU fallback')
    policy, pre, post, env_pre, env_post = load_policy_stack(
        str(args.policy), env_config(protocol['task_ids'][0]), checkpoint['n_action_steps'], args.device)
    if policy.config.num_steps != 10 or policy.config.n_action_steps != 1:
        raise RuntimeError('loaded policy does not preserve source NAS1/NS10')
    runtime = {'packages': package_versions(), 'assets': assets_pin,
               'gpu': torch.cuda.get_device_name(0) if args.device.startswith('cuda') else args.device,
               'policy_config_sha256': file_hash(args.policy / 'config.json'),
               'own_preprocessor_sha256': file_hash(args.policy / 'policy_preprocessor.json'),
               'own_postprocessor_sha256': file_hash(args.policy / 'policy_postprocessor.json'),
               'evaluator_source_sha256': file_hash(Path(__file__)),
               'source_hashes': {name: file_hash(Path(importlib.import_module(name).__file__)) for name in
                                 ('lerobot.envs.libero', 'lerobot.processor.env_processor',
                                  'lerobot.policies.smolvla.modeling_smolvla', 'lerobot.policies.factory')},
               'renderer': os.environ['MUJOCO_GL'], 'reference_differences': protocol['reference_differences']}
    if args.position_jitter:
        runtime['position_setup_source_sha256'] = file_hash(Path(__file__).with_name('run_position_pilot.py'))
        runtime['position_protocol_source_sha256'] = file_hash(Path(__file__).with_name('position_protocol.py'))
    runtime_sha = digest(runtime)
    runtime_path = args.out / 'runtime.json'
    if runtime_path.exists() and json.loads(runtime_path.read_text()) != runtime:
        raise RuntimeError('runtime changed; do not combine trials')
    atomic_json(runtime_path, runtime)
    rows = completed_trials(args.out, protocol, runtime_sha)
    new_count = 0
    for task_id in protocol['task_ids']:
        task_trials = [row for row in pending if row['task_id'] == task_id]
        if not task_trials:
            continue
        vec = make_env(env_config(task_id), n_envs=1)[protocol['suite']][task_id]
        env = vec.envs[0]
        try:
            if env._init_states is None or max(protocol['initial_states']) >= len(env._init_states):
                raise RuntimeError('requested initial states are absent; modulo wrapping is forbidden')
            initial_hashes = {row['initial_state_id']: row['initial_native_state_sha256']
                              for row in rows if row['task_id'] == task_id}
            retained = {row['success'] for row in rows if row['task_id'] == task_id and row.get('video_ref')}
            for trial in task_trials:
                directory = args.out / trial['trial_id']
                directory.mkdir(exist_ok=True)
                env.init_state_id = trial['initial_state_id']
                observation, _ = vec.reset(seed=[trial['env_seed']])
                position_setup = None
                if args.position_jitter:
                    from run_position_pilot import perturb_target_position
                    observation, position_setup = perturb_target_position(env, trial, args.position_jitter)
                torch.manual_seed(trial['policy_rng_seed'])
                if args.device.startswith('cuda'):
                    torch.cuda.manual_seed_all(trial['policy_rng_seed'])
                policy.reset()
                problem = env._env.env
                xml = problem.sim.model.get_xml()
                (directory / 'scene.xml').write_text(xml)
                initial = snapshot_state(problem)
                numerical_hash = native_state_hash(initial)
                if trial['initial_state_id'] in initial_hashes and initial_hashes[trial['initial_state_id']] != numerical_hash:
                    raise RuntimeError('paired policy seeds did not receive identical native initial state')
                initial_hashes[trial['initial_state_id']] = numerical_hash
                np.savez_compressed(directory / 'initial_state.npz', **initial)
                objects = initial_objects(problem)
                writer = VideoWriter(directory / 'rollout.mp4', args.camera_size*2, args.camera_size, 20)
                trace_path = directory / 'trace.jsonl.gz'
                success, steps, terminal = False, 0, False
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
                                raise RuntimeError('invalid policy action')
                            applied = np.clip(requested, -1, 1)
                            observation, _, terminated, truncated, _ = vec.step(applied)
                            steps += 1
                            success = bool(env._env.check_success())
                            terminal = bool(terminated[0] or truncated[0])
                            trace.write(json.dumps({'step': step, 'observation_state_before_action': before,
                                                    'postprocessed_policy_action_requested': requested[0].tolist(),
                                                    'executed_action': applied[0].tolist(), 'control_source': 'policy',
                                                    'native_success_after_action': success,
                                                    'sim_time_after_action': float(problem.sim.data.time)})+'\n')
                            if step == 0 or steps % 10 == 0:
                                atomic_json(args.out / 'progress.json', {
                                    'status': 'running', 'model_loaded': True,
                                    'actual_native_policy_action_verified': True,
                                    'trial_id': trial['trial_id'], 'trial_step': steps,
                                    'trial_horizon': protocol['horizon'], 'completed_trials': len(rows),
                                    'expected_trials': len(selected_trials(protocol)),
                                    'native_sim_time': float(problem.sim.data.time),
                                    'trial_elapsed_seconds': time.monotonic()-started,
                                    'last_updated_unix_seconds': time.time(),
                                    'n_action_steps': 1, 'num_steps': 10,
                                    'position_setup': position_setup,
                                    'claim_scope': 'actual evaluation running; no completed50-run result yet',
                                })
                            if success or terminal:
                                break
                        pixels = observation['pixels']
                        writer.write(np.concatenate([pixels['image'][0][::-1, ::-1], pixels['image2'][0][::-1, ::-1]], axis=1).copy())
                        final = observation_batch(observation, vec, env_pre, preprocess_observation)
                        trace.write(json.dumps({'kind': 'terminal_observation', 'step': steps,
                                                'observation_state': final['observation.state'].detach().cpu().numpy()[0].tolist(),
                                                'native_success': success})+'\n')
                finally:
                    writer.close()
                np.savez_compressed(directory / 'terminal_state.npz', **snapshot_state(problem))
                keep_video = success not in retained
                if keep_video:
                    retained.add(success)
                else:
                    (directory / 'rollout.mp4').unlink()
                row = {'schema_version': 'werescue.hfvla_check_trial.v1', **trial,
                       'protocol_sha256': protocol['protocol_sha256'], 'runtime_sha256': runtime_sha,
                       'policy': protocol['policy'], 'suite': protocol['suite'], 'language': env.task_description,
                       'success': success, 'success_predicate_verified': True,
                       'success_predicate': 'native_LIBERO_check_success',
                       'executed_steps': steps, 'seconds': time.monotonic()-started,
                       'termination': 'native_success' if success else ('environment_terminal' if terminal else 'horizon'),
                       'control_source': 'policy', 'contains_human_intervention': False,
                       'contains_reset_during_rollout': False, 'initial_objects': objects,
                       'initial_native_state_sha256': numerical_hash,
                       'initial_state_sha256': file_hash(directory / 'initial_state.npz'),
                       'initial_xml_sha256': file_hash(directory / 'scene.xml'),
                       'trace_sha256': file_hash(trace_path), 'video_ref': 'rollout.mp4' if keep_video else None,
                       'position_setup': position_setup}
                validate_done(row, protocol, trial, directory, runtime_sha)
                atomic_json(directory / 'done.json', row)
                rows.append(row)
                new_count += 1
                aggregate = summary(rows, protocol)
                atomic_json(args.out / 'summary.json', aggregate)
                atomic_json(args.out / 'progress.json', {
                    'status': 'complete' if aggregate['complete'] else 'running',
                    'model_loaded': True, 'actual_native_policy_action_verified': True,
                    'last_completed_trial': trial['trial_id'], 'completed_trials': len(rows),
                    'expected_trials': aggregate['expected_trials'], 'successes': aggregate['successes'],
                    'last_updated_unix_seconds': time.time(),
                })
                if not (args.out / 'ready.json').exists():
                    atomic_json(args.out / 'ready.json', {'model_loaded': True, 'native_inference_verified': True,
                                'first_completed_trial': row['trial_id'], 'actual_success': success,
                                'policy': protocol['policy'], 'n_action_steps': 1, 'num_steps': 10,
                                'trial_evidence': str(directory), 'expected_trials': aggregate['expected_trials'],
                                'completed_trials_at_readiness': len(rows), 'evaluation_complete': aggregate['complete'],
                                'claim_scope': protocol['claim_scope']})
                print(json.dumps({'trial': row['trial_id'], 'success': success, 'steps': steps,
                                  'completed': len(rows), 'expected': aggregate['expected_trials'],
                                  'successes': aggregate['successes']}), flush=True)
                if args.stop_after and new_count >= args.stop_after:
                    atomic_json(args.out / 'progress.json', {
                        'status': 'diagnostic_stop_after_requested', 'model_loaded': True,
                        'actual_native_policy_action_verified': True,
                        'last_completed_trial': trial['trial_id'], 'completed_trials': len(rows),
                        'expected_trials': aggregate['expected_trials'], 'evaluation_complete': aggregate['complete'],
                        'last_updated_unix_seconds': time.time(),
                    })
                    return
        finally:
            vec.close()


if __name__ == '__main__':
    main()
