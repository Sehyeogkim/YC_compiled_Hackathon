#!/usr/bin/env python3
"""Evaluate verified PUBLIC human LIBERO task fine-tune on P's held-out starts.

This owns a separate output directory and refuses an incomplete or mismatched
P baseline. Trial IDs, initial numerical states and can positions must pair.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

from position_protocol import (BASE_MODEL_ID, BASE_MODEL_REVISION, MODEL_ID, MODEL_REVISION,
                      POLICY_SEEDS, STATE_SPLITS, digest, make_protocol,
                      parse_selection, summarize, trials, validate_done,
                      choose_target_xy, admissible_xy)


def verify_finetune(path: Path) -> dict:
    record_path = path / 'WERESCUE_HUMAN_FINETUNE.json'
    if not record_path.is_file() or not (path / 'model.safetensors').is_file() or not (path / 'config.json').is_file():
        raise ValueError('P+public-H checkpoint requires model, config, and WERESCUE_HUMAN_FINETUNE.json')
    record = json.loads(record_path.read_text())
    if (record.get('schema_version') != 'werescue.human_finetune.v1'
            or record.get('arm') != 'task'
            or record.get('base_policy') != {'repo_id': MODEL_ID, 'revision': MODEL_REVISION}
            or Path(record.get('checkpoint', '')).resolve() != path.resolve()
            or record.get('normalization') != 'retain_P_statistics'):
        raise ValueError('checkpoint is not a validated public human task fine-tune')
    provenance = record.get('human_dataset_provenance') or {}
    if (record.get('training_source') != 'public_LIBERO_human_teleoperation'
            or record.get('recovery_claim') is not False
            or provenance.get('source_repo') != 'yifengzhu-hf/LIBERO-datasets'
            or provenance.get('revision') != 'f13aa24a3da8c43c7225569f28c562979fa0e35a'
            or provenance.get('control_alignment_verified') is not True
            or provenance.get('simulator_control_hz') != 20):
        raise ValueError('checkpoint lacks pinned public LIBERO human provenance')
    selected = provenance.get('selected_demo_indices', provenance.get('selected_demos'))
    if not isinstance(selected, list) or not selected or not all(isinstance(i, int) and 0 <= i < 25 for i in selected):
        raise ValueError('public-H demos must be confined to reserved training indices 0..24')
    return {'label': 'P+public-H', 'base_policy': record['base_policy'],
            'checkpoint_sha256': file_hash(path / 'model.safetensors'),
            'finetune_manifest_sha256': file_hash(record_path),
            'dataset': record.get('dataset'), 'steps': record.get('steps')}


def paired_protocol(args, states, finetune: dict, baseline: dict) -> dict:
    p = make_protocol(args.phase, states, parse_selection(args.seeds), args.suite,
                      args.task_id, args.horizon, 'libero', args.position_jitter)
    comparison_keys = ('phase', 'suite', 'task_id', 'initial_states', 'policy_rng_seeds',
                       'env_seed_rule', 'horizon', 'init_states', 'hard_reset', 'control_mode',
                       'control_hz', 'chunk_size', 'n_action_steps', 'applied_action_rule',
                       'camera_hw', 'success_predicate', 'control_source',
                       'human_interventions_allowed', 'resets_within_rollout_allowed', 'condition')
    if (baseline.get('arm') != 'libero' or baseline.get('policy') !=
            {'repo_id': MODEL_ID, 'revision': MODEL_REVISION}
            or any(baseline.get(key) != p.get(key) for key in comparison_keys)):
        raise ValueError('P baseline uses a different held-out evaluation contract')
    p['arm'] = 'P+public-H'
    p['policy'] = finetune
    p['paired_baseline_protocol_sha256'] = baseline['protocol_sha256']
    p['protocol_sha256'] = digest({key: value for key, value in p.items() if key != 'protocol_sha256'})
    return p


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def verify_pin(path: Path, repo: str, revision: str):
    pin = path / 'WERESCUE_PIN.json'
    if pin.exists():
        if json.loads(pin.read_text()) != {'repo_id': repo, 'revision': revision}:
            raise ValueError(f'{path}: checkpoint pin differs from protocol')
    elif path.name != revision:
        raise ValueError(f'{path}: requires an exact revision-named HF snapshot or WERESCUE_PIN.json')
    if not (path / 'config.json').is_file() or not (path / 'model.safetensors').is_file():
        raise ValueError(f'{path}: checkpoint files missing')


def prepare_assets(assets, out):
    """Use a private config directory; never write the inference service config."""
    pin_path = assets / 'WERESCUE_ASSETS_PIN.json'
    if not pin_path.is_file():
        raise ValueError('assets must be a pinned completed native asset directory')
    spec = importlib.util.find_spec('libero')
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError('native hf-libero package is not installed')
    package = Path(next(iter(spec.submodule_search_locations))) / 'libero'
    config_dir = out / '.libero-config'
    config_dir.mkdir(exist_ok=True)
    os.environ['LIBERO_CONFIG_PATH'] = str(config_dir.resolve())
    import yaml
    config = {'benchmark_root': str(package), 'bddl_files': str(package / 'bddl_files'),
              'init_states': str(package / 'init_files'), 'datasets': str(config_dir / 'datasets'),
              'assets': str(assets.resolve())}
    (config_dir / 'config.yaml').write_text(yaml.safe_dump(config))
    module = importlib.import_module('libero.libero')
    module._assets_path_cache = str(assets.resolve())
    return json.loads(pin_path.read_text())


def read_completed(out, protocol, runtime_sha=None):
    completed = []
    for trial in trials(protocol):
        directory = out / trial['trial_id']
        path = directory / 'done.json'
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        validate_done(data, protocol, trial)
        if runtime_sha is not None and data['runtime_sha256'] != runtime_sha:
            raise ValueError(f'{trial["trial_id"]}: runtime changed; choose a fresh run directory')
        for name, key in [('trace.jsonl.gz', 'trace_sha256'),
                          ('initial_state.npz', 'initial_state_sha256'), ('scene.xml', 'initial_xml_sha256')]:
            if not (directory / name).is_file() or file_hash(directory / name) != data[key]:
                raise ValueError(f'{trial["trial_id"]}: missing or changed evidence {name}')
        if data.get('video_ref') and not (directory / data['video_ref']).is_file():
            raise ValueError(f'{trial["trial_id"]}: retained representative video missing')
        completed.append(data)
    return completed


def observation_batch(observation, vec, env_pre, preprocess_observation):
    batch = preprocess_observation(observation)
    batch['task'] = list(vec.call('task_description'))
    return env_pre(batch)


def initial_objects(problem):
    return {name: [float(value) for value in problem.sim.data.body_xpos[body]]
            for name, body in problem.obj_body_id.items()}


def native_state_hash(state):
    """Hash numerical state content, independently of npz archive timestamps."""
    result = hashlib.sha256()
    for name, array in sorted(state.items()):
        result.update(name.encode())
        result.update(str(array.dtype).encode())
        result.update(str(array.shape).encode())
        result.update(array.tobytes(order='C'))
    return result.hexdigest()


def perturb_target_position(env, trial, jitter):
    """Set up one reproducible displaced can before policy evaluation begins."""
    import numpy as np
    from lerobot.envs.libero import get_libero_dummy_action
    problem = env._env.env
    target = next((name for name in problem.obj_of_interest if 'alphabet_soup' in name.lower()), None)
    basket = next((name for name in problem.obj_of_interest if 'basket' in name.lower()), None)
    if target is None or basket is None:
        raise RuntimeError('position-jitter condition requires native alphabet_soup and basket objects')
    if target not in problem.objects_dict or target not in problem.obj_body_id:
        raise RuntimeError('native target object is missing a body/freejoint')
    sim = problem.sim
    joint = problem.objects_dict[target].joints[-1]
    qpos = np.array(sim.data.get_joint_qpos(joint), copy=True)
    if qpos.shape != (7,):
        raise RuntimeError(f'target joint must be a 7D freejoint, got {qpos.shape}')
    original_position = np.array(sim.data.body_xpos[problem.obj_body_id[target]], copy=True)
    others = {name: np.array(sim.data.body_xpos[body][:2], copy=True)
              for name, body in problem.obj_body_id.items() if name != target}
    selected = choose_target_xy(original_position[:2], others, basket,
                                trial['layout_rng_seed'], jitter, max_attempts=100)
    qpos[:2] = selected['requested_xy']
    sim.data.set_joint_qpos(joint, qpos)
    sim.forward()
    raw = None
    for _ in range(10):
        raw, _, _, _ = env._env.step(get_libero_dummy_action())
    actual = np.array(sim.data.body_xpos[problem.obj_body_id[target]], copy=True)
    actual_others = {name: np.array(sim.data.body_xpos[body][:2], copy=True)
                     for name, body in problem.obj_body_id.items() if name != target}
    if not admissible_xy(actual[:2], actual_others, basket):
        raise RuntimeError('target settled outside the declared workspace/clearance condition')
    if float(np.linalg.norm(actual[:2] - original_position[:2])) <= 1e-5:
        raise RuntimeError('target did not measurably move; randomization setup is invalid')
    if env._env.check_success():
        raise RuntimeError('setup unexpectedly satisfied the task before any policy action')

    def batched(tree):
        if isinstance(tree, dict):
            return {key: batched(value) for key, value in tree.items()}
        return np.expand_dims(np.asarray(tree), 0)

    record = {**selected, 'target_object': target, 'basket_object': basket,
              'original_position': original_position.tolist(),
              'actual_settled_position': actual.tolist(),
              'actual_delta_xy': (actual[:2] - original_position[:2]).tolist(),
              'settle_native_noop_steps': 10, 'vector_policy_steps_during_setup': 0,
              'setup_before_rollout': True, 'target_z_and_quaternion_preserved_on_assignment': True}
    return batched(env._format_raw_obs(raw)), record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True, help='Pinned local weights for the selected arm.')
    parser.add_argument('--paired-baseline', type=Path, required=True,
                        help='Completed P baseline directory using identical held-out starts.')
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--suite', default='libero_object')
    parser.add_argument('--task-id', type=int, default=0)
    parser.add_argument('--initial-states', help='Comma-separated IDs and inclusive ranges, e.g. 25-49.')
    parser.add_argument('--seeds', default='1000,1001', help='Policy RNG seeds; environment seeds stay fixed per layout.')
    parser.add_argument('--horizon', type=int, default=280)
    parser.add_argument('--position-jitter', type=float, default=0.04,
                        help='Maximum absolute XY proposal displacement per axis in metres; default ±4cm.')
    parser.add_argument('--phase', choices=['pilot', 'final'], default='pilot')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    try:
        states = parse_selection(args.initial_states) if args.initial_states else STATE_SPLITS[args.phase]
        finetune = verify_finetune(args.policy)
        baseline = json.loads((args.paired_baseline / 'protocol.json').read_text())
        protocol = paired_protocol(args, states, finetune, baseline)
        baseline_rows = read_completed(args.paired_baseline, baseline)
        if len(baseline_rows) != len(trials(baseline)):
            raise ValueError('P baseline must finish all paired trials before P+public-H evaluation')
        baseline_by_trial = {row['trial_id']: row for row in baseline_rows}
    except ValueError as error:
        parser.error(str(error))
    args.out.mkdir(parents=True, exist_ok=True)
    protocol_path = args.out / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        parser.error('output directory contains a different protocol; choose a fresh directory')
    atomic_json(protocol_path, protocol)
    existing = read_completed(args.out, protocol)
    atomic_json(args.out / 'summary.json', summarize(existing, protocol))
    pending = [trial for trial in trials(protocol) if trial['trial_id'] not in {row['trial_id'] for row in existing}]
    if not pending:
        print(json.dumps(summarize(existing, protocol)), flush=True)
        return
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    asset_pin = prepare_assets(args.assets, args.out)
    import numpy as np
    import torch
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.factory import make_env
    from lerobot.envs.utils import preprocess_observation
    # Reuse the tested stack loader only. Its old CLI main is deliberately not called.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
    from native_smoke import VideoWriter, load_policy_stack, package_versions, snapshot_state

    cfg = LiberoEnvConfig(task=args.suite, task_ids=[args.task_id], observation_height=256,
                         observation_width=256, control_mode='relative', fps=20,
                         episode_length=args.horizon, init_states=True, hard_reset=True)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no CPU/scripted fallback')
    policy, pre, post, env_pre, env_post = load_policy_stack(str(args.policy), cfg, 10, args.device)
    adaptation = {'fine_tuned_from_P': True, 'source_manifest_sha256': finetune['finetune_manifest_sha256']}
    if policy.config.chunk_size != 50 or policy.config.n_action_steps != 10:
        raise RuntimeError('loaded model violates fixed action-chunk protocol')
    runtime = {'packages': package_versions(), 'assets': asset_pin,
               'policy_config_sha256': file_hash(args.policy / 'config.json'),
               'gpu': torch.cuda.get_device_name(0) if args.device.startswith('cuda') else args.device,
               'source_hashes': {name: file_hash(Path(importlib.import_module(name).__file__)) for name in
                                 ('lerobot.envs.libero', 'lerobot.processor.env_processor',
                                  'lerobot.policies.smolvla.modeling_smolvla', 'lerobot.policies.factory')},
               'evaluator_source_sha256': file_hash(Path(__file__)), 'adaptation': adaptation}
    runtime_sha = digest(runtime)
    runtime_path = args.out / 'runtime.json'
    if runtime_path.exists() and json.loads(runtime_path.read_text()) != runtime:
        raise RuntimeError('runtime differs from prior run; do not combine results')
    atomic_json(runtime_path, runtime)
    existing = read_completed(args.out, protocol, runtime_sha)
    retained_outcomes = {row['success'] for row in existing if row.get('video_ref')}
    envs = make_env(cfg, n_envs=1)
    vec = envs[args.suite][args.task_id]
    env = vec.envs[0]
    if env._init_states is None or max(states) >= len(env._init_states):
        vec.close()
        raise RuntimeError('native initial-state set does not cover requested IDs; modulo wrapping is forbidden')
    initial_hashes = {row['initial_state_id']: row['initial_native_state_sha256'] for row in existing}
    try:
        for trial in pending:
            directory = args.out / trial['trial_id']
            directory.mkdir(exist_ok=True)
            env.init_state_id = trial['initial_state_id']
            observation, _ = vec.reset(seed=[trial['env_seed']])
            observation, position_setup = perturb_target_position(env, trial, args.position_jitter)
            # Reset establishes layout RNG independently; only torch controls policy sampling.
            torch.manual_seed(trial['policy_rng_seed'])
            if args.device.startswith('cuda'):
                torch.cuda.manual_seed_all(trial['policy_rng_seed'])
            policy.reset()
            problem = env._env.env
            language = env.task_description.lower().replace('_', ' ')
            if args.suite == 'libero_object' and args.task_id == 0 and not ('alphabet soup' in language and 'basket' in language):
                raise RuntimeError('native task ordering differs from expected soup can→basket')
            xml = problem.sim.model.get_xml()
            (directory / 'scene.xml').write_text(xml)
            initial_state = snapshot_state(problem)
            numerical_hash = native_state_hash(initial_state)
            previous_hash = initial_hashes.get(trial['initial_state_id'])
            if previous_hash is not None and previous_hash != numerical_hash:
                raise RuntimeError(f"layout {trial['initial_state_id']}: paired policy seeds did not receive identical native start state")
            initial_hashes[trial['initial_state_id']] = numerical_hash
            baseline_row = baseline_by_trial[trial['trial_id']]
            if baseline_row['initial_native_state_sha256'] != numerical_hash:
                raise RuntimeError(f"{trial['trial_id']}: P and P+R native start states differ")
            if baseline_row['position_setup']['actual_settled_position'] != position_setup['actual_settled_position']:
                raise RuntimeError(f"{trial['trial_id']}: P and P+R can positions differ")
            np.savez_compressed(directory / 'initial_state.npz', **initial_state)
            objects = initial_objects(problem)
            writer = VideoWriter(directory / 'rollout.mp4', 512, 256, 20)
            trace_path = directory / 'trace.jsonl.gz'
            success, executed_steps, terminated_by_env = False, 0, False
            started = time.monotonic()
            try:
                with gzip.open(trace_path, 'wt') as trace:
                    for step in range(args.horizon):
                        batch = observation_batch(observation, vec, env_pre, preprocess_observation)
                        pixels = observation['pixels']
                        writer.write(np.concatenate([pixels['image'][0][::-1, ::-1],
                                                     pixels['image2'][0][::-1, ::-1]], axis=1).copy())
                        before_state = batch['observation.state'].detach().cpu().numpy()[0].tolist()
                        with torch.inference_mode():
                            action = policy.select_action(pre(batch))
                        requested = env_post({'action': post(action)})['action'].detach().cpu().numpy()
                        if requested.shape != (1, 7) or not np.isfinite(requested).all():
                            raise RuntimeError(f'invalid actual action shape/values: {requested.shape}')
                        executed = np.clip(requested, -1.0, 1.0)
                        observation, _, terminated, truncated, _ = vec.step(executed)
                        success = bool(env._env.check_success())
                        executed_steps += 1
                        trace.write(json.dumps({'step': step, 'control_source': 'policy',
                                                'observation_state_before_action': before_state,
                                                'postprocessed_policy_action_requested': requested[0].tolist(),
                                                'executed_action': executed[0].tolist(),
                                                'sim_time_after_action': float(problem.sim.data.time),
                                                'native_success_after_action': success}) + '\n')
                        terminated_by_env = bool(terminated[0] or truncated[0])
                        if success or terminated_by_env:
                            break
                    pixels = observation['pixels']
                    writer.write(np.concatenate([pixels['image'][0][::-1, ::-1],
                                                 pixels['image2'][0][::-1, ::-1]], axis=1).copy())
                    final_batch = observation_batch(observation, vec, env_pre, preprocess_observation)
                    trace.write(json.dumps({'kind': 'terminal_observation', 'step': executed_steps,
                                            'observation_state': final_batch['observation.state'].detach().cpu().numpy()[0].tolist(),
                                            'native_success': success}) + '\n')
            finally:
                writer.close()
            np.savez_compressed(directory / 'terminal_state.npz', **snapshot_state(problem))
            keep_video = success not in retained_outcomes
            if keep_video:
                retained_outcomes.add(success)
            else:
                (directory / 'rollout.mp4').unlink()
            result = {'schema_version': 'werescue.baseline_trial.v1', **trial,
                      'protocol_sha256': protocol['protocol_sha256'], 'runtime_sha256': runtime_sha,
                      'policy': protocol['policy'], 'phase': args.phase, 'arm': 'P+public-H',
                      'suite': args.suite, 'task_id': args.task_id, 'language': env.task_description,
                      'success': success, 'success_predicate_verified': True,
                      'success_predicate': 'native_LIBERO_check_success',
                      'executed_steps': executed_steps, 'seconds': time.monotonic()-started,
                      'termination': 'native_success' if success else ('environment_terminal' if terminated_by_env else 'horizon'),
                      'control_source': 'policy', 'contains_human_intervention': False,
                      'contains_reset_during_rollout': False, 'initial_objects': objects,
                      'position_setup': position_setup,
                      'initial_state_sha256': file_hash(directory / 'initial_state.npz'),
                      'initial_native_state_sha256': numerical_hash,
                      'initial_xml_sha256': file_hash(directory / 'scene.xml'),
                      'trace_sha256': file_hash(trace_path), 'trace_ref': f"{trial['trial_id']}/trace.jsonl.gz",
                      'video_ref': 'rollout.mp4' if keep_video else None}
            validate_done(result, protocol, trial)
            atomic_json(directory / 'done.json', result)
            existing.append(result)
            summary = summarize(existing, protocol)
            atomic_json(args.out / 'summary.json', summary)
            print(json.dumps({'trial': trial['trial_id'], 'success': success,
                              'steps': executed_steps, 'completed': len(existing),
                              'successes': summary['successes'], 'expected': summary['expected_trials']}), flush=True)
    finally:
        vec.close()


if __name__ == '__main__':
    main()
