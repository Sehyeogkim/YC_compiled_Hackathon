#!/usr/bin/env python3
"""Run one native LIBERO policy episode and keep real camera/action/state evidence.

Uses the same public LeRobot policy/processor path as lerobot-eval. API pattern
adapted from Jeff's WEMINE game1016/lerobot_export/libero_failure_eval.py.
No perturbations, human actions, scripted task completion, or fabricated results.
Imports heavy dependencies only after argument parsing so --help works anywhere.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time

RENAME_MAP = {
    'observation.images.image': 'observation.images.camera1',
    'observation.images.image2': 'observation.images.camera2',
}


def package_versions() -> dict:
    result = {}
    for name in ('lerobot', 'hf-libero', 'torch', 'transformers', 'mujoco', 'robosuite', 'numpy', 'av'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def load_policy_stack(policy_path, env_cfg, n_action_steps=10, device='cuda'):
    # Populate the config registry before deserializing a SmolVLA config.
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    config = PreTrainedConfig.from_pretrained(policy_path)
    config.pretrained_path = policy_path
    config.device = device
    config.n_action_steps = n_action_steps
    # Both official repos exist, with different camera keys; inspect the actual pin.
    rename = RENAME_MAP if 'observation.images.camera1' in config.input_features else {}
    policy = make_policy(cfg=config, env_cfg=env_cfg, rename_map=rename)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=config, pretrained_path=policy_path,
        preprocessor_overrides={
            'device_processor': {'device': device},
            'rename_observations_processor': {'rename_map': rename},
        },
    )
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=config)
    return policy, pre, post, env_pre, env_post


class VideoWriter:
    def __init__(self, path, width, height, fps):
        import av
        self.av = av
        self.container = av.open(str(path), mode='w')
        self.stream = self.container.add_stream('libx264', rate=fps)
        self.stream.width, self.stream.height = width, height
        self.stream.pix_fmt = 'yuv420p'

    def write(self, frame):
        for packet in self.stream.encode(self.av.VideoFrame.from_ndarray(frame, format='rgb24')):
            self.container.mux(packet)

    def close(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def snapshot_state(problem):
    """Native integration state plus actuator/controller evidence for later replay.

    This does not claim that a standalone qpos vector restores controller state.
    """
    import numpy as np
    sim = problem.sim
    fields = {'flattened_state': np.array(sim.get_state().flatten(), copy=True)}
    for key in ('qpos', 'qvel', 'act', 'ctrl', 'qacc_warmstart', 'mocap_pos', 'mocap_quat'):
        value = getattr(sim.data, key, None)
        if value is not None:
            fields[key] = np.array(value, copy=True)
    fields['sim_time'] = np.asarray(float(sim.data.time))
    controller = problem.robots[0].controller
    for key in ('goal_pos', 'goal_ori', 'ee_pos', 'ee_ori_mat'):
        value = getattr(controller, key, None)
        if value is not None:
            fields['controller_' + key] = np.array(value, copy=True)
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', required=True, help='Prefer a local pinned snapshot directory.')
    parser.add_argument('--suite', default='libero_object')
    parser.add_argument('--task-id', type=int, default=0)
    parser.add_argument('--init-state', type=int, default=0)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--n-action-steps', type=int, default=10)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.n_action_steps < 1:
        parser.error('steps and n-action-steps must be positive')
    if args.out.exists():
        parser.error('output directory already exists; choose a fresh run path')
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

    import numpy as np
    import torch
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.factory import make_env
    from lerobot.envs.utils import add_envs_task, preprocess_observation

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = LiberoEnvConfig(task=args.suite, task_ids=[args.task_id],
                         observation_height=256, observation_width=256)
    if hasattr(cfg, 'control_mode'):
        cfg.control_mode = 'relative'
    if hasattr(cfg, 'fps'):
        cfg.fps = 20
    policy, pre, post, env_pre, env_post = load_policy_stack(
        args.policy, cfg, args.n_action_steps, args.device)
    envs = make_env(cfg, n_envs=1)
    vec = envs[args.suite][args.task_id]
    env = vec.envs[0]
    env.init_state_id = args.init_state
    observation, _ = vec.reset(seed=[args.seed])
    problem = env._env.env
    if args.suite == 'libero_object' and args.task_id == 0:
        language = env.task_description.lower().replace('_', ' ')
        if 'alphabet soup' not in language or 'basket' not in language:
            vec.close()
            raise RuntimeError(f'Task ordering changed; expected soup can→basket, got {language!r}')

    args.out.mkdir(parents=True)
    xml = problem.sim.model.get_xml()
    (args.out / 'scene.xml').write_text(xml)
    manifest = {
        'schema_version': 'werescue.native_smoke.v1', 'policy': args.policy,
        'suite': args.suite, 'task_id': args.task_id, 'task': env.task,
        'language': env.task_description, 'initial_state_id': args.init_state,
        'seed': args.seed, 'n_action_steps': policy.config.n_action_steps,
        'chunk_size': policy.config.chunk_size, 'control_mode': 'relative', 'fps': 20,
        'packages': package_versions(), 'xml_sha256': hashlib.sha256(xml.encode()).hexdigest(),
        'control_source': 'policy', 'contains_human_intervention': False,
        'contains_reset_during_rollout': False, 'image_transform': 'flip H and W, same as LeRobot LiberoProcessorStep',
        'alignment': 'observation/state/video[t] precedes executed_action[t]; final observation has no action',
        'state_restore_status': 'full integration state and selected controller fields recorded; replay equivalence not yet tested',
    }
    (args.out / 'run.json').write_text(json.dumps(manifest, indent=2))
    policy.reset()
    writers = [VideoWriter(args.out / f'{name}.mp4', width, 256, 20)
               for name, width in [('agentview', 256), ('wrist', 256), ('side_by_side', 512)]]
    states, proprio, actions, wall_times = [], [], [], []
    success = False
    started = time.monotonic()
    try:
        for step in range(args.steps + 1):
            images = observation['pixels']
            main_rgb = np.ascontiguousarray(images['image'][0][::-1, ::-1])
            wrist_rgb = np.ascontiguousarray(images['image2'][0][::-1, ::-1])
            for writer, image in zip(writers, (main_rgb, wrist_rgb, np.concatenate([main_rgb, wrist_rgb], axis=1))):
                writer.write(image)
            states.append(snapshot_state(problem))
            wall_times.append(time.monotonic() - started)
            batch = env_pre(add_envs_task(vec, preprocess_observation(observation)))
            proprio.append(batch['observation.state'].detach().cpu().numpy()[0].copy())
            if success or step == args.steps:
                break
            with torch.inference_mode():
                action = policy.select_action(pre(batch))
            executed = env_post({'action': post(action)})['action'].detach().cpu().numpy()
            if executed.shape != (1, 7) or not np.isfinite(executed).all():
                raise RuntimeError(f'Invalid policy output: shape={executed.shape}')
            # This is the exact postprocessed action passed to the simulator.
            actions.append(executed[0].copy())
            observation, _, terminated, truncated, _ = vec.step(executed)
            success = bool(env._env.check_success())
            if bool(terminated[0] or truncated[0]) and not success:
                # Keep terminal observation without taking another autoresetting step.
                final = env_pre(add_envs_task(vec, preprocess_observation(observation)))
                proprio.append(final['observation.state'].detach().cpu().numpy()[0].copy())
                states.append(snapshot_state(problem))
                wall_times.append(time.monotonic() - started)
                terminal_main = np.ascontiguousarray(observation['pixels']['image'][0][::-1, ::-1])
                terminal_wrist = np.ascontiguousarray(observation['pixels']['image2'][0][::-1, ::-1])
                for writer, image in zip(writers, (terminal_main, terminal_wrist, np.concatenate([terminal_main, terminal_wrist], axis=1))):
                    writer.write(image)
                break
            if (step + 1) % 25 == 0:
                print(json.dumps({'step': step + 1, 'success': success, 'seconds': round(time.monotonic() - started, 2)}), flush=True)
        np.savez_compressed(args.out / 'native_states.npz', **{
            key: np.stack([row[key] for row in states]) for key in states[0]})
        np.savez_compressed(args.out / 'trajectory.npz',
                            observation_state=np.asarray(proprio), executed_action=np.asarray(actions),
                            elapsed_seconds=np.asarray(wall_times))
        summary = {
            **manifest, 'success': success, 'success_predicate_verified': True,
            'success_predicate': 'native LIBERO check_success()', 'executed_steps': len(actions),
            'observations': len(states), 'duration_seconds': time.monotonic() - started,
            'terminal_reason': 'native_task_success' if success else 'episode_or_smoke_horizon',
            'claim_scope': 'one actual baseline rollout, not benchmark success rate',
        }
        (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
    finally:
        for writer in writers:
            writer.close()
        vec.close()


if __name__ == '__main__':
    main()
