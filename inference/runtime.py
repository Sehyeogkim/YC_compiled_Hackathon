"""GPU model and native environment owner; called only from one worker thread."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import inspect
import importlib.util
import importlib.metadata
from io import BytesIO
import json
import math
import mimetypes
import os
import secrets
from pathlib import Path
import time
import uuid
import xml.etree.ElementTree as ET

MODEL_ID = 'lerobot/smolvla_libero'
MODEL_REVISION = 'a24f1f380461b665158d52acd140152c79e8b36d'
FRAMEWORK_REVISION = 'b9cb121cb7d4e3e26ec5c906d08dda68d151ba52'
RENAME_MAP = {
    'observation.images.image': 'observation.images.camera1',
    'observation.images.image2': 'observation.images.camera2',
}


class NotReady(RuntimeError):
    pass


class SessionConflict(RuntimeError):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def finite_vector(value, length, name, bound=None):
    if not isinstance(value, (tuple, list)) or len(value) != length:
        raise ValueError(f'{name} must contain exactly {length} values')
    numbers = [float(x) for x in value]
    if not all(math.isfinite(x) for x in numbers):
        raise ValueError(f'{name} contains non-finite values')
    if bound is not None and any(abs(x) > bound for x in numbers):
        raise ValueError(f'{name} values must be in [-{bound}, {bound}]')
    return numbers


class NativeRuntime:
    def __init__(self):
        self.model_id = MODEL_ID
        self.model_revision = MODEL_REVISION
        self.model_provenance = None
        self.status = 'not_loaded'
        self.error = None
        self.device = os.environ.get('WERESCUE_DEVICE', 'cuda')
        self.policy_path = os.environ.get('WERESCUE_POLICY_PATH')
        self.session_id = None
        self.vec = self.env = self.observation = None
        self.policy = self.pre = self.post = self.env_pre = self.env_post = None
        self.step_count = 0
        self.done = self.success = False
        self.source = 'policy'
        self.segment = 'task'
        self.history = []
        self.scene_cache = None
        self.session_dir = None
        self.last_saved_observation = -1
        self.suite = os.environ.get('WERESCUE_SUITE', 'libero_object')
        self.task_id = int(os.environ.get('WERESCUE_TASK_ID', '0'))
        self.horizon = int(os.environ.get('WERESCUE_HORIZON', '280'))
        self.gpu_name = None
        self.native_verified = False
        self.native_step_verified = False

    def health(self):
        return {
            'service': 'werescue.native.v1', 'status': self.status,
            'ready': self.status == 'ready', 'model_ready': self.status == 'ready', 'error': self.error,
            'last_api_error': getattr(self, 'last_api_error', None),
            'device': self.device, 'gpu': self.gpu_name,
            'model': {'repo_id': self.model_id, 'revision': self.model_revision,
                      'state_dim': 8, 'action_dim': 7, 'chunk_size': 50,
                      'execute_steps': 10},
            'model_provenance': self.model_provenance,
            'environment': {'suite': self.suite, 'task_id': self.task_id,
                            'control_mode': 'relative', 'control_hz': 20,
                            'episode_horizon_steps': self.horizon,
                            'native_ready': self.native_verified,
                            'native_step_verified': self.native_step_verified},
            'active_session': self.session_id,
            'inference_verified': getattr(self, 'inference_verified', False),
            'runtime_versions': getattr(self, 'runtime_versions', {}),
            'cpu_threads': getattr(self, 'cpu_thread_config', None),
            'training_completed': False,
        }

    def load(self):
        if self.status == 'ready':
            return self.health()
        self.status = 'loading'
        try:
            os.environ.setdefault('MUJOCO_GL', 'egl')
            os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
            os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
            # hf-libero prompts on first import without a config. Keep ours isolated
            # from the image's older environment and write the equivalent defaults.
            libero_spec = importlib.util.find_spec('libero')
            if libero_spec and libero_spec.submodule_search_locations:
                package = Path(next(iter(libero_spec.submodule_search_locations))) / 'libero'
                config_dir = Path(os.environ.setdefault('LIBERO_CONFIG_PATH',
                    str(Path(os.environ.get('WERESCUE_RECORD_ROOT', 'inference/runs')).parent / 'libero-config')))
                config_dir.mkdir(parents=True, exist_ok=True)
                config_path = config_dir / 'config.yaml'
                if not config_path.exists():
                    import yaml
                    config_path.write_text(yaml.safe_dump({
                        'benchmark_root': str(package),
                        'bddl_files': str(package / 'bddl_files'),
                        'init_states': str(package / 'init_files'),
                        'datasets': str(config_dir / 'datasets'),
                        'assets': str(package / 'assets'),
                    }))
                assets_override = os.environ.get('WERESCUE_LIBERO_ASSETS')
                if assets_override:
                    import yaml
                    assets_root = Path(assets_override)
                    if not (assets_root / 'WERESCUE_ASSETS_PIN.json').is_file():
                        raise RuntimeError('Explicit native assets directory is not completely downloaded/pinned')
                    # hf-libero 0.1.4's documented asset resolver caches one root.
                    # Set that root to the identical, revision-recorded Hub snapshot.
                    module = importlib.import_module('libero.libero')
                    module._assets_path_cache = str(assets_root)
                    paths = yaml.safe_load(config_path.read_text())
                    paths['assets'] = str(assets_root)
                    config_path.write_text(yaml.safe_dump(paths))
            import numpy as np
            import torch
            # Tiny camera/state operations become slower with this host's
            # default 48 CPU threads. Configure before model/preprocessor work.
            threads = int(os.environ.get('WERESCUE_TORCH_THREADS', '4'))
            if not 1 <= threads <= 16:
                raise ValueError('WERESCUE_TORCH_THREADS must be between 1 and 16')
            torch.set_num_threads(threads)
            torch.set_num_interop_threads(1)
            self.cpu_thread_config = {'torch_intraop': torch.get_num_threads(),
                                      'torch_interop': torch.get_num_interop_threads()}
            from huggingface_hub import snapshot_download
            from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
            from lerobot.envs.factory import make_env_pre_post_processors
            from lerobot.policies.factory import make_policy, make_pre_post_processors

            self.np, self.torch = np, torch
            if self.device.startswith('cuda'):
                if not torch.cuda.is_available():
                    raise RuntimeError('CUDA requested but torch.cuda.is_available() is false; no CPU/scripted fallback')
                self.gpu_name = torch.cuda.get_device_name(0)
            if self.policy_path is None:
                self.policy_path = snapshot_download(repo_id=self.model_id, revision=self.model_revision)
            else:
                pin = Path(self.policy_path) / 'WERESCUE_PIN.json'
                if pin.exists() and json.loads(pin.read_text()) != {'repo_id': self.model_id, 'revision': self.model_revision}:
                    raise RuntimeError('Local checkpoint pin differs from the experiment')
                if not pin.exists() and Path(self.policy_path).name != self.model_revision:
                    raise RuntimeError('Local checkpoint needs WERESCUE_PIN.json or an exact revision-named Hub snapshot directory')
            cfg = LiberoEnvConfig(task=self.suite, task_ids=[self.task_id],
                                  observation_height=256, observation_width=256,
                                  control_mode='relative', fps=20,
                                  episode_length=self.horizon)
            config = PreTrainedConfig.from_pretrained(self.policy_path)
            config.pretrained_path = self.policy_path
            config.device = self.device
            config.n_action_steps = 10
            state_shape = tuple(config.input_features['observation.state'].shape)
            action_shape = tuple(config.output_features['action'].shape)
            if state_shape != (8,) or action_shape != (7,) or config.chunk_size != 50:
                raise RuntimeError(f'Checkpoint contract changed: state={state_shape}, action={action_shape}, chunk={config.chunk_size}')
            rename = RENAME_MAP if 'observation.images.camera1' in config.input_features else {}
            self.policy = make_policy(cfg=config, env_cfg=cfg, rename_map=rename)
            self.policy.eval()
            self.pre, self.post = make_pre_post_processors(
                policy_cfg=config, pretrained_path=self.policy_path,
                preprocessor_overrides={
                    'device_processor': {'device': self.device},
                    'rename_observations_processor': {'rename_map': rename},
                })
            self.env_pre, self.env_post = make_env_pre_post_processors(env_cfg=cfg, policy_cfg=config)
            self.env_cfg = cfg
            self.policy.reset()
            self.runtime_versions = {}
            for name in ('lerobot', 'torch', 'torchvision', 'torchcodec', 'transformers', 'hf-libero', 'robosuite', 'mujoco', 'numpy'):
                try:
                    self.runtime_versions[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    self.runtime_versions[name] = None
            self.status = 'ready'
            self.error = None
            return self.health()
        except Exception as exc:
            self.status = 'error'
            self.error = f'{type(exc).__name__}: {exc}'
            raise

    def require_ready(self):
        if self.status != 'ready':
            raise NotReady(self.error or f'Model is {self.status}')

    def require_session(self, session_id):
        self.require_ready()
        if not self.session_id or session_id != self.session_id:
            raise SessionConflict('Session is absent or stale; reset/connect to the current session')

    def reset_session(self, initial_state_id=0, seed=123, source='policy', layout_seed=None):
        self.require_ready()
        from lerobot.envs.factory import make_env
        jitter = float(os.environ.get('WERESCUE_POSITION_JITTER_M', '0'))
        if not math.isfinite(jitter) or not 0 <= jitter <= 0.10:
            raise ValueError('WERESCUE_POSITION_JITTER_M must be between 0 and 0.10 metres')
        if jitter and (self.suite, self.task_id) != ('libero_object', 0):
            raise ValueError('Position jitter is supported only for LIBERO Object task 0')
        self.native_verified = False
        if self.session_id:
            self._record_observation()
            self._event('session_reset_requested', was_terminal=self.done,
                        reset_during_active_segment=not self.done)
        if self.vec is None:
            envs = make_env(self.env_cfg, n_envs=1)
            self.vec = envs[self.suite][self.task_id]
            self.env = self.vec.envs[0]
        self.torch.manual_seed(seed)
        self.np.random.seed(seed)
        self.policy.reset()
        self.env.init_state_id = initial_state_id
        self.observation, _ = self.vec.reset(seed=[seed])
        self.problem = self.env._env.env
        language = self.env.task_description.lower().replace('_', ' ')
        if self.suite == 'libero_object' and self.task_id == 0 and not ('alphabet soup' in language and 'basket' in language):
            raise RuntimeError(f'Unexpected native task ordering: {language!r}')
        position_setup = None
        if jitter:
            try:
                from .position_jitter import perturb_target_position
            except ImportError:
                from position_jitter import perturb_target_position
            if layout_seed is None:
                layout_seed = secrets.randbits(63)
            self.observation, position_setup = perturb_target_position(
                self.env, layout_seed, jitter)
        self.session_id = str(uuid.uuid4())
        self.step_count = 0
        self.done = self.success = False
        self.last_action = None
        self.source = source
        self.segment = 'recovery' if source == 'operator' else 'task'
        self.history = []
        self.scene_cache = None
        self.last_saved_observation = -1
        self.session_dir = None
        if os.environ.get('WERESCUE_RECORD', '1') != '0':
            self.session_dir = Path(os.environ.get('WERESCUE_RECORD_ROOT', 'inference/runs')) / self.session_id
            self.session_dir.mkdir(parents=True, exist_ok=False)
        xml = self.problem.sim.model.get_xml()
        self.metadata = {
            'schema': 'werescue.native_session.v1', 'session_id': self.session_id,
            'created_at': utc_now(), 'model': {'repo_id': self.model_id, 'revision': self.model_revision},
            'model_provenance': self.model_provenance,
            'framework_revision_expected': FRAMEWORK_REVISION,
            'runtime_versions': self.runtime_versions,
            'cpu_threads': self.cpu_thread_config,
            'storage': {'png_compress_level': 1, 'png_lossless': True,
                        'state_format': 'numpy_savez_compressed_unchanged'},
            'assets_pin': json.loads((Path(os.environ['WERESCUE_LIBERO_ASSETS']) / 'WERESCUE_ASSETS_PIN.json').read_text()) if os.environ.get('WERESCUE_LIBERO_ASSETS') else None,
            'suite': self.suite, 'task_id': self.task_id, 'task': self.env.task_description,
            'initial_state_id': initial_state_id, 'seed': seed, 'control_hz': 20,
            'position_setup': position_setup,
            'standard_libero_benchmark': position_setup is None,
            'episode_horizon_steps': self.horizon,
            'control_mode': 'relative', 'n_action_steps': 10, 'chunk_size': 50,
            'scene_sha256': hashlib.sha256(xml.encode()).hexdigest(),
            'alignment': 'observation[t], state[t], images[t] precede executed_action[t]',
            'image_transform': 'raw native image flipped on H and W, matching LeRobot LIBERO processor',
            'claim_scope': 'interactive native session; not a benchmark success rate',
        }
        if self.session_dir:
            (self.session_dir / 'metadata.json').write_text(json.dumps(self.metadata, indent=2))
            (self.session_dir / 'scene.xml').write_text(xml)
        self._event('session_reset', source=source, position_setup=position_setup)
        response = self.get_frame(self.session_id)
        self.native_verified = True
        return response

    def _batch(self):
        from lerobot.envs.utils import preprocess_observation
        observation = preprocess_observation(self.observation)
        # Match the pinned LeRobot evaluation rollout: task descriptions are
        # attached to the vector batch before environment-specific processing.
        observation['task'] = list(self.vec.call('task_description'))
        return self.env_pre(observation)

    def _rgb(self):
        pixels = self.observation['pixels']
        return {name: self.np.ascontiguousarray(pixels[key][0][::-1, ::-1])
                for name, key in [('agentview', 'image'), ('wrist', 'image2')]}

    def _camera_payload(self, rgb):
        from PIL import Image
        buffer = BytesIO()
        Image.fromarray(rgb).save(buffer, format='JPEG', quality=90)
        return {'mime': 'image/jpeg', 'data': base64.b64encode(buffer.getvalue()).decode(),
                'width': rgb.shape[1], 'height': rgb.shape[0]}

    def _names(self, kind):
        model = self.problem.sim.model
        plural = 'body_names' if kind == 'body' else f'{kind}_names'
        names = getattr(model, plural, None)
        if names is not None:
            return [n.decode() if isinstance(n, bytes) else str(n) for n in names]
        count = int(getattr(model, f'n{kind}'))
        return [getattr(model, f'{kind}_id2name')(i) or f'{kind}_{i}' for i in range(count)]

    def frame(self):
        batch = self._batch()
        state = batch['observation.state'].detach().cpu().numpy()[0].tolist()
        if len(state) != 8:
            raise RuntimeError(f'Native state is not 8D: {len(state)}')
        sim, robot = self.problem.sim, self.problem.robots[0]
        data = sim.data
        body_positions = getattr(data, 'body_xpos', getattr(data, 'xpos', None))
        body_quaternions = getattr(data, 'body_xquat', getattr(data, 'xquat', None))
        eef_site_id = getattr(robot, 'eef_site_id', None)
        if isinstance(eef_site_id, dict):
            eef_site_id = next(iter(eef_site_id.values()))
        orientation = data.site_xmat[eef_site_id].tolist() if eef_site_id is not None else []
        return {
            'cameras': {name: self._camera_payload(image) for name, image in self._rgb().items()},
            'state': state, 'joints': self.np.asarray(robot._joint_positions).tolist(),
            'eef': state[:3], 'eef_orientation': self.np.asarray(orientation).reshape(-1).tolist(),
            'gripper': state[-2:], 'qpos': self.np.asarray(data.qpos).tolist(),
            'qvel': self.np.asarray(data.qvel).tolist(),
            'body_poses': {'names': self._names('body'),
                           'xpos': self.np.asarray(body_positions).tolist(),
                           'xquat': self.np.asarray(body_quaternions).tolist()},
            'sim_time': float(data.time), 'step': self.step_count,
            'success': self.success, 'done': self.done,
            'control_source': self.source, 'segment': self.segment,
            'task': self.env.task_description, 'last_action': getattr(self, 'last_action', None),
        }

    def get_frame(self, session_id):
        self.require_session(session_id)
        return {'session_id': session_id, 'scene_sha256': self.metadata['scene_sha256'],
                'frame': self.frame()}

    def control(self, session_id, source):
        self.require_session(session_id)
        if self.done:
            raise SessionConflict('Episode is terminal; reset explicitly before further control')
        self._set_control_source(source)
        return self.get_frame(session_id)

    def _set_control_source(self, source):
        if source != self.source:
            self.policy.reset()
            self.source = source
            self.segment = 'recovery' if source == 'operator' else 'task'
            self._event('control_handoff', source=source, queue_reset=True)

    def _event(self, kind, **fields):
        item = {'kind': kind, 'sequence': len(self.history), 'utc': utc_now(),
                'step': self.step_count, 'segment': self.segment, 'control_source': self.source, **fields}
        self.history.append(item)
        if self.session_dir:
            with (self.session_dir / 'events.jsonl').open('a') as stream:
                stream.write(json.dumps(item, allow_nan=False) + '\n')

    def _record_observation(self):
        if not self.session_dir or self.last_saved_observation == self.step_count:
            return
        from PIL import Image
        prefix = self.session_dir / f'{self.step_count:06d}'
        for name, rgb in self._rgb().items():
            Image.fromarray(rgb).save(f'{prefix}.{name}.png', compress_level=1)
        sim, controller = self.problem.sim, self.problem.robots[0].controller
        arrays = {'flattened_state': self.np.array(sim.get_state().flatten(), copy=True),
                  'observation_state': self._batch()['observation.state'].detach().cpu().numpy()[0]}
        for key in ('qpos', 'qvel', 'act', 'ctrl', 'qacc_warmstart', 'mocap_pos', 'mocap_quat'):
            value = getattr(sim.data, key, None)
            if value is not None:
                arrays[key] = self.np.array(value, copy=True)
        for key in ('goal_pos', 'goal_ori', 'ee_pos', 'ee_ori_mat'):
            value = getattr(controller, key, None)
            if value is not None:
                arrays['controller_' + key] = self.np.array(value, copy=True)
        arrays['sim_time'] = self.np.asarray(float(sim.data.time))
        self.np.savez_compressed(f'{prefix}.state.npz', **arrays)
        self.last_saved_observation = self.step_count

    def step(self, session_id, source, action, steps):
        self.require_session(session_id)
        if self.done:
            raise SessionConflict('Episode is terminal; reset explicitly')
        if source == 'operator':
            values = finite_vector(action, 7, 'operator action', bound=1)
        elif action is not None:
            raise ValueError('Policy steps cannot supply an operator action')
        self._set_control_source(source)
        actions = []
        inference_ms = 0.0
        policy_call_ms = 0.0
        fresh_predictions = 0
        for _ in range(steps):
            self._record_observation()
            if source == 'policy':
                fresh_prediction = len(self.policy._queues.get('action', [])) == 0
                before = time.perf_counter()
                with self.torch.inference_mode():
                    predicted = self.policy.select_action(self.pre(self._batch()))
                    executed = self.env_post({'action': self.post(predicted)})['action']
                if self.device.startswith('cuda'):
                    self.torch.cuda.synchronize()
                call_ms = (time.perf_counter() - before) * 1000
                policy_call_ms += call_ms
                if fresh_prediction:
                    inference_ms += call_ms
                    fresh_predictions += 1
                executed = executed.detach().cpu().numpy()
                if executed.shape != (1, 7) or not self.np.isfinite(executed).all():
                    raise RuntimeError(f'Invalid actual model output: {executed.shape}')
                if fresh_prediction:
                    self.inference_verified = True
            else:
                fresh_prediction = False
                executed = self.np.asarray([values], dtype=self.np.float32)
            self.last_action = executed[0].tolist()
            self._event('action_requested', action=self.last_action,
                        observation_step=self.step_count, native_action_dim=7)
            observation, _, terminated, truncated, _ = self.vec.step(executed)
            self.native_step_verified = True
            self._event('executed_action', action=self.last_action,
                        observation_step=self.step_count, native_action_dim=7,
                        fresh_prediction=fresh_prediction,
                        meaning='native_simulator_step_input; not measured achieved displacement')
            self.observation = observation
            self.step_count += 1
            self.success = bool(self.env._env.check_success())
            self.done = bool(self.success or terminated[0] or truncated[0])
            actions.append(self.last_action)
            if self.done:
                self._record_observation()
                self._event('episode_terminal', success=self.success,
                            predicate='native_LIBERO_check_success')
                self.policy.reset()
                break
        return {**self.get_frame(session_id), 'executed_actions': actions,
                'inference_ms': inference_ms, 'policy_call_ms': policy_call_ms,
                'fresh_predictions': fresh_predictions, 'model_inference': fresh_predictions > 0,
                'action_source': 'pretrained_policy' if source == 'policy' else 'operator'}

    def mark(self, session_id, marker, detail):
        self.require_session(session_id)
        if marker == 'recovery_complete':
            if self.source != 'operator' or self.segment != 'recovery':
                raise SessionConflict('Recovery boundary requires active human recovery control')
            self._event('recovery_complete_annotation', detail=detail,
                        independently_verified=False, control_retained=True)
            self.segment = 'task'
            self._event('human_task_continuation_start', detail=detail)
        else:
            self._event('note', detail=detail)
        return self.get_frame(session_id)

    def export(self, session_id):
        self.require_session(session_id)
        return {'metadata': self.metadata, 'events': self.history,
                'server_recording_enabled': self.session_dir is not None,
                'observations_saved': self.last_saved_observation + 1,
                'state_restore_verified': False,
                'success': self.success, 'done': self.done}

    def scene(self, session_id):
        self.require_session(session_id)
        if self.scene_cache:
            return self.scene_cache
        # Only asset paths already referenced by the trusted simulator model are read.
        # There is deliberately no general filesystem or arbitrary-path asset endpoint.
        xml = self.problem.sim.model.get_xml()
        root = ET.fromstring(xml)
        compiler = root.find('compiler')
        meshdir = compiler.get('meshdir', '') if compiler is not None else ''
        texturedir = compiler.get('texturedir', '') if compiler is not None else ''
        candidates = [Path.cwd()]
        try:
            from libero.libero import get_libero_path
            candidates.append(Path(get_libero_path('assets')))
        except ImportError:
            pass
        assets, seen = [], {}
        for element in root.iter():
            file = element.get('file')
            if not file:
                continue
            directory = meshdir if element.tag == 'mesh' else texturedir if element.tag == 'texture' else ''
            path = Path(file)
            if not path.is_absolute():
                options = [Path(directory) / file, *[p / directory / file for p in candidates], *[p / file for p in candidates]]
                path = next((p for p in options if p.is_file()), path)
            if not path.is_file():
                raise RuntimeError(f'Native model references an unavailable asset: {file}')
            key = str(path.resolve())
            if key not in seen:
                content = path.read_bytes()
                safe = f'assets/{hashlib.sha256(content).hexdigest()[:16]}{path.suffix}'
                seen[key] = safe
                assets.append({'path': safe, 'data': base64.b64encode(content).decode(),
                               'mime': mimetypes.guess_type(path.name)[0] or 'application/octet-stream'})
            element.set('file', seen[key])
        if compiler is not None:
            compiler.attrib.pop('meshdir', None)
            compiler.attrib.pop('texturedir', None)
        model = self.problem.sim.model
        controller = self.problem.robots[0].controller
        out_max = self.np.asarray(controller.output_max, dtype=float).reshape(-1)
        out_min = self.np.asarray(controller.output_min, dtype=float).reshape(-1)
        in_max = self.np.asarray(controller.input_max, dtype=float).reshape(-1)
        in_min = self.np.asarray(controller.input_min, dtype=float).reshape(-1)
        if in_max.size == 1:
            in_max = self.np.repeat(in_max, 6)
        if in_min.size == 1:
            in_min = self.np.repeat(in_min, 6)
        if any(array.size != 6 for array in (out_max, out_min, in_max, in_min)):
            raise RuntimeError('Native OSC controller does not expose the expected 6D scaling')
        scale = (out_max - out_min) / (in_max - in_min)
        offset = (out_max + out_min) / 2 - scale * (in_max + in_min) / 2
        gripper = self.problem.robots[0].gripper
        if isinstance(gripper, dict):
            gripper = next(iter(gripper.values()))
        gripper_source = inspect.getsource(type(gripper).format_action)
        gripper_holds_zero = 'np.sign(action)' in gripper_source and 'current_action' in gripper_source
        self.scene_cache = {'xml': ET.tostring(root, encoding='unicode'), 'assets': assets,
            'nq': int(model.nq), 'nv': int(model.nv),
            'body_names': self._names('body'), 'geom_names': self._names('geom'),
            'site_names': self._names('site'), 'cameras': self._names('camera'),
            'camera_observations': ['agentview', 'wrist'],
            'coordinate_frame': 'native_mujoco_z_up_meters',
            'control': {
                'action_order': ['dx', 'dy', 'dz', 'rx', 'ry', 'rz', 'gripper'],
                'action_dim': 7, 'control_hz': 20, 'dt_seconds': 0.05,
                'mode': 'relative_OSC_POSE', 'position_frame': 'world',
                'translation_scale_m': scale[:3].tolist(),
                'rotation_scale_rad': scale[3:6].tolist(),
                'translation_offset_m': offset[:3].tolist(),
                'rotation_offset_rad': offset[3:6].tolist(),
                'input_min': in_min.tolist(), 'input_max': in_max.tolist(),
                'output_min': out_min.tolist(), 'output_max': out_max.tolist(),
                'gripper_open': -1, 'gripper_close': 1,
                'gripper_neutral': 0 if gripper_holds_zero else None,
                'gripper_type': type(gripper).__name__,
                'gripper_neutral_verified_from': 'installed_format_action_source' if gripper_holds_zero else None,
            },
            'scene_sha256': self.metadata['scene_sha256']}
        return self.scene_cache

    def predict(self, camera1, camera2, state, task, reset):
        self.require_ready()
        if not reset:
            raise ValueError('Standalone prediction is stateless: reset must be true; use native sessions for queued execution')
        if self.session_id:
            raise SessionConflict('Standalone predict is isolated from active native sessions; use /session/step')
        from PIL import Image
        values = finite_vector(state, 8, 'state')
        batch = {'observation.state': self.torch.tensor([values], dtype=self.torch.float32), 'task': [task]}
        image_keys = list(self.policy.config.image_features)
        if len(image_keys) != 2:
            raise RuntimeError(f'Expected exactly two policy cameras, got {image_keys}')
        for key, encoded in zip(image_keys, [camera1, camera2]):
            try:
                raw = encoded.split(',', 1)[1] if encoded.startswith('data:') else encoded
                image = Image.open(BytesIO(base64.b64decode(raw, validate=True))).convert('RGB')
            except Exception as exc:
                raise ValueError('Camera must contain a valid base64 JPEG or PNG') from exc
            if image.size != (256, 256):
                raise ValueError(f'Camera must be 256x256, got {image.size}')
            array = self.np.asarray(image).copy()
            batch[key] = self.torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float() / 255
        # Every standalone chunk starts with clean temporal/action queues.
        self.policy.reset()
        before = time.perf_counter()
        try:
            with self.torch.inference_mode():
                chunk = self.policy.predict_action_chunk(self.pre(batch))
                actions = []
                for index in range(chunk.shape[1]):
                    action = self.env_post({'action': self.post(chunk[:, index, :])})['action']
                    actions.append(action.detach().cpu().numpy()[0].tolist())
            if self.device.startswith('cuda'):
                self.torch.cuda.synchronize()
            if len(actions) != 50 or any(len(a) != 7 or not all(math.isfinite(v) for v in a) for a in actions):
                raise RuntimeError('Actual model output does not match the 50x7 action contract')
            self.inference_verified = True
            return {'actions': actions, 'action_dim': 7, 'chunk_size': 50,
                    'execute_steps': 10, 'inference_ms': (time.perf_counter() - before) * 1000,
                    'model': {'repo_id': self.model_id, 'revision': self.model_revision},
                    'model_provenance': self.model_provenance,
                    'queues_reset': True}
        finally:
            self.policy.reset()

    def close(self):
        if self.vec is not None:
            self.vec.close()
            self.vec = None
