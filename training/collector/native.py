"""Native LIBERO adapter. Heavy dependencies are imported only at construction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class NativeLibero:
    def __init__(self, suite_name: str, task_id: int, size: int = 256):
        import numpy as np
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        from robosuite.utils.transform_utils import quat2axisangle

        self.np, self.quat2axisangle = np, quat2axisangle
        suite = benchmark.get_benchmark_dict()[suite_name]()
        self.task = suite.get_task(task_id)
        bddl = Path(get_libero_path('bddl_files')) / self.task.problem_folder / self.task.bddl_file
        self.env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=size,
                                      camera_widths=size, control_freq=20)
        self.obs = self.env.reset()
        for robot in self.env.robots:
            robot.controller.use_delta = True
        self.problem = self.env.env
        self.metadata = {'environment': 'native_libero', 'suite': suite_name, 'task_id': task_id,
                         'language': self.task.language, 'bddl_file': str(bddl),
                         'bddl_sha256': hashlib.sha256(bddl.read_bytes()).hexdigest(),
                         'control_freq_hz': 20, 'camera_size': size,
                         'cameras': ['agentview_image', 'robot0_eye_in_hand_image'],
                         'state_schema': 'eef_xyz3+eef_axisangle3+gripper_qpos2',
                         'action_schema': 'native_relative_OSC_POSE_delta_xyz3_rotation3_gripper1',
                         'gripper_action': '-1=open,+1=close',
                         'images': 'stored exactly as native raw observations; preview rotated180 only'}

    def restore(self, path: Path) -> None:
        state = self.np.load(path, allow_pickle=False)
        if state.ndim != 1 or not self.np.isfinite(state).all():
            raise ValueError('Failure state must be a finite 1D native get_sim_state array')
        self.env.reset()
        self.obs = self.env.set_init_state(state)
        for robot in self.env.robots:
            robot.controller.use_delta = True

    def state8(self, obs=None):
        obs = self.obs if obs is None else obs
        return self.np.concatenate([obs['robot0_eef_pos'], self.quat2axisangle(obs['robot0_eef_quat']),
                                    obs['robot0_gripper_qpos']]).astype(self.np.float32)

    def sim_state(self):
        return self.np.asarray(self.env.get_sim_state()).copy()

    def scene(self) -> dict:
        problem = self.problem
        objects = {name: self.np.asarray(problem.sim.data.body_xpos[body]).tolist()
                   for name, body in problem.obj_body_id.items()}
        return {'eef_pos': self.np.asarray(self.obs['robot0_eef_pos']).tolist(),
                'eef_quat': self.np.asarray(self.obs['robot0_eef_quat']).tolist(),
                'gripper_qpos': self.np.asarray(self.obs['robot0_gripper_qpos']).tolist(),
                'object_positions': objects,
                'max_joint_speed': float(self.np.max(self.np.abs(problem.sim.data.qvel))),
                'sim_time': float(problem.sim.data.time)}

    def step(self, action):
        action = self.np.asarray(action, dtype=self.np.float32)
        if action.shape != (7,) or not self.np.isfinite(action).all() or (self.np.abs(action) > 1).any():
            raise ValueError('Action must be seven finite native normalized controls in [-1, 1]')
        obs, reward, done, info = self.env.step(action)
        self.obs = obs
        return {'reward': float(reward), 'done': bool(done), 'task_success': bool(self.env.check_success()),
                'ctrl': self.np.asarray(self.problem.sim.data.ctrl).copy()}

    def recovery_predicate(self, goal: dict) -> dict:
        """Conservative task-specific resumable-state check, not a hardware safety certificate."""
        np = self.np
        scene = self.scene()
        for field in ('eef_pos', 'eef_quat', 'gripper_qpos', 'object_positions'):
            if field not in goal:
                raise ValueError(f'Goal requires {field}; pose-only matching is insufficient')
        if not goal['object_positions']:
            raise ValueError('Goal requires at least the target object position')
        pos_error = float(np.linalg.norm(np.asarray(scene['eef_pos']) - np.asarray(goal['eef_pos'])))
        q1, q2 = np.asarray(scene['eef_quat']), np.asarray(goal['eef_quat'])
        if q1.shape != (4,) or q2.shape != (4,) or np.linalg.norm(q2) < 1e-6:
            raise ValueError('Goal quaternion must be four nonzero XYZW values')
        rot_error = float(2 * np.arccos(np.clip(abs(np.dot(q1 / np.linalg.norm(q1), q2 / np.linalg.norm(q2))), 0, 1)))
        grip_error = float(np.linalg.norm(np.asarray(scene['gripper_qpos']) - np.asarray(goal['gripper_qpos'])))
        width = float(abs(scene['gripper_qpos'][0] - scene['gripper_qpos'][1]))
        object_errors = {}
        for name, target in goal['object_positions'].items():
            if name not in scene['object_positions']:
                raise ValueError(f'Unknown goal object: {name}')
            object_errors[name] = float(np.linalg.norm(np.asarray(scene['object_positions'][name]) - np.asarray(target)))
        checks = {'eef_near_goal': pos_error <= goal.get('eef_tolerance_m', .025),
                  'orientation_near_goal': rot_error <= goal.get('rotation_tolerance_rad', .20),
                  'gripper_near_goal': grip_error <= goal.get('gripper_tolerance_m', .015),
                  'gripper_open': width >= goal.get('minimum_open_width_m', .04),
                  'scene_near_goal': max(object_errors.values()) <= goal.get('object_tolerance_m', .025),
                  'motion_settled': scene['max_joint_speed'] <= goal.get('max_joint_speed', .15)}
        return {'predicate': 'native_resumable_pose_scene_open_settled_v1', 'passed': all(checks.values()),
                'checks': checks, 'eef_error_m': pos_error, 'rotation_error_rad': rot_error,
                'gripper_error_m': grip_error, 'object_errors_m': object_errors, 'scene': scene}

    def close(self):
        self.env.close()
