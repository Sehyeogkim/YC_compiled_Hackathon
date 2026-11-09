"""Collector control/recording tests with a fake adapter, not native physics tests."""
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from server import CollectionDesk, load_cases


class FakeAdapter:
    metadata = {'language': 'Synthetic unit-test task'}

    def __init__(self):
        self.i = 0
        self.goal_pass = True
        self.success_at = 4
        self.actions = []
        self.obs = {'agentview_image': np.zeros((4, 4, 3), np.uint8),
                    'robot0_eye_in_hand_image': np.zeros((4, 4, 3), np.uint8)}

    def restore(self, path):
        self.i = 0

    def recovery_predicate(self, goal):
        return {'passed': self.goal_pass, 'predicate': 'fake_test_only'}

    def state8(self, obs=None):
        return np.full(8, self.i, np.float32)

    def sim_state(self):
        return np.full(10, self.i, np.float32)

    def scene(self):
        return {'sim_time': self.i / 20}

    def step(self, action):
        self.actions.append(list(action))
        self.i += 1
        return {'reward': 0, 'done': False, 'task_success': self.i >= self.success_at,
                'ctrl': np.zeros(9, np.float32)}


class CollectorFlowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'state.npy').write_bytes(b'fake adapter input only')
        (self.root / 'goal.json').write_text('{}')
        self.case = {'case_id': 'case-0', 'parent_episode_id': 'parent-0',
                     'initial_state_id': 'libero_object/task0/init0', 'split': 'train',
                     'failure_state_ref': str(self.root / 'state.npy'),
                     'goal_ref': str(self.root / 'goal.json')}
        self.adapter = FakeAdapter()
        self.desk = CollectionDesk(self.adapter, {'cases': {'case-0': self.case}}, self.root / 'out', 'human', 20)
        self.desk._video_optional = lambda: None

    def tearDown(self):
        self.tmp.cleanup()

    def test_recovery_boundary_retains_human_and_slices_correct_frames(self):
        self.desk.start('case-0')
        self.desk.tick()
        self.desk.tick()
        self.desk.recovery_complete()
        self.assertTrue(self.desk.active)
        self.assertEqual(self.desk.phase, 'task')
        self.assertEqual(self.desk.status()['controller'], 'human_operator')
        self.desk.tick()
        self.desk.tick()
        self.assertFalse(self.desk.active)
        meta = json.loads((self.desk.directory / 'episode.json').read_text())
        self.assertEqual(meta['qualification'], 'qualified')
        self.assertEqual(meta['phase_boundary_frame'], 2)
        with np.load(self.desk.directory / 'R_prime.npz') as recovery:
            self.assertEqual(recovery['phase'].tolist(), [0, 0])
            self.assertEqual(recovery['observation_state'][0, 0], 0)
            self.assertEqual(recovery['next_observation_state'][-1, 0], 2)
        with np.load(self.desk.directory / 'R.npz') as task:
            self.assertEqual(task['phase'].tolist(), [1, 1])
            self.assertEqual(task['observation_state'][0, 0], 2)
        self.assertTrue((self.desk.directory / 'agentview_terminal.png').is_file())

    def test_expired_controls_stop_motion_but_hold_gripper(self):
        self.desk.start('case-0')
        self.desk.set_action([1, 0, 0, 0, 0, 0, 1])
        self.assertEqual(self.desk.action[0], .4)
        self.desk.deadline = time.monotonic() - 1
        self.desk.tick()
        self.assertEqual(self.adapter.actions[-1], [0, 0, 0, 0, 0, 0, 1])
        self.desk.finish(False)

    def test_cannot_restore_during_recording(self):
        self.desk.start('case-0')
        with self.assertRaisesRegex(ValueError, 'resets during recording'):
            self.desk.start('case-0')
        self.desk.finish(False)

    def test_boundary_refuses_unverified_goal(self):
        self.desk.start('case-0')
        self.desk.tick()
        self.adapter.goal_pass = False
        with self.assertRaisesRegex(ValueError, 'predicate has not passed'):
            self.desk.recovery_complete()
        self.assertEqual(self.desk.phase, 'recovery')
        self.assertIsNone(self.desk.phase_boundary)
        self.desk.finish(False)

    def test_success_before_boundary_cannot_be_training_qualified(self):
        self.adapter.success_at = 1
        self.desk.start('case-0')
        self.desk.tick()
        self.assertFalse(self.desk.active)
        meta = json.loads((self.desk.directory / 'episode.json').read_text())
        self.assertEqual(meta['qualification'], 'unqualified')

    def test_case_manifest_rejects_same_init_across_splits(self):
        other = {**self.case, 'case_id': 'case-1', 'parent_episode_id': 'parent-1', 'split': 'test'}
        path = self.root / 'cases.json'
        path.write_text(json.dumps({'schema_version': 'werescue.collection_cases.v1', 'cases': [self.case, other]}))
        with self.assertRaisesRegex(ValueError, 'initial_state_id leaks'):
            load_cases(path)


if __name__ == '__main__':
    unittest.main()
