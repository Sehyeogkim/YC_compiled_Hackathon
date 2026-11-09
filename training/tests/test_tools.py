"""Synthetic fixtures test bookkeeping only; no model performance is asserted."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from manifests import ValidationError, validate_collection, validate_recovery_video
from evaluate import EvaluationError, chart_svg, summarize


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'asset').write_text('synthetic metadata fixture only')
        phase = dict(start_frame=0, end_frame=10, control_source='operator', success=True,
                     success_predicate='native_predicate_v1', predicate_verified=True,
                     evidence_ref='asset', contains_reset=False)
        self.episode = dict(episode_id='a', parent_episode_id='parent-a', initial_state_id='state-a',
                            split='train', task_id='task', environment_id='env', robot_id='robot',
                            operator_id='human', action_schema='native_eef_delta7', actions_are='executed',
                            frame_count=20, fps=20, camera_refs={'agentview': 'asset', 'wrist': 'asset'},
                            state_ref='asset', action_ref='asset', timestamps_ref='asset',
                            recovery=phase, task={**phase, 'start_frame': 10, 'end_frame': 20})

    def tearDown(self):
        self.temp.cleanup()

    def manifest(self, *episodes):
        return {'schema_version': 'werescue.collection.v1', 'episodes': list(episodes or [self.episode])}

    def test_segments_are_deterministic_and_do_not_cross_handoff(self):
        first = validate_collection(self.manifest(), self.root)
        self.assertEqual(first, validate_collection(self.manifest(), self.root))
        by_dataset = {s['dataset']: s for s in first['segments']}
        self.assertEqual(by_dataset['R_prime']['end_frame'], by_dataset['R']['start_frame'])
        self.assertEqual(by_dataset['R_prime']['assets']['action']['path'], str((self.root / 'asset').resolve()))

    def test_same_parent_cannot_leak_between_splits(self):
        other = {**self.episode, 'episode_id': 'b', 'initial_state_id': 'state-b', 'split': 'test'}
        with self.assertRaisesRegex(ValidationError, 'parent_episode_id.*leaks'):
            validate_collection(self.manifest(self.episode, other), self.root)

    def test_same_initial_state_cannot_leak_between_splits(self):
        other = {**self.episode, 'episode_id': 'b', 'parent_episode_id': 'parent-b', 'split': 'val'}
        with self.assertRaisesRegex(ValidationError, 'initial_state_id.*leaks'):
            validate_collection(self.manifest(self.episode, other), self.root)

    def test_requested_commands_are_not_executed_demonstrations(self):
        self.episode['actions_are'] = 'requested'
        with self.assertRaisesRegex(ValidationError, 'executed actions'):
            validate_collection(self.manifest(), self.root)

    def test_missing_rgb_cannot_generate_training_manifest(self):
        self.episode['camera_refs'] = {}
        with self.assertRaisesRegex(ValidationError, 'camera_refs'):
            validate_collection(self.manifest(), self.root)

    def test_reset_is_not_recovery(self):
        self.episode['recovery']['contains_reset'] = True
        with self.assertRaisesRegex(ValidationError, 'contains_reset'):
            validate_collection(self.manifest(), self.root)

    def test_phase_overlap_is_rejected(self):
        self.episode['task']['start_frame'] = 9
        with self.assertRaisesRegex(ValidationError, 'handoff'):
            validate_collection(self.manifest(), self.root)

    def test_missing_evidence_rejected(self):
        self.episode['recovery']['evidence_ref'] = 'missing'
        with self.assertRaisesRegex(ValidationError, 'missing local file'):
            validate_collection(self.manifest(), self.root)

    def test_recovery_video_cannot_be_scripted_demo(self):
        data = dict(schema_version='werescue.recovery_video.v1', checkpoint='model',
                    failure_state_id='failed', recovery_state_id='safe', task_id='task',
                    recovery_predicate='safe_v1', control_source='demo_controller')
        with self.assertRaisesRegex(ValidationError, 'policy execution'):
            validate_recovery_video(data, self.root)


class EvaluationTests(unittest.TestCase):
    def fixture(self):
        rows = []
        for arm in ('P', 'P+R'):
            for i in range(3):
                rows.append(dict(arm=arm, split='test', control_source='policy',
                                 contains_human_intervention=False, success=(i > (0 if arm == 'P+R' else 1)),
                                 success_predicate_verified=True, seed=i, task_id='task',
                                 start_state_id=f'state-{i}', checkpoint=f'{arm}-checkpoint',
                                 environment_id='env', trace_ref=f'{arm}/{i}.json'))
        return dict(schema_version='werescue.task_eval.v1', protocol_id='synthetic-unit-test', trials=rows)

    def test_paired_counts_and_denominators(self):
        summary = summarize(self.fixture())
        self.assertEqual(summary['arms']['P']['successes'], 1)
        self.assertEqual(summary['arms']['P+R']['successes'], 2)
        self.assertEqual(summary['paired_outcomes'], {'00': 1, '01': 1, '10': 0, '11': 1})
        self.assertIn('1/3', chart_svg(summary))
        self.assertIn('Wilson', chart_svg(summary))

    def test_mismatched_starts_rejected(self):
        data = self.fixture()
        data['trials'][-1]['start_state_id'] = 'different'
        with self.assertRaisesRegex(EvaluationError, 'same held-out starts'):
            summarize(data)

    def test_duplicate_trial_rejected(self):
        data = self.fixture()
        data['trials'].append(copy.deepcopy(data['trials'][0]))
        with self.assertRaisesRegex(EvaluationError, 'duplicate trial'):
            summarize(data)

    def test_training_start_rejected(self):
        data = self.fixture()
        data['training_initial_state_ids'] = ['state-0']
        with self.assertRaisesRegex(EvaluationError, 'overlaps training'):
            summarize(data)

    def test_numeric_training_ids_cannot_bypass_string_id_exclusion(self):
        data = self.fixture()
        data['training_initial_state_ids'] = [0]
        with self.assertRaisesRegex(EvaluationError, 'canonical string IDs'):
            summarize(data)

    def test_empty_results_cannot_make_chart(self):
        data = self.fixture()
        data['trials'] = []
        with self.assertRaisesRegex(EvaluationError, 'measured trials required'):
            summarize(data)

    def test_human_assistance_cannot_enter_autonomous_task_comparison(self):
        data = self.fixture()
        data['trials'][0]['contains_human_intervention'] = True
        with self.assertRaisesRegex(EvaluationError, 'human intervention'):
            summarize(data)


if __name__ == '__main__':
    unittest.main()
