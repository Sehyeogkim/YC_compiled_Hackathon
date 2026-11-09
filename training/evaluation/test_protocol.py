"""Protocol/split/evidence tests that require no CUDA, NumPy or ML libraries."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocol import make_protocol, parse_selection, summarize, trials, validate_done
from run_baseline import file_hash, read_completed


def valid_trial(protocol, trial):
    return {'schema_version': 'werescue.baseline_trial.v1', **trial,
            'protocol_sha256': protocol['protocol_sha256'], 'policy': protocol['policy'],
            'success': False, 'control_source': 'policy', 'contains_human_intervention': False,
            'contains_reset_during_rollout': False, 'success_predicate_verified': True,
            'success_predicate': 'native_LIBERO_check_success', 'executed_steps': 280,
            'trace_sha256': 'a'*64, 'initial_state_sha256': 'b'*64,
            'initial_native_state_sha256': 'e'*64,
            'initial_xml_sha256': 'c'*64, 'runtime_sha256': 'd'*64}


class SelectionTests(unittest.TestCase):
    def test_pilot_and_final_are_disjoint_with_expected_sizes(self):
        pilot = make_protocol('pilot', list(range(10)), [1000, 1001])
        final = make_protocol('final', list(range(25, 50)), [1000, 1001])
        self.assertEqual(len(trials(pilot)), 20)
        self.assertEqual(len(trials(final)), 50)
        self.assertTrue(set(pilot['initial_states']).isdisjoint(final['initial_states']))
        self.assertEqual(len({row['trial_id'] for row in trials(final)}), 50)

    def test_policy_seed_does_not_change_layout_environment_seed(self):
        protocol = make_protocol('final', [25, 26], [1000, 1001])
        selected = trials(protocol)
        self.assertEqual(selected[0]['env_seed'], selected[1]['env_seed'])
        self.assertNotEqual(selected[0]['policy_rng_seed'], selected[1]['policy_rng_seed'])
        self.assertNotEqual(selected[0]['env_seed'], selected[2]['env_seed'])

    def test_final_rejects_training_or_selection_starts(self):
        for states in ([0], [10], [19], [20], [24], [25, 50]):
            with self.assertRaisesRegex(ValueError, 'reserved IDs'):
                make_protocol('final', states, [1000, 1001])
        with self.assertRaisesRegex(ValueError, 'reserved IDs'):
            make_protocol('pilot', [25], [1000, 1001])

    def test_shared_protocol_is_fixed_and_adapter_label_is_explicit(self):
        libero = make_protocol('pilot', [0], [1000, 1001])
        base = make_protocol('pilot', [0], [1000, 1001], arm='base-adapted')
        self.assertEqual(trials(libero), trials(base))
        self.assertNotEqual(libero['protocol_sha256'], base['protocol_sha256'])
        self.assertEqual(base['policy']['gradient_updates'], 0)
        self.assertIn('not raw zero-shot', base['policy']['label'])
        self.assertEqual(base['raw_smolvla_base']['status'], 'incompatible_NA')
        for horizon, seeds in [(281, [1000, 1001]), (280, [1000]), (280, [1000, 1001, 1002])]:
            with self.assertRaises(ValueError):
                make_protocol('pilot', [0], seeds, horizon=horizon)

    def test_range_parser_rejects_duplicates_and_bad_ranges(self):
        self.assertEqual(parse_selection('25-27,30'), [25, 26, 27, 30])
        for raw in ('0-2,2', '3-1', '-1', '1,,2'):
            with self.assertRaises(ValueError):
                parse_selection(raw)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.protocol = make_protocol('pilot', [0], [1000, 1001])
        self.trial = trials(self.protocol)[0]

    def test_only_matching_verified_autonomous_done_file_can_resume(self):
        data = valid_trial(self.protocol, self.trial)
        validate_done(data, self.protocol, self.trial)
        for key, value in [('protocol_sha256', 'other'), ('success', 'false'),
                           ('contains_human_intervention', True), ('contains_reset_during_rollout', True),
                           ('success_predicate_verified', False), ('executed_steps', 281),
                           ('initial_native_state_sha256', None)]:
            changed = {**data, key: value}
            with self.assertRaises(ValueError, msg=key):
                validate_done(changed, self.protocol, self.trial)

    def test_resume_rehashes_evidence_and_checks_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            directory = out / self.trial['trial_id']
            directory.mkdir()
            row = valid_trial(self.protocol, self.trial)
            for name, key in [('trace.jsonl.gz', 'trace_sha256'),
                              ('initial_state.npz', 'initial_state_sha256'), ('scene.xml', 'initial_xml_sha256')]:
                (directory / name).write_bytes(b'evidence-placeholder-for-io-contract-test')
                row[key] = file_hash(directory / name)
            (directory / 'done.json').write_text(json.dumps(row))
            self.assertEqual(len(read_completed(out, self.protocol, 'd'*64)), 1)
            with self.assertRaisesRegex(ValueError, 'runtime changed'):
                read_completed(out, self.protocol, 'x'*64)
            (directory / 'trace.jsonl.gz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed evidence'):
                read_completed(out, self.protocol)

    def test_empty_summary_has_no_invented_rate(self):
        result = summarize([], self.protocol)
        self.assertIsNone(result['success_rate'])
        self.assertIsNone(result['wilson_95'])
        self.assertEqual(result['completed_trials'], 0)


if __name__ == '__main__':
    unittest.main()
