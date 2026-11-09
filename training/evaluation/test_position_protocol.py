"""Pure jitter sampling/selection checks; no simulator or GPU required."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from position_protocol import (admissible_xy, choose_target_xy, make_protocol,
                               summarize, trials)


class PositionTests(unittest.TestCase):
    def test_position_seed_is_shared_by_both_policy_seeds_and_arms(self):
        first = make_protocol('pilot', [0, 1], [1000, 1001], arm='libero')
        second = make_protocol('pilot', [0, 1], [1000, 1001], arm='base-adapted')
        self.assertEqual(trials(first), trials(second))
        self.assertEqual(trials(first)[0]['layout_rng_seed'], trials(first)[1]['layout_rng_seed'])
        self.assertNotEqual(trials(first)[0]['layout_rng_seed'], trials(first)[2]['layout_rng_seed'])
        self.assertFalse(first['condition']['standard_LIBERO_benchmark'])

    def test_deterministic_bounded_proposal_and_real_variation(self):
        starts = []
        others = {'basket': [0.14, 0.28], 'distractor': [-0.24, -0.29]}
        for state in range(10):
            a = choose_target_xy([0.0, 0.0], others, 'basket', 50000 + state)
            b = choose_target_xy([0.0, 0.0], others, 'basket', 50000 + state)
            self.assertEqual(a, b)
            self.assertTrue(all(abs(value) <= 0.04 for value in a['requested_delta_xy']))
            self.assertTrue(admissible_xy(a['requested_xy'], others, 'basket'))
            starts.append(tuple(a['requested_xy']))
        self.assertEqual(len(set(starts)), 10)

    def test_workspace_and_object_clearance_rejections(self):
        self.assertFalse(admissible_xy([0.2, 0.0], {}, 'basket'))
        self.assertFalse(admissible_xy([0.0, 0.0], {'basket': [0.14, 0.0]}, 'basket'))
        self.assertFalse(admissible_xy([0.0, 0.0], {'other': [0.07, 0.0]}, 'basket'))
        with self.assertRaisesRegex(ValueError, '100 proposals'):
            choose_target_xy([0.0, 0.0], {'basket': [0.0, 0.0]}, 'basket', 50000)

    def test_invalid_jitter_is_not_silently_accepted(self):
        for jitter in (0, -0.04, float('nan'), 0.2):
            with self.assertRaises(ValueError):
                choose_target_xy([0.0, 0.0], {}, 'basket', 50000, jitter)

    def test_actual_settled_positions_must_vary(self):
        protocol = make_protocol('pilot', [0, 1], [1000, 1001])
        rows = [{'initial_state_id': state, 'success': False,
                 'position_setup': {'actual_settled_position': [0.01, 0.02, 0.03]}}
                for state in (0, 1)]
        with self.assertRaisesRegex(ValueError, 'no material settled position variation'):
            summarize(rows, protocol)
        rows[1]['position_setup']['actual_settled_position'][0] += 0.01
        result = summarize(rows, protocol)
        self.assertTrue(result['position_variation']['verified'])
        self.assertAlmostEqual(result['position_variation']['actual_settled_xy_span_m'][0], 0.01)


if __name__ == '__main__':
    unittest.main()
