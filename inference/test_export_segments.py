"""Synthetic alignment/qualification checks, not real collection evidence."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from inference.export_segments import export_session


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.raw = Path(self.temp.name) / 'raw'
        self.out = Path(self.temp.name) / 'out'
        self.raw.mkdir()
        (self.raw / 'metadata.json').write_text(json.dumps({'session_id': 'synthetic',
            'created_at': '2026-11-09T00:00:00+00:00', 'initial_state_id': 0}))
        self.events = []
        for i in range(4):
            if i == 2:
                self.events.append(self.event('recovery_complete_annotation', i,
                    independently_verified=False))
            for kind in ('action_requested', 'executed_action'):
                self.events.append(self.event(kind, i, observation_step=i, action=[i / 10] * 7))
        self.events.append(self.event('episode_terminal', 4, success=True,
                                      predicate='native_LIBERO_check_success'))
        for i in range(5):
            np.savez(self.raw / f'{i:06d}.state.npz', observation_state=np.full(8, i),
                flattened_state=np.full(12, i), ctrl=np.full(9, i), sim_time=i * .05)
            for camera in ('agentview', 'wrist'):
                (self.raw / f'{i:06d}.{camera}.png').write_bytes(b'synthetic-reference-only')
        self.save_events()

    def tearDown(self):
        self.temp.cleanup()

    def event(self, kind, step, **fields):
        return {'kind': kind, 'step': step, 'control_source': 'operator',
            'segment': 'recovery' if step < 2 else 'task',
            'utc': f'2026-11-09T00:00:0{step}+00:00', **fields}

    def save_events(self):
        (self.raw / 'events.jsonl').write_text('\n'.join(json.dumps(e) for e in self.events))

    def test_annotation_requires_opt_in_and_boundary_aligns_exactly(self):
        strict = export_session(self.raw, self.out, operator_id='test')
        self.assertFalse(strict['training_eligible'])
        accepted = export_session(self.raw, self.out, operator_id='test', accept_annotations=True)
        self.assertEqual(accepted['qualification'], 'operator_annotated')
        self.assertFalse(accepted['recovery_predicate_verified'])
        with np.load(accepted['R_prime'], allow_pickle=False) as a, np.load(accepted['R'], allow_pickle=False) as b:
            np.testing.assert_array_equal(a['next_observation_state'][-1], b['observation_state'][0])
            np.testing.assert_array_equal(a['next_png_refs'][-1], b['png_refs'][0])
            self.assertEqual(a['action'].shape, (2, 7))
            self.assertEqual(b['action'].shape, (2, 7))

    def test_success_is_required_even_with_annotation_opt_in(self):
        self.events[-1]['success'] = False
        self.save_events()
        result = export_session(self.raw, self.out, operator_id='test', accept_annotations=True)
        self.assertFalse(result['training_eligible'])

    def test_policy_contamination_is_rejected(self):
        self.events[3]['control_source'] = 'policy'
        self.save_events()
        result = export_session(self.raw, self.out, operator_id='test', accept_annotations=True)
        self.assertFalse(result['training_eligible'])
        self.assertIn('Policy actions inside human intervention', result['reasons'])

    def test_missing_next_observation_is_never_synthesized(self):
        (self.raw / '000004.state.npz').unlink()
        result = export_session(self.raw, self.out, operator_id='test', accept_annotations=True)
        self.assertFalse(result['training_eligible'])
        self.assertNotIn('R', result)


if __name__ == '__main__':
    unittest.main()
