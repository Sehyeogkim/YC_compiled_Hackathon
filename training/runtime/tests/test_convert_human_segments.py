"""Synthetic UI export fixtures; they do not claim collected human data."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from inference.export_segments import export_session
from training.runtime.convert_human_segments import validate_sources, segment_manifest, policy_rgb


class HumanConversionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.raw_root = self.root / 'raw'
        self.export_root = self.root / 'exports'
        self.raw_root.mkdir()
        selections = []
        for session in range(10):
            sid = f'synthetic-{session:02d}'
            raw = self.raw_root / sid
            raw.mkdir()
            meta = {'schema': 'werescue.native_session.v1', 'session_id': sid,
                    'created_at': '2026-11-09T00:00:00+00:00',
                    'suite': 'libero_object', 'task_id': 0, 'initial_state_id': session,
                    'control_hz': 20, 'control_mode': 'relative',
                    'task': 'pick up the alphabet soup and place it in the basket',
                    'image_transform': 'raw native image flipped on H and W, matching LeRobot LIBERO processor'}
            (raw / 'metadata.json').write_text(json.dumps(meta))
            events = []
            for i in range(4):
                if i == 2:
                    events.append({'kind': 'recovery_complete_annotation', 'step': i})
                for kind in ('action_requested', 'executed_action'):
                    events.append({'kind': kind, 'step': i, 'observation_step': i,
                                   'control_source': 'operator',
                                   'segment': 'recovery' if i < 2 else 'task',
                                   'utc': f'2026-11-09T00:00:0{i}+00:00',
                                   'action': [i / 10] * 7})
            events.append({'kind': 'episode_terminal', 'step': 4, 'success': True,
                           'predicate': 'native_LIBERO_check_success'})
            (raw / 'events.jsonl').write_text('\n'.join(json.dumps(e) for e in events) + '\n')
            for i in range(5):
                np.savez_compressed(raw / f'{i:06d}.state.npz',
                                    observation_state=np.full(8, i, dtype=np.float32),
                                    flattened_state=np.full(12, i, dtype=np.float32),
                                    ctrl=np.full(9, i, dtype=np.float32), sim_time=i * .05)
                for camera in ('agentview', 'wrist'):
                    Image.fromarray(np.full((256, 256, 3), i, dtype=np.uint8)).save(
                        raw / f'{i:06d}.{camera}.png')
            selected = export_session(raw, self.export_root,
                                      accept_annotations=True, operator_id='synthetic_test')
            self.assertTrue(selected['training_eligible'])
            selections.append(selected)
        manifest = {'schema_version': 'werescue.native_segment_selection.v1',
                    'sessions': selections, 'training_selection': selections,
                    'accept_operator_annotations': True}
        self.manifest = self.export_root / 'manifest.json'
        self.manifest.write_text(json.dumps(manifest))

    def tearDown(self):
        self.tmp.cleanup()

    def test_ten_operator_sessions_convert_to_twenty_auditable_segments(self):
        specs, _ = validate_sources(self.manifest, self.raw_root, self.export_root)
        self.assertEqual(len(specs), 10)
        result = segment_manifest(specs, self.manifest)
        self.assertEqual(len(result['segments']), 20)
        self.assertEqual({row['dataset'] for row in result['segments']}, {'R', 'R_prime'})
        self.assertTrue(all(row['control_source'] == 'operator' for row in result['segments']))
        self.assertTrue(all(not row['predicate_verified'] for row in result['segments']
                            if row['dataset'] == 'R_prime'))
        self.assertTrue(all(row['predicate_verified'] for row in result['segments']
                            if row['dataset'] == 'R'))

    def test_partial_qualified_collection_keeps_actual_count(self):
        manifest = json.loads(self.manifest.read_text())
        manifest['training_selection'] = manifest['training_selection'][:1]
        self.manifest.write_text(json.dumps(manifest))
        specs, _ = validate_sources(self.manifest, self.raw_root, self.export_root)
        self.assertEqual(len(specs), 1)
        self.assertEqual(len(segment_manifest(specs, self.manifest)['segments']), 2)

    def test_source_event_edit_rejected_after_export(self):
        events = self.raw_root / 'synthetic-00' / 'events.jsonl'
        events.write_text(events.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'event log changed'):
            validate_sources(self.manifest, self.raw_root, self.export_root)

    def test_policy_action_contamination_rejected(self):
        path = self.export_root / 'synthetic-00' / 'R.npz'
        with np.load(path, allow_pickle=False) as source:
            arrays = {key: source[key] for key in source.files}
        arrays['control_source'][0] = 'policy'
        np.savez_compressed(path, **arrays)
        with self.assertRaisesRegex(ValueError, 'policy/action contamination'):
            validate_sources(self.manifest, self.raw_root, self.export_root)

    def test_missing_exact_camera_frame_rejected(self):
        (self.raw_root / 'synthetic-00' / '000002.agentview.png').unlink()
        with self.assertRaisesRegex(ValueError, 'missing native image'):
            validate_sources(self.manifest, self.raw_root, self.export_root)

    def test_ui_png_is_reversed_once_for_policy_input(self):
        path = self.raw_root / 'synthetic-00' / '000000.agentview.png'
        pixels = np.zeros((256, 256, 3), dtype=np.uint8)
        pixels[0, 0] = [255, 1, 2]
        pixels[-1, -1] = [3, 4, 250]
        Image.fromarray(pixels).save(path)
        native = policy_rgb(path)
        self.assertEqual(native[0, 0].tolist(), [3, 4, 250])
        self.assertEqual(native[-1, -1].tolist(), [255, 1, 2])


if __name__ == '__main__':
    unittest.main()
