from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from base_adapter import (
    BASE_MODEL_ID, BASE_REVISION, _check_snapshot_pin,
    adapted_config, adapter_manifest,
)


def configs():
    common = dict(vlm_model_name='same', num_vlm_layers=16, num_expert_layers=0,
                  expert_width_multiplier=.75, attention_mode='cross_attn',
                  self_attn_every_n_layers=2, max_state_dim=32, max_action_dim=32,
                  chunk_size=50, n_action_steps=50, device='cpu',
                  normalization_mapping={'STATE': 'MEAN_STD', 'ACTION': 'MEAN_STD'})
    base = SimpleNamespace(**deepcopy(common), input_features={
        'observation.state': {'shape': [6]},
        **{f'observation.images.camera{i}': {'shape': [3, 256, 256]} for i in range(1, 4)},
    }, output_features={'action': {'shape': [6]}})
    native = SimpleNamespace(**deepcopy(common), input_features={
        'observation.state': {'shape': [8]},
        **{f'observation.images.camera{i}': {'shape': [3, 256, 256]} for i in range(1, 3)},
    }, output_features={'action': {'shape': [7]}})
    return base, native


class AdapterTests(unittest.TestCase):
    def test_contract_copy_does_not_mutate_either_source(self):
        base, native = configs()
        original_base, original_native = deepcopy(base), deepcopy(native)
        result = adapted_config(base, native)
        self.assertEqual(base, original_base)
        self.assertEqual(native, original_native)
        self.assertEqual(result.input_features, native.input_features)
        self.assertEqual(result.output_features, native.output_features)
        self.assertEqual(result.n_action_steps, 10)
        result.input_features['observation.state']['shape'][0] = 99
        self.assertEqual(native.input_features['observation.state']['shape'], [8])

    def test_incompatible_architecture_rejected_before_weight_load(self):
        base, native = configs()
        native.max_action_dim = 64
        with self.assertRaisesRegex(ValueError, 'Architecture differs'):
            adapted_config(base, native)

    def test_no_implicit_action_truncation(self):
        base, native = configs()
        base.output_features['action']['shape'] = [7]
        with self.assertRaisesRegex(ValueError, 'base action6'):
            adapted_config(base, native)

    def test_changed_camera_contract_rejected(self):
        base, native = configs()
        native.input_features['observation.images.camera3'] = {'shape': [3, 256, 256]}
        with self.assertRaisesRegex(ValueError, 'exactly'):
            adapted_config(base, native)

    def test_manifest_discloses_adaptation_and_no_training(self):
        record = adapter_manifest()
        self.assertFalse(record['raw_checkpoint_zero_shot'])
        self.assertEqual(record['additional_gradient_steps'], 0)
        self.assertEqual(record['weights']['revision'], BASE_REVISION)
        self.assertIn('unvalidated', record['limitation'])

    def test_unpinned_or_wrong_snapshot_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            with self.assertRaisesRegex(ValueError, 'requires WERESCUE_PIN'):
                _check_snapshot_pin(path, BASE_MODEL_ID, BASE_REVISION)
            (path/'WERESCUE_PIN.json').write_text(json.dumps({'repo_id': BASE_MODEL_ID, 'revision': 'wrong'}))
            with self.assertRaisesRegex(ValueError, 'pin mismatch'):
                _check_snapshot_pin(path, BASE_MODEL_ID, BASE_REVISION)


if __name__ == '__main__':
    unittest.main()
