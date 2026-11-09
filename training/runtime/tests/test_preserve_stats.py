"""Contract checks for the tiny-data normalization wrapper; no ML deps needed."""
import copy
import importlib.util
from pathlib import Path
import unittest

module_path = Path(__file__).resolve().parents[1] / 'train_preserve_stats.py'
spec = importlib.util.spec_from_file_location('train_preserve_stats', module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PreserveStatsTests(unittest.TestCase):
    def test_removes_new_dataset_stats_without_mutating_input(self):
        kwargs = {
            'policy_cfg': 'config-sentinel', 'pretrained_path': '/models/P',
            'dataset_stats': {'action': {'mean': [99.0]}},
            'preprocessor_overrides': {
                'device_processor': {'device': 'cuda'},
                'rename_observations_processor': {'rename_map': {'old': 'camera1'}},
                'normalizer_processor': {'stats': {'sentinel': 99}, 'features': {'state': [8]},
                                         'norm_map': {'STATE': 'MEAN_STD'}},
            },
            'postprocessor_overrides': {
                'unnormalizer_processor': {'stats': {'sentinel': 99}, 'features': {'action': [7]}},
            },
        }
        original = copy.deepcopy(kwargs)
        result = module.keep_pretrained_stats(kwargs)
        self.assertEqual(kwargs, original)
        self.assertNotIn('dataset_stats', result)
        self.assertNotIn('stats', result['preprocessor_overrides']['normalizer_processor'])
        self.assertNotIn('stats', result['postprocessor_overrides']['unnormalizer_processor'])
        self.assertEqual(result['pretrained_path'], '/models/P')
        self.assertEqual(result['policy_cfg'], 'config-sentinel')
        self.assertEqual(result['preprocessor_overrides']['device_processor'], {'device': 'cuda'})
        self.assertEqual(result['preprocessor_overrides']['rename_observations_processor'],
                         {'rename_map': {'old': 'camera1'}})
        self.assertEqual(result['preprocessor_overrides']['normalizer_processor'],
                         {'features': {'state': [8]}, 'norm_map': {'STATE': 'MEAN_STD'}})
        self.assertEqual(result['postprocessor_overrides']['unnormalizer_processor'],
                         {'features': {'action': [7]}})

    def test_requires_pretrained_path(self):
        for kwargs in ({}, {'pretrained_path': None}, {'pretrained_path': ''}):
            with self.assertRaisesRegex(RuntimeError, 'requires a pretrained P'):
                module.keep_pretrained_stats(kwargs)

    def test_none_overrides_stay_none(self):
        result = module.keep_pretrained_stats({'pretrained_path': '/models/P',
                                               'preprocessor_overrides': None})
        self.assertIsNone(result['preprocessor_overrides'])


if __name__ == '__main__':
    unittest.main()
