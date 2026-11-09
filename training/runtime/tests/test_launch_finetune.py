"""Check dry-run branches against experiment settings; no ML deps or GPU needed."""
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

runtime = Path(__file__).resolve().parents[1]
experiment = json.loads((runtime.parent / 'experiment.json').read_text())


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pretrained = self.root / 'P'
        self.pretrained.mkdir()
        (self.pretrained / 'WERESCUE_PIN.json').write_text(json.dumps({
            'repo_id': experiment['model']['repo_id'], 'revision': experiment['model']['revision']}))
        (self.pretrained / 'config.json').write_text(json.dumps({
            'n_action_steps': 50, 'num_steps': 10, 'chunk_size': 50,
            'input_features': {'observation.state': {'shape': [8]},
                               'observation.images.camera1': {'shape': [3, 256, 256]},
                               'observation.images.camera2': {'shape': [3, 256, 256]}},
            'output_features': {'action': {'shape': [7]}},
        }))
        self.dataset = self.root / 'data'
        (self.dataset / 'meta').mkdir(parents=True)
        self.info = {'fps': 20, 'total_episodes': 10, 'features': {
            'observation.state': {'shape': [8]}, 'action': {'shape': [7]},
            'observation.images.image': {'shape': [256, 256, 3]},
            'observation.images.image2': {'shape': [256, 256, 3]},
        }}

    def launch(self, arm='task'):
        (self.dataset / 'meta/info.json').write_text(json.dumps(self.info))
        name = 'R' if arm == 'task' else 'R_prime'
        count = self.info['total_episodes'] if type(self.info['total_episodes']) is int and 1 <= self.info['total_episodes'] <= 10 else 10
        segments = [{'segment_id': f'e{i}/{name}', 'dataset': name, 'split': 'train',
                     'control_source': 'operator', 'fps': 20, 'environment_id': 'libero_object',
                     'task_id': '0', 'episode_id': f'e{i}',
                     'source_failure_policy': {'repo_id': experiment['model']['repo_id'],
                                               'revision': experiment['model']['revision']},
                     'image_transform': 'reverse_UI_preview_flip_once',
                     'dataset_image_orientation': 'native policy RGB after one H and W reversal of source PNG'}
                    for i in range(count)]
        manifest = {'schema_version': 'werescue.segments.v1', 'segments': segments,
                    'validation_scope': 'metadata_and_local_file_hashes_only; tensor_alignment_and_native_replay_not_verified'}
        canonical = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
        manifest['manifest_sha256'] = hashlib.sha256(canonical).hexdigest()
        manifest_bytes = json.dumps(manifest).encode()
        (self.dataset / 'WERESCUE_SEGMENTS.json').write_bytes(manifest_bytes)
        (self.dataset / 'WERESCUE_HUMAN_DATASET.json').write_text(json.dumps({
            'schema_version': 'werescue.human_lerobot_dataset.v1',
            'kind': 'qualified_operator_segments', 'arm': arm, 'dataset': name,
            'control_source': 'operator', 'practice_data_excluded': True,
            'source_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
            'segment_ids': [row['segment_id'] for row in segments],
        }))
        return subprocess.run([sys.executable, str(runtime / 'launch_finetune.py'),
                               '--arm', arm, '--P', str(self.pretrained),
                               '--dataset-root', str(self.dataset), '--dataset-id', 'test/dataset',
                               '--output', str(self.root / arm)], capture_output=True, text=True)

    def test_both_arms_start_from_same_P_and_match_proposal(self):
        for arm, label in [('task', 'P+R'), ('recovery', 'P+R_prime')]:
            result = self.launch(arm)
            self.assertEqual(result.returncode, 0, result.stderr)
            output, _ = json.JSONDecoder().raw_decode(result.stdout)
            self.assertEqual(output['arm'], label)
            self.assertFalse(output['execute'])
            command = output['command']
            self.assertIn(f'--policy.path={self.pretrained.resolve()}', command)
            self.assertIn(f"--seed={experiment['training_proposal']['seed']}", command)
            self.assertIn(f"--policy.scheduler_warmup_steps={experiment['training_proposal']['warmup_steps']}", command)
            self.assertIn('--policy.n_action_steps=10', command)
            self.assertIn('--rename_map={"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}', command)
            self.assertEqual(output['human_source_segments'], 10)
            self.assertEqual(len(output['human_source_intervention_ids']), 10)
            self.assertFalse(output['cross_policy_failure_source'])
            self.assertFalse(any('eval_freq' in flag for flag in command))
            self.assertFalse((self.root / arm).exists())

    def test_wrong_control_rate_rejected(self):
        self.info['fps'] = 30
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('fps must be exactly 20', result.stderr)

    def test_partial_qualified_subset_accepted(self):
        self.info['total_episodes'] = 5
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        output, _ = json.JSONDecoder().raw_decode(result.stdout)
        self.assertEqual(output['human_source_segments'], 5)
        self.assertEqual(output['human_source_intervention_ids'], [f'e{i}' for i in range(5)])

    def test_wrong_or_noninteger_episode_count_rejected(self):
        for value in (0, -1, True, 1.5, 11):
            self.info['total_episodes'] = value
            result = self.launch()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('1..10 verified human', result.stderr)

    def test_practice_dataset_rejected_even_with_human_sidecar(self):
        (self.dataset / 'WERESCUE_PRACTICE.json').write_text('{}')
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('practice self-imitation data cannot enter', result.stderr)

    def test_evaluation_render_size_is_not_human_dataset_size(self):
        self.info['features']['observation.images.image']['shape'] = [360, 360, 3]
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('must match P RGB feature [256, 256, 3]', result.stderr)

    def test_source_failure_policy_mismatch_is_disclosed(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest_path = self.dataset / 'WERESCUE_SEGMENTS.json'
        manifest = json.loads(manifest_path.read_text())
        for row in manifest['segments']:
            row['source_failure_policy'] = {'repo_id': 'HuggingFaceVLA/smolvla_libero', 'revision': 'other'}
        body = {k: v for k, v in manifest.items() if k != 'manifest_sha256'}
        manifest['manifest_sha256'] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        manifest_bytes = json.dumps(manifest).encode()
        manifest_path.write_bytes(manifest_bytes)
        data_path = self.dataset / 'WERESCUE_HUMAN_DATASET.json'
        data = json.loads(data_path.read_text())
        data['source_manifest_sha256'] = hashlib.sha256(manifest_bytes).hexdigest()
        data_path.write_text(json.dumps(data))
        command = [sys.executable, str(runtime / 'launch_finetune.py'), '--arm', 'task',
                   '--P', str(self.pretrained), '--dataset-root', str(self.dataset),
                   '--dataset-id', 'test/dataset', '--output', str(self.root / 'task')]
        rerun = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(rerun.returncode, 0, rerun.stderr)
        output, _ = json.JSONDecoder().raw_decode(rerun.stdout)
        self.assertTrue(output['cross_policy_failure_source'])


if __name__ == '__main__':
    unittest.main()
