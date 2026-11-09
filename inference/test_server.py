"""API/error-boundary regression checks; not evidence of model execution."""
import os
import base64
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from inference.runtime import NativeRuntime, NotReady, SessionConflict, finite_vector
from inference.server import create_app


class FakeRuntime:
    def health(self):
        return {'ready': False, 'status': 'not_loaded', 'inference_verified': False}

    def close(self):
        pass

    def reset_session(self, *_args):
        raise NotReady('No model loaded in API test')

    def get_frame(self, session_id):
        raise SessionConflict('Session is stale')

    def step(self, *_args):
        raise ValueError('operator action must contain exactly 7 values')


class FakeGripper:
    def format_action(self, action):
        # Source inspection tests the recognized neutral-hold contract only.
        return self.current_action + np.sign(action)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'WERESCUE_AUTOLOAD': '0', 'WERESCUE_API_TOKEN': ''})
        self.env.start()
        self.client_context = TestClient(create_app(FakeRuntime))
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        self.env.stop()

    def test_unloaded_health_does_not_claim_inference(self):
        data = self.client.get('/health').json()
        self.assertFalse(data['ready'])
        self.assertFalse(data['inference_verified'])
        self.assertEqual(self.client.post('/session/reset', json={}).status_code, 503)

    def test_stale_session_and_action_shape_fail_explicitly(self):
        self.assertEqual(self.client.get('/session/frame?session_id=old').status_code, 409)
        reply = self.client.post('/session/step', json={'session_id': 'old', 'source': 'operator', 'action': [0] * 8})
        self.assertEqual(reply.status_code, 422)

    def test_bearer_token_required_when_configured(self):
        with patch.dict(os.environ, {'WERESCUE_API_TOKEN': 'test-only-secret'}):
            self.assertEqual(self.client.get('/health').status_code, 401)
            self.assertEqual(self.client.get('/health', headers={'Authorization': 'Bearer test-only-secret'}).status_code, 200)

    def test_unknown_fields_and_unbounded_batch_rejected(self):
        self.assertEqual(self.client.post('/session/reset', json={'pretend_success': True}).status_code, 422)
        self.assertEqual(self.client.post('/session/step', json={'session_id': 'x', 'source': 'policy', 'steps': 11}).status_code, 422)

    def test_finite_bounded_native_action_contract(self):
        for action in ([0] * 8, [float('nan')] * 7, [2] * 7):
            with self.assertRaises(ValueError):
                finite_vector(action, 7, 'action', bound=1)
        self.assertEqual(finite_vector([0] * 6 + [-1], 7, 'action', bound=1), [0.0] * 6 + [-1.0])

    def test_native_runtime_does_not_fall_back_when_unloaded(self):
        runtime = NativeRuntime()
        self.assertFalse(runtime.health()['ready'])
        with self.assertRaises(NotReady):
            runtime.reset_session()

    def test_frame_envelope_exposes_geometry_hash_for_reset_cache(self):
        runtime = NativeRuntime()
        runtime.status, runtime.session_id = 'ready', 'test'
        runtime.metadata = {'scene_sha256': 'same-native-geometry'}
        runtime.frame = lambda: {'step': 0}
        result = runtime.get_frame('test')
        self.assertEqual(result['scene_sha256'], 'same-native-geometry')
        self.assertEqual(result['frame'], {'step': 0})

    def test_scene_bundles_only_model_references_and_reports_actual_scales(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            mesh = Path(directory) / 'shape.stl'
            mesh.write_bytes(b'test-model-mesh')
            xml = f'<mujoco><compiler meshdir="{directory}"/><asset><mesh name="part" file="shape.stl"/></asset></mujoco>'
            model = SimpleNamespace(get_xml=lambda: xml, nq=3, nv=3,
                body_names=['world', 'part'], geom_names=['part'], site_names=['eef'], camera_names=['agentview'])
            controller = SimpleNamespace(input_min=-1.0, input_max=1.0,
                output_min=np.array([-.02, -.03, -.04, -.2, -.3, -.4]),
                output_max=np.array([.02, .03, .04, .2, .3, .4]))
            runtime = NativeRuntime()
            runtime.status, runtime.session_id, runtime.np = 'ready', 'test', np
            runtime.problem = SimpleNamespace(sim=SimpleNamespace(model=model), robots=[SimpleNamespace(controller=controller,gripper=FakeGripper())])
            runtime.metadata = {'scene_sha256': 'test-scene'}
            result = runtime.scene('test')
            self.assertEqual(len(result['assets']), 1)
            self.assertEqual(base64.b64decode(result['assets'][0]['data']), b'test-model-mesh')
            self.assertNotIn(directory, result['xml'])
            self.assertEqual(result['control']['translation_scale_m'], [.02, .03, .04])
            self.assertEqual(result['control']['rotation_scale_rad'], [.2, .3, .4])


if __name__ == '__main__':
    unittest.main()
