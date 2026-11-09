#!/usr/bin/env python3
"""Private native LIBERO human collection desk; launch on GPU host, use SSH forwarding.

python training/collector/server.py --cases cases.json --out recordings
All MuJoCo/EGL work stays on the HTTP server's main thread.
"""
from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
from pathlib import Path
import time
from urllib.parse import urlparse, parse_qs
import uuid


def load_cases(path: Path) -> dict:
    data = json.loads(path.read_text())
    if data.get('schema_version') != 'werescue.collection_cases.v1':
        raise ValueError('Expected schema_version werescue.collection_cases.v1')
    cases = data.get('cases', [])
    if not cases:
        raise ValueError('At least one saved native failure case required')
    result = {}
    for case in cases:
        for field in ('case_id', 'initial_state_id', 'parent_episode_id', 'failure_state_ref', 'goal_ref'):
            if not isinstance(case.get(field), str) or not case[field]:
                raise ValueError(f'case requires {field}')
        if case.get('split') not in ('train', 'val', 'test'):
            raise ValueError('Each case requires explicit train/val/test split')
        if case['case_id'] in result:
            raise ValueError('Duplicate case_id')
        case = dict(case)
        for field in ('failure_state_ref', 'goal_ref'):
            case[field] = str((path.parent / case[field]).resolve())
            if not Path(case[field]).is_file():
                raise ValueError(f'Missing {field}: {case[field]}')
        result[case['case_id']] = case
    # Repeated attempts from a failure lineage must remain together.
    seen = {}
    for case in result.values():
        for field in ('initial_state_id', 'parent_episode_id'):
            key = (field, case[field])
            if key in seen and seen[key] != case['split']:
                raise ValueError(f'{field} leaks across splits')
            seen[key] = case['split']
    return {**data, 'cases': result}


class CollectionDesk:
    def __init__(self, adapter, config, out: Path, operator: str, max_steps: int):
        self.adapter, self.config, self.out = adapter, config, out.resolve()
        self.operator, self.max_steps = operator, max_steps
        self.active = False
        self.phase = 'idle'
        self.action = [0., 0., 0., 0., 0., 0., -1.]
        self.deadline = 0.
        self.steps = 0
        self.phase_boundary = None
        self.message = 'Choose a saved failure and begin human recovery.'
        self.frames = []
        self.last_manifest = None
        self.last_predicate = None

    def status(self):
        return {'active': self.active, 'phase': self.phase, 'steps': self.steps,
                'sim_seconds': self.steps / 20, 'message': self.message,
                'case_id': getattr(self, 'case', {}).get('case_id'),
                'language': self.adapter.metadata['language'], 'scene': self.adapter.scene(),
                'last_predicate': self.last_predicate, 'last_manifest': self.last_manifest,
                'cases': [{'case_id': c['case_id'], 'split': c['split']} for c in self.config['cases'].values()],
                'controller': 'human_operator', 'policy_connected': False}

    def start(self, case_id):
        if self.active:
            raise ValueError('Finish or abort the active episode; resets during recording are forbidden')
        if case_id not in self.config['cases']:
            raise ValueError('Unknown case_id')
        self.case = self.config['cases'][case_id]
        self.goal = json.loads(Path(self.case['goal_ref']).read_text())
        self.adapter.restore(Path(self.case['failure_state_ref']))  # before recording starts
        self.adapter.recovery_predicate(self.goal)  # validate goal fields before creating episode
        self.episode_id = uuid.uuid4().hex
        self.directory = self.out / self.episode_id
        self.directory.mkdir(parents=True)
        for camera in ('agentview', 'wrist'):
            (self.directory / camera).mkdir()
        self.frames, self.steps, self.phase_boundary = [], 0, None
        self.last_predicate, self.last_manifest = None, None
        self.phase = 'recovery'
        self.started = time.monotonic()
        self.started_utc = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        self.active = True
        self.action = [0., 0., 0., 0., 0., 0., -1.]
        self.deadline = 0
        self.message = 'Recording R′: move from failed state to the verified resumable goal.'
        self._persist_metadata('recording')

    def set_action(self, values):
        import math
        if not self.active:
            raise ValueError('Start an episode first')
        if not isinstance(values, list) or len(values) != 7:
            raise ValueError('Expected action list with 7 native normalized values')
        if any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1 for v in values):
            raise ValueError('Action values must be finite numbers in [-1, 1]')
        # Human UI bounds translation/rotation; preserve gripper convention.
        self.action = [max(-.4, min(.4, float(v))) for v in values[:6]] + [float(values[6])]
        self.deadline = time.monotonic() + .25

    def recovery_complete(self):
        if not self.active or self.phase != 'recovery' or self.steps < 1:
            raise ValueError('An active nonempty recovery phase is required')
        self.last_predicate = self.adapter.recovery_predicate(self.goal)
        if not self.last_predicate['passed']:
            raise ValueError('Resumable-state predicate has not passed; inspect the reported checks')
        self.phase_boundary = self.steps
        (self.directory / 'recovery_evidence.json').write_text(json.dumps(self.last_predicate, indent=2))
        self.phase = 'task'
        self.action[:6] = [0.] * 6
        self.deadline = 0
        self.message = 'Recording R: YOU still control the robot. Complete the native task.'
        self._persist_metadata('recording')

    def tick(self):
        if not self.active:
            return
        import numpy as np
        from PIL import Image
        action = self.action.copy()
        if time.monotonic() > self.deadline:
            action[:6] = [0.] * 6
        before = self.adapter.obs
        state_before = self.adapter.state8(before)
        sim_before = self.adapter.sim_state()
        mono_before = time.monotonic() - self.started
        sim_time_before = self.adapter.scene()['sim_time']
        for camera, raw_name in (('agentview', 'agentview_image'), ('wrist', 'robot0_eye_in_hand_image')):
            Image.fromarray(np.asarray(before[raw_name], dtype=np.uint8)).save(
                self.directory / camera / f'{self.steps:06d}.png', compress_level=1)
        dispatch_timestamp = time.monotonic() - self.started
        result = self.adapter.step(action)
        next_timestamp = time.monotonic() - self.started
        frame = {'observation_state': state_before, 'action': np.asarray(action, dtype=np.float32),
                 'next_observation_state': self.adapter.state8(), 'sim_state_before': sim_before,
                 'sim_state_after': self.adapter.sim_state(), 'actuator_ctrl_after': result['ctrl'],
                 'frame_index': self.steps, 'timestamp': mono_before, 'sim_timestamp': sim_time_before,
                 'action_dispatch_timestamp': dispatch_timestamp, 'next_timestamp': next_timestamp,
                 'next_sim_timestamp': self.adapter.scene()['sim_time'],
                 'phase': 0 if self.phase == 'recovery' else 1, 'task_success': result['task_success']}
        self.frames.append(frame)
        self.steps += 1
        # Each tick is durable; no in-memory-only demonstrations if the process exits.
        np.savez_compressed(self.directory / f'frame_{self.steps-1:06d}.npz', **frame)
        if result['task_success']:
            if self.phase == 'task' and self.steps > self.phase_boundary:
                self.finish(True, 'native_task_success')
            else:
                self.finish(False, 'task_completed_before_recovery_handoff')
        elif result['done'] or self.steps >= self.max_steps:
            self.finish(False, 'native_done_or_collection_limit')

    def finish(self, success=False, reason='operator_abort'):
        if not self.active:
            raise ValueError('No active episode')
        self.active = False
        self.action[:6] = [0.] * 6
        self.deadline = 0
        self.phase = 'completed' if success else 'aborted'
        self.message = f'{self.phase}: {reason}; raw episode retained.'
        import numpy as np
        if self.frames:
            keys = self.frames[0].keys()
            arrays = {key: np.asarray([row[key] for row in self.frames]) for key in keys}
            np.savez_compressed(self.directory / 'trajectory.npz', **arrays)
            if self.phase_boundary is not None:
                for name, lo, hi in (('R_prime', 0, self.phase_boundary), ('R', self.phase_boundary, self.steps)):
                    np.savez_compressed(self.directory / f'{name}.npz', **{key: value[lo:hi] for key, value in arrays.items()})
        evidence = {'predicate': 'native_libero_check_success', 'passed': success, 'reason': reason,
                    'scene': self.adapter.scene(), 'steps': self.steps}
        (self.directory / 'task_evidence.json').write_text(json.dumps(evidence, indent=2))
        self._persist_metadata('qualified' if success and self.phase_boundary is not None else 'unqualified')
        self.last_manifest = str(self.directory / 'episode.json')
        # Native next/terminal images stored separately; no invented terminal action.
        from PIL import Image
        for camera, raw_name in (('agentview', 'agentview_image'), ('wrist', 'robot0_eye_in_hand_image')):
            Image.fromarray(np.asarray(self.adapter.obs[raw_name], dtype=np.uint8)).save(self.directory / f'{camera}_terminal.png')
        self._video_optional()

    def _persist_metadata(self, qualification):
        payload = {'schema_version': 'werescue.native_raw_episode.v1', 'episode_id': self.episode_id,
                   'case': self.case, 'operator_id': self.operator, 'started_utc': self.started_utc,
                   'metadata': self.adapter.metadata, 'goal': self.goal, 'steps': self.steps,
                   'phase_boundary_frame': self.phase_boundary, 'qualification': qualification,
                   'control_source': 'operator', 'model_inference': False, 'contains_reset': False,
                   'failure_restore_before_recording': True,
                   'failure_state_sha256': hashlib.sha256(Path(self.case['failure_state_ref']).read_bytes()).hexdigest(),
                   'action_provenance': 'normalized_native_commands_accepted_by_env_step; actuator_ctrl_also_recorded',
                   'recording_rate': '20Hz simulation; wall timestamps stored separately; no forced wall-time catchup',
                   'next_images': 'next tick PNG, except terminal image saved separately',
                   'full_state_scope': 'native_get_sim_state before/after; controller internal cache not serialized',
                   'training_ready': False}
        (self.directory / 'episode.json').write_text(json.dumps(payload, indent=2))

    def _video_optional(self):
        try:
            import imageio.v2 as imageio
            import numpy as np
            for name, start, end in [('episode', 0, self.steps), ('R_prime', 0, self.phase_boundary or 0),
                                     ('R', self.phase_boundary or self.steps, self.steps)]:
                if end <= start:
                    continue
                with imageio.get_writer(str(self.directory / f'{name}.mp4'), fps=20, codec='libx264') as writer:
                    for i in range(start, end):
                        pair = np.concatenate([imageio.imread(self.directory / camera / f'{i:06d}.png')[::-1, ::-1]
                                               for camera in ('agentview', 'wrist')], axis=1)
                        writer.append_data(pair)
        except Exception as error:
            (self.directory / 'video_note.txt').write_text(f'Optional video encoding unavailable; lossless PNG source retained.\n{error}\n')


def serve(desk, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, data, code=200, content_type='application/json'):
            payload = json.dumps(data).encode() if content_type == 'application/json' else data
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == '/':
                return self.respond(Path(__file__).with_name('operator.html').read_bytes(), content_type='text/html; charset=utf-8')
            if parsed.path == '/api/state':
                return self.respond(desk.status())
            if parsed.path == '/api/rgb':
                from PIL import Image
                import numpy as np
                name = parse_qs(parsed.query).get('camera', ['agentview'])[0]
                raw_name = {'agentview': 'agentview_image', 'wrist': 'robot0_eye_in_hand_image'}.get(name)
                if raw_name is None:
                    return self.respond({'error': 'unknown camera'}, 400)
                stream = io.BytesIO()
                Image.fromarray(np.asarray(desk.adapter.obs[raw_name])[::-1, ::-1]).save(stream, format='JPEG', quality=85)
                return self.respond(stream.getvalue(), content_type='image/jpeg')
            return self.respond({'error': 'not found'}, 404)

        def do_POST(self):
            # Loopback binding alone is not sufficient against malicious websites.
            origin = self.headers.get('Origin')
            host = self.headers.get('Host', '')
            if origin and origin not in (f'http://{host}', f'https://{host}'):
                return self.respond({'error': 'cross-origin requests forbidden'}, 403)
            if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                return self.respond({'error': 'JSON required'}, 415)
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if length < 0 or length > 8192:
                    raise ValueError('request too large')
                body = json.loads(self.rfile.read(length) or b'{}')
                if self.path == '/api/start':
                    desk.start(body.get('case_id'))
                elif self.path == '/api/action':
                    desk.set_action(body.get('action'))
                elif self.path == '/api/recovery-complete':
                    desk.recovery_complete()
                elif self.path == '/api/abort':
                    desk.finish(False, 'operator_abort')
                else:
                    return self.respond({'error': 'not found'}, 404)
                return self.respond(desk.status())
            except (ValueError, KeyError, TypeError) as error:
                return self.respond({'error': str(error), 'state': desk.status()}, 400)

    server = HTTPServer(('127.0.0.1', port), Handler)
    server.timeout = .005
    next_tick = time.monotonic()
    print(f'Private collector: http://127.0.0.1:{port} (SSH forward this port)', flush=True)
    try:
        while True:
            server.handle_request()
            now = time.monotonic()
            if now >= next_tick:
                desk.tick()
                next_tick = max(next_tick + .05, time.monotonic())
    except KeyboardInterrupt:
        if desk.active:
            desk.finish(False, 'server_shutdown')
    finally:
        server.server_close()
        desk.adapter.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--operator', default='jeff')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--max-steps', type=int, default=1200)
    args = parser.parse_args()
    config = load_cases(args.cases.resolve())
    from native import NativeLibero
    adapter = NativeLibero(config.get('suite', 'libero_object'), int(config.get('task_id', 0)))
    desk = CollectionDesk(adapter, config, args.out, args.operator, args.max_steps)
    serve(desk, args.port)


if __name__ == '__main__':
    main()
