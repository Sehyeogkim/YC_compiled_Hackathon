"""Dependency-free selection, provenance and resumability rules for baseline runs."""
from __future__ import annotations

import hashlib
import json
import math

MODEL_ID = 'lerobot/smolvla_libero'
MODEL_REVISION = 'a24f1f380461b665158d52acd140152c79e8b36d'
BASE_MODEL_ID = 'lerobot/smolvla_base'
BASE_MODEL_REVISION = '5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6'
POLICY_SEEDS = [1000, 1001]
STATE_SPLITS = {'pilot': list(range(10)), 'final': list(range(25, 50))}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def parse_selection(raw: str) -> list[int]:
    result = []
    for part in raw.split(','):
        part = part.strip()
        if not part:
            raise ValueError('empty selection element')
        low, dash, high = part.partition('-')
        start, end = int(low), int(high) if dash else int(low)
        if start < 0 or end < start:
            raise ValueError('selection must contain nonnegative ascending ranges')
        result.extend(range(start, end + 1))
    if len(set(result)) != len(result):
        raise ValueError('duplicate selection values')
    return sorted(result)


def make_protocol(phase: str, initial_states: list[int], seeds: list[int],
                  suite='libero_object', task_id=0, horizon=280, arm='libero') -> dict:
    if phase not in STATE_SPLITS:
        raise ValueError('phase must be pilot or final')
    if not initial_states or len(initial_states) != len(set(initial_states)):
        raise ValueError('initial states must be nonempty and unique')
    if not set(initial_states).issubset(STATE_SPLITS[phase]):
        raise ValueError(f'{phase} initial states must stay in reserved IDs {STATE_SPLITS[phase]}')
    if sorted(seeds) != POLICY_SEEDS:
        raise ValueError('this protocol requires policy RNG seeds 1000 and 1001')
    if horizon != 280:
        raise ValueError('this frozen baseline protocol requires horizon=280')
    if not isinstance(task_id, int) or task_id < 0:
        raise ValueError('task-id must be a nonnegative integer')
    if arm not in ('libero', 'base-adapted'):
        raise ValueError('arm must be libero or base-adapted')
    body = {
        'schema_version': 'werescue.baseline_protocol.v1', 'phase': phase, 'arm': arm,
        'suite': suite, 'task_id': task_id, 'initial_states': sorted(initial_states),
        'policy_rng_seeds': POLICY_SEEDS, 'env_seed_rule': '10000 + native_initial_state_id',
        'horizon': horizon, 'init_states': True, 'hard_reset': True,
        'control_mode': 'relative', 'control_hz': 20, 'chunk_size': 50, 'n_action_steps': 10,
        'applied_action_rule': 'clip_postprocessed_native_action_to_-1_1; retain_unclipped_request_in_trace',
        'camera_hw': [256, 256], 'success_predicate': 'native_LIBERO_check_success',
        'control_source': 'policy', 'human_interventions_allowed': False,
        'resets_within_rollout_allowed': False,
        'policy': {'repo_id': MODEL_ID, 'revision': MODEL_REVISION} if arm == 'libero' else {
            'repo_id': BASE_MODEL_ID, 'revision': BASE_MODEL_REVISION,
            'adapter': 'LIBERO_state8_action7_two_cameras_with_LIBERO_checkpoint_processors',
            'calibration_repo_id': MODEL_ID, 'calibration_revision': MODEL_REVISION,
            'gradient_updates': 0, 'label': 'adapted-base diagnostic; not raw zero-shot',
        },
        'heldout_scope': 'final states held out from intervention tuning; not claimed unseen in P pretraining',
        'raw_smolvla_base': {
            'status': 'incompatible_NA',
            'reason': 'official raw checkpoint exposes state6/action6/three cameras and SO100 statistics; native LIBERO requires state8/action7/two cameras',
            'not_a_measured_zero_success_rate': True,
        },
    }
    body['protocol_sha256'] = digest(body)
    return body


def trials(protocol):
    return [{'trial_id': f'init-{state:02d}_rng-{seed}', 'initial_state_id': state,
             'policy_rng_seed': seed, 'env_seed': 10000 + state}
            for state in protocol['initial_states'] for seed in protocol['policy_rng_seeds']]


def validate_done(data: dict, protocol: dict, trial: dict):
    if data.get('schema_version') != 'werescue.baseline_trial.v1':
        raise ValueError('invalid completed-trial schema')
    if data.get('protocol_sha256') != protocol['protocol_sha256']:
        raise ValueError('completed trial belongs to a different protocol')
    for key, value in trial.items():
        if data.get(key) != value:
            raise ValueError(f'completed trial {key} differs')
    if data.get('policy') != protocol['policy']:
        raise ValueError('completed trial checkpoint differs')
    if type(data.get('success')) is not bool:
        raise ValueError('completed trial requires a measured boolean success')
    if data.get('control_source') != 'policy' or data.get('contains_human_intervention') is not False:
        raise ValueError('completed trial is not autonomous policy execution')
    if data.get('contains_reset_during_rollout') is not False:
        raise ValueError('reset occurred inside completed trial')
    if data.get('success_predicate_verified') is not True:
        raise ValueError('native success verification missing')
    if data.get('success_predicate') != protocol['success_predicate']:
        raise ValueError('native success predicate changed')
    if not isinstance(data.get('executed_steps'), int) or not 1 <= data['executed_steps'] <= protocol['horizon']:
        raise ValueError('invalid executed-step count')
    for key in ('trace_sha256', 'initial_state_sha256', 'initial_xml_sha256',
                'initial_native_state_sha256', 'runtime_sha256'):
        if not isinstance(data.get(key), str) or len(data[key]) != 64:
            raise ValueError(f'missing evidence digest {key}')


def summarize(rows: list[dict], protocol: dict) -> dict:
    successes, count = sum(row['success'] for row in rows), len(rows)
    interval = None
    if count:
        z, p = 1.959963984540054, successes / count
        d = 1 + z*z/count
        center = (p + z*z/(2*count))/d
        half = z*math.sqrt(p*(1-p)/count + z*z/(4*count*count))/d
        interval = [max(0, center-half), min(1, center+half)]
    return {'schema_version': 'werescue.baseline_summary.v1',
            'protocol_sha256': protocol['protocol_sha256'], 'phase': protocol['phase'], 'arm': protocol['arm'],
            'policy': protocol['policy'], 'expected_trials': len(trials(protocol)),
            'completed_trials': count, 'successes': successes,
            'success_rate': successes/count if count else None, 'wilson_95': interval,
            'raw_smolvla_base': protocol['raw_smolvla_base'],
            'selection_warning': 'Use pilot only for task/condition selection; final cannot be used to tune the choice.',
            'uncertainty_note': 'Two policy RNG repetitions share each layout; pooled Wilson is descriptive and does not account for within-layout clustering.',
            'trials': rows}
