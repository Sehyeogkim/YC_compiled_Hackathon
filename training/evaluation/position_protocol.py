"""Explicit target-position jitter condition; not the unmodified LIBERO benchmark."""
from __future__ import annotations

import math
import random

from protocol import (BASE_MODEL_ID, BASE_MODEL_REVISION, MODEL_ID, MODEL_REVISION,
                      POLICY_SEEDS, STATE_SPLITS, digest, parse_selection, validate_done)
from protocol import make_protocol as standard_protocol
from protocol import summarize as standard_summary
from protocol import trials as standard_trials

WORKSPACE_X = (-0.25, 0.15)
WORKSPACE_Y = (-0.30, 0.30)
OTHER_CLEARANCE = 0.08
BASKET_CLEARANCE = 0.15


def admissible_xy(xy, others, basket_name):
    x, y = xy
    if not (WORKSPACE_X[0] <= x <= WORKSPACE_X[1] and WORKSPACE_Y[0] <= y <= WORKSPACE_Y[1]):
        return False
    return all(math.hypot(x-float(pos[0]), y-float(pos[1])) >=
               (BASKET_CLEARANCE if name == basket_name else OTHER_CLEARANCE)
               for name, pos in others.items())


def choose_target_xy(start_xy, others, basket_name, seed, jitter=0.04, max_attempts=100):
    """Deterministic uniform XY proposals with explicit rejection, no ML deps."""
    if not math.isfinite(jitter) or jitter <= 0 or jitter > 0.10:
        raise ValueError('position jitter must be positive and at most 0.10 metres per axis')
    rng = random.Random(seed)
    for attempt in range(1, max_attempts + 1):
        delta = [rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)]
        xy = [float(start_xy[0]) + delta[0], float(start_xy[1]) + delta[1]]
        if admissible_xy(xy, others, basket_name):
            return {'layout_rng_seed': seed, 'requested_delta_xy': delta,
                    'requested_xy': xy, 'sampling_attempts': attempt,
                    'proposal_distribution': 'independent uniform[-jitter,+jitter] in X and Y; rejection conditioned on workspace/clearance'}
    raise ValueError(f'no admissible target position in {max_attempts} proposals; do not silently use the original position')


def make_protocol(phase, initial_states, seeds, suite='libero_object', task_id=0,
                  horizon=280, arm='libero', position_jitter=0.04):
    if not math.isfinite(position_jitter) or not 0 < position_jitter <= 0.10:
        raise ValueError('position jitter must be positive and at most 0.10 metres per axis')
    body = standard_protocol(phase, initial_states, seeds, suite, task_id, horizon, arm)
    body.pop('protocol_sha256')
    body['condition'] = {
        'name': 'alphabet_soup_target_position_jitter',
        'standard_LIBERO_benchmark': False,
        'jitter_max_abs_xy_m': position_jitter,
        'layout_rng_seed_rule': '50000 + native_initial_state_id',
        'proposal_distribution': 'uniform square XY proposals with workspace/clearance rejection',
        'max_sampling_attempts': 100, 'workspace_x': list(WORKSPACE_X), 'workspace_y': list(WORKSPACE_Y),
        'other_object_clearance_m': OTHER_CLEARANCE, 'basket_clearance_m': BASKET_CLEARANCE,
        'setup': 'move only target freejoint XY before policy rollout; preserve Z/quaternion; forward then10 native no-op settle steps',
        'setup_steps_count_as_policy_actions': False,
        'position_variation_verification': 'actual settled XY span must exceed1e-4m when multiple native IDs finish',
    }
    body['protocol_sha256'] = digest(body)
    return body


def trials(protocol):
    return [{**trial, 'layout_rng_seed': 50000 + trial['initial_state_id']}
            for trial in standard_trials(protocol)]


def summarize(rows, protocol):
    result = standard_summary(rows, protocol)
    result['condition'] = protocol['condition']
    positions = {}
    for row in rows:
        metadata = row.get('position_setup')
        if metadata is None:
            raise ValueError('position-jitter trial is missing its measured position setup')
        state = row['initial_state_id']
        position = metadata['actual_settled_position'][:2]
        if state in positions and any(abs(a-b) > 1e-10 for a, b in zip(positions[state], position)):
            raise ValueError('paired policy seeds received different settled target positions')
        positions[state] = position
    span = [max(p[i] for p in positions.values()) - min(p[i] for p in positions.values())
            for i in range(2)] if positions else [0.0, 0.0]
    if len(positions) > 1 and max(span) <= 1e-4:
        raise ValueError('requested target randomization produced no material settled position variation')
    result['position_variation'] = {'verified': len(positions) > 1,
                                    'unique_start_ids_recorded': len(positions),
                                    'actual_settled_xy_span_m': span,
                                    'actual_settled_xy_by_initial_id': positions}
    return result
