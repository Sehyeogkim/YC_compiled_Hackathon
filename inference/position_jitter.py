"""Reproducible, clearance-checked target-can placement for collection resets."""
from __future__ import annotations

import math
import random

WORKSPACE_X = (-0.25, 0.15)
WORKSPACE_Y = (-0.30, 0.30)
OTHER_CLEARANCE = 0.08
BASKET_CLEARANCE = 0.15


def admissible_xy(xy, others, basket_name):
    x, y = map(float, xy)
    return (WORKSPACE_X[0] <= x <= WORKSPACE_X[1]
            and WORKSPACE_Y[0] <= y <= WORKSPACE_Y[1]
            and all(math.hypot(x - float(pos[0]), y - float(pos[1])) >=
                    (BASKET_CLEARANCE if name == basket_name else OTHER_CLEARANCE)
                    for name, pos in others.items()))


def choose_target_xy(start_xy, others, basket_name, seed, jitter=0.04, max_attempts=100):
    if not math.isfinite(jitter) or not 0 < jitter <= 0.10:
        raise ValueError('position jitter must be positive and at most 0.10 metres per axis')
    rng = random.Random(seed)
    for attempt in range(1, max_attempts + 1):
        delta = [rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)]
        xy = [float(start_xy[0]) + delta[0], float(start_xy[1]) + delta[1]]
        if math.hypot(*delta) > 1e-4 and admissible_xy(xy, others, basket_name):
            return {'layout_rng_seed': seed, 'requested_delta_xy': delta,
                    'requested_xy': xy, 'sampling_attempts': attempt,
                    'jitter_max_abs_xy_m': jitter,
                    'proposal_distribution': 'independent uniform XY with workspace/clearance rejection'}
    raise ValueError('no admissible target position in 100 proposals')


def perturb_target_position(env, layout_seed, jitter):
    """Set can XY, settle physics, return vector-style observation and provenance."""
    import numpy as np
    from lerobot.envs.libero import get_libero_dummy_action

    problem = env._env.env
    target = next((name for name in problem.obj_of_interest if 'alphabet_soup' in name.lower()), None)
    basket = next((name for name in problem.obj_of_interest if 'basket' in name.lower()), None)
    if target is None or basket is None or target not in problem.objects_dict or target not in problem.obj_body_id:
        raise RuntimeError('position jitter requires alphabet_soup and basket native objects')
    sim = problem.sim
    joint = problem.objects_dict[target].joints[-1]
    qpos = np.array(sim.data.get_joint_qpos(joint), copy=True)
    if qpos.shape != (7,):
        raise RuntimeError(f'target must have a 7D freejoint, got {qpos.shape}')
    original = np.array(sim.data.body_xpos[problem.obj_body_id[target]], copy=True)
    others = {name: np.array(sim.data.body_xpos[body][:2], copy=True)
              for name, body in problem.obj_body_id.items() if name != target}
    selected = choose_target_xy(original[:2], others, basket, layout_seed, jitter)
    qpos[:2] = selected['requested_xy']
    sim.data.set_joint_qpos(joint, qpos)
    sim.forward()
    raw = None
    for _ in range(10):
        raw, _, _, _ = env._env.step(get_libero_dummy_action())
    actual = np.array(sim.data.body_xpos[problem.obj_body_id[target]], copy=True)
    actual_others = {name: np.array(sim.data.body_xpos[body][:2], copy=True)
                     for name, body in problem.obj_body_id.items() if name != target}
    if not admissible_xy(actual[:2], actual_others, basket):
        raise RuntimeError('target settled outside workspace/clearance')
    if float(np.linalg.norm(actual[:2] - original[:2])) <= 1e-5:
        raise RuntimeError('target did not move after settling')
    if env._env.check_success():
        raise RuntimeError('position setup satisfied task before policy action')

    def batched(tree):
        if isinstance(tree, dict):
            return {key: batched(value) for key, value in tree.items()}
        return np.expand_dims(np.asarray(tree), 0)

    return batched(env._format_raw_obs(raw)), {
        **selected, 'target_object': target, 'basket_object': basket,
        'original_position': original.tolist(), 'actual_settled_position': actual.tolist(),
        'actual_delta_xy': (actual[:2] - original[:2]).tolist(),
        'settle_native_noop_steps': 10, 'vector_policy_steps_during_setup': 0,
        'setup_before_rollout': True,
        'target_z_and_quaternion_preserved_on_assignment': True,
    }
