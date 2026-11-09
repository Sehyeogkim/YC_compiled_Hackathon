"""Explicit, untrained LIBERO I/O adapter around pinned SmolVLA base weights.

This is a diagnostic adapted-base system, not the raw checkpoint's zero-shot
LIBERO performance. It borrows the LIBERO checkpoint's feature contract and
normalization processors, but loads model weights ONLY from smolvla_base.
Heavy imports are deferred so contract tests run without PyTorch or LeRobot.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

BASE_MODEL_ID = 'lerobot/smolvla_base'
BASE_REVISION = '5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6'
LIBERO_MODEL_ID = 'lerobot/smolvla_libero'
LIBERO_REVISION = 'a24f1f380461b665158d52acd140152c79e8b36d'
RENAME_MAP = {
    'observation.images.image': 'observation.images.camera1',
    'observation.images.image2': 'observation.images.camera2',
}
_ARCHITECTURE_FIELDS = (
    'vlm_model_name', 'num_vlm_layers', 'num_expert_layers',
    'expert_width_multiplier', 'attention_mode', 'self_attn_every_n_layers',
    'max_state_dim', 'max_action_dim',
)


def _shape(features, key):
    if key not in features:
        raise ValueError(f'Missing feature: {key}')
    feature = features[key]
    return tuple(feature['shape'] if isinstance(feature, dict) else feature.shape)


def _camera_keys(config):
    return {key for key in config.input_features if key.startswith('observation.images.')}


def adapted_config(base_config, libero_config, *, n_action_steps=10, device='cuda'):
    """Copy only the declared native contract into a copy of base's configuration.

    The internal projections use 32 padded channels, so changing exposed shapes
    does not add a randomly initialized seventh-output head. Its semantics are
    nevertheless unvalidated: this cannot turn SO-100 pretraining into learned
    Panda end-effector control.
    """
    if type(n_action_steps) is not int or not 1 <= n_action_steps <= 50:
        raise ValueError('n_action_steps must be an integer between 1 and 50')
    if _shape(base_config.input_features, 'observation.state') != (6,):
        raise ValueError('Expected the pinned base state6 contract')
    if _shape(base_config.output_features, 'action') != (6,):
        raise ValueError('Expected the pinned base action6 contract')
    if _camera_keys(base_config) != {
        'observation.images.camera1', 'observation.images.camera2',
        'observation.images.camera3',
    }:
        raise ValueError('Expected the pinned base three-camera contract')
    if _shape(libero_config.input_features, 'observation.state') != (8,):
        raise ValueError('LIBERO state must have 8 features')
    if _shape(libero_config.output_features, 'action') != (7,):
        raise ValueError('LIBERO action must have 7 features')
    if _camera_keys(libero_config) != set(RENAME_MAP.values()):
        raise ValueError('LIBERO must expose exactly agentview and wrist camera keys')
    for key in set(RENAME_MAP.values()):
        if _shape(libero_config.input_features, key) != (3, 256, 256):
            raise ValueError(f'Unexpected camera shape: {key}')
    for key in _ARCHITECTURE_FIELDS:
        if getattr(base_config, key, None) != getattr(libero_config, key, None):
            raise ValueError(f'Architecture differs at {key}; do not partially load weights')
    if base_config.max_state_dim != 32 or base_config.max_action_dim != 32:
        raise ValueError('Expected the pinned internal 32-channel padded architecture')
    if base_config.chunk_size != 50 or libero_config.chunk_size != 50:
        raise ValueError('Both pinned models must have chunk_size=50')
    result = deepcopy(base_config)
    result.input_features = deepcopy(libero_config.input_features)
    result.output_features = deepcopy(libero_config.output_features)
    result.normalization_mapping = deepcopy(libero_config.normalization_mapping)
    result.device = device
    result.n_action_steps = n_action_steps
    return result


def _check_snapshot_pin(path, repo_id, revision):
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f'Expected a local pinned checkpoint directory: {path}')
    pin_file = path / 'WERESCUE_PIN.json'
    if pin_file.exists():
        pin = json.loads(pin_file.read_text())
        if pin.get('repo_id') != repo_id or pin.get('revision') != revision:
            raise ValueError(f'Checkpoint pin mismatch: {path}')
    elif path.name != revision:
        raise ValueError(f'Checkpoint requires WERESCUE_PIN.json or a revision-named Hub snapshot: {path}')
    for name in ('config.json', 'model.safetensors'):
        if not (path / name).is_file():
            raise ValueError(f'Checkpoint file missing: {path / name}')


def adapter_manifest(base_snapshot=None, libero_snapshot=None):
    return {
        'arm': 'base-adapted',
        'display_name': 'SmolVLA base weights + declared LIBERO I/O and normalization adapter',
        'weights': {'repo_id': BASE_MODEL_ID, 'revision': BASE_REVISION},
        'processor_and_feature_contract': {'repo_id': LIBERO_MODEL_ID, 'revision': LIBERO_REVISION},
        'additional_gradient_steps': 0,
        'raw_checkpoint_zero_shot': False,
        'weights_modified': False,
        'source_contract': {'state_dim': 6, 'action_dim': 6, 'camera_count': 3},
        'adapted_contract': {'state_dim': 8, 'action_dim': 7, 'camera_count': 2},
        'action_adapter': 'Expose seven native OSC channels from the existing internal padded32 output; no six-to-seven action truncation or synthetic head.',
        'third_camera': 'Removed from the exposed feature contract; two actual native cameras remain.',
        'normalization': 'Pinned LIBERO processor statistics, not SO-100 base statistics.',
        'limitation': 'Base weights were not exposed as a trained seven-dimensional LIBERO OSC controller. Seventh-channel and robot/state/action semantics are unvalidated. This diagnostic measures the declared adapted system only.',
        **({'base_snapshot': str(base_snapshot)} if base_snapshot is not None else {}),
        **({'libero_snapshot': str(libero_snapshot)} if libero_snapshot is not None else {}),
    }


def load_base_adapted_stack(base_snapshot, libero_snapshot, env_cfg,
                            n_action_steps=10, device='cuda'):
    """Return (policy, pre, post, env_pre, env_post) without updating weights."""
    _check_snapshot_pin(base_snapshot, BASE_MODEL_ID, BASE_REVISION)
    _check_snapshot_pin(libero_snapshot, LIBERO_MODEL_ID, LIBERO_REVISION)
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.envs.factory import make_env_pre_post_processors

    base_cfg = PreTrainedConfig.from_pretrained(str(base_snapshot))
    libero_cfg = PreTrainedConfig.from_pretrained(str(libero_snapshot))
    config = adapted_config(base_cfg, libero_cfg, n_action_steps=n_action_steps, device=device)
    config.pretrained_path = str(base_snapshot)
    # The base architecture is retained and every parameter must load. There is
    # no partial load, random new output head, finetuning, or copied LIBERO weight.
    policy = SmolVLAPolicy.from_pretrained(str(base_snapshot), config=config, strict=True)
    policy.eval()
    features = {**config.input_features, **config.output_features}
    pre, post = make_pre_post_processors(
        policy_cfg=config, pretrained_path=str(libero_snapshot),
        preprocessor_overrides={
            'device_processor': {'device': device},
            'rename_observations_processor': {'rename_map': RENAME_MAP},
            'normalizer_processor': {'features': features, 'norm_map': config.normalization_mapping},
        },
        postprocessor_overrides={
            'unnormalizer_processor': {
                'features': config.output_features, 'norm_map': config.normalization_mapping,
            },
        },
    )
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=config)
    return policy, pre, post, env_pre, env_post


# Earlier messages used these spellings; both expose the same explicit adapter.
load_base_adapter_stack = load_base_adapted_stack
base_adapter_provenance = adapter_manifest
