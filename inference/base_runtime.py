"""Isolated collection runtime using SmolVLA base weights with native LIBERO I/O.

The original LIBERO service stays on its own process and port. This mode is an
adapted-base diagnostic: it does not claim that raw SmolVLA base supports Panda.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from .runtime import NativeRuntime
    from .base_adapter import (
        BASE_MODEL_ID, BASE_REVISION, LIBERO_MODEL_ID, LIBERO_REVISION,
        adapter_manifest, load_base_adapted_stack,
    )
except ImportError:
    from runtime import NativeRuntime
    from base_adapter import (
        BASE_MODEL_ID, BASE_REVISION, LIBERO_MODEL_ID, LIBERO_REVISION,
        adapter_manifest, load_base_adapted_stack,
    )


class BaseAdaptedRuntime(NativeRuntime):
    """Replace only policy weights and processors after loading the native stack."""

    def __init__(self):
        super().__init__()

    def load(self):
        if self.status == 'ready':
            return self.health()

        # Super loads the verified native simulator, GPU dependencies and the
        # exact LIBERO checkpoint whose normalization processors we reuse.
        self.model_id = LIBERO_MODEL_ID
        self.model_revision = LIBERO_REVISION
        super().load()
        self.status = 'loading'
        from huggingface_hub import snapshot_download

        try:
            base_path = os.environ.get('WERESCUE_BASE_POLICY_PATH')
            base_snapshot = Path(base_path) if base_path else Path(snapshot_download(
                repo_id=BASE_MODEL_ID, revision=BASE_REVISION))
            libero_snapshot = Path(self.policy_path)
            policy, pre, post, env_pre, env_post = load_base_adapted_stack(
                base_snapshot, libero_snapshot, self.env_cfg,
                n_action_steps=10, device=self.device)
            previous_policy = self.policy
            self.policy, self.pre, self.post = policy, pre, post
            self.env_pre, self.env_post = env_pre, env_post
            del previous_policy
            if self.device.startswith('cuda'):
                self.torch.cuda.empty_cache()
            self.policy.reset()
            self.model_id = BASE_MODEL_ID
            self.model_revision = BASE_REVISION
            self.model_provenance = adapter_manifest(base_snapshot, libero_snapshot)
            self.status = 'ready'
            return self.health()
        except Exception as exc:
            self.status = 'error'
            self.error = f'{type(exc).__name__}: {exc}'
            raise
