#!/usr/bin/env python3
"""Run LeRobot finetuning while retaining pretrained P normalization statistics.

Pass ordinary lerobot-train flags after this script. Both task and recovery runs
must independently set --policy.path to the same local pinned P directory.

This wrapper changes only runtime function bindings, not the framework checkout.
It removes replacement dataset stats from pretrained processor-loading calls.
It checks the actual policy source before bridging an old padding-mask key.

Example:
  python train_preserve_stats.py --policy.path=/workspace/models/P \
    --dataset.repo_id=werescue/task_R --dataset.root=/workspace/data/task_R \
    --output_dir=/workspace/runs/task_R --steps=1000 --batch_size=16 \
    --policy.freeze_vision_encoder=true --policy.train_expert_only=true \
    --policy.optimizer_lr=0.00003 --policy.scheduler_warmup_steps=100 \
    --policy.scheduler_decay_steps=1000 --policy.scheduler_decay_lr=0.000003 \
    --policy.push_to_hub=false --wandb.enable=false

Use --lerobot-help to inspect the installed framework's exact options.
"""
from __future__ import annotations

import inspect
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys


def keep_pretrained_stats(kwargs: dict) -> dict:
    """Pure transformation, separately testable without importing ML packages."""
    result = dict(kwargs)
    if not result.get('pretrained_path'):
        raise RuntimeError('This experiment requires a pretrained P processor path.')
    result.pop('dataset_stats', None)
    for name in ('preprocessor_overrides', 'postprocessor_overrides'):
        if name not in result or result[name] is None:
            continue
        result[name] = {key: dict(value) for key, value in result[name].items()}
        for key in ('normalizer_processor', 'unnormalizer_processor'):
            if key in result[name]:
                result[name][key].pop('stats', None)
    return result


def install_patches():
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.scripts import lerobot_train

    original_processors = lerobot_train.make_pre_post_processors

    def pretrained_processors(*args, **kwargs):
        if args:
            raise RuntimeError('Unexpected positional processor call; review this framework version.')
        return original_processors(**keep_pretrained_stats(kwargs))

    lerobot_train.make_pre_post_processors = pretrained_processors
    source = inspect.getsource(SmolVLAPolicy.forward)
    needs_old_alias = 'batch.get("actions_id_pad")' in source or "batch.get('actions_id_pad')" in source
    uses_current_key = 'batch.get("action_is_pad")' in source or "batch.get('action_is_pad')" in source
    if needs_old_alias and not uses_current_key:
        original_forward = SmolVLAPolicy.forward

        def forward_with_old_mask(self, batch, *args, **kwargs):
            if 'action_is_pad' in batch:
                batch = {**batch, 'actions_id_pad': batch['action_is_pad']}
            return original_forward(self, batch, *args, **kwargs)

        SmolVLAPolicy.forward = forward_with_old_mask
        mask_note = 'old actions_id_pad reader detected: alias action_is_pad at forward boundary'
    elif uses_current_key:
        mask_note = 'current action_is_pad reader detected: no mask patch applied'
    else:
        raise RuntimeError('Unknown SmolVLA padding-mask implementation; inspect before training short segments.')
    source_path = Path(inspect.getfile(SmolVLAPolicy))
    try:
        version = importlib.metadata.version('lerobot')
    except importlib.metadata.PackageNotFoundError:
        version = 'unregistered_checkout'
    print(json.dumps({
        'werescue_training_wrapper': 'preserve_pretrained_stats',
        'framework_version': version,
        'trainer_source': str(Path(inspect.getfile(lerobot_train)).resolve()),
        'smolvla_source': str(source_path.resolve()),
        'smolvla_source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'padding_mask_path': mask_note,
        'framework_files_modified_by_wrapper': False,
    }), flush=True)
    return lerobot_train


def main():
    if sys.argv[1:] in (['--help'], ['-h']):
        print(__doc__)
        return
    if '--lerobot-help' in sys.argv:
        sys.argv[sys.argv.index('--lerobot-help')] = '--help'
    # Training script import registers built-in policy classes before CLI parsing.
    trainer = install_patches()
    if hasattr(trainer, 'main'):
        trainer.main()
    else:
        trainer.train()


if __name__ == '__main__':
    main()
