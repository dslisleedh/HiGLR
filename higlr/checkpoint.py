"""Atomic epoch checkpoints and complete per-rank random state."""
import os
import random
from pathlib import Path

import numpy as np
import torch


def random_state(generator, *, use_cuda=False):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), loader=generator.get_state(),
                cuda=torch.cuda.get_rng_state() if use_cuda else None)


def restore_random_state(state, generator):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    generator.set_state(state['loader'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'])


def save_checkpoint(directory, state, milestone=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    latest, previous = directory / 'latest.pt', directory / 'previous.pt'
    temporary = directory / '.latest.tmp'
    with temporary.open('wb') as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    if latest.exists():
        link = directory / '.previous.tmp'
        link.unlink(missing_ok=True)
        os.link(latest, link)
        os.replace(link, previous)
    os.replace(temporary, latest)
    if milestone:
        target = directory / f"epoch_{state['epoch']:03d}.pt"
        # Never replace a milestone belonging to an already saved epoch.
        os.link(latest, target)


def model_weights(checkpoint):
    """Extract model weights from a training checkpoint or a model-only state dict."""
    if 'model' in checkpoint:
        return checkpoint['model']
    if 'state_dict' in checkpoint:
        return checkpoint['state_dict']
    return checkpoint
