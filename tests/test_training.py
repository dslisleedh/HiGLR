import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from higlr import training
from higlr.data import class_ids


class TinyDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        target = torch.rand(3, 32, 32) * .5 + .2
        image = (target + np.random.rand() * .03 + random.random() * .02).clamp(0, 1)
        return ['synthetic', index], image, target


def assert_tree(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for av, bv in zip(a, b):
            assert_tree(av, bv)
    else:
        assert a == b


def test_epoch_resume_and_finetune(tmp_path, monkeypatch):
    # Keep the replay oracle independent of MKLDNN's CPU reduction choices.
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.backends.mkldnn, 'enabled', False)
    monkeypatch.setattr(training, 'build_dataset', lambda *args: TinyDataset())
    config = json.loads((Path(__file__).parents[1] / 'configs/aio3.json').read_text())
    config['global_batch_size'] = 1
    config['stages']['pretrain'].update(epochs=2, patch_size=32, warmup_epochs=.5, checkpoint_every=1)
    config['stages']['finetune'].update(epochs=1, patch_size=32, warmup_epochs=.5, checkpoint_every=1)
    path = tmp_path / 'diagnostic.json'
    path.write_text(json.dumps(config))
    def run(output, stage='pretrain', extra=()):
        training.main(['--config', str(path), '--stage', stage, '--data-root', str(tmp_path),
                       '--output', str(output), '--device', 'cpu', '--backend', 'torch',
                       '--micro-batch', '1', '--workers', '0', *extra])
    uninterrupted, interrupted = tmp_path / 'full', tmp_path / 'resumed'
    run(uninterrupted)
    save = training.save_checkpoint
    def stop_after_epoch(directory, state, **kwargs):
        save(directory, state, **kwargs)
        raise InterruptedError('test interruption after durable epoch checkpoint')
    monkeypatch.setattr(training, 'save_checkpoint', stop_after_epoch)
    with pytest.raises(InterruptedError):
        run(interrupted)
    monkeypatch.setattr(training, 'save_checkpoint', save)
    run(interrupted, extra=['--resume', str(interrupted / 'checkpoints/latest.pt')])
    load = lambda directory: torch.load(directory / 'checkpoints/latest.pt', weights_only=False, map_location='cpu')
    a, b = load(uninterrupted), load(interrupted)
    for key in ('model', 'ema', 'optimizer', 'scheduler', 'epoch', 'step'):
        assert_tree(a[key], b[key])
    assert (interrupted / 'checkpoints/previous.pt').is_file()
    assert (interrupted / 'checkpoints/epoch_001.pt').is_file()
    assert (interrupted / 'checkpoints/epoch_002.pt').is_file()
    run(tmp_path / 'ft', stage='finetune', extra=['--init-checkpoint', str(interrupted / 'checkpoints/latest.pt')])
    ft = load(tmp_path / 'ft')
    assert ft['step'] == 2 and ft['epoch'] == 1
    assert all(value['step'] == 2 for value in ft['optimizer']['state'].values())
    with pytest.raises(RuntimeError, match='must be empty'):
        run(uninterrupted)


def test_schedule_and_label_mapping():
    optimizer = json.loads((Path(__file__).parents[1] / 'configs/aio3.json').read_text())['optimizer']
    assert training.learning_rate(0, 400, 75, optimizer) == 1e-6
    assert training.learning_rate(75, 400, 75, optimizer) == .001
    assert training.learning_rate(399, 400, 75, optimizer) == 1e-6
    assert class_ids(torch.arange(7), 'aio5').tolist() == [0, 0, 0, 1, 2, 3, 4]
    assert class_ids(torch.arange(11), 'cdd11').tolist() == list(range(11))
