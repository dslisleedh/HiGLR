"""Single-device/DDP training, explicitly separated pretraining and fine-tuning."""
import argparse
from contextlib import nullcontext
from copy import deepcopy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .checkpoint import model_weights, random_state, restore_random_state, save_checkpoint
from .data import build_dataset, class_ids
from .losses import restoration_loss
from .model import HiGLR


def learning_rate(step, total_steps, warmup_steps, config):
    if step < warmup_steps:
        return config['start_lr'] + (config['lr'] - config['start_lr']) * step / warmup_steps
    fraction = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
    return config['min_lr'] + .5 * (config['lr'] - config['min_lr']) * (1 + math.cos(math.pi * min(fraction, 1)))


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


@torch.no_grad()
def update_ema(ema, model, decay):
    for name, value in model.state_dict().items():
        if value.is_floating_point():
            ema[name].lerp_(value.detach(), 1 - decay)
        else:
            ema[name].copy_(value)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=['pretrain', 'finetune', 'train'], required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--data-lists', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--backend', choices=['torch', 'triton'], default='triton')
    parser.add_argument('--micro-batch', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--init-checkpoint', type=Path)
    parser.add_argument('--print-config', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = json.loads(args.config.read_text())
    expected_model = dict(block_size=8, dsffn_expansion=3, spatial_ranks=[16,16,32,64])
    if any(config['model'].get(key) != value for key, value in expected_model.items()):
        raise ValueError('This implementation requires B8, DSFFN gamma3, ranks16/16/32/64')
    if config['optimizer']['name'] != 'AdamW':
        raise ValueError('This implementation uses AdamW')
    if config['dataset'] not in ('aio3', 'aio5', 'cdd11'):
        raise ValueError('Only AIO3, AIO5 and CDD11 are supported')
    if args.stage not in config['stages']:
        raise ValueError(f"Stage {args.stage} is not defined in this config")
    stage = config['stages'][args.stage]
    if args.print_config:
        print(json.dumps(dict(config=config, stage=stage), indent=2))
        return
    if args.data_root is None or args.output is None:
        raise ValueError('--data-root and --output are required for training')
    if args.resume and args.init_checkpoint:
        raise ValueError('Choose full-state --resume or weights-only --init-checkpoint')
    if args.stage == 'finetune' and not (args.resume or args.init_checkpoint):
        raise ValueError('Fine-tuning requires a pretrained checkpoint')
    if args.micro_batch < 1 or args.workers < 0:
        raise ValueError('Invalid micro-batch/worker count')
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device('cuda', local_rank) if args.device == 'cuda' else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    elif args.backend != 'torch':
        raise ValueError('CPU training requires --backend torch')
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    lock = None
    try:
        # All ranks reach the same error before entering model collectives.
        output = args.output.resolve()
        error = [None]
        if rank == 0:
            try:
                output.mkdir(parents=True, exist_ok=True)
                lock = (output / '.writer.lock').open('a')
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if not args.resume and any(p.name != '.writer.lock' for p in output.iterdir()):
                    raise ValueError('Fresh output directory must be empty')
                if args.resume and args.resume.resolve() != output / 'checkpoints' / 'latest.pt':
                    raise ValueError('Resume requires this output directory\'s latest.pt; use a new output for weight transfer')
            except Exception as exc:
                error[0] = str(exc)
        if world > 1:
            dist.broadcast_object_list(error)
        if error[0]:
            raise RuntimeError(error[0])
        global_batch = config['global_batch_size']
        if global_batch % (args.micro_batch * world):
            raise ValueError('global_batch_size must equal micro_batch * world_size * integer accumulation')
        accumulation = global_batch // (args.micro_batch * world)
        seed = config['seed']
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        model = HiGLR(num_classes=config['num_classes'], backend=args.backend,
                      remote_k=config['model']['train_remote_k'], eval_remote_k=config['model']['eval_remote_k']).to(device)
        ema = deepcopy(model.state_dict())
        # Same dataset ordering across ranks; unique augmentation RNG afterward.
        dataset = build_dataset(config['dataset'], args.data_root, stage['patch_size'], args.data_lists)
        sampler = DistributedSampler(dataset, world, rank, shuffle=True, seed=seed, drop_last=True)
        generator = torch.Generator().manual_seed(seed + rank)
        loader = DataLoader(dataset, batch_size=args.micro_batch, sampler=sampler, drop_last=True,
                            num_workers=args.workers, pin_memory=device.type == 'cuda',
                            worker_init_fn=seed_worker, generator=generator, persistent_workers=False)
        updates_per_epoch = len(loader) // accumulation
        if not updates_per_epoch:
            raise ValueError('Dataset is smaller than a global batch')
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['optimizer']['lr'],
                                     betas=tuple(config['optimizer']['betas']), weight_decay=config['optimizer']['weight_decay'])
        protocol = dict(config=config, stage=args.stage, world_size=world, micro_batch=args.micro_batch,
                        accumulation=accumulation, workers=args.workers, backend=args.backend,
                        data_root=str(args.data_root.resolve()), data_lists=str(args.data_lists.resolve()) if args.data_lists else None,
                        dataset_length=len(dataset), updates_per_epoch=updates_per_epoch,
                        torch_version=torch.__version__,
                        source_sha256={str(path.relative_to(Path(__file__).parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                                       for path in sorted(Path(__file__).parent.rglob('*.py'))})
        epoch_start, step = 0, 0
        random.seed(seed + rank); np.random.seed(seed + rank); torch.manual_seed(seed + rank)
        if args.init_checkpoint:
            checkpoint = torch.load(args.init_checkpoint, map_location='cpu', weights_only=False)
            model.load_state_dict(model_weights(checkpoint), strict=True)
            # RAW initializes RAW, EMA initializes EMA; no optimizer/scheduler transfer.
            ema = {key: value.to(device).clone() for key, value in checkpoint.get('ema', model.state_dict()).items()}
        resume_state = None
        if args.resume:
            resume_state = torch.load(args.resume, map_location='cpu', weights_only=False)
            if resume_state['protocol'] != protocol:
                raise ValueError('Resume protocol differs; keep config, data, backend, world size, batch and workers fixed')
            model.load_state_dict(resume_state['model'], strict=True)
            optimizer.load_state_dict(resume_state['optimizer'])
            ema = {key: value.to(device) for key, value in resume_state['ema'].items()}
            epoch_start, step = resume_state['epoch'], resume_state['step']
            if step != epoch_start * updates_per_epoch:
                raise ValueError('Expected a complete epoch checkpoint')
        net = DistributedDataParallel(model, device_ids=[local_rank] if device.type == 'cuda' else None) if world > 1 else model
        if resume_state:
            restore_random_state(resume_state['rng'][rank], generator)
        if rank == 0:
            (output / 'config.json').write_text(json.dumps(protocol, indent=2) + '\n')
        total_steps = stage['epochs'] * updates_per_epoch
        warmup_steps = stage['warmup_epochs'] * updates_per_epoch
        log = (output / 'train.jsonl').open('a') if rank == 0 else None
        try:
            for epoch in range(epoch_start, stage['epochs']):
                sampler.set_epoch(epoch)
                net.train()
                optimizer.zero_grad(set_to_none=True)
                accumulated_loss = 0.
                for batch_index, ((_, labels), inputs, targets) in enumerate(loader):
                    if batch_index >= updates_per_epoch * accumulation:
                        break
                    sync = (batch_index + 1) % accumulation == 0
                    context = net.no_sync() if world > 1 and not sync else nullcontext()
                    with context:
                        restored, scores = net(inputs.to(device), return_scores=True)
                        loss = restoration_loss(restored, targets.to(device), scores,
                                                class_ids(labels.to(device), config['dataset']), stage['loss'])
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Non-finite training loss')
                        (loss / accumulation).backward()
                        accumulated_loss += float(loss.detach()) / accumulation
                    if sync:
                        lr = learning_rate(step, total_steps, warmup_steps, config['optimizer'])
                        for group in optimizer.param_groups:
                            group['lr'] = lr
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        update_ema(ema, model, config['ema_decay'])
                        step += 1
                        if log and (step % 10 == 0 or step == 1):
                            log.write(json.dumps(dict(epoch=epoch+1, step=step, loss=accumulated_loss, lr=lr, time=time.time()))+'\n')
                            log.flush()
                        accumulated_loss = 0.
                local_rng = random_state(generator, use_cuda=device.type == 'cuda')
                all_rng = [None] * world
                if world > 1:
                    dist.all_gather_object(all_rng, local_rng)
                else:
                    all_rng[0] = local_rng
                if rank == 0:
                    state = dict(format_version=1, model=model.state_dict(), ema=ema, optimizer=optimizer.state_dict(),
                                 epoch=epoch+1, step=step, rng=all_rng, protocol=protocol,
                                 scheduler=dict(step=step, total_steps=total_steps, warmup_steps=warmup_steps), sampler_epoch=epoch+1)
                    save_checkpoint(output / 'checkpoints', state,
                                    milestone=(epoch+1) % stage['checkpoint_every'] == 0 or epoch+1 == stage['epochs'])
                    print(f'epoch={epoch+1} step={step} checkpoint saved', flush=True)
                if world > 1:
                    dist.barrier()
        finally:
            if log:
                log.close()
    finally:
        if lock:
            lock.close()
        if dist.is_initialized():
            dist.destroy_process_group()
