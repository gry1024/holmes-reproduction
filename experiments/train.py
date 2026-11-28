"""Train the original baseline or an ablation with the same training loop."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

p = argparse.ArgumentParser()
p.add_argument('--dataset', choices=['cha', 'act', 'tvr'], required=True)
p.add_argument('--variant', choices=['baseline', 'pooling', 'gating', 'combined'], required=True)
p.add_argument('--multi', action='store_true')
p.add_argument('--data-root', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--workers', type=int, default=4)
p.add_argument('--smoke', action='store_true')
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'ICML26-Holmes' / ('multisrc' if a.multi else 'src')))
import numpy as np
import torch
from torch.nn import functional as F
from Configs.builder import get_configs
from Datasets.builder import get_datasets
from Models.builder import get_models
from Losses.builder import get_losses
from Opts.builder import get_opts
from Validations.builder import get_validations
from Utils.utils import gpu, set_seed
from adapters import extend_model, extend_validation


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temp.replace(path)


def save(path, value):
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    temp.replace(path)


def main():
    a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = get_configs(a.dataset).copy()
    cfg.update(data_root=str(a.data_root), root=str(a.output), num_workers=a.workers)
    set_seed(cfg['seed'])
    cfg, train, vv, vq, tv, tq = get_datasets(cfg)
    if a.variant == 'baseline' and not a.smoke:
        peer = a.output.parent/'pooling'/'config.json'
        if peer.exists():
            previous = json.loads(peer.read_text())
            differences = {k: (previous['config'].get(k), v) for k, v in cfg.items()
                           if k != 'root' and previous['config'].get(k) != v}
            if differences or previous['torch'] != torch.__version__:
                raise ValueError(f'Baseline training environment/config mismatch: {differences}')
    model = get_models(cfg)
    if a.variant != 'baseline':
        model = extend_model(model, a.variant, cfg, a.multi)
    model = model.cuda()
    gating = a.variant in ('gating', 'combined')
    criterion = get_losses(cfg)
    validator = get_validations(cfg)
    if a.variant != 'baseline':
        validator = extend_validation(validator)
    optimizer = get_opts(cfg, model, train)
    metadata = dict(config=cfg, variant=a.variant, multi=a.multi, torch=torch.__version__,
        gpu=torch.cuda.get_device_name(), cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),
        precision='FP32, TF32 disabled', pooling_k=3, pooling_peak_weight=0.5,
        fusion_loss_weight=0.1 if gating else 0, fusion_temperature=0.1,
        train_videos=len(train.dataset), train_batches=len(train),
        source_hashes={str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
                       for folder in [root/'experiments', root/'ICML26-Holmes'/'src', root/'ICML26-Holmes'/'multisrc']
                       for f in folder.rglob('*.py')})
    write_json(a.output/'config.json', metadata)
    print(json.dumps({k:v for k,v in metadata.items() if k != 'source_hashes'}), flush=True)
    best, stale, start_epoch = [0.]*5, 0, 0
    latest = a.output/'last.pt'
    if latest.exists() and not a.smoke:
        ck = torch.load(latest, map_location='cpu', weights_only=False)
        if ck['variant'] != a.variant or ck['multi'] != a.multi:
            raise ValueError('Checkpoint experiment mismatch')
        model.load_state_dict(ck['state_dict'], strict=True)
        optimizer.load_state_dict(ck['optimizer'])
        best, stale, start_epoch = ck['best'], ck['stale'], ck['epoch']+1
        random.setstate(ck['python_rng'])
        np.random.set_state(ck['numpy_rng'])
        torch.set_rng_state(ck['torch_rng'])
        torch.cuda.set_rng_state_all(ck['cuda_rng'])
        del ck
    started = time.time()
    epochs = [0, cfg['warmup_epoch']] if a.smoke else range(start_epoch, cfg['n_epoch'])
    for epoch in epochs:
        if not a.smoke and stale > cfg['max_es_cnt'] >= 0:
            break
        criterion.cfg['warmup_end'] = epoch >= cfg['warmup_epoch']
        criterion.cfg['use_hard_negative'] = epoch >= cfg['hard_negative_start_epoch']
        model.train()
        torch.cuda.reset_peak_memory_stats()
        epoch_start, loss_sum, count = time.time(), 0., 0
        for step, batch in enumerate(train):
            batch = gpu(batch)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss = criterion(output, batch)
            if gating:
                labels = torch.tensor(batch['text_labels'], device='cuda')
                loss = loss + 0.1 * F.cross_entropy(model.last_fused / 0.1, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            if a.smoke and gating:
                grad = model.scale_gate[-1].weight.grad
                assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
            optimizer.step()
            loss_sum += loss.item()
            count += 1
            if step % 10 == 0:
                state = dict(status='training', epoch=epoch, step=step, batches=len(train),
                    loss=loss.item(), seconds=time.time()-started, updated_at=time.time(),
                    peak_gib=torch.cuda.max_memory_allocated()/2**30)
                write_json(a.output/'status.json', state)
                print(json.dumps(state), flush=True)
            del output, loss, batch
            model.last_fused = None
            if a.smoke and step >= 1:
                break
        if a.smoke:
            print('SMOKE_TRAIN_OK', epoch, torch.cuda.max_memory_allocated()/2**30, flush=True)
            continue
        with torch.no_grad():
            metrics = [float(x) for x in validator(model, vv, vq)[:5]]
        improved = metrics[4] > best[4]
        best, stale = (metrics, 0) if improved else (best, stale+1)
        checkpoint = dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
            config=cfg, variant=a.variant, multi=a.multi, epoch=epoch, best=best, stale=stale,
            python_rng=random.getstate(), numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
        save(latest, checkpoint)
        if improved:
            save(a.output/'best.pt', dict(state_dict=model.state_dict(), epoch=epoch, metrics=metrics))
        record = dict(epoch=epoch, loss=loss_sum/count, metrics=metrics, best=best,
            epoch_seconds=time.time()-epoch_start, peak_gib=torch.cuda.max_memory_allocated()/2**30)
        with (a.output/'epochs.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        print('EPOCH_COMPLETE', json.dumps(record), flush=True)
    if a.smoke:
        model.eval()
        with torch.no_grad():
            metrics = validator(model, vv, vq)[:5]
        save(a.output/'smoke.pt', model.state_dict())
        model.load_state_dict(torch.load(a.output/'smoke.pt', weights_only=True), strict=True)
        write_json(a.output/'status.json', dict(status='smoke_passed', metrics=metrics))
        print('SMOKE_PASSED', metrics, flush=True)
        return
    ck = torch.load(a.output/'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(ck['state_dict'], strict=True)
    with torch.no_grad():
        val_metrics = validator(model, vv, vq)[:5]
        test_metrics = validator(model, tv, tq)[:5]
    result = dict(status='completed', best_epoch=ck['epoch'], val=val_metrics, test=test_metrics,
                  seconds_this_invocation=time.time()-started, updated_at=time.time())
    write_json(a.output/'result.json', result)
    write_json(a.output/'status.json', result)
    print('COMPLETED', json.dumps(result), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        a.output.mkdir(parents=True, exist_ok=True)
        write_json(a.output/'status.json', dict(status='failed', error=traceback.format_exc(), updated_at=time.time()))
        raise
