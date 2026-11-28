"""Read-only CPU diagnostics of saved experiment logs and query gates."""
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/'ICML26-Holmes/src'))
import torch
from Configs.builder import get_configs
from Datasets.builder import get_datasets
from Models.builder import get_models
from adapters import extend_model

torch.set_num_threads(2)
report = {'checked_at':time.time(), 'curves':{}, 'gates':{}}
for path in sorted((root/'runs').glob('*/*/epochs.jsonl')):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    key = str(path.parent.relative_to(root/'runs'))
    best = max(rows, key=lambda row:row['metrics'][4])
    selected = [r for r in rows if r['epoch'] in [0,18,19,20,21,22,30,40,60,80,rows[-1]['epoch']]]
    report['curves'][key] = dict(completed_epochs=len(rows), best_epoch=best['epoch'],
        best_metrics=best['metrics'], samples=[{k:r[k] for k in ['epoch','loss','metrics']} for r in selected])

for dataset in ['cha','act']:
    cfg = get_configs(dataset).copy()
    cfg.update(data_root=str(root/'data'), num_workers=0)
    for variant in ['gating','combined']:
        path = root/'runs'/dataset/variant/'best.pt'
        if not path.exists():
            continue
        cfg, train, vv, vq, tv, tq = get_datasets(cfg)
        model = extend_model(get_models(cfg),variant,cfg)
        ck = torch.load(path, map_location='cpu', weights_only=True)
        model.load_state_dict(ck['state_dict'],strict=True)
        model.eval()
        weights = []
        with torch.no_grad():
            for batch in vq:
                q = model.encode_query(batch[0],batch[1])
                weights.append(model.scale_gate(q).sigmoid().flatten())
        w = torch.cat(weights)
        report['gates'][f'{dataset}/{variant}'] = dict(checkpoint_epoch=ck['epoch'],
            queries=w.numel(),mean=w.mean().item(),std=w.std().item(),
            minimum=w.min().item(),maximum=w.max().item(),
            p05=w.quantile(.05).item(),median=w.median().item(),p95=w.quantile(.95).item(),
            fraction_below_01=(w<.1).float().mean().item(),fraction_above_09=(w>.9).float().mean().item())
        del model,ck,train,vv,vq,tv,tq
target=root/'results/diagnosis.json'
target.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
