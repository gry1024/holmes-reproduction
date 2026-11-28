"""Compact, read-only snapshot for estimating remaining queue runtime."""
import json
from pathlib import Path
import statistics
import time
root=Path(__file__).resolve().parents[1]
rows=[]
for d in sorted((root/'runs').glob('*/*')):
    if not d.is_dir():
        continue
    def read(name):
        p=d/name
        return json.loads(p.read_text()) if p.exists() else None
    cfg=read('config.json')
    epochs=[]
    if (d/'epochs.jsonl').exists():
        for line in (d/'epochs.jsonl').read_text().splitlines():
            try: epochs.append(json.loads(line))
            except json.JSONDecodeError: pass
    row=dict(run=str(d.relative_to(root/'runs')),status=read('status.json'),result=read('result.json'),
             completed_epochs=len(epochs),last_epoch=epochs[-1] if epochs else None)
    if epochs:
        row['median_epoch_seconds']=statistics.median(e['epoch_seconds'] for e in epochs[-10:])
        row['total_epoch_seconds']=sum(e['epoch_seconds'] for e in epochs)
    if cfg:
        row.update(max_epochs=cfg['config']['n_epoch'],patience=cfg['config']['max_es_cnt'],
                   train_batches=cfg['train_batches'],train_videos=cfg['train_videos'])
    rows.append(row)
print(json.dumps(dict(timestamp=time.time(),runs=rows),indent=2))
