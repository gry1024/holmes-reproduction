"""Evaluate author checkpoints through the unmodified author evaluator."""
import argparse
import hashlib
import json
import sys
import time
import platform
import subprocess
from pathlib import Path

sys.dont_write_bytecode = True

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['cha', 'act', 'tvr'], required=True)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--multi', action='store_true')
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
source = root / 'ICML26-Holmes' / ('multisrc' if args.multi else 'src')
sys.path.insert(0, str(source))

import torch
from Configs.builder import get_configs
from Datasets.builder import get_datasets
from Models.builder import get_models
from Utils.utils import set_seed
from Validations.builder import get_validations

torch.set_num_threads(4)
checkpoint = torch.load(args.checkpoint, map_location='cpu')
cfg = get_configs(args.dataset).copy()
differences = {key: [value, checkpoint['config'].get(key)] for key, value in cfg.items()
               if key in checkpoint['config'] and value != checkpoint['config'][key]}
cfg['data_root'] = str(root / 'reproduction' / 'data')
cfg['root'] = str(root / 'reproduction' / 'results')
cfg['num_workers'] = 0
set_seed(cfg['seed'])
cfg, _, val_videos, val_queries, test_videos, test_queries = get_datasets(cfg)
model = get_models(cfg)
model.load_state_dict(checkpoint['state_dict'], strict=True)
del checkpoint['optimizer']
model = model.cuda().eval()
criterion = get_validations(cfg)
with args.checkpoint.open('rb') as checkpoint_file:
    checkpoint_hash = hashlib.file_digest(checkpoint_file, 'sha256').hexdigest()
result = {'dataset': args.dataset, 'multi': args.multi, 'torch': torch.__version__,
          'python': platform.python_version(), 'platform': platform.platform(),
          'source_commit': subprocess.check_output(['git', '-C', str(root / 'ICML26-Holmes'), 'rev-parse', 'HEAD'], text=True).strip(),
          'gpu': torch.cuda.get_device_name(), 'checkpoint_epoch': checkpoint['epoch'],
          'checkpoint_reported_metrics': checkpoint['model_val'], 'config': cfg,
          'checkpoint_config_differences': differences,
          'checkpoint_sha256': checkpoint_hash,
          'splits': {}}
for name, videos, queries in [('val', val_videos, val_queries), ('test', test_videos, test_queries)]:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.monotonic()
    with torch.no_grad():
        metrics = criterion(model, videos, queries)
    torch.cuda.synchronize()
    result['splits'][name] = {'queries': len(queries.dataset), 'videos': len(videos.dataset),
                              'metrics': metrics[:5], 'seconds': time.monotonic() - start,
                              'peak_allocated_gpu_bytes': torch.cuda.max_memory_allocated()}
    print(name, result['splits'][name], flush=True)
    output = root / 'reproduction' / 'results' / (args.dataset + ('-multi' if args.multi else '') + '-metrics.json')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
