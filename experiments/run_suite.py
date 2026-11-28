"""Run the three variants serially on one GPU; preserve failures and checkpoints."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--work', type=Path, required=True)
p.add_argument('--gpu', required=True, help='Physical GPU UUID')
p.add_argument('--variants', nargs='+', choices=['baseline', 'pooling', 'gating', 'combined'],
               default=['pooling', 'gating', 'combined'])
p.add_argument('--queue-name', choices=['queue', 'baseline-queue'], default='queue')
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
a.work.mkdir(parents=True, exist_ok=True)
lock = (a.work/f'{a.queue_name}.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1')


def status(**values):
    values.update(updated_at=time.time(), gpu=a.gpu)
    temp = a.work/f'{a.queue_name}-status.tmp'
    temp.write_text(json.dumps(values, indent=2))
    temp.replace(a.work/f'{a.queue_name}-status.json')
    print(json.dumps(values), flush=True)


jobs = [('cha','charades',False), ('act','activitynet',False), ('tvr','tvr',False), ('tvr','tvr',True)]
failures = []
for dataset, folder, multi in jobs:
    label = dataset + ('-multi' if multi else '')
    ready = a.work/'ready'/f'{folder}.json'
    while not ready.exists():
        status(status='waiting_for_data', dataset=label, failures=failures)
        time.sleep(30)
    for variant in a.variants:
        out = a.work/'runs'/label/variant
        if (out/'result.json').exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        # Do not launch another task on a GPU occupied by other compute jobs.
        while True:
            listing = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'], text=True)
            if not any(a.gpu in row for row in listing.splitlines()):
                break
            status(status='waiting_for_gpu', dataset=label, variant=variant, failures=failures)
            time.sleep(30)
        command = [sys.executable, '-u', str(root/'experiments'/'train.py'),
                   '--dataset',dataset,'--variant',variant,'--data-root',str(a.work/'data'),'--output',str(out)]
        if multi:
            command.append('--multi')
        status(status='running', dataset=label, variant=variant, log=str(out/'train.log'), failures=failures)
        with (out/'train.log').open('a') as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            failures.append(dict(dataset=label, variant=variant, exit_code=result.returncode))
        status(status='task_finished', dataset=label, variant=variant, exit_code=result.returncode, failures=failures)
status(status='finished_with_failures' if failures else 'completed', failures=failures)
