"""Wait for the baseline queue, then scan its four best models on the released GPU."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
p=argparse.ArgumentParser()
p.add_argument('--work',type=Path,required=True)
p.add_argument('--gpu',required=True)
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
out=a.work/'experiments/results/fusion-sensitivity-retrained'
out.mkdir(parents=True,exist_ok=True)
lock=(a.work/'retrained-fusion.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def status(state,**extra):
    info=dict(status=state,updated_at=time.time(),gpu=a.gpu,**extra)
    temp=a.work/'retrained-fusion-status.tmp'
    temp.write_text(json.dumps(info,indent=2))
    temp.replace(a.work/'retrained-fusion-status.json')
    print(json.dumps(info),flush=True)
jobs=[('cha',False),('act',False),('tvr',False),('tvr',True)]
labels=[d+('-multi' if m else '') for d,m in jobs]
try:
    while True:
        missing=[d for d in labels if not (a.work/'runs'/d/'baseline/result.json').exists()]
        qpath=a.work/'baseline-queue-status.json'
        q=json.loads(qpath.read_text()) if qpath.exists() else {}
        if q.get('status')=='finished_with_failures' and missing:
            raise RuntimeError(f'Baseline queue failed; missing: {missing}')
        if not missing and q.get('status')=='completed': break
        status('waiting_for_baselines',remaining=missing)
        time.sleep(30)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    failures=[]
    for dataset,multi in jobs:
        label=dataset+('-multi' if multi else '')
        target=out/f'{label}.json'
        if target.exists() and json.loads(target.read_text()).get('baseline_matches_checkpoint'):continue
        while a.gpu in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True):
            status('waiting_for_gpu',dataset=label)
            time.sleep(30)
        command=[sys.executable,'-B','-u',str(root/'experiments/sweep_fusion.py'),'--dataset',dataset,
                 '--checkpoint',str(a.work/'runs'/label/'baseline/best.pt'),'--data-root',str(a.work/'data'),'--output',str(out)]
        if multi:command.append('--multi')
        status('running',dataset=label)
        with (out/f'{label}.log').open('a') as log:
            r=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:failures.append(label)
    if failures:raise RuntimeError(f'Sweep failures: {failures}')
    status('plotting')
    subprocess.run([sys.executable,'-B',str(root/'experiments/plot_fusion_sweep.py'),'--retrained','--output',str(out)],check=True)
    status('completed',output=str(out))
except Exception as error:
    status('failed',error=str(error))
    raise
