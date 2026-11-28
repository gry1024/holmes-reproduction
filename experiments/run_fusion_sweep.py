"""Run the four official checkpoints in isolated Python processes."""
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parents[1]
out=root/'experiments/results/fusion-sensitivity'
out.mkdir(parents=True,exist_ok=True)
for dataset,multi in [('cha',False),('act',False),('tvr',False),('tvr',True)]:
    label=dataset+('-multi' if multi else '')
    command=[sys.executable,'-B','-u',str(root/'experiments/sweep_fusion.py'),'--dataset',dataset]
    if multi:
        command.append('--multi')
    print('START',label,flush=True)
    with (out/f'{label}.log').open('w') as f:
        subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
    print('FINISHED',label,flush=True)
