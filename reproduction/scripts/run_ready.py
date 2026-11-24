"""Run each complete official archive through preparation and author evaluation."""
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
python = Path(sys.executable)
jobs = [
    ('charades', 'cha', 'charades-parallel.zip', 3267453090, 'Holmes-cha-parallel.zip', False),
    ('activitynet', 'act', 'activitynet.zip', 18527330684, 'Holmes-act.zip', False),
    ('tvr', 'tvr', 'tvr.zip', 17787514489, 'Holmes-tvr-multi.zip', True),
    ('tvr', 'tvr', 'tvr.zip', 17787514489, 'Holmes-tvr.zip', False),
]

def run(command, log):
    print('RUN', command, flush=True)
    with log.open('w', encoding='utf-8') as output:
        result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Exit {result.returncode}: {log}')

for dataset, code, archive_name, archive_size, weight_name, multi in jobs:
    label = code + ('-multi' if multi else '')
    data = root / 'downloads' / archive_name
    weights = root / 'downloads' / weight_name
    while not (data.exists() and data.stat().st_size == archive_size and zipfile.is_zipfile(data)
               and weights.exists() and zipfile.is_zipfile(weights)):
        time.sleep(30)
    manifest = root / 'results' / (dataset + '-data-manifest.json')
    if not manifest.exists():
        run([str(python), str(root / 'scripts' / 'prepare_data.py'), str(data), dataset],
            root / 'results' / (dataset + '-preparation.log'))
    with zipfile.ZipFile(weights) as archive:
        checkpoints = [entry for entry in archive.infolist() if entry.filename.endswith('.ckpt')]
        assert len(checkpoints) == 1, [entry.filename for entry in checkpoints]
        target = root / 'checkpoints' / (label + '.ckpt')
        target.parent.mkdir(exist_ok=True)
        with archive.open(checkpoints[0]) as source, target.open('wb') as output:
            while block := source.read(1024 * 1024):
                output.write(block)
    command = [str(python), str(root / 'scripts' / 'evaluate.py'), '--dataset', code,
               '--checkpoint', str(target)]
    if multi:
        command.append('--multi')
    run(command, root / 'results' / (label + '-evaluation.log'))
    print('EVALUATION_COMPLETE', label, flush=True)
run([str(python), str(root / 'scripts' / 'compare_results.py')], root / 'results' / 'comparison.log')
print('ALL_EVALUATIONS_COMPLETE', flush=True)
