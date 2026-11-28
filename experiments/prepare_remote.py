"""Wait for atomically transferred archives, then use the audited converter."""
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
work = Path(sys.argv[1])
(work/'ready').mkdir(exist_ok=True)
for folder, filename in [('charades','charades-parallel.zip'), ('activitynet','activitynet.zip'), ('tvr','tvr.zip')]:
    archive = work/'downloads'/filename
    marker = work/'ready'/f'{folder}.json'
    if marker.exists():
        continue
    while not archive.exists() or archive.with_suffix(archive.suffix+'.uploading').exists():
        time.sleep(30)
    with (work/'results'/f'{folder}-preparation.log').open('w') as log:
        result = subprocess.run([sys.executable,str(root/'reproduction/scripts/prepare_data.py'),str(archive),folder],stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Preparation failed for {folder}, see log')
    marker.write_text(json.dumps(dict(dataset=folder, prepared_at=time.time())))
    print('PREPARED', folder, flush=True)
