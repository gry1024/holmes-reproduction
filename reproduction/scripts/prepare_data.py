"""Extract original features and convert BigFile rows without changing values/order."""
import argparse
import ast
import hashlib
import json
import zipfile
import fcntl
from pathlib import Path

import h5py
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('archive', type=Path)
parser.add_argument('dataset', choices=['charades', 'activitynet', 'tvr'])
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
destination = root / 'data' / args.dataset
destination.mkdir(parents=True, exist_ok=True)
# Preparation can run ahead of the sequential evaluator. Serialize writers to
# the same dataset if the evaluator reaches it before preparation has finished.
preparation_lock = (root / 'results' / (args.dataset + '-preparation.lock')).open('a')
fcntl.flock(preparation_lock, fcntl.LOCK_EX)
manifest_path = root / 'results' / (args.dataset + '-data-manifest.json')
if manifest_path.exists():
    existing = json.loads(manifest_path.read_text())
    if (existing.get('archive') == str(args.archive)
            and existing.get('archive_crc_verified')
            and existing.get('conversion', {}).get('all_values_and_frame_order_verified')):
        print('ALREADY_PREPARED', args.dataset, flush=True)
        raise SystemExit(0)
manifest = {'archive': str(args.archive), 'files': {}, 'splits': {}}
with args.archive.open('rb') as source:
    manifest['archive_sha256'] = hashlib.file_digest(source, 'sha256').hexdigest()
with zipfile.ZipFile(args.archive) as archive:
    invalid_member = archive.testzip()
    assert invalid_member is None, f'Archive CRC failure: {invalid_member}'
    manifest['archive_crc_verified'] = True
    parts = args.archive.with_suffix(args.archive.suffix + '.parts')
    if parts.exists():
        assert parts.resolve().parent == (root / 'downloads').resolve()
        for part in parts.iterdir():
            assert part.is_file() and part.stem.isdecimal(), part
            part.unlink()
        parts.rmdir()
    for member in archive.infolist():
        if member.is_dir() or 'clip_vit' in member.filename.lower():
            continue
        relative = Path(member.filename)
        if relative.parts[0] == args.dataset:
            relative = Path(*relative.parts[1:])
        target = destination / relative
        assert target.resolve().is_relative_to(destination.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.stat().st_size != member.file_size:
            with archive.open(member) as source, target.open('wb') as output:
                while block := source.read(1024 * 1024):
                    output.write(block)
        with target.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        manifest['files'][str(relative)] = {'bytes': target.stat().st_size, 'sha256': digest}
        print('EXTRACTED', relative, flush=True)
for captions in destination.rglob('*.caption.txt'):
    lines = captions.read_text().splitlines()
    query_ids = [line.split(' ', 1)[0] for line in lines]
    video_ids = {query.split('#')[0] for query in query_ids}
    manifest['splits'][str(captions.relative_to(destination))] = {
        'queries': len(lines), 'videos': len(video_ids),
        'query_set_sha256': hashlib.sha256('\n'.join(sorted(query_ids)).encode()).hexdigest(),
        'video_set_sha256': hashlib.sha256('\n'.join(sorted(video_ids)).encode()).hexdigest(),
        'caption_set_sha256': hashlib.sha256('\n'.join(sorted(lines)).encode()).hexdigest(),
        'sha256': hashlib.sha256(captions.read_bytes()).hexdigest()}
feature_dir = next(destination.rglob('feature.bin')).parent
rows, width = map(int, (feature_dir / 'shape.txt').read_text().split())
names = (feature_dir / 'id.txt').read_text(encoding='ISO-8859-1').split()
assert len(names) == rows
assert (feature_dir / 'feature.bin').stat().st_size == rows * width * 4
indices = {name: index for index, name in enumerate(names)}
video_frames = ast.literal_eval((feature_dir / 'video2frames.txt').read_text())
features = np.memmap(feature_dir / 'feature.bin', mode='r', dtype=np.float32, shape=(rows, width))
converted = feature_dir / 'feature.hdf5'
with h5py.File(converted, 'w') as output:
    for video, frames in video_frames.items():
        selected = [indices[name] for name in frames]
        # The referenced FAWL converter builds NumPy arrays from Python floats.
        # Preserve its float64 intermediate precision before author preprocessing.
        output.create_dataset(video, data=features[selected].astype(np.float64))
with h5py.File(converted, 'r') as output:
    assert len(output) == len(video_frames)
    for video, frames in video_frames.items():
        assert np.array_equal(output[video][...], features[[indices[name] for name in frames]])
manifest['conversion'] = {'videos': len(video_frames), 'rows': rows, 'width': width,
                          'all_values_and_frame_order_verified': True}
(root / 'results' / (args.dataset + '-data-manifest.json')).write_text(json.dumps(manifest, indent=2))
print('PREPARED', args.dataset, flush=True)
