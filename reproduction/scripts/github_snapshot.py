"""Transfer and restore the binary portion of the Holmes workspace snapshot."""

import argparse
import hashlib
import json
import shutil
import subprocess
import time
import zipfile
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[2]
REPO = 'gry1024/holmes-reproduction'
TAG = 'workspace-files-v1'
WORK = ROOT / 'reproduction' / '.github-transfer'
STATE = WORK / 'snapshot-manifest.json'
MANIFEST = STATE.name
BUNDLE = 'small-files.zip'
CHUNK_SIZE = 1024 ** 3
BUFFER_SIZE = 8 * 1024 ** 2
ROOTS = ('reproduction/checkpoints', 'reproduction/downloads', 'reproduction/data',
         'reproduction/archive/待清理')


def gh(*args, capture=False):
    result = subprocess.run(['gh', *args], check=True, text=True,
                            encoding='utf-8', capture_output=capture)
    return result.stdout


def assets():
    release = gh('api', f'repos/{REPO}/releases/tags/{TAG}', '--jq', '.id', capture=True).strip()
    pages = json.loads(gh('api', '--paginate', '--slurp',
                          f'repos/{REPO}/releases/{release}/assets?per_page=100', capture=True))
    return {asset['name']: asset for page in pages for asset in page}


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(BUFFER_SIZE), b''):
            digest.update(block)
    return digest.hexdigest()


def entry(path):
    return {'path': path.relative_to(ROOT).as_posix(), 'size': path.stat().st_size}


def save_state(manifest):
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(STATE)


def make_manifest():
    paths = [path for directory in ROOTS for path in sorted((ROOT / directory).rglob('*'))]
    files = [path for path in paths if path.is_file()]
    bundled = [path for path in files if path.stat().st_size < 100 * 1024 ** 2]
    large = [path for path in files if path not in bundled]
    return {'version': 1, 'repo': REPO, 'tag': TAG,
            'empty_directories': [p.relative_to(ROOT).as_posix() for p in paths
                                  if p.is_dir() and not any(p.iterdir())],
            'bundled': [entry(path) for path in bundled],
            'files': [dict(entry(path), parts=[]) for path in large]}


def same_asset(remote, part):
    return (remote.get('state') == 'uploaded' and remote.get('size') == part['size']
            and remote.get('digest') == 'sha256:' + part['sha256'])


def upload_asset(path, part, remote):
    for attempt in range(4):
        current = remote.get(path.name, {})
        if same_asset(current, part):
            return
        if current and current.get('state') == 'uploaded':
            raise RuntimeError(f'Existing asset differs: {path.name}')
        if current:
            gh('api', '-X', 'DELETE', f'repos/{REPO}/releases/assets/{current["id"]}')
        try:
            gh('release', 'upload', TAG, str(path), '-R', REPO)
        except subprocess.CalledProcessError:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
        remote.update(assets())
        if same_asset(remote.get(path.name, {}), part):
            print(f'Uploaded and verified {path.name} ({part["size"]:,} bytes)', flush=True)
            return
    raise RuntimeError(f'Could not verify uploaded asset: {path.name}')


def asset_info(path):
    return {'name': path.name, 'size': path.stat().st_size, 'sha256': file_hash(path)}


def upload():
    WORK.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(STATE.read_text(encoding='utf-8')) if STATE.exists() else make_manifest()
    save_state(manifest)
    remote = assets()
    for index, file in enumerate(manifest['files'], start=1):
        source_path = ROOT / file['path']
        if source_path.stat().st_size != file['size']:
            raise RuntimeError(f'Source size changed: {source_path}')
        with source_path.open('rb') as source:
            for offset in range(0, file['size'], CHUNK_SIZE):
                number = offset // CHUNK_SIZE
                size = min(CHUNK_SIZE, file['size'] - offset)
                path = WORK / f'{index:03d}-{source_path.name}.part{number:03d}'
                if number < len(file['parts']) and same_asset(remote.get(path.name, {}), file['parts'][number]):
                    source.seek(offset + size)
                    continue
                with path.open('wb') as target:
                    remaining = size
                    while remaining:
                        block = source.read(min(BUFFER_SIZE, remaining))
                        if not block:
                            raise RuntimeError(f'Source shortened: {source_path}')
                        target.write(block)
                        remaining -= len(block)
                part = asset_info(path)
                if number == len(file['parts']):
                    file['parts'].append(part)
                elif file['parts'][number] != part:
                    raise RuntimeError(f'Source content changed: {source_path}')
                save_state(manifest)
                upload_asset(path, part, remote)
                path.unlink()
    if 'bundle' not in manifest or not same_asset(remote.get(BUNDLE, {}), manifest['bundle']):
        bundle = WORK / BUNDLE
        with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_STORED) as archive:
            for file in manifest['bundled']:
                source = ROOT / file['path']
                if source.stat().st_size != file['size']:
                    raise RuntimeError(f'Source size changed: {source}')
                archive.write(source, file['path'])
        manifest['bundle'] = asset_info(bundle)
        save_state(manifest)
        upload_asset(bundle, manifest['bundle'], remote)
        bundle.unlink()
    upload_asset(STATE, asset_info(STATE), remote)
    print('Full snapshot uploaded and verified.', flush=True)


def destination(relative):
    path = PurePosixPath(relative)
    if path.as_posix() != relative or '..' in path.parts or '\\' in relative or ':' in relative:
        raise ValueError(f'Invalid snapshot path: {relative}')
    target = (ROOT / relative).resolve()
    if not target.is_relative_to(ROOT.resolve()) or not any(
            target.is_relative_to((ROOT / directory).resolve()) for directory in ROOTS):
        raise ValueError(f'Snapshot path is outside the expected directories: {relative}')
    return target


def download(name, directory):
    gh('release', 'download', TAG, '-R', REPO, '--pattern', name, '--dir', str(directory))
    return directory / name


def download_checked(part, directory):
    path = download(part['name'], directory)
    if path.stat().st_size != part['size'] or file_hash(path) != part['sha256']:
        raise RuntimeError(f'Asset verification failed: {path.name}')
    return path


def restore():
    WORK.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=WORK) as tmp:
        directory = Path(tmp)
        manifest = json.loads(download(MANIFEST, directory).read_text(encoding='utf-8'))
        targets = [destination(file['path']) for file in manifest['files'] + manifest['bundled']]
        existing = [path for path in targets if path.exists()]
        if existing:
            raise FileExistsError(f'Restore into a fresh clone. First existing path: {existing[0]}')
        for file in manifest['files']:
            target = destination(file['path'])
            temporary = directory / 'restoring.tmp'
            with temporary.open('wb') as output:
                for part in file['parts']:
                    path = download_checked(part, directory)
                    with path.open('rb') as source:
                        shutil.copyfileobj(source, output, BUFFER_SIZE)
                    path.unlink()
            if temporary.stat().st_size != file['size']:
                raise RuntimeError(f'Restored size mismatch: {target}')
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary.replace(target)
            print(f'Restored {file["path"]}', flush=True)
        bundle = download_checked(manifest['bundle'], directory)
        with zipfile.ZipFile(bundle) as archive:
            for file in manifest['bundled']:
                target = destination(file['path'])
                if archive.getinfo(file['path']).file_size != file['size']:
                    raise RuntimeError(f'Bundled size mismatch: {target}')
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(file['path']) as source, target.open('xb') as output:
                    shutil.copyfileobj(source, output, BUFFER_SIZE)
        for relative in manifest['empty_directories']:
            destination(relative).mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['upload', 'restore'])
    args = parser.parse_args()
    (upload if args.action == 'upload' else restore)()
