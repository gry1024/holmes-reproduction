"""Download a public author artifact with verified, resumable HTTP ranges."""
import concurrent.futures
import html
import re
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

file_id, destination = sys.argv[1:3]
target = Path(destination)
session = requests.Session()
response = session.get(f'https://drive.google.com/uc?export=download&id={file_id}', timeout=60, stream=True)
if 'text/html' in response.headers.get('Content-Type', ''):
    page = response.text
    fields = dict(re.findall(r'name="([^"]+)" value="([^"]*)"', page))
    url = 'https://drive.usercontent.google.com/download'
    request = requests.Request('GET', url, params=fields).prepare()
    url = request.url
else:
    url = response.url
response.close()
probe = session.get(url, headers={'Range': 'bytes=0-0'}, timeout=90, stream=True)
assert probe.status_code == 206, (probe.status_code, probe.headers)
total = int(probe.headers['Content-Range'].split('/')[-1])
probe.close()
parts = target.with_suffix(target.suffix + '.parts')
parts.mkdir(parents=True, exist_ok=True)
size = 8 * 1024 * 1024

def fetch(index):
    start = index * size
    end = min(total, start + size) - 1
    part = parts / str(index)
    if part.exists() and part.stat().st_size == end - start + 1:
        return
    for attempt in range(8):
        try:
            offset = part.stat().st_size if part.exists() else 0
            resume = start + offset
            if os.name == 'nt':
                pending = part.with_suffix('.pending')
                headers = part.with_suffix('.headers')
                if pending.exists() and headers.exists():
                    previous_ranges = re.findall(r'(?im)^content-range:\s*([^\r\n]+)', headers.read_text())
                    if (previous_ranges and previous_ranges[-1].strip() == f'bytes {resume}-{end}/{total}'
                            and 0 < pending.stat().st_size <= end - resume + 1):
                        with part.open('ab') as output, pending.open('rb') as source:
                            while block := source.read(1024 * 1024):
                                output.write(block)
                        pending.unlink()
                        headers.unlink()
                        resume = start + part.stat().st_size
                        if resume == end + 1:
                            return
                transfer = subprocess.run(['curl.exe', '--silent', '--show-error', '--fail',
                    '--ssl-revoke-best-effort', '--noproxy', '*', '--location',
                    '--connect-timeout', '30', '--max-time', '300',
                    '--max-filesize', str(end - resume + 1), '--range', f'{resume}-{end}',
                    '--dump-header', str(headers), '--output', str(pending), url])
                ranges = re.findall(r'(?im)^content-range:\s*([^\r\n]+)', headers.read_text())
                assert ranges[-1].strip() == f'bytes {resume}-{end}/{total}'
                assert 0 < pending.stat().st_size <= end - resume + 1
                with part.open('ab') as output, pending.open('rb') as source:
                    while block := source.read(1024 * 1024):
                        output.write(block)
                pending.unlink()
                headers.unlink()
                if part.stat().st_size != end - start + 1:
                    raise IOError(f'Incomplete range, retained valid prefix; curl exit {transfer.returncode}')
                return
            with requests.get(url, headers={'Range': f'bytes={resume}-{end}'}, stream=True, timeout=(30, 90)) as result:
                assert result.status_code == 206
                assert result.headers['Content-Range'] == f'bytes {resume}-{end}/{total}'
                with part.open('ab') as output:
                    for block in result.iter_content(65536):
                        output.write(block)
            assert part.stat().st_size == end - start + 1
            return
        except Exception as error:
            print(f'{target.name}: part {index} retry {attempt + 1}: {type(error).__name__}', flush=True)
            if attempt == 7:
                raise
            time.sleep(2 + attempt)

count = (total + size - 1) // size
print(f'{target.name}: {total} bytes in {count} parts', flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    for done, future in enumerate(concurrent.futures.as_completed([pool.submit(fetch, i) for i in range(count)]), 1):
        future.result()
        print(f'{target.name}: {done}/{count}', flush=True)
with target.open('wb') as output:
    for index in range(count):
        with (parts / str(index)).open('rb') as source:
            while block := source.read(1024 * 1024):
                output.write(block)
assert target.stat().st_size == total
print('COMPLETE', flush=True)
