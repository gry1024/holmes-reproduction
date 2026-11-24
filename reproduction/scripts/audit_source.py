"""Record author-source identity, allowing Windows line-ending conversion only."""
import hashlib
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
repo = root.parent / 'ICML26-Holmes'

def git(*args):
    return subprocess.check_output(['git', '-C', str(repo), *args])

files = []
for relative in git('ls-files', '-z', 'src', 'multisrc').decode().split('\0'):
    if not relative.endswith('.py'):
        continue
    expected = git('show', 'HEAD:' + relative)
    actual = (repo / relative).read_bytes()
    same = expected.replace(b'\r\n', b'\n') == actual.replace(b'\r\n', b'\n')
    files.append({'file': relative, 'matches_commit_ignoring_crlf': same,
                  'local_sha256': hashlib.sha256(actual).hexdigest(),
                  'commit_sha256': hashlib.sha256(expected).hexdigest()})
result = {'commit': git('rev-parse', 'HEAD').decode().strip(),
          'all_author_python_sources_unchanged': all(item['matches_commit_ignoring_crlf'] for item in files),
          'files': files}
(root / 'results' / 'source-audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
assert result['all_author_python_sources_unchanged'], 'Author source differs from the recorded commit'
print(f'Verified {len(files)} author Python files against {result["commit"]}')
