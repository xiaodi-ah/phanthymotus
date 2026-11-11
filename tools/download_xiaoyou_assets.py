"""Download model assets from JuiceFS HTTP, checking bytes and SHA256."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import urllib.parse
import urllib.request

def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):value.update(block)
    return value.hexdigest()

def download(manifest,target):
    target=Path(target).resolve()
    base=manifest['base_url'].rstrip('/')+'/'
    if not base.startswith(('http://','https://')):raise ValueError('Invalid asset URL')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for item in manifest['files']:
        name=PurePosixPath(item['path'])
        if name.is_absolute() or '..' in name.parts or not name.parts or not re.fullmatch('[0-9a-f]{64}',item['sha256']) or item['bytes']<=0:raise ValueError('Invalid manifest entry')
        path=target/str(name)
        temporary=path.with_suffix(path.suffix+'.part')
        if any(target not in p.resolve().parents for p in [path,temporary]):raise ValueError('Asset path escapes destination')
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and path.stat().st_size==item['bytes'] and digest(path)==item['sha256']:continue
        size=0
        with opener.open(base+urllib.parse.quote(str(name)),timeout=30) as source,temporary.open('wb') as output:
            for block in iter(lambda:source.read(1<<20),b''):
                size+=len(block)
                if size>item['bytes']:raise RuntimeError('Asset exceeds expected size: '+str(name))
                output.write(block)
        if temporary.stat().st_size!=item['bytes'] or digest(temporary)!=item['sha256']:raise RuntimeError('Asset size/SHA mismatch: '+str(name))
        temporary.replace(path);print('Verified '+str(name),flush=True)
    (target/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--destination',required=True);args=parser.parse_args()
    manifest=json.loads(Path(args.manifest).read_text()); target=Path(args.destination)
    download(manifest,target)
if __name__=='__main__':main()
