"""Download immutable trial assets with size/SHA checks; no Git-hosted weights."""
import argparse,hashlib,json,re
from pathlib import Path,PurePosixPath
import urllib.request

def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):value.update(block)
    return value.hexdigest()

def download(manifest,target):
    base=manifest['base_url'].rstrip('/')+'/'
    if not base.startswith(('http://','https://')):raise ValueError('Invalid asset URL')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for item in manifest['files']:
        name=PurePosixPath(item['path'])
        if name.is_absolute() or '..' in name.parts or not name.parts or not re.fullmatch('[0-9a-f]{64}',item['sha256']) or item['bytes']<=0:raise ValueError('Invalid manifest entry')
        path=target/str(name);path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and path.stat().st_size==item['bytes'] and digest(path)==item['sha256']:continue
        temporary=path.with_suffix(path.suffix+'.part')
        with opener.open(base+str(name),timeout=30) as source,temporary.open('wb') as output:
            for block in iter(lambda:source.read(1<<20),b''):output.write(block)
        if temporary.stat().st_size!=item['bytes'] or digest(temporary)!=item['sha256']:raise RuntimeError('Asset size/SHA mismatch: '+str(name))
        temporary.replace(path);print('Verified '+str(name),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--destination',required=True);args=parser.parse_args()
    download(json.loads(Path(args.manifest).read_text()),Path(args.destination))
if __name__=='__main__':main()
