"""Install immutable, version-matched frontend archives without pip/network resolution."""
import argparse,json,shutil,sys,tarfile
from pathlib import Path,PurePosixPath
from download_xiaoyou_assets import download

def extract(archive,target):
 target=Path(target).resolve();target.mkdir(parents=True,exist_ok=True)
 with tarfile.open(archive,'r:gz') as stream:
  members=stream.getmembers();assert sum(m.size for m in members)<512*1024**2
  for m in members:
   name=PurePosixPath(m.name)
   if name.is_absolute() or '..' in name.parts or not (m.isfile() or m.isdir()):raise ValueError('Invalid runtime archive member')
   path=target/str(name)
   if target not in path.resolve().parents:raise ValueError('Runtime path escapes destination')
   if m.isdir():path.mkdir(parents=True,exist_ok=True);continue
   path.parent.mkdir(parents=True,exist_ok=True)
   with stream.extractfile(m) as source,path.open('wb') as output:shutil.copyfileobj(source,output)

def main():
 p=argparse.ArgumentParser();p.add_argument('--jp-version',required=True);a=p.parse_args()
 version={'6':'61','61':'61','5':'511','511':'511'}.get(a.jp_version)
 if not version:raise ValueError('JP_VERSION must be 61/6 or 511/5')
 manifest=json.loads(Path('/opt/runtime-manifests/xiaoyou_runtime_'+version+'.json').read_text())
 assert '.'.join(map(str,sys.version_info[:2]))==manifest['python_abi']
 import numpy
 assert numpy.__version__==manifest['numpy_version']
 target=Path('/opt/runtime-downloads');download(manifest,target)
 for entry in manifest['files']:extract(target/entry['path'],'/opt/xiaoyou-python')
 shutil.rmtree(target)
if __name__=='__main__':main()
