"""Freeze only the existing frontend package dependency closure, without models or credentials."""
import argparse,importlib.metadata as metadata,json,tarfile
from pathlib import Path
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
 roots=['misaki','spacy','phonemizer-fork','espeakng-loader','num2words','cn2an','jieba','ordered-set','pypinyin']
 pending=list(roots);seen=set();records=[];paths={}
 while pending:
  name=canonicalize_name(pending.pop())
  if name in seen or name in ['torch','numpy']:continue
  seen.add(name);d=metadata.distribution(name);base=Path(d.locate_file('')).resolve()
  records.append({'name':name,'version':d.version})
  for raw in d.requires or []:
   req=Requirement(raw)
   if req.marker is None or req.marker.evaluate({'extra':''}):pending.append(req.name)
  for f in d.files or []:
   if '..' in f.parts or f.name=='direct_url.json' or '__pycache__' in f.parts or f.suffix=='.pyc':continue
   source=Path(d.locate_file(f)).resolve()
   if source.is_file() and base in source.parents:paths[str(f)]=source
 output=Path(a.output);assert not output.exists()
 with tarfile.open(output,'w:gz') as t:
  for name,source in sorted(paths.items()):
   assert not Path(name).is_absolute() and '..' not in Path(name).parts
   info=t.gettarinfo(str(source),arcname=name);info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0
   assert info.isfile()
   with source.open('rb') as stream:t.addfile(info,stream)
 report={'packages':sorted(records,key=lambda x:x['name']),'files':len(paths),'numpy_from_base':metadata.version('numpy')}
 output.with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
