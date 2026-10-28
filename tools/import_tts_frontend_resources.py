"""Rebuild public Chinese frontend data from SHA-verified, locally downloaded source snapshots."""
import argparse,hashlib,json,re,shutil
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--sources',type=Path,required=True)
args=parser.parse_args()
repo=Path(__file__).resolve().parents[1];ROOT=repo.parent
runtime=repo/'perception/plugins/moss_tts_runtime';public=args.sources.resolve()
manifest=json.loads((public/'manifest.json').read_text())
for name,meta in manifest.items():
 if name not in ('wetext','glossary','cmudict'):raise ValueError('Unexpected source')
 for file,entry in meta['files'].items():
  path=Path(file)
  if path.is_absolute() or '..' in path.parts:raise ValueError('Invalid source path')
  if hashlib.sha256((public/name/path).read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Source checksum mismatch: '+file)
def tsv(path):return dict(line.split('\t',1) for line in path.read_text().splitlines() if line and not line.startswith('#'))
units=tsv(public/'wetext/tn/chinese/data/measure/units_en.tsv')
exclude={'ds':'upstream says milliseconds; ds is not ms','ω':'lowercase omega is a variable, not the ohm symbol Ω','tl':'regional/ambiguous unit outside the supported engineering inventory'}
canonical={'gb':'GB','ghz':'GHz','gpa':'GPa','gwh':'GWh','gw':'GW','gy':'Gy','hz':'Hz','khz':'kHz','kpa':'kPa','kwh':'kWh','kw':'kW','mhz':'MHz','mpa':'MPa','mv':'mV','mw':'mW','pa':'Pa','sv':'Sv','tb':'TB','tj':'TJ','v':'V','ml':'mL'}
units={canonical.get(k,k):v for k,v in units.items() if k not in exclude}
units.update({'amu':'原子质量单位','km':'千米','Sv':'希沃特','%':'百分之','％':'百分之','‰':'千分之','‱':'万分之','K':'开尔文','A':'安培','W':'瓦特','N':'牛顿','J':'焦耳','C':'库仑','F':'法拉','H':'亨利','S':'西门子','Ω':'欧姆','Bq':'贝克勒尔','Wb':'韦伯','T':'特斯拉','lm':'流明','lx':'勒克斯','cd':'坎德拉','sr':'球面度','t':'吨','bit':'比特','B':'字节','KB':'千字节','MB':'兆字节','GB':'吉字节','TB':'太字节','KiB':'千二进制字节','MiB':'兆二进制字节','GiB':'吉二进制字节','TiB':'太二进制字节','Wh':'瓦时','kWh':'千瓦时','MWh':'兆瓦时','kW':'千瓦','MW':'兆瓦','GW':'吉瓦','mW':'毫瓦','μW':'微瓦','V':'伏特','mV':'毫伏','μV':'微伏','kV':'千伏','MV':'兆伏','A':'安培','mA':'毫安','μA':'微安','kA':'千安','Pa':'帕斯卡','hPa':'百帕','kPa':'千帕','MPa':'兆帕','GPa':'吉帕','kN':'千牛顿','mJ':'毫焦耳','kJ':'千焦耳','MJ':'兆焦耳','kΩ':'千欧姆','MΩ':'兆欧姆','μF':'微法','nF':'纳法','pF':'皮法','mH':'毫亨','μH':'微亨','GHz':'吉赫兹','MHz':'兆赫兹','kHz':'千赫兹','THz':'太赫兹','Hz':'赫兹','mHz':'毫赫兹','kbps':'千比特每秒','Mbps':'兆比特每秒','Gbps':'吉比特每秒','ms':'毫秒','μs':'微秒','ns':'纳秒','ps':'皮秒','μg':'微克','μm':'微米','m²':'平方米','m³':'立方米','s²':'平方秒','°':'度'})
for key,value in list(units.items()):
 if 'μ' in key:units[key.replace('μ','µ')]=value
 if key.endswith(('²','³')):units[key[:-1]+'^'+{'²':'2','³':'3'}[key[-1]]]=value
operators=tsv(public/'wetext/tn/chinese/data/math/operator.tsv')
operators.update({'==':'等于','!=':'不等于','≠':'不等于','≈':'约等于','≃':'约等于','>':'大于','<':'小于','~':'到','～':'到',':':'比','∶':'比'})
functions={'sin':'正弦','cos':'余弦','tan':'正切','cot':'余切','sec':'正割','csc':'余割','asin':'反正弦','acos':'反余弦','atan':'反正切','ln':'自然对数','log':'对数','exp':'指数函数','abs':'绝对值','max':'最大值','min':'最小值'}
greek=dict(zip('αβγδεζηθικλμνξοπρστυφχψω',['阿尔法','贝塔','伽马','德尔塔','艾普西龙','泽塔','伊塔','西塔','约塔','卡帕','拉姆达','缪','纽','克西','奥密克戎','派','柔','西格玛','陶','宇普西龙','斐','希','普西','欧米伽']))
greek.update({k.upper():v for k,v in list(greek.items())})
greek_commands=dict(zip('alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega'.split(),'αβγδεζηθικλμνξοπρστυφχψω'))
wetext=json.loads((public/'wetext/commit.json').read_text())['sha']
data=dict(schema_version=1,units=dict(sorted(units.items())),operators=operators,functions=functions,greek=greek,greek_commands=greek_commands,source_notes={'wetext_revision':wetext,'imported':['tn/chinese/data/math/operator.tsv','tn/chinese/data/measure/units_en.tsv'],'canonical_case_and_additions':'SI symbol case checked against https://www.nist.gov/pml/special-publication-330/sp-330-section-3 ; local engineering additions and standard mathematical readings are not upstream data','excluded_upstream_units':exclude})
(runtime/'zh_frontend_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
print('units',len(units),'operators',len(operators),'functions',len(functions),'Greek',len(greek))

manifest=json.loads((public/'manifest.json').read_text());assert set(manifest)=={'wetext','glossary','cmudict'}
for name,m in manifest.items():
 for file,entry in m['files'].items():assert hashlib.sha256((public/name/file).read_bytes()).hexdigest()==entry['sha256']
asset=runtime/'pronunciation_lexicon.json';data=json.loads(asset.read_text());entries=data['entries'];existing={(e['language'],e['term'].casefold()):e for e in entries}
# The previous Chinese 27-term A/B candidate was accepted by the user; do not activate foreign-language trials.
accepted=set('BERT|Transformer|Self-Attention|Multi-Head Attention|Cross-Attention|GPT|T5|Encoder|Decoder|Embedding|Tokenization|Positional Encoding|Layer Normalization|Feed-Forward Network|Residual Connection|Mixture of Experts|Diffusion Model|Vision Transformer|CLIP|Pre-training|Fine-tuning|LoRA|RLHF|DPO|RAG|Knowledge Distillation|Quantization'.split('|'))
for e in entries:
 if e['language']=='zh' and e['term'] in accepted:
  e['status']='approved'
  note=' User accepted the current listening comparison on 2026-10-28; not a guarantee for every context.'
  if note not in e['evidence']:e['evidence']+=note
pronunciations={}
for line in (public/'cmudict/cmudict.dict').read_text().splitlines():
 if not line or line.startswith(';;;'):continue
 name,phones=line.split(' ',1);pronunciations.setdefault(re.sub(r'\([0-9]+\)$','',name),phones.strip())
words={'BERT':'Bert','CLIP':'clip','RAG':'rag','BLEU':'bleu','ROUGE':'rouge','ONNX':'ONNX','FLUX':'Flux'}
initials={'vLLM','MoE','CoT','ToT','GPU','NPU','CPU','DCU','MLU','API','TTS','RTF','CNCF'}
nums=['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen']
def en_number(n):
 return nums[n] if n<20 else ['','','twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety'][n//10]+(' '+nums[n%10] if n%10 else '') if n<100 else ' '.join(nums[int(c)] for c in str(n))
def reading(name):
 if name in words:return words[name]
 if name in initials or re.fullmatch('[A-Z][A-Z0-9]{1,9}',name):
  return ' '.join(en_number(int(p)) if p.isdigit() else ' '.join(p.upper()) for p in re.findall('[A-Za-z]+|[0-9]+',name))
 value=name.replace('-', ' ')
 value=re.sub(r'\b(KV|GPU|API|AI|LLM|VRAM)\b',lambda m:' '.join(m[0]),value)
 return value
catalog=[];new=0
for file in sorted((public/'glossary/docs').glob('*.md')):
 text=file.read_text();category=file.stem
 sections=re.split(r'^### ',text,flags=re.M)[1:]
 for section in sections:
  heading=section.splitlines()[0];m=re.search(r'^\*\*英文\*\*：([^|\n]+)',section,re.M);assert m,(file,heading)
  en=m[1].strip();terms=[]
  for phrase in en.split(' / '):
   if '(' in phrase:
    outside,inside=re.fullmatch(r'(.+?)\s*\(([^()]*)\)',phrase).groups();terms.extend([outside.strip(),inside.strip()])
   else:terms.append(phrase.strip())
  # Only names and explicit acronym inventory are imported; definitions are not TTS readings.
  acronyms=set(re.findall(r'\b[A-Z][A-Z0-9]{1,9}\b',section))
  terms+=sorted(acronyms)
  terms=[t for t in terms if re.fullmatch(r'[A-Za-z][A-Za-z0-9 +.\-/]*',t)]
  terms=list(dict.fromkeys(terms));assert terms
  source=manifest['glossary']['repository']+'/blob/'+manifest['glossary']['revision']+'/docs/'+file.name
  catalog.append(dict(category=category,heading=heading,english=en,terms=terms,source=source))
  for term in terms:
   if term in ('Top','AI/LLM'):continue
   key=('zh',term.casefold());e=existing.get(key)
   if e:
    if source!=e.get('source'):
     e.setdefault('public_sources',[])
     if source not in e['public_sources']:e['public_sources'].append(source)
    # Existing pending product names retain their spelling; Qwen's version grammar remains separate.
    if e['status']=='pending' and not e.get('reading') and e.get('scope','token')=='token':e.update(reading=reading(term),status='approved',evidence='Public terminology inventory; deterministic text spelling policy. Not individual listening approval.')
   else:
    e=dict(term=term,language='zh',reading=reading(term),status='approved',source=source,evidence='Public AI/LLM glossary inventory, case-sensitive text reading policy; initialisms spelled, word identities preserved. Not per-term listening approval.',category=category)
    entries.append(e);existing[key]=e;new+=1
   if term in words and e['source'].startswith(manifest['glossary']['repository']) and e['reading']==' '.join(term):
    e['reading']=words[term]
    e['evidence']+=' CMUdict whole-word entry retained instead of initialism spelling.'
   phones={w:pronunciations[w.casefold()] for w in re.findall('[A-Za-z]+',term) if w.casefold() in pronunciations}
   if phones:e['pronunciation_reference']=dict(format='CMU ARPABET, metadata only; not passed to Nano',words=phones,revision=manifest['cmudict']['revision'])
# Existing domain acronyms use the same deterministic policy after regression testing.
for e in entries:
 if e['language']=='zh' and e['term'] in initials and e['status']=='trial':
  e.update(status='approved',evidence=e['evidence']+' Included by the public Chinese initialism policy; native pronunciation not established.')
assert len(catalog)==148,len(catalog)
asset.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
folder=repo/'docs/tts_frontend_resources';folder.mkdir(exist_ok=True)
(folder/'terminology_catalog.json').write_text(json.dumps(dict(source_terms=len(catalog),categories=12,entries=catalog),ensure_ascii=False,indent=2)+'\n')
(folder/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
shutil.copy2(public/'glossary/LICENSE',runtime/'GLOSSARY_LICENSE');shutil.copy2(public/'cmudict/LICENSE',runtime/'CMUDICT_LICENSE')
notice=runtime/'NOTICE.md'
prefix=notice.read_text().split('Chinese frontend public resources (2026-10-28):')[0].rstrip()
with notice.open('w') as f:
 f.write(prefix+'\n')
 f.write('\nChinese frontend public resources (2026-10-28):\n')
 for name,m in manifest.items():f.write('- '+name+': '+m['repository']+' @ '+m['revision']+'\n')
 f.write('WeTextProcessing operator/unit data: Copyright (c) 2022 Zhendong Peng, Apache-2.0 (LICENSE). Symbol case, SI additions and the bounded expression parser are local adaptations.\nAI/LLM glossary term inventory: Copyright (c) 2026 JingHao-Leon, MIT (GLOSSARY_LICENSE).\nSelected CMUdict pronunciation data: Copyright (C) 1993-2015 Carnegie Mellon University (CMUDICT_LICENSE). ARPABET is reference metadata, not a supported Nano input.\nPinned source-file SHA256 and the complete 148-item imported inventory: docs/tts_frontend_resources in the source repository. No external code is executed or downloaded at inference time.\n')
print('source records',len(catalog),'new entries',new,'total entries',len(entries))
