"""Build the small runtime tables from pinned CLDR, SRE and NeMo snapshots."""
import argparse,hashlib,json,re,shutil,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUNTIME=ROOT/'perception/plugins/moss_tts_runtime'
LANGUAGES=('en','ja','de','fr','ru','es')
UNITS={
 'm':'length-meter','km':'length-kilometer','cm':'length-centimeter','mm':'length-millimeter','µm':'length-micrometer','μm':'length-micrometer','nm':'length-nanometer',
 's':'duration-second','ms':'duration-millisecond','µs':'duration-microsecond','μs':'duration-microsecond','ns':'duration-nanosecond','min':'duration-minute','h':'duration-hour',
 'g':'mass-gram','kg':'mass-kilogram','mg':'mass-milligram','t':'mass-tonne','L':'volume-liter','l':'volume-liter','mL':'volume-milliliter',
 'V':'electric-volt','A':'electric-ampere','mA':'electric-milliampere','Ω':'electric-ohm','W':'power-watt','mW':'power-milliwatt','kW':'power-kilowatt','MW':'power-megawatt','GW':'power-gigawatt',
 'Hz':'frequency-hertz','kHz':'frequency-kilohertz','MHz':'frequency-megahertz','GHz':'frequency-gigahertz',
 'B':'digital-byte','kB':'digital-kilobyte','KB':'digital-kilobyte','MB':'digital-megabyte','GB':'digital-gigabyte','TB':'digital-terabyte','PB':'digital-petabyte',
 'bit':'digital-bit','kbit':'digital-kilobit','Mbit':'digital-megabit','Gbit':'digital-gigabit',
 '℃':'temperature-celsius','°C':'temperature-celsius','℉':'temperature-fahrenheit','°F':'temperature-fahrenheit','K':'temperature-kelvin',
 '%':'concentr-percent','％':'concentr-percent','‰':'concentr-permille','m²':'area-square-meter','km²':'area-square-kilometer','cm²':'area-square-centimeter','m³':'volume-cubic-meter',
 'Pa':'pressure-pascal','kPa':'pressure-kilopascal','MPa':'pressure-megapascal','bar':'pressure-bar','J':'energy-joule','kJ':'energy-kilojoule','kWh':'energy-kilowatt-hour','N':'force-newton','°':'angle-degree','rad':'angle-radian',
 'km/h':'speed-kilometer-per-hour','m/s':'speed-meter-per-second'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--sources',type=Path,required=True);a=p.parse_args();data={};used=[]
 expected=json.loads((a.sources/'manifest.json').read_text())
 def read(path):
  name,relative=path.relative_to(a.sources).parts[0],str(path.relative_to(a.sources)).split('/',1)[1]
  assert hashlib.sha256(path.read_bytes()).hexdigest()==expected[name]['files'][relative]['sha256'],str(path)
  used.append(path);return path.read_text(encoding='utf-8')
 for lang in LANGUAGES:
  root=ET.fromstring(read(a.sources/f'cldr/common/main/{lang}.xml'));en=ET.fromstring(read(a.sources/'cldr/common/main/en.xml'));units={}
  for symbol,key in UNITS.items():
   node=root.find(f".//unitLength[@type='long']/unit[@type='{key}']")
   if node is None:continue
   patterns={x.attrib['count']:re.sub(r'\s+',' ',x.text or '') for x in node.findall('unitPattern') if 'case' not in x.attrib and x.text and '↑' not in x.text}
   if lang=='ja' and symbol in ('s','min','h','°'):
    patterns={'other':'{0}'+{'s':'秒','min':'分','h':'時間','°':'度'}[symbol]}
   if not patterns or 'other' not in patterns:continue
   # CLDR Japanese percent uses a symbol; use NeMo's spoken label instead.
   if lang=='ja' and symbol in ('%','％','‰'):patterns={'other':'{0}'+('パーセント' if symbol!='‰' else 'パーミル')}
   if lang=='ja' and key=='temperature-celsius':patterns={'other':'摂氏{0}度'}
   if lang=='ja' and key=='temperature-fahrenheit':patterns={'other':'華氏{0}度'}
   units[symbol]={'id':key,'patterns':patterns,'gender':node.findtext('gender') or 'masculine'}
  greek={};functions={};operators={}
  if lang in ('en','de','fr','es'):
   base=a.sources/'sre/mathmaps'/lang
   alph=json.loads(read(base/'messages/alphabets.json'))['messages']
   chars='∇αβγδεζηθικλμνξοπρςστυφχψω∂ϵϑϰϕϱϖ'
   greek.update(zip(chars,alph['greekSmall']));greek.update(zip('ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡϴΣΤΥΦΧΨΩ',alph['greekCap']))
   for path in sorted((base/'functions').glob('*.json')):
    for entry in json.loads(read(path)):
     if 'key' in entry:
      value=entry.get('mappings',{}).get('default',{}).get('default')
      if isinstance(value,str):functions[entry['key']]=value
   for path in sorted((base/'symbols').glob('math_*.json')):
    for entry in json.loads(read(path)):
     try:char=chr(int(entry.get('key',''),16))
     except ValueError:continue
     value=entry.get('mappings',{}).get('default',{}).get('default')
     if isinstance(value,str):operators[char]=value
  letters={}
  if lang=='ja':
   for line in read(a.sources/'nemo/nemo_text_processing/text_normalization/ja/data/latin/letters.tsv').splitlines():
    k,v=line.split('\t');letters[k]=v
  elif lang=='ru':
   for line in read(a.sources/'nemo/nemo_text_processing/text_normalization/ru/data/latin_to_cyrillic.tsv').splitlines():
    k,v=line.split('\t');letters.setdefault(k.upper(),v)
  letter_names={
   'en':list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
   'de':'A Be Ze De E Ef Ge Ha I Jot Ka El Em En O Pe Ku Er Es Te U Fau We Iks Ypsilon Zett'.split(),
   'fr':'a bé cé dé e effe gé ache i ji ka elle emme enne o pé ku erre esse té u vé double-vé ixe i-grec zède'.split(),
   'es':'a be ce de e efe ge hache i jota ka ele eme ene o pe cu erre ese te u uve doble-uve equis ye zeta'.split()}
  if lang in letter_names:letters=dict(zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ',letter_names[lang]))
  if lang=='ru':letters.update(L='эл',N='эн',R='ар',Z='зед')
  operators={k:v for k,v in operators.items() if k in '+-−±×÷=<>≤≥≠≈≃:∶'}
  math_letters={}
  if lang=='ru':
   snapshot=json.loads((a.sources/'mathcat-snapshot.json').read_text())
   for name,sha in snapshot['files'].items():
    source=a.sources/name;assert hashlib.sha256(source.read_bytes()).hexdigest()==sha,name;used.append(source)
   definitions=(a.sources/'mathcat-ru-definitions.yaml').read_text()
   block=re.search(r'RussianLatinLetterNames:\s*\{(.*?)\}',definitions,re.S)[1]
   math_letters={k.upper():v for k,v in re.findall(r'"([b-z])":\s*"([^"]+)"',block)}
   assert len(math_letters)==25;math_letters['A']='а'
  data[lang]={'math_letters':math_letters,'units':units,'greek':greek,'functions':{k:v for k,v in functions.items() if k in ('sin','cos','tan','cot','sec','csc','ln','log','exp','abs','max','min')},'operators':operators,'letters':letters}
 (RUNTIME/'multilingual_frontend_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
 manifest={'sources':{name:{'revision':(a.sources/name/'revision.txt').read_text().strip(),'url':url} for name,url in [('cldr','https://github.com/unicode-org/cldr'),('sre','https://github.com/speech-rule-engine/speech-rule-engine'),('nemo','https://github.com/NVIDIA/NeMo-text-processing')]},'files':{str(p.relative_to(a.sources)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(used))},'unit_symbols':{lang:len(x['units']) for lang,x in data.items()},'notes':'Native ICU spellout at runtime; localized scope wording is authored. Japanese/Russian math phrases are project rules, not NeMo math grammars.'}
 manifest['sources']['mathcat']=json.loads((a.sources/'mathcat-snapshot.json').read_text())
 shutil.copyfile(a.sources/'mathcat-LICENSE',RUNTIME/'MATHCAT_LICENSE')
 (ROOT/'docs/tts_frontend_resources/multilingual_sources.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 for name in ('cldr','sre','nemo'):shutil.copyfile(a.sources/name/'LICENSE',RUNTIME/(name.upper()+'_LICENSE'))
 from import_tts_multilingual_terms import update
 update(ROOT)
 print(manifest['unit_symbols'])
if __name__=='__main__':main()
