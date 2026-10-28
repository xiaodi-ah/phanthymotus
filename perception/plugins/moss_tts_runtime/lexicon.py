"""Approved text readings and opt-in trials; no phoneme model."""
import json,re
from functools import lru_cache
from pathlib import Path

LEFT=r'(?<![A-Za-z0-9_./\\-])'
RIGHT=r'(?![A-Za-z0-9_/-]|\.[A-Za-z0-9_]|\\)'
PATH=Path(__file__).with_name('pronunciation_lexicon.json')

@lru_cache(maxsize=4)
def load_lexicon(path=PATH):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('schema_version')!=1 or not isinstance(data.get('entries'),list):
        raise ValueError('Unsupported pronunciation lexicon schema')
    seen={};entries=[]
    for entry in data['entries']:
        if not isinstance(entry,dict):raise ValueError('Invalid lexicon entry')
        for key in ('term','language','reading','status','source','evidence'):
            if not isinstance(entry.get(key),str) or (key!='reading' and not entry[key].strip()):
                raise ValueError('Invalid lexicon field: '+key)
        if entry['language'] not in ('zh','en','ja','de','fr','es','ru','th') or entry['status'] not in ('approved','trial','pending','rejected'):
            raise ValueError('Invalid lexicon language/status')
        if entry['status'] in ('approved','trial') and not entry['reading'].strip():raise ValueError('Enabled reading is empty')
        if entry.get('scope','token') not in ('token','version'):raise ValueError('Invalid lexicon scope')
        if not isinstance(entry.get('case_sensitive',True),bool):raise ValueError('Invalid case sensitivity')
        aliases=entry.get('aliases',[])
        if not isinstance(aliases,list) or any(not isinstance(alias,str) or not alias.strip() for alias in aliases):
            raise ValueError('Invalid lexicon aliases')
        for name in [entry['term']]+aliases:
            key=(entry['language'],name.casefold())
            if key in seen:raise ValueError('Duplicate/ambiguous lexicon alias: '+name)
            seen[key]=entry['term']
        entries.append(entry)
    return tuple(entries)

@lru_cache(maxsize=16)
def _rules(language,include_trials):
    names=[]
    for entry in load_lexicon():
        if entry['language']==language and (entry['status']=='approved' or include_trials and entry['status']=='trial') and entry.get('scope','token')=='token':
            for name in [entry['term']]+entry.get('aliases',[]):names.append((name,entry))
    names.sort(key=lambda item:len(item[0]),reverse=True)
    if not names:return None,()
    parts=[('('+re.escape(name)+')' if entry.get('case_sensitive',True) else '((?i:'+re.escape(name)+'))') for name,entry in names]
    return re.compile(LEFT+'(?:'+'|'.join(parts)+')'+RIGHT),tuple(entry['reading'] for _,entry in names)

def rewrite_terms(text,language,*,include_trials=True):
    pattern,readings=_rules(language,include_trials)
    # One pass prevents a replacement from becoming another term in this call.
    return pattern.sub(lambda match:readings[match.lastindex-1],text) if pattern else text

def version_reading(term,language):
    return next((entry['reading'] for entry in load_lexicon() if entry['term']==term and entry['language']==language and entry['status']=='trial' and entry.get('scope')=='version'),None)
