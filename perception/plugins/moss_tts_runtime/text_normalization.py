"""Conservative spoken quantities and pronunciation candidates for EN/DE/JA."""
import re
from decimal import Decimal

_EN=['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen']
_DE=['null','eins','zwei','drei','vier','fünf','sechs','sieben','acht','neun','zehn','elf','zwölf','dreizehn','vierzehn','fünfzehn','sechzehn','siebzehn','achtzehn','neunzehn']
_JA=['ぜろ','いち','に','さん','よん','ご','ろく','なな','はち','きゅう']
# Only explicit quantities; leave dates, software versions, identifiers and bare numbers intact.
_QUANTITY=re.compile(r'(?<![A-Za-z0-9_.,/\\-])([−+-]?\d{1,6}(?:[.,]\d{1,3})?)\s*(℃|°\s*C|%|％|km/h|m/s|GHz|MHz|kHz|Hz|GB|MB|km|cm|mm|kg|ms|m|s|g)(?![A-Za-z0-9_/-]|\.[A-Za-z0-9_]|\\)')
_PROTECTED=re.compile(r'https?://[^\s]+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|`[^`]*`')
_TERMS={
    'en':{'GPU':'G P U','CPU':'C P U','TTS':'T T S','RTF':'R T F'},
    'de':{'NVIDIA':'Envidia','Jetson':'Dschetson','GPU':'Ge Pe U','CPU':'Ze Pe U','TTS':'Te Te Es','RTF':'Er Te Ef'},
    'ja':{'NVIDIA':'エヌビディア','Jetson':'ジェットソン','GPU':'ジーピーユー','CPU':'シーピーユー','TTS':'ティーティーエス','RTF':'アールティーエフ'}}
_UNITS={
    'en':{'C':('degree Celsius','degrees Celsius'),'%':('percent','percent'),'km/h':('kilometer per hour','kilometers per hour'),'m/s':('meter per second','meters per second'),'GHz':('gigahertz','gigahertz'),'MHz':('megahertz','megahertz'),'kHz':('kilohertz','kilohertz'),'Hz':('hertz','hertz'),'GB':('gigabyte','gigabytes'),'MB':('megabyte','megabytes'),'km':('kilometer','kilometers'),'cm':('centimeter','centimeters'),'mm':('millimeter','millimeters'),'kg':('kilogram','kilograms'),'ms':('millisecond','milliseconds'),'m':('meter','meters'),'s':('second','seconds'),'g':('gram','grams')},
    'de':{'C':('Grad Celsius','Grad Celsius'),'%':('Prozent','Prozent'),'km/h':('Kilometer pro Stunde','Kilometer pro Stunde'),'m/s':('Meter pro Sekunde','Meter pro Sekunde'),'GHz':('Gigahertz','Gigahertz'),'MHz':('Megahertz','Megahertz'),'kHz':('Kilohertz','Kilohertz'),'Hz':('Hertz','Hertz'),'GB':('Gigabyte','Gigabyte'),'MB':('Megabyte','Megabyte'),'km':('Kilometer','Kilometer'),'cm':('Zentimeter','Zentimeter'),'mm':('Millimeter','Millimeter'),'kg':('Kilogramm','Kilogramm'),'ms':('Millisekunde','Millisekunden'),'m':('Meter','Meter'),'s':('Sekunde','Sekunden'),'g':('Gramm','Gramm')},
    'ja':{'C':'ど','%':'パーセント','km/h':'キロメートルまいじ','m/s':'メートルまいびょう','GHz':'ギガヘルツ','MHz':'メガヘルツ','kHz':'キロヘルツ','Hz':'ヘルツ','GB':'ギガバイト','MB':'メガバイト','km':'キロメートル','cm':'センチメートル','mm':'ミリメートル','kg':'キログラム','ms':'ミリびょう','m':'メートル','s':'びょう','g':'グラム'}}


def integer_words(value,language):
    if not 0<=value<=999999:raise ValueError('Quantity is outside the supported range')
    if language=='en':
        if value<20:return _EN[value]
        if value<100:return ['','','twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety'][value//10]+(' '+_EN[value%10] if value%10 else '')
        base=100 if value<1000 else 1000
        return integer_words(value//base,language)+(' hundred' if base==100 else ' thousand')+(' '+integer_words(value%base,language) if value%base else '')
    if language=='de':
        if value<20:return _DE[value]
        if value<100:
            tens=['','','zwanzig','dreißig','vierzig','fünfzig','sechzig','siebzig','achtzig','neunzig'][value//10]
            return (('ein' if value%10==1 else _DE[value%10])+'und' if value%10 else '')+tens
        base=100 if value<1000 else 1000
        prefix='ein' if value//base==1 else integer_words(value//base,language)
        return prefix+('hundert' if base==100 else 'tausend')+(integer_words(value%base,language) if value%base else '')
    if language!='ja':raise ValueError('Unsupported quantity language')
    if value<10:return _JA[value]
    if value>=10000:return integer_words(value//10000,language)+'まん'+(integer_words(value%10000,language) if value%10000 else '')
    result=''
    for base,unit,exceptions in [(1000,'せん',{3:'さんぜん',8:'はっせん'}),(100,'ひゃく',{3:'さんびゃく',6:'ろっぴゃく',8:'はっぴゃく'}),(10,'じゅう',{})]:
        digit,value=divmod(value,base)
        if digit:result+=exceptions.get(digit,('' if digit==1 else _JA[digit])+unit)
    return result+(_JA[value] if value else '')


def normalize_text(text,language,*,terms=True):
    if not isinstance(text,str):raise TypeError('text must be a string')
    if language not in _TERMS:return text
    def quantity(match):
        raw,unit=match.groups();unit='C' if '°' in unit or unit=='℃' else unit.replace('％','%')
        # Keep ambiguous locale-specific grouping/decimal notation intact.
        if (language in ('en','ja') and ',' in raw) or (language=='de' and '.' in raw):return match[0]
        number=raw.replace('−','-').replace(',','.');value=Decimal(number)
        sign=''
        if number[0] in '+-':
            sign=({'en':'minus ','de':'minus ','ja':'マイナス'} if number[0]=='-' else {'en':'plus ','de':'plus ','ja':'プラス'})[language];number=number[1:]
        whole,point,fraction=number.partition('.')
        if len(whole)>1 and whole.startswith('0'):return match[0]
        spoken=integer_words(int(whole),language)
        if point:
            separator={'en':' point ','de':' Komma ','ja':'てん'}[language]
            digits={'en':_EN,'de':_DE,'ja':_JA}[language]
            spoken+=separator+('' if language=='ja' else ' ').join(digits[int(c)] for c in fraction)
        if language=='ja':return sign+spoken+_UNITS[language][unit]
        singular=abs(value)==1 and not point
        # shortcut: German singular unit forms do not infer sentence case; add grammar rules when needed.
        if language=='de' and not point and singular:spoken='eine' if unit in ('s','ms') else 'ein'
        return sign+spoken+' '+_UNITS[language][unit][0 if singular else 1]
    def segment(value):
        value=_QUANTITY.sub(quantity,value)
        if terms:
            names=_TERMS[language]
            pattern=r'(?<![A-Za-z0-9_./\\-])('+ '|'.join(map(re.escape,names))+r')(?![A-Za-z0-9_/-]|\.[A-Za-z0-9_]|\\)'
            value=re.sub(pattern,lambda m:names[m[0]],value)
        return value
    pieces=[];start=0
    for match in _PROTECTED.finditer(text):
        pieces.extend([segment(text[start:match.start()]),match[0]]);start=match.end()
    pieces.append(segment(text[start:]));return ''.join(pieces)
