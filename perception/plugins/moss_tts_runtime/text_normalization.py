"""Spoken quantities and bounded, language-specific pronunciation candidates."""
import re
from decimal import Decimal
from .lexicon import rewrite_terms,version_reading

_EN=['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen']
_DE=['null','eins','zwei','drei','vier','fünf','sechs','sieben','acht','neun','zehn','elf','zwölf','dreizehn','vierzehn','fünfzehn','sechzehn','siebzehn','achtzehn','neunzehn']
_JA=['ぜろ','いち','に','さん','よん','ご','ろく','なな','はち','きゅう']
_ZH='零一二三四五六七八九'
_FR='zéro un deux trois quatre cinq six sept huit neuf dix onze douze treize quatorze quinze seize'.split()
_RU='ноль один два три четыре пять шесть семь восемь девять десять одиннадцать двенадцать тринадцать четырнадцать пятнадцать шестнадцать семнадцать восемнадцать девятнадцать'.split()
# Only explicit quantities; leave dates, software versions, identifiers and bare numbers intact.
_QUANTITY=re.compile(r'(?<![A-Za-z0-9_.,/\\-])([−+-]?\d{1,6}(?:[.,]\d{1,3})?)\s*(℃|°\s*C|%|％|km/h|m/s|GHz|MHz|kHz|Hz|GB|MB|km|cm|mm|kg|ms|m|s|g)(?![A-Za-z0-9_/-]|\.[A-Za-z0-9_]|\\)')
_PROTECTED=re.compile(r'https?://[^\s]+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|`[^`]*`')
_TOKEN_LEFT=r'(?<![A-Za-z0-9_./\\-])'
_TOKEN_RIGHT=r'(?![A-Za-z0-9_/-]|\.[A-Za-z0-9_]|\\)'
_QWEN=re.compile(_TOKEN_LEFT+r'Qwen[ \t]*(\d{1,6}(?:\.\d{1,3})?)'+_TOKEN_RIGHT,re.IGNORECASE)
_IDENTIFIER=re.compile(r'((?:\b(?:order\s+(?:ID|number)|serial\s+number|verification\s+code|ID)\b|订单号|订单\s*ID|验证码|序列号|编号)[ \t]*[:：#]?[ \t]*)(\d{2,20})(?![A-Za-z0-9_/-]|[.,]\d)',re.IGNORECASE)
_UNITS={
    'fr':{'C':('degré Celsius','degrés Celsius'),'%':('pour cent','pour cent')},
    'ru':{'C':('градус Цельсия','градуса Цельсия','градусов Цельсия'),'%':('процент','процента','процентов')},
    'zh':{'C':'摄氏度','%':'百分之','km/h':'千米每小时','m/s':'米每秒','GHz':'吉赫兹','MHz':'兆赫兹','kHz':'千赫兹','Hz':'赫兹','GB':'吉字节','MB':'兆字节','km':'千米','cm':'厘米','mm':'毫米','kg':'千克','ms':'毫秒','m':'米','s':'秒','g':'克'},
    'en':{'C':('degree Celsius','degrees Celsius'),'%':('percent','percent'),'km/h':('kilometer per hour','kilometers per hour'),'m/s':('meter per second','meters per second'),'GHz':('gigahertz','gigahertz'),'MHz':('megahertz','megahertz'),'kHz':('kilohertz','kilohertz'),'Hz':('hertz','hertz'),'GB':('gigabyte','gigabytes'),'MB':('megabyte','megabytes'),'km':('kilometer','kilometers'),'cm':('centimeter','centimeters'),'mm':('millimeter','millimeters'),'kg':('kilogram','kilograms'),'ms':('millisecond','milliseconds'),'m':('meter','meters'),'s':('second','seconds'),'g':('gram','grams')},
    'de':{'C':('Grad Celsius','Grad Celsius'),'%':('Prozent','Prozent'),'km/h':('Kilometer pro Stunde','Kilometer pro Stunde'),'m/s':('Meter pro Sekunde','Meter pro Sekunde'),'GHz':('Gigahertz','Gigahertz'),'MHz':('Megahertz','Megahertz'),'kHz':('Kilohertz','Kilohertz'),'Hz':('Hertz','Hertz'),'GB':('Gigabyte','Gigabyte'),'MB':('Megabyte','Megabyte'),'km':('Kilometer','Kilometer'),'cm':('Zentimeter','Zentimeter'),'mm':('Millimeter','Millimeter'),'kg':('Kilogramm','Kilogramm'),'ms':('Millisekunde','Millisekunden'),'m':('Meter','Meter'),'s':('Sekunde','Sekunden'),'g':('Gramm','Gramm')},
    'ja':{'C':'ど','%':'パーセント','km/h':'キロメートルまいじ','m/s':'メートルまいびょう','GHz':'ギガヘルツ','MHz':'メガヘルツ','kHz':'キロヘルツ','Hz':'ヘルツ','GB':'ギガバイト','MB':'メガバイト','km':'キロメートル','cm':'センチメートル','mm':'ミリメートル','kg':'キログラム','ms':'ミリびょう','m':'メートル','s':'びょう','g':'グラム'}}


def integer_words(value,language):
    if not 0<=value<=999999:raise ValueError('Quantity is outside the supported range')
    if language in ('fr','ru'):
        # shortcut: screen common temperature/percentage readings through 100 before extending the range.
        if value>100:raise ValueError('French/Russian quantity candidate supports 0..100')
        if language=='ru':
            if value<20:return _RU[value]
            if value==100:return 'сто'
            return ['','','двадцать','тридцать','сорок','пятьдесят','шестьдесят','семьдесят','восемьдесят','девяносто'][value//10]+(' '+_RU[value%10] if value%10 else '')
        if value<17:return _FR[value]
        if value<20:return 'dix-'+_FR[value-10]
        if value==100:return 'cent'
        if value>=80:return 'quatre-vingt'+('-'+integer_words(value-80,'fr') if value>80 else 's')
        tens=value//10*10 if value<70 else 60;tail=value-tens
        prefix={20:'vingt',30:'trente',40:'quarante',50:'cinquante',60:'soixante'}[tens]
        return prefix+('-et-' if tail in (1,11) else '-' if tail else '')+(integer_words(tail,'fr') if tail else '')
    if language=='zh':
        def section(number):
            result='';zero=False
            for base,unit in ((1000,'千'),(100,'百'),(10,'十'),(1,'')):
                digit,number=divmod(number,base)
                if digit:
                    if zero:result+='零'
                    result+=_ZH[digit]+unit;zero=False
                elif result and number:zero=True
            return result
        high,low=divmod(value,10000)
        result=(section(high)+'万'+('零' if 0<low<1000 else '') if high else '')+section(low)
        return (result[1:] if result.startswith('一十') else result) or '零'
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
    if language not in _UNITS:return text
    # Keep the released quantity-only mode unchanged while screening the Chinese frontend.
    if language in ('zh','fr','ru') and not terms:return text
    def quantity(match):
        raw,unit=match.groups();unit='C' if '°' in unit or unit=='℃' else unit.replace('％','%')
        if unit not in _UNITS[language]:return match[0]
        # Keep ambiguous locale-specific grouping/decimal notation intact.
        if (language in ('zh','en','ja') and ',' in raw) or (language=='de' and '.' in raw):return match[0]
        number=raw.replace('−','-').replace(',','.');value=Decimal(number)
        if language in ('fr','ru') and (abs(value)>100 or language=='ru' and '.' in number):return match[0]
        sign=''
        if number[0] in '+-':
            sign=({'zh':'负','en':'minus ','de':'minus ','ja':'マイナス','fr':'moins ','ru':'минус '} if number[0]=='-' else {'zh':'正','en':'plus ','de':'plus ','ja':'プラス','fr':'plus ','ru':'плюс '})[language];number=number[1:]
        whole,point,fraction=number.partition('.')
        if len(whole)>1 and whole.startswith('0'):return match[0]
        spoken=integer_words(int(whole),language)
        if point:
            separator={'zh':'点','en':' point ','de':' Komma ','ja':'てん','fr':' virgule '}[language]
            digits={'zh':_ZH,'en':_EN,'de':_DE,'ja':_JA,'fr':_FR}[language]
            spoken+=separator+('' if language in ('zh','ja') else ' ').join(digits[int(c)] for c in fraction)
        if language=='zh':return sign+('百分之'+spoken if unit=='%' else spoken+_UNITS[language][unit])
        if language=='ja':return sign+spoken+_UNITS[language][unit]
        if language=='ru':
            count=int(whole);form=2 if 11<=count%100<=14 else 0 if count%10==1 else 1 if 2<=count%10<=4 else 2
            return sign+spoken+' '+_UNITS[language][unit][form]
        singular=abs(value)==1 and not point
        if language=='fr':singular=abs(value)<2
        # shortcut: German singular unit forms do not infer sentence case; add grammar rules when needed.
        if language=='de' and not point and singular:spoken='eine' if unit in ('s','ms') else 'ein'
        return sign+spoken+' '+_UNITS[language][unit][0 if singular else 1]
    def segment(value):
        if terms and language in ('zh','en'):
            def identifier(match):
                digits=_ZH if language=='zh' else _EN
                return match[1]+('' if language=='zh' else ' ').join(digits[int(c)] for c in match[2])
            value=_IDENTIFIER.sub(identifier,value)
            def qwen(match):
                whole,point,fraction=match[1].partition('.')
                if len(whole)>1 and whole.startswith('0'):return match[0]
                reading=version_reading('Qwen','zh')
                return reading+''.join(_ZH[int(c)] if c.isdigit() else '点' for c in match[1]) if reading else match[0]
            if language=='zh':
                value=_QWEN.sub(qwen,value)
        value=_QUANTITY.sub(quantity,value)
        if terms:
            value=rewrite_terms(value,language)
        return value
    pieces=[];start=0
    for match in _PROTECTED.finditer(text):
        pieces.extend([segment(text[start:match.start()]),match[0]]);start=match.end()
    pieces.append(segment(text[start:]));return ''.join(pieces)
