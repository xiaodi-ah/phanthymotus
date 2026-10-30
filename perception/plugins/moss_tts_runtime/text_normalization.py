"""Spoken quantities and bounded, language-specific pronunciation candidates."""
import re
from .lexicon import rewrite_terms,version_reading
from . import zh_math, multilingual_math, icu_numbers, daily_years

_EN=['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen']
_ZH='零一二三四五六七八九'
_PROTECTED=re.compile(r'```[\s\S]*?```|https?://[^\s]+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|`[^`]*`')
_TOKEN_LEFT=r'(?<![A-Za-z0-9_./\\-])'
_TOKEN_RIGHT=r'(?![A-Za-z0-9_/-]|\.[A-Za-z0-9_]|\\)'
_QWEN=re.compile(_TOKEN_LEFT+r'Qwen[ \t]*(\d{1,6}(?:\.\d{1,3})?)'+_TOKEN_RIGHT,re.IGNORECASE)
_IDENTIFIER=re.compile(r'((?:\b(?:order\s+(?:ID|number)|serial\s+number|verification\s+code|ID)\b|订单号|订单\s*ID|验证码|序列号|编号)[ \t]*[:：#]?[ \t]*)(\d{2,20})(?![A-Za-z0-9_/-]|[.,]\d)',re.IGNORECASE)
def integer_words(value,language):
    return zh_math.integer(value) if language=='zh' else icu_numbers.integer(value,language)


def normalize_text(text,language,*,terms=True):
    if not isinstance(text,str):raise TypeError('text must be a string')
    if language not in ('zh','en','ja','de','fr','ru','es'):return text
    def segment(value):
        value=daily_years.normalize(value,language)
        if language=='zh':value=zh_math.normalize(value)
        else:value=multilingual_math.normalize(value,language)
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
        return rewrite_terms(value,language,include_trials=terms)
    pieces=[];start=0
    for match in _PROTECTED.finditer(text):
        pieces.extend([segment(text[start:match.start()]),match[0]]);start=match.end()
    pieces.append(segment(text[start:]));return ''.join(pieces)
