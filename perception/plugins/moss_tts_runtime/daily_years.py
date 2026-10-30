"""Spell out explicit German/Russian year contexts using installed ICU rules."""
import re
from functools import lru_cache
from .icu_numbers import Formatter

_YEAR=r'([12][0-9]{3})(?![0-9A-Za-z_/\\-]|\.[0-9A-Za-z_])'
_GERMAN=re.compile(r'\b((?:im(?:\s+Jahr(?:e)?)?|in|Ende|Anfang|seit)\s+)'+_YEAR,re.I)
_RUSSIAN_LOCATIVE=re.compile(r'\b(в\s+)'+_YEAR+r'(?=\s+году\b)',re.I)
_RUSSIAN_GENITIVE=re.compile(r'(?<![\w./\\-])'+_YEAR+r'(?=\s+года\b)',re.I)

@lru_cache(maxsize=3)
def formatter(language,case=''):
    return Formatter(language,'%spellout-ordinal-masculine-'+case if language=='ru' else '%spellout-numbering-year')

def russian_year(value,case):
    result=formatter('ru',case).format(value)
    # ICU70/72's locative forty ending differs from the corrected pinned CLDR rule.
    if case=='locative' and value%100==40 and result.endswith('сороковой'):
        result=result[:-len('сороковой')]+'сороковом'
    return result

def normalize(text,language):
    if language=='de':
        return _GERMAN.sub(lambda m:m[1]+formatter('de').format(int(m[2])),text)
    if language=='ru':
        text=_RUSSIAN_LOCATIVE.sub(lambda m:m[1]+russian_year(int(m[2]),'locative'),text)
        return _RUSSIAN_GENITIVE.sub(lambda m:russian_year(int(m[1]),'genitive'),text)
    return text
