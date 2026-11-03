"""Shared spoken-text preparation; keep neutral identifiers separate from prose numbers."""
import re
import unicodedata
from .moss_tts_runtime.text_normalization import normalize_text, integer_words


def standalone_digits(text):
    value=unicodedata.normalize('NFKC',text).strip()
    return ''.join(value.split()) if re.fullmatch(r'[0-9]+(?:[ \t]+[0-9]+)*',value) else None


def spoken_identifier(text,language):
    digits=standalone_digits(text)
    if digits is None or len(digits)<5 and not (len(digits)>1 and digits.startswith('0')):
        return text
    # Without quantity context, long digit-only inputs are identifiers; preserve every digit.
    if language=='zh':
        words=''.join('零一二三四五六七八九'[int(c)] for c in digits)
        return '，'.join([words[:3]]+[words[i:i+4] for i in range(3,len(words),4)])+'。' if len(words)>=10 else words
    return ' '.join(integer_words(int(c),language) for c in digits)


def supertonic_text(text,language):
    from .tts_entities import spoken_entities
    # Expand powers before the native NFKD frontend loses superscript information.
    # Let the existing quantity grammar handle units and percentages first.
    return spoken_entities(normalize_text(spoken_identifier(text,language),language,terms=False),language)
