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


def frontend_segments(text,language,*,terms=False):
    from .tts_entities import entity_segments,prose_decimals,TextSegment
    if not isinstance(text,str):raise TypeError('text must be a string')
    if language in ('zh','en','ja','de','fr','ru','es'):
        standalone=spoken_identifier(text,language)
        if standalone!=text:return [TextSegment('identifier',0,len(text),text,standalone)]
    segments=[]
    for part in entity_segments(text,language):
        # Only prose reaches math/quantity/term rules; replacement speech is final.
        spoken=part.spoken
        if part.kind=='text':
            # A separator left after an entity is not the start of a math expression.
            prefix=re.match(r'^[\s;；]+',part.raw)
            prefix=prefix[0] if prefix else ''
            spoken=prefix+prose_decimals(normalize_text(part.raw[len(prefix):],language,terms=terms),language)
        segments.append(TextSegment(part.kind,part.start,part.end,part.raw,spoken))
    return segments


def prepare_text(text,language,*,terms=False):
    return ''.join(part.spoken for part in frontend_segments(text,language,terms=terms))


def supertonic_text(text,language):
    # Expand powers before the native NFKD frontend loses superscript information.
    return prepare_text(text,language)
