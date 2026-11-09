"""Bound actual model tokens, including tags, without dropping oversized sentences."""
import re
import unicodedata

_SENTENCE=re.compile(r'[。！？!?]+[”’"）)]*|\.[”’"）)]*(?=\s|$)|\n+')
_CLAUSE=re.compile(r'[；，、：]|[;,:](?=\s|$)')
_ATOMIC=re.compile(r'[A-Za-z0-9]+(?:[._+@/\-][A-Za-z0-9]+)*|[ァ-ヶー]+')
_ABBREVIATIONS={'mr','mrs','ms','dr','prof','sr','jr','vs','etc','e.g','i.e','no','fig','inc'}


def continuation_text(text,final):
    if final:return text
    if text.endswith(('、','，','；','：')):return text[:-1]+{'、':',','，':',','；':';','：':':'}[text[-1]]
    if not re.search(r'''[.!?;:,'"')\]}…。」』】〉》›»]$''',text):
        return text+','
    return text


def split_point(text, limit, language=None):
    end = min(len(text), limit)
    if end == len(text):
        return end
    from ..tts_entities import spoken_identifier_spans
    spans=[span for span in spoken_identifier_spans(text,language) if span[1]-span[0]<=end]
    def allowed(position):return not any(start<position<stop for start,stop in spans)
    sentences=[]
    for match in _SENTENCE.finditer(text):
        if match.end()>end:break
        if match[0].startswith('.'):
            word=re.search(r'([^\W\d_]+(?:\.[^\W\d_]+)*)\.$',text[:match.start()+1])
            if word and (word[1].casefold() in _ABBREVIATIONS or re.fullmatch(r'(?:[A-Za-z]\.)*[A-Za-z]',word[1])):continue
        if allowed(match.end()):sentences.append(match.end())
    clauses=[m.end() for m in _CLAUSE.finditer(text) if m.end()<=end and m.end()>=min(20,end//2) and allowed(m.end())]
    spaces=[m.end() for m in re.finditer(r'\s+',text) if m.end()<=end and allowed(m.end())]
    if sentences:end=sentences[-1]
    elif clauses:end=clauses[-1]
    elif spaces:end=spaces[-1]
    else:
        for start,stop in spans:
            if 0<start<end<stop:end=start;break
        for match in _ATOMIC.finditer(text):
            if 0<match.start()<end<match.end():end=match.start();break
    while end > 1 and end < len(text) and unicodedata.combining(text[end]):
        end -= 1
    return end


def text_chunks(text, language, processor):
    remaining = text.strip()
    first = True
    while remaining:
        # A short first clause bounds time to real audio; later blocks bound scratch memory.
        limit = (40 if first else 80) if language == 'ja' else (120 if first else 160)
        end = split_point(remaining, limit, language)
        while True:
            chunk = remaining[:end].strip()
            ids, _ = processor([continuation_text(chunk,end==len(remaining))], [language])
            if chunk and ids.shape[-1] <= 384:
                break
            if end <= 1:
                raise ValueError('A single text character exceeds the supported token budget')
            end = split_point(remaining, max(1, end // 2), language)
        yield chunk
        remaining = remaining[end:].lstrip()
        first = False
