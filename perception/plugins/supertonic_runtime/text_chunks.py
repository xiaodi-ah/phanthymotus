"""Bound actual model tokens, including tags, without dropping oversized sentences."""
import re
import unicodedata


def split_point(text, limit):
    end = min(len(text), limit)
    if end == len(text):
        return end
    prefix = text[:end]
    punctuation = [m.end() for m in re.finditer(r'[。！？；，、：]|[!?;,:.](?=\s)', prefix)]
    for boundaries in (punctuation, [m.end() for m in re.finditer(r'\s+', prefix)]):
        if boundaries and boundaries[-1] >= end // 2:
            end = boundaries[-1]
            break
    while end > 1 and end < len(text) and unicodedata.combining(text[end]):
        end -= 1
    return end


def text_chunks(text, language, processor):
    remaining = text.strip()
    first = True
    while remaining:
        # A short first clause bounds time to real audio; later blocks bound scratch memory.
        limit = (40 if first else 80) if language == 'ja' else (120 if first else 160)
        end = split_point(remaining, limit)
        while True:
            chunk = remaining[:end].strip()
            ids, _ = processor([chunk], [language])
            if chunk and ids.shape[-1] <= 384:
                break
            if end <= 1:
                raise ValueError('A single text character exceeds the supported token budget')
            end = split_point(remaining, max(1, end // 2))
        yield chunk
        remaining = remaining[end:].lstrip()
        first = False
