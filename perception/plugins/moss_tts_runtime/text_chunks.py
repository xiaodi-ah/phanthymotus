"""Sentence boundaries and bounded text input for each independent synthesis."""
import re

_END=re.compile(r'[。！？!?；;]+[”’"）)]*|\.[”’"）)]*(?=\s|$)|\n+')
_ABBREVIATIONS={'mr','mrs','ms','dr','prof','sr','jr','vs','etc','e.g','i.e','no','fig','inc'}


def text_chunks(text,tokenizer):
    sentences=[];start=0
    for match in _END.finditer(text):
        if match.group().startswith('.'):
            word=re.search(r'([A-Za-z.]+)\.$',text[:match.start()+1])
            # shortcut: protect common English abbreviations; extend for observed domain abbreviations.
            if word and (word[1].lower() in _ABBREVIATIONS or re.fullmatch(r'(?:[A-Za-z]\.)*[A-Za-z]',word[1])):
                continue
        sentence=text[start:match.end()].strip()
        if sentence:sentences.append(sentence)
        start=match.end()
    if text[start:].strip():sentences.append(text[start:].strip())
    chunks=[]
    for sentence in sentences:
        while sentence:
            if len(tokenizer.encode(sentence,out_type=int))<=75:
                chunks.append(sentence);break
            low=1;high=len(sentence);cut=0
            while low<=high:
                middle=(low+high)//2
                if len(tokenizer.encode(sentence[:middle],out_type=int))<=75:cut=middle;low=middle+1
                else:high=middle-1
            boundaries=list(re.finditer(r'[\s,，:：、]',sentence[:cut]))
            if boundaries and boundaries[-1].end()>=cut//2:cut=boundaries[-1].end()
            # A shorter clause prefix can have more SentencePiece tokens; recheck after trimming.
            while cut and len(tokenizer.encode(sentence[:cut].strip(),out_type=int))>75:cut-=1
            if not cut:raise ValueError('A single character exceeds the text token budget')
            piece=sentence[:cut].strip()
            if piece:chunks.append(piece)
            sentence=sentence[cut:].strip()
    return chunks
