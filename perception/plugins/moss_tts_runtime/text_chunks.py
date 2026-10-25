"""Sentence boundaries and bounded text input for each independent synthesis."""
import re,unicodedata

_END=re.compile(r'[。！？!?；;]+[”’"）)]*|\.[”’"）)]*(?=\s|$)|\n+')
_ABBREVIATIONS={'mr','mrs','ms','dr','prof','sr','jr','vs','etc','e.g','i.e','no','fig','inc'}


def text_chunks(text,tokenizer):
    sentences=[];start=0
    for match in _END.finditer(text):
        if match.group().startswith('.'):
            # Match the whole Unicode word; the ASCII suffix of «plaît» is not an initial.
            word=re.search(r'([^\W\d_]+(?:\.[^\W\d_]+)*)\.$',unicodedata.normalize('NFC',text[:match.start()+1]))
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
            # Prefer complete clauses and leave enough context for the final independent call.
            boundaries=list(re.finditer(r'(?<!\d)[,，:：、]\s*|[,，:：、](?!\d)\s*|\s+',sentence[:cut]))
            valid=[]
            for boundary in boundaries:
                end=boundary.end()
                left=len(tokenizer.encode(sentence[:end].strip(),out_type=int))
                right=len(tokenizer.encode(sentence[end:].strip(),out_type=int))
                if 20<=left<=75 and right>=12:
                    valid.append((end,bool(re.match(r'[,，:：、]',boundary.group()))))
            clauses=[end for end,is_clause in valid if is_clause]
            if clauses:cut=clauses[-1]
            elif valid:cut=valid[-1][0]
            elif len(tokenizer.encode(sentence[cut:].strip(),out_type=int))<12:
                for end in range(cut-1,0,-1):
                    if 12<=len(tokenizer.encode(sentence[:end].strip(),out_type=int))<=75 and len(tokenizer.encode(sentence[end:].strip(),out_type=int))>=12:
                        cut=end;break
            # A shorter clause prefix can have more SentencePiece tokens; recheck after trimming.
            while cut and len(tokenizer.encode(sentence[:cut].strip(),out_type=int))>75:cut-=1
            if not cut:raise ValueError('A single character exceeds the text token budget')
            piece=sentence[:cut].strip()
            if piece:chunks.append(piece)
            sentence=sentence[cut:].strip()
    return chunks
