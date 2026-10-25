"""Check streaming resampling and language handling without constructing a model."""
import sys,re
from pathlib import Path
import numpy as np
from scipy.signal import sosfilt
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'perception') if (root/'perception').exists() else '/work')
from plugins.moss_tts_runtime import StreamingPCM,language_for
from plugins.moss_tts_runtime.text_chunks import text_chunks

def main():
    class Characters:
        def encode(self,text,out_type=int):return list(text)
    tokenizer=Characters()
    paragraph='现在开始测试。机器人前往会议室！最后一句：播报结束。'
    assert text_chunks(paragraph,tokenizer)==['现在开始测试。','机器人前往会议室！','最后一句：播报结束。']
    english='Dr. Smith uses model.onnx version3.2 in the U.S. office. It costs3.5 dollars.'
    assert text_chunks(english,tokenizer)==['Dr. Smith uses model.onnx version3.2 in the U.S. office.','It costs3.5 dollars.']
    for sentence in ('Merci, s’il vous plaît.', 'La réunion.', 'La habitación.', 'Der Akku ist grün.', 'Это окно.', 'Il est pre\u0302t.'):
        assert text_chunks(sentence+' Next.',tokenizer)==[sentence,'Next.'],sentence
    assert text_chunks('J. Smith uses the U.S. office. Next.',tokenizer)==['J. Smith uses the U.S. office.','Next.']
    assert text_chunks('数字是3. 下一句。',tokenizer)==['数字是3.','下一句。']
    assert text_chunks('他说：“你好！” 然后离开。',tokenizer)==['他说：“你好！”','然后离开。']
    for paragraph in ['长句没有标点'*50,'word '*100,'这是长句，'*40,english,paragraph]:
        chunks=text_chunks(paragraph,tokenizer)
        assert chunks and all(0<len(tokenizer.encode(chunk))<=75 for chunk in chunks)
        assert re.sub(r'\s','', ''.join(chunks))==re.sub(r'\s','',paragraph),'Text lost during splitting'
    class NonMonotone(Characters):
        def encode(self,text,out_type=int):return [0]*76 if text.endswith(',') else list(text)
    chunks=text_chunks('x'*39+','+'y'*80,NonMonotone())
    assert all(len(NonMonotone().encode(chunk))<=75 for chunk in chunks)
    assert ''.join(chunks)=='x'*39+','+'y'*80
    # A nearly full chunk must not leave a two-word independent synthesis.
    orphan='The resources return to the computing pool in time to be allocated to other tasks.'
    chunks=text_chunks(orphan,tokenizer)
    assert len(chunks)==2 and all(len(chunk)>=12 for chunk in chunks),chunks
    clause='A'*32+', '+'B'*32+' and '+'C'*35+'.'
    chunks=text_chunks(clause,tokenizer)
    assert chunks[0].endswith(',') and ''.join(chunks).replace(' ','')==clause.replace(' ',''),chunks
    assert all(0<len(tokenizer.encode(chunk))<=75 for chunk in chunks)
    for value in ('123,456,789','12:30'):
        sentence='This long sentence refers to value '+value+' and continues to describe the remaining resources for other jobs.'
        assert any(value in chunk for chunk in text_chunks(sentence,tokenizer))
    rng=np.random.default_rng(42);audio=rng.normal(scale=.1,size=(2,10001))
    converter=StreamingPCM();pieces=[];start=0
    for count in [1,2,31,3840,500,17,5610]:
        piece=audio[:,start:start+count];start+=count;pieces.append(converter.process(piece))
    assert start==audio.shape[1]
    actual=b''.join(pieces);single=StreamingPCM().process(audio)
    assert actual==single,'Streaming FIR loses continuity/decimation phase'
    assert len(actual)==((audio.shape[1]+2)//3)*2
    time=np.arange(48000)/48000
    low=np.sin(2*np.pi*2000*time);high=np.sin(2*np.pi*10000*time)
    def rms(signal):
        samples=np.frombuffer(StreamingPCM().process(np.stack([signal,signal])),dtype='<i2').astype(np.float64)/32768
        return np.sqrt(np.mean(samples[100:]**2))
    assert rms(high)<rms(low)*.02,'Insufficient anti-alias attenuation'
    assert language_for('你好')=='zh' and language_for('こんにちは')=='ja' and language_for('สวัสดี')=='th'
    assert language_for('Здравствуйте')=='ru' and language_for('Bonjour','fr-FR')=='fr'
    try:language_for('test','xx')
    except ValueError:pass
    else:raise AssertionError('Unsupported language accepted')
    try:StreamingPCM().process(np.full((2,10),np.nan))
    except ValueError:pass
    else:raise AssertionError('Nonfinite audio accepted')
    print('PASS sentence splitting/token bounds/text preservation and streaming PCM continuity, 16kHz anti-aliasing, finite audio and language validation')
if __name__=='__main__':main()
