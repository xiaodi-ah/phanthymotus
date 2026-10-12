"""Check streaming resampling and language handling without constructing a model."""
import sys
from pathlib import Path
import numpy as np
from scipy.signal import sosfilt
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'perception') if (root/'perception').exists() else '/work')
from plugins.moss_tts_runtime import StreamingPCM,language_for

def main():
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
    print('PASS streaming PCM continuity, 16kHz anti-aliasing, finite audio and language validation')
if __name__=='__main__':main()
