"""Stateful 48 kHz stereo -> 16 kHz mono PCM, with anti-alias filtering."""
import numpy as np
from scipy.signal import butter,sosfilt

class StreamingPCM:
    def __init__(self):
        self.sos=butter(10,7000,fs=48000,output='sos')
        self.state=np.zeros((self.sos.shape[0],2));self.phase=0
    def process(self,audio):
        audio=np.asarray(audio)
        if audio.ndim!=2 or audio.shape[0]!=2 or not np.isfinite(audio).all():
            raise ValueError('Expected finite stereo codec output')
        filtered,self.state=sosfilt(self.sos,audio.mean(axis=0,dtype=np.float64),zi=self.state)
        samples=filtered[(-self.phase)%3::3];self.phase=(self.phase+audio.shape[1])%3
        return np.rint(np.clip(samples,-1,32767/32768)*32768).astype('<i2').tobytes()

class SentenceEnding:
    """Hold 12ms of PCM so a sentence can end at zero without deleting speech."""
    def __init__(self):self.tail=b'';self.closed=False
    def process(self,pcm):
        if self.closed:raise RuntimeError('Sentence already finished')
        if len(pcm)%2:raise ValueError('Expected complete PCM16 samples')
        pcm=self.tail+pcm;count=max(0,len(pcm)-384)
        self.tail=pcm[count:];return pcm[:count]
    def finish(self):
        if self.closed:raise RuntimeError('Sentence already finished')
        self.closed=True;samples=np.frombuffer(self.tail,dtype='<i2').astype(np.float64);self.tail=b''
        if samples.size:samples*=np.linspace(1.,0.,samples.size) if samples.size>1 else 0.
        return np.rint(samples).astype('<i2').tobytes()
