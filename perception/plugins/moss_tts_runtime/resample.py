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
