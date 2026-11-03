"""Padding removal must preserve quiet speech, interior pauses and streaming cleanup."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'perception'))
from plugins.supertonic_runtime import SupertonicAdapter
from plugins.supertonic_runtime.audio_join import INTER_BLOCK_PAUSE_SECONDS,trim_zero_padding


class Tests(unittest.TestCase):
    def test_removes_only_long_zero_edges(self):
        voice=np.array([1,-1,800,-800,0,0,1],dtype='<i2').tobytes()
        source=bytes(6400*2)+voice+bytes(4800*2)
        self.assertEqual(trim_zero_padding(source),bytes(320*2)+voice+bytes(320*2))

    def test_internal_and_short_pauses_are_preserved(self):
        source=np.array([1,-1],dtype='<i2').tobytes()+bytes(16000)+np.array([1],dtype='<i2').tobytes()
        self.assertEqual(trim_zero_padding(source),source)
        short=bytes(800*2)+source+bytes(800*2)
        self.assertEqual(trim_zero_padding(short),short)

    def test_single_sample_and_silence(self):
        for source in [bytes(16000),np.array([1],dtype='<i2').tobytes()]:
            self.assertEqual(trim_zero_padding(source),source)
        for source in [b'',b'\x01']:
            with self.assertRaises(ValueError):trim_zero_padding(source)

    def test_native_rate_guard(self):
        source=bytes(44100)+np.array([1,-1],dtype='<i2').tobytes()+bytes(44100)
        self.assertEqual(trim_zero_padding(source,44100),bytes(882*2)+np.array([1,-1],dtype='<i2').tobytes()+bytes(882*2))

    def adapter(self):
        adapter=SupertonicAdapter.__new__(SupertonicAdapter);self.closed=[]
        def blocks(*args):
            try:
                for _ in range(2):
                    yield np.r_[np.zeros(4410),np.full(4410,.1),np.zeros(4410)].astype(np.float32)
            finally:self.closed.append(True)
        adapter.native_blocks=blocks
        return adapter

    def test_stream_join_and_nonzero_samples(self):
        from scipy.signal import resample_poly
        from plugins.supertonic_runtime import pcm16
        adapter=self.adapter();expected=[]
        for audio in adapter.native_blocks('', 'en'):
            expected.append(trim_zero_padding(pcm16(resample_poly(audio,160,441))))
        actual=list(adapter.synthesize_stream_language('', 'en'))
        self.assertEqual(actual,[expected[0],bytes(round(16000*INTER_BLOCK_PAUSE_SECONDS)*2),expected[1]])
        self.assertEqual(self.closed,[True,True])

    def test_cancellation_closes_source(self):
        adapter=self.adapter();stream=adapter.synthesize_stream_language('', 'en')
        next(stream);stream.close();self.assertEqual(self.closed,[True])


if __name__=='__main__':unittest.main()
