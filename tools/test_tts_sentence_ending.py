"""Finalization preserves all PCM samples and affects only the final 12ms."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'perception'))
from plugins.moss_tts_runtime.resample import SentenceEnding

class EndingTests(unittest.TestCase):
    def test_arbitrary_chunking_matches_whole_and_preserves_duration(self):
        rng=np.random.default_rng(42)
        for count in (0,1,2,191,192,193,1280,2049):
            pcm=rng.integers(-32768,32768,count,dtype=np.int16).astype('<i2').tobytes()
            whole=SentenceEnding();expected=whole.process(pcm)+whole.finish()
            stream=SentenceEnding();parts=[];offset=0
            for size in (2,4,18,300,2000,4096):
                parts.append(stream.process(pcm[offset:offset+size]));offset+=size
            parts.append(stream.process(pcm[offset:]));parts.append(stream.finish());actual=b''.join(parts)
            self.assertEqual(actual,expected);self.assertEqual(len(actual),len(pcm))
            self.assertEqual(actual[:max(0,len(pcm)-384)],pcm[:max(0,len(pcm)-384)])
            if count:self.assertEqual(actual[-2:],b'\0\0')
    def test_finished_and_misaligned_input_rejected(self):
        stream=SentenceEnding()
        with self.assertRaises(ValueError):stream.process(b'x')
        stream.finish()
        with self.assertRaises(RuntimeError):stream.finish()
        with self.assertRaises(RuntimeError):stream.process(b'')
    def test_ramp_never_amplifies_or_changes_sign(self):
        stream=SentenceEnding();pcm=np.full(1280,-10000,dtype='<i2').tobytes();actual=np.frombuffer(stream.process(pcm)+stream.finish(),'<i2')
        self.assertTrue(np.all(actual<=0));self.assertTrue(np.all(actual>=-10000));self.assertTrue(np.all(np.diff(actual[-192:])>=0))

if __name__=='__main__':unittest.main()
