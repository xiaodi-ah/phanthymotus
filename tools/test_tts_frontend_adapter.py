"""The real threaded adapter applies the selected frontend before tokenization."""
import os,sys,threading,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'perception'))
from plugins.moss_tts_runtime import MossAdapter,VOICES

class FrontendTests(unittest.TestCase):
    def exercise(self,text,language,frontend,next_request=None):
        encoded=[]
        def encode(text,out_type=int):encoded.append(text);return [1,2]
        def generate(request,on_frame):
            self.assertEqual(model.runtime.greedy_stop,language=='en' or language=='auto' and text.isascii())
            frame=[0]*16;on_frame([frame],0,frame);return [frame]
        model=MossAdapter.__new__(MossAdapter)
        model.text_frontend=frontend;model.stop_mode='greedy';model.language='auto';model.seed=42;model.lock=threading.Lock();model.reset_codec=lambda:None
        model.tokenizer=SimpleNamespace(encode=encode);model.voices={name:[[0]*16] for name in VOICES.values()}
        model.runtime=SimpleNamespace(manifest={'generation_defaults':{}},build_voice_clone_request_rows=lambda ref,tokens:{'inputIds':[[0]*17]*10},generate_audio_frames=generate)
        model.codec=SimpleNamespace(run_frames=lambda frames:(np.zeros((1,2,3840),np.float32),3840))
        result=model.synthesize(text,language)
        self.assertEqual(len(result),2560);self.assertTrue(encoded)
        if next_request:
            text,language=next_request
            self.assertEqual(len(model.synthesize(text,language)),2560)
        return encoded
    def test_explicit_off_preserves_raw_text(self):
        self.assertTrue(all(x=='GPU: 23°C' for x in self.exercise('GPU: 23°C','en','off')))
    def test_numbers_are_spoken_before_tokenization(self):
        encoded=self.exercise('GPU: 23°C','en','numbers')
        self.assertTrue(encoded);self.assertTrue(all(x=='GPU: twenty three degrees Celsius' for x in encoded))
    def test_terms_are_separate_from_number_only_mode(self):
        encoded=self.exercise('GPU: 23°C','en','numbers_terms')
        self.assertTrue(all(x=='G P U: twenty three degrees Celsius' for x in encoded))
    def test_auto_japanese_resolves_before_frontend(self):
        encoded=self.exercise('温度は23℃です。','auto','numbers')
        self.assertTrue(all(x=='温度はにじゅうさんどです。' for x in encoded))
    def test_other_languages_keep_sampled_stop(self):
        for language,text in [('de','Der Akku ist zu 85% geladen.'),('zh','电量是85%。'),('fr','Bonjour.')]:
            self.exercise(text,language,'numbers')
    def test_reused_model_resets_stop_policy_between_languages(self):
        self.exercise('Hello.','en','numbers',next_request=('你好。','zh'))
        self.exercise('你好。','zh','numbers',next_request=('Hello.','en'))
    def test_invalid_mode_fails_before_model_loading(self):
        with patch.dict(os.environ,{'MOSS_TTS_DELIVERY':'optimized','MOSS_TTS_TEXT_FRONTEND':'invalid'}):
            with self.assertRaisesRegex(ValueError,'MOSS_TTS_TEXT_FRONTEND'):MossAdapter('/missing')

if __name__=='__main__':unittest.main()
