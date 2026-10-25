"""The real threaded adapter applies the selected frontend before tokenization."""
import os,sys,threading,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'perception'))
from plugins.moss_tts_runtime import MossAdapter,VOICES

class FrontendTests(unittest.TestCase):
    def exercise(self,text,language,frontend,next_request=None,stop_mode='greedy',audio_value=0.,resolved_language=None):
        encoded=[]
        def encode(text,out_type=int):encoded.append(text);return [1,2]
        def generate(request,on_frame):
            english=(resolved_language or language)=='en'
            self.assertEqual(model.runtime.greedy_stop,stop_mode in ('greedy','guarded') and english)
            self.assertEqual(model.runtime.guard_stop,stop_mode=='guarded' and not english)
            frame=[0]*16;on_frame([frame],0,frame);return [frame]
        model=MossAdapter.__new__(MossAdapter)
        model.text_frontend=frontend;model.stop_mode=stop_mode;model.language='auto';model.seed=42;model.lock=threading.Lock();model.reset_codec=lambda:None
        model.tokenizer=SimpleNamespace(encode=encode);model.voices={name:[[0]*16] for name in VOICES.values()}
        model.runtime=SimpleNamespace(manifest={'generation_defaults':{}},build_voice_clone_request_rows=lambda ref,tokens:{'inputIds':[[0]*17]*10},generate_audio_frames=generate)
        model.codec=SimpleNamespace(run_frames=lambda frames:(np.full((1,2,3840),audio_value,np.float32),3840))
        result=model.synthesize(text,language)
        self.assertEqual(len(result),2560);self.assertTrue(encoded)
        self.assertEqual(result[-2:],b'\0\0')
        if audio_value:self.assertGreater(np.max(np.abs(np.frombuffer(result,dtype='<i2').astype(int))),0)
        if next_request:
            text,language=next_request
            self.assertEqual(len(model.synthesize(text,language)),2560)
        return encoded
    def test_sentence_tail_is_flushed_in_real_threaded_adapter(self):
        self.exercise('Hello.','en','numbers',audio_value=.1)
    def test_explicit_off_preserves_raw_text(self):
        self.assertTrue(all(x=='GPU: 23°C' for x in self.exercise('GPU: 23°C','en','off')))
    def test_numbers_are_spoken_before_tokenization(self):
        encoded=self.exercise('GPU: 23°C','en','numbers')
        self.assertTrue(encoded);self.assertTrue(all(x=='GPU: twenty three degrees Celsius' for x in encoded))
    def test_terms_are_separate_from_number_only_mode(self):
        encoded=self.exercise('ID: 0012.','en','numbers_terms')
        self.assertTrue(all(x=='I D: zero zero one two.' for x in encoded))
    def test_chinese_terms_and_quantities_precede_chunking(self):
        encoded=self.exercise('HAMi 使用 Qwen3，温度23℃。','zh','numbers_terms')
        self.assertTrue(all(x=='哈米 使用 千问三，温度二十三摄氏度。' for x in encoded))
    def test_english_product_names_are_preserved(self):
        encoded=self.exercise('Qwen3 uses ModelHubXC.','en','numbers_terms')
        self.assertTrue(all(x=='Qwen3 uses ModelHubXC.' for x in encoded))
    def test_auto_japanese_resolves_before_frontend(self):
        encoded=self.exercise('温度は23℃です。','auto','numbers')
        self.assertTrue(all(x=='温度はにじゅうさんどです。' for x in encoded))
    def test_other_languages_keep_sampled_stop(self):
        for language,text in [('de','Der Akku ist zu 85% geladen.'),('zh','电量是85%。'),('fr','Bonjour.')]:
            self.exercise(text,language,'numbers')
    def test_latin_auto_resolves_before_numbers_and_stopping(self):
        encoded=self.exercise('Die Temperatur ist 23°C.','auto','numbers',resolved_language='de')
        self.assertTrue(all(x=='Die Temperatur ist dreiundzwanzig Grad Celsius.' for x in encoded))
        for code,text in [('fr','Le robot est prêt.'),('es','El robot está listo.')]:
            self.exercise(text,'auto','numbers',stop_mode='guarded',resolved_language=code)
    def test_french_russian_quantity_candidates_remain_opt_in(self):
        for code,expected in [('fr','vingt-trois degrés Celsius, quatre-vingt-cinq pour cent.'),('ru','двадцать три градуса Цельсия, восемьдесят пять процентов.')]:
            self.assertTrue(all(x=='23°C, 85%.' for x in self.exercise('23°C, 85%.',code,'numbers')))
            self.assertTrue(all(x==expected for x in self.exercise('23°C, 85%.',code,'numbers_terms')))
    def test_reused_model_resets_stop_policy_between_languages(self):
        self.exercise('Hello.','en','numbers',next_request=('你好。','zh'))
        self.exercise('你好。','zh','numbers',next_request=('Hello.','en'))
    def test_guard_resets_between_languages(self):
        self.exercise('你好。','zh','numbers',next_request=('Hello.','en'),stop_mode='guarded')
        self.exercise('Hello.','en','numbers',next_request=('Bonjour.','fr'),stop_mode='guarded')
    def test_invalid_mode_fails_before_model_loading(self):
        with patch.dict(os.environ,{'MOSS_TTS_DELIVERY':'optimized','MOSS_TTS_TEXT_FRONTEND':'invalid'}):
            with self.assertRaisesRegex(ValueError,'MOSS_TTS_TEXT_FRONTEND'):MossAdapter('/missing')

if __name__=='__main__':unittest.main()
