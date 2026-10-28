"""The real threaded adapter applies the selected frontend before tokenization."""
import os,sys,threading,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'perception'))
from plugins.moss_tts_runtime import MossAdapter,VOICES

class FrontendTests(unittest.TestCase):
    def exercise(self,text,language,frontend,next_request=None,stop_mode='greedy',audio_value=0.,resolved_language=None,switching=False):
        encoded=[]
        def encode(text,out_type=int):encoded.append(text);return [1,2]
        def generate(request,on_frame):
            code=resolved_language or language
            greedy=stop_mode in ('greedy','guarded') and code=='en' or stop_mode=='guarded' and code=='ja'
            self.assertEqual(model.runtime.greedy_stop,greedy)
            self.assertEqual(model.runtime.guard_stop,stop_mode=='guarded' and not greedy)
            self.assertEqual(model.runtime.force_likely_eos,stop_mode=='guarded' and code in ('de','fr'))
            self.assertEqual(model.runtime.stop_guard_threshold,.95 if code=='es' else .99)
            if switching:
                expected=[1.,.8,.95,1.2] if code=='fr' else [1.5,1.7,.8,1.]
                self.assertEqual([model.runtime.manifest['generation_defaults'][k] for k in ('text_temperature','audio_temperature','audio_top_p','audio_repetition_penalty')],expected)
            frame=[0]*16;on_frame([frame],0,frame);return [frame]
        model=MossAdapter.__new__(MossAdapter)
        model.french_frame=None
        model.text_frontend=frontend;model.stop_mode=stop_mode;model.language='auto';model.seed=42;model.lock=threading.Lock();model.reset_codec=lambda:None
        model.tokenizer=SimpleNamespace(encode=encode);model.voices={name:[[0]*16] for name in VOICES.values()}
        model.runtime=SimpleNamespace(manifest={'generation_defaults':{}},build_voice_clone_request_rows=lambda ref,tokens:{'inputIds':[[0]*17]*10},generate_audio_frames=generate)
        if switching:
            model.french_frame=Path('frame_fr.plan');model.legacy_frame=Path('frame.plan');model.frame_profile='legacy';model.cuda_graph=True
            model.runtime.sessions={'local_fixed_sampled_frame':object()}
            def load(path,cuda_graph):
                self.assertNotIn('local_fixed_sampled_frame',model.runtime.sessions)
                return SimpleNamespace(path=path)
            self.enterContext(patch.dict(sys.modules,{'plugins.moss_tts_runtime.trt_fused_session':SimpleNamespace(FusedSession=load)}))
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
    def test_russian_chunks_finish_with_a_period_before_encoding(self):
        self.assertEqual(self.exercise('ModelHub XC,','ru','numbers')[-1],'ModelHub XC.')
        self.assertEqual(self.exercise('ModelHub XC,','ru','off')[-1],'ModelHub XC,')
        self.assertEqual(self.exercise('ModelHub XC, next sentence.','ru','numbers')[-1],'ModelHub XC, next sentence.')
        for language in ('en','fr'):
            self.assertEqual(self.exercise('ModelHub XC,',language,'numbers')[-1],'ModelHub XC,')
    def test_explicit_off_preserves_raw_text(self):
        self.assertTrue(all(x=='GPU: 23°C' for x in self.exercise('GPU: 23°C','en','off')))
    def test_numbers_are_spoken_before_tokenization(self):
        encoded=self.exercise('GPU: 23°C','en','numbers')
        self.assertTrue(encoded);self.assertTrue(all(x=='GPU: twenty-three degrees Celsius' for x in encoded))
    def test_terms_are_separate_from_number_only_mode(self):
        encoded=self.exercise('ID: 0012.','en','numbers_terms')
        self.assertTrue(all(x=='I D: zero zero one two.' for x in encoded))
    def test_approved_readings_apply_in_default_frontend(self):
        encoded=self.exercise('HaMi vGPU 使用 Qwen3。','zh','numbers')
        self.assertTrue(all(x=='哈密 V G P U 使用 Qwen3。' for x in encoded))
    def test_chinese_terms_and_quantities_precede_chunking(self):
        encoded=self.exercise('HAMi 使用 Qwen3，温度23℃。','zh','numbers_terms')
        self.assertTrue(all(x=='哈密 使用 千问三，温度二十三摄氏度。' for x in encoded))
    def test_chinese_math_reaches_actual_model_in_default_mode(self):
        encoded=self.exercise('y = 2x - 3，误差±0.5℃，电压5V。','zh','numbers')
        self.assertTrue(all(x=='Y等于二乘以X减三，误差正负零点五摄氏度，电压五伏特。' for x in encoded))
        self.assertTrue(all(x=='y = 2x - 3。' for x in self.exercise('y = 2x - 3。','zh','off')))
    def test_english_product_names_are_preserved(self):
        encoded=self.exercise('Qwen3 uses ModelHubXC.','en','numbers_terms')
        self.assertTrue(all(x=='Qwen3 uses ModelHubXC.' for x in encoded))
    def test_auto_japanese_resolves_before_frontend(self):
        encoded=self.exercise('温度は23℃です。','auto','numbers')
        self.assertTrue(all(x=='温度は摂氏二十三度です。' for x in encoded))
    def test_explicit_sample_keeps_sampled_stop(self):
        for language,text in [('de','Der Akku ist zu 85% geladen.'),('zh','电量是85%。'),('fr','Bonjour.')]:
            self.exercise(text,language,'numbers',stop_mode='sample')
    def test_default_stopping_policy_for_all_languages(self):
        for language in VOICES:
            self.exercise('Test.',language,'numbers',stop_mode='guarded')
    def test_legacy_greedy_keeps_english_only_policy(self):
        for language in VOICES:
            self.exercise('Test.',language,'numbers',stop_mode='greedy')
    def test_latin_auto_resolves_before_numbers_and_stopping(self):
        encoded=self.exercise('Die Temperatur ist 23°C.','auto','numbers',resolved_language='de')
        self.assertTrue(all(x=='Die Temperatur ist dreiundzwanzig Grad Celsius.' for x in encoded))
        for code,text in [('fr','Le robot est prêt.'),('es','El robot está listo.')]:
            self.exercise(text,'auto','numbers',stop_mode='guarded',resolved_language=code)
    def test_all_six_language_math_precedes_tokenization(self):
        for code,expected in [('fr','vingt-trois degrés Celsius, quatre-vingt-cinq pour cent.'),('ru','двадцать три градуса Цельсия, восемьдесят пять процентов.')]:
            self.assertTrue(all(x==expected for x in self.exercise('23°C, 85%.',code,'numbers')))
            self.assertTrue(all(x==expected for x in self.exercise('23°C, 85%.',code,'numbers_terms')))
        for code,expected in [('es','equis igual a dos'),('de','Iks gleich zwei'),('ja','エックスイコール二'),('en','X equals two')]:
            self.assertTrue(all(x==expected for x in self.exercise('x = 2',code,'numbers')))
    def test_reused_model_resets_stop_policy_between_languages(self):
        self.exercise('Hello.','en','numbers',next_request=('你好。','zh'))
        self.exercise('你好。','zh','numbers',next_request=('Hello.','en'))
    def test_guard_resets_between_languages(self):
        self.exercise('Bonjour.','fr','numbers',next_request=('Здравствуйте.','ru'),stop_mode='guarded')
        self.exercise('Bonjour.','fr','numbers',next_request=('Hello.','en'),stop_mode='guarded')
        self.exercise('Hola.','es','numbers',next_request=('Bonjour.','fr'),stop_mode='guarded')
        self.exercise('你好。','zh','numbers',next_request=('Hello.','en'),stop_mode='guarded')
        self.exercise('Hello.','en','numbers',next_request=('Bonjour.','fr'),stop_mode='guarded')
        self.exercise('Hallo.','de','numbers',next_request=('Bonjour.','fr'),stop_mode='guarded')
        self.exercise('你好。','zh','numbers',next_request=('こんにちは。','ja'),stop_mode='guarded')
    def test_invalid_mode_fails_before_model_loading(self):
        with patch.dict(os.environ,{'MOSS_TTS_DELIVERY':'optimized','MOSS_TTS_TEXT_FRONTEND':'invalid'}):
            with self.assertRaisesRegex(ValueError,'MOSS_TTS_TEXT_FRONTEND'):MossAdapter('/missing')
    def test_frame_switch_resets_sampling_without_retaining_two_engines(self):
        for first,second in [('fr','en'),('en','fr'),('fr','zh'),('de','fr')]:
            self.exercise('Test.',first,'numbers',next_request=('Test.',second),stop_mode='guarded',switching=True)
    def test_failed_engine_switch_is_reported_and_next_request_can_retry(self):
        model=MossAdapter.__new__(MossAdapter);model.french_frame=Path('fr.plan');model.legacy_frame=Path('legacy.plan');model.cuda_graph=True;model.frame_profile='legacy'
        model.runtime=SimpleNamespace(sessions={'local_fixed_sampled_frame':object()},manifest={'generation_defaults':{}})
        with patch.dict(sys.modules,{'plugins.moss_tts_runtime.trt_fused_session':SimpleNamespace(FusedSession=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('load failed')))}):
            with self.assertRaisesRegex(RuntimeError,'load failed'):model.select_frame('fr')
        self.assertIsNone(model.frame_profile);self.assertNotIn('local_fixed_sampled_frame',model.runtime.sessions)
        restored=object()
        with patch.dict(sys.modules,{'plugins.moss_tts_runtime.trt_fused_session':SimpleNamespace(FusedSession=lambda *a,**k:restored)}):model.select_frame('en')
        self.assertEqual(model.frame_profile,'legacy');self.assertIs(model.runtime.sessions['local_fixed_sampled_frame'],restored)

if __name__=='__main__':unittest.main()
