"""Lexicon statuses, alias validation, single-pass matching and shipped asset."""
import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'perception'))
from plugins.moss_tts_runtime import lexicon
from plugins.moss_tts_runtime.text_normalization import normalize_text

class LexiconTests(unittest.TestCase):
    def tearDown(self):lexicon._rules.cache_clear()
    def test_shipped_asset_preserves_candidate_behaviour(self):
        entries=lexicon.load_lexicon();self.assertGreater(len(entries),50)
        self.assertEqual(normalize_text('HAMi、hami 和 Qwen 2.5、个人IP。','zh'),'哈米、哈米 和 千问二点五、个人I P。')
        self.assertEqual(normalize_text('ModelHubXC GPU NPU DCU MLU TensorRT ONNX','en'),'ModelHubXC GPU NPU DCU MLU TensorRT ONNX')
    def test_default_mode_does_not_apply_library(self):
        for language in ('zh','en','ja','de','fr','es','ru','th'):
            self.assertEqual(normalize_text('HAMi Qwen3 NVIDIA GPU IP',language,terms=False),'HAMi Qwen3 NVIDIA GPU IP')
    def test_protected_names_and_ordinary_words(self):
        for text in ('https://host/HAMi/Qwen3','`HAMi ID GPU`','a@HAMi.com','models/HAMi/file','myHAMi','HAMi-v2'):
            self.assertEqual(normalize_text(text,'zh'),text)
        self.assertEqual(normalize_text('send us it, US IT','en'),'send us it, US IT')
    def test_longest_alias_and_no_cascading(self):
        entries=tuple(dict(term=name,reading=reading,language='en',status='trial',aliases=aliases) for name,reading,aliases in [('Model','short',[]),('ModelHub','ID',['Model Hub']),('ID','I D',[])])
        with patch.object(lexicon,'load_lexicon',return_value=entries):
            self.assertEqual(lexicon.rewrite_terms('ModelHub Model Hub ID Model','en'),'ID ID I D short')
    def test_pending_and_rejected_are_not_applied(self):
        entries=tuple(dict(term=name,reading='bad',language='en',status=status) for name,status in [('Foo','pending'),('Bar','rejected')])
        with patch.object(lexicon,'load_lexicon',return_value=entries):self.assertEqual(lexicon.rewrite_terms('Foo Bar','en'),'Foo Bar')
        with patch.object(lexicon,'load_lexicon',return_value=()):self.assertEqual(normalize_text('Qwen3','zh'),'Qwen3')
    def test_invalid_asset_fails_instead_of_silently_changing_pronunciation(self):
        entry=dict(term='Foo',language='en',reading='foo',status='trial',source='test',evidence='test')
        for change in ({'status':'enabled'},{'reading':''},{'language':'xx'},{'aliases':['FOO']},{'aliases':'Foo'},{'case_sensitive':'false'}):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'terms.json';path.write_text(json.dumps(dict(schema_version=1,entries=[dict(entry,**change)])))
                with self.assertRaises(ValueError):lexicon.load_lexicon(path)

if __name__=='__main__':unittest.main()
