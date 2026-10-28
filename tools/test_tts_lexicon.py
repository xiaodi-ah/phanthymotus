"""Lexicon statuses, alias validation, single-pass matching and shipped asset."""
import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'perception'))
from plugins.moss_tts_runtime import lexicon
from plugins.moss_tts_runtime.text_normalization import normalize_text

class LexiconTests(unittest.TestCase):
    def setUp(self):lexicon._rules.cache_clear()
    def tearDown(self):lexicon._rules.cache_clear()
    def test_shipped_asset_preserves_candidate_behaviour(self):
        entries=lexicon.load_lexicon();self.assertGreater(len(entries),50)
        self.assertEqual(normalize_text('HAMi、hami 和 Qwen 2.5、个人IP。','zh'),'哈密、哈密 和 千问二点五、个人I P。')
        self.assertEqual(normalize_text('ModelHubXC GPU NPU DCU MLU TensorRT ONNX','en',terms=False),'ModelHubXC GPU NPU DCU MLU TensorRT ONNX')
    def test_default_mode_applies_only_approved_readings(self):
        for language in ('zh','en','ja','de','fr','es','ru','th'):
            self.assertEqual(normalize_text('HAMi Qwen3 NVIDIA GPU IP',language,terms=False),('哈密 Qwen3 NVIDIA G P U IP' if language=='zh' else 'HAMi Qwen3 NVIDIA GPU IP'))
    def test_chinese_technical_term_candidates_preserve_word_identity(self):
        raw='GPT、BERT、T5、LoRA、GPU、RLHF、DPO、RAG、Self-Attention。'
        self.assertEqual(normalize_text(raw,'zh',terms=False),'G P T、Bert、T five、Lo Ra、G P U、R L H F、D P O、rag、Self Attention。')
        self.assertEqual(normalize_text(raw,'zh',terms=True),'G P T、Bert、T five、Lo Ra、G P U、R L H F、D P O、rag、Self Attention。')
        self.assertEqual(normalize_text('FLUX GPT','zh',terms=False),'Flux G P T')
        for raw in ('myGPT','T5-v2','models/GPT/weights','`RLHF GPT`','https://host/GPT','RAG@example.com'):
            self.assertEqual(normalize_text(raw,'zh',terms=True),raw)
    def test_user_reading_and_letter_spelling_preserve_other_text(self):
        for spelling in ('HAMi','HaMi','hami'):
            self.assertEqual(normalize_text(spelling+' vGPU 使用 23℃ 和 Qwen3。','zh',terms=False),'哈密 V G P U 使用 二十三摄氏度 和 Qwen3。')
        self.assertEqual(normalize_text('HAMi vGPU uses 2GB.','en',terms=False),'HAMi V G P U uses two gigabytes.')
        for text in ('https://host/HAMi/vGPU','`HaMi vGPU`','hami@vgpu.com','models/vGPU/weights','vGPU-v2','myvGPU'):
            self.assertEqual(normalize_text(text,'zh',terms=False),text)
    def test_approved_rules_do_not_enable_unreviewed_trials(self):
        entries=tuple(dict(term=name,reading=value,language='en',status=status) for name,value,status in [('Foo','Bar','approved'),('Bar','trial','trial')])
        with patch.object(lexicon,'load_lexicon',return_value=entries):
            self.assertEqual(lexicon.rewrite_terms('Foo Bar','en',include_trials=False),'Bar Bar')
            self.assertEqual(lexicon.rewrite_terms('Foo Bar','en',include_trials=True),'Bar trial')
    def test_protected_names_and_ordinary_words(self):
        for text in ('https://host/HAMi/Qwen3','`HAMi ID GPU`','a@HAMi.com','models/HAMi/file','myHAMi','HAMi-v2'):
            self.assertEqual(normalize_text(text,'zh'),text)
        self.assertEqual(normalize_text('send us it, US IT','en'),'send us it, US IT')
    def test_multilingual_candidate_inventory_preserves_words_and_rejected_english(self):
        for language in ('en','ja','de','fr','ru','es'):
            entries=[e for e in lexicon.load_lexicon() if e['language']==language and e['status']=='trial']
            self.assertGreaterEqual(len(entries),60)
            self.assertEqual(normalize_text('BERT CLIP RAG FLUX CUDA ONNX',language),'BERT CLIP RAG FLUX CUDA ONNX')
            self.assertNotEqual(normalize_text('GPT LLM API ASR MCP',language), 'GPT LLM API ASR MCP')
        self.assertEqual(normalize_text('GPU NPU DCU MLU','en'),'GPU NPU DCU MLU')
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
