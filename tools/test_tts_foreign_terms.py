"""Candidate names stay disabled; selected trial rules respect structured boundaries."""
import sys,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'perception'))
from plugins.moss_tts_runtime import VOICES,lexicon
from plugins.tts_text_frontend import prepare_text

TARGETS={('4Paradigm','ja'),('ModelHub','fr'),('ModelHubXC','fr'),('Qianfan','en'),('4Paradigm','en')}


class Tests(unittest.TestCase):
    def setUp(self):lexicon._rules.cache_clear()
    def tearDown(self):lexicon._rules.cache_clear()
    def selected(self,text,language):
        entries=tuple(dict(e,status='approved') if (e['term'],e['language']) in TARGETS else e for e in lexicon.load_lexicon())
        with patch.object(lexicon,'load_lexicon',return_value=entries):return prepare_text(text,language)
    def test_trials_are_inactive_in_production(self):
        for language,text in [('ja','4Paradigmへようこそ。'),('fr','ModelHub XC et ModelHub.'),('en',"Baidu Qianfan and 4Paradigm's platform.")]:
            self.assertEqual(prepare_text(text,language),text)
    def test_selected_rules_keep_possessive_suffixes(self):
        self.assertEqual(self.selected("4Paradigm's and 4Paradigm’s models.",'en'),"four paradigm's and four paradigm’s models.")
    def test_longest_product_alias_is_one_pass(self):
        self.assertEqual(self.selected('ModelHub XC et ModelHub.','fr'),'Model Hub ixe cé et Model Hub.')
        self.assertEqual(self.selected('4Paradigmの製品です。','ja'),'フォーパラダイムの製品です。')
    def test_urls_code_paths_and_larger_words_are_protected(self):
        for language,name in [('ja','4Paradigm'),('fr','ModelHub'),('en','Qianfan')]:
            text=f'https://host/{name} `{name}` my{name} models/{name}/file'
            self.assertEqual(self.selected(text,language),text)
    def test_selected_chinese_reference_preserves_term_processing(self):
        self.assertEqual(VOICES['zh'],'Zhiming')
        self.assertEqual(prepare_text('哈密使用 vGPU。','zh'),'哈密使用 V G P U。')


if __name__=='__main__':unittest.main()
