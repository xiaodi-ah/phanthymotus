"""Explicit years, grammatical cases and non-year boundaries."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'perception'))
from plugins.moss_tts_runtime.text_normalization import normalize_text
from plugins.moss_tts_runtime.daily_years import russian_year

class YearTests(unittest.TestCase):
    def test_german_contexts(self):
        self.assertEqual(normalize_text('Im Jahr 1998 erschien das Buch.','de',terms=False),'Im Jahr neunzehnhundertachtundneunzig erschien das Buch.')
        self.assertEqual(normalize_text('Ende 2002.','de',terms=False),'Ende zweitausendzwei.')
    def test_russian_cases(self):
        self.assertEqual(normalize_text('В 1998 году.','ru',terms=False),'В одна тысяча девятьсот девяносто восьмом году.')
        self.assertEqual(normalize_text('С 1998 года.','ru',terms=False),'С одна тысяча девятьсот девяносто восьмого года.')
        self.assertEqual(normalize_text('Конец 2000 года.','ru',terms=False),'Конец двухтысячного года.')
    def test_old_icu_forty_ending(self):
        self.assertTrue(russian_year(1940,'locative').endswith('сороковом'))
        self.assertTrue(russian_year(2040,'locative').endswith('сороковом'))
        self.assertTrue(russian_year(1940,'genitive').endswith('сорокового'))
        self.assertTrue(russian_year(1941,'locative').endswith('первом'))
    def test_not_a_supported_year_context(self):
        for language in ('de','ru'):
            for text in ('1998','ID 1998','in 01998','in 1998.5','in 1998-2000','2026-10-29','к 1998 году','версия1998 года','foo1998 года','in 1998er','model/1998 года','https://host/en/1998','`in 1998`','`в 1998 году`'):
                with self.subTest(language=language,text=text):self.assertEqual(normalize_text(text,language,terms=False),text)
    def test_other_languages_unchanged(self):
        for language in ('en','fr','es','ja','th'):
            for text in ('in 1998','en 1998','1998年','в 1998 году'):
                self.assertEqual(normalize_text(text,language,terms=False),text)
    def test_idempotence(self):
        for language,text in [('de','Im Jahr 1998.'),('ru','В 1940 году. С 2000 года.')]:
            spoken=normalize_text(text,language,terms=False)
            self.assertEqual(normalize_text(spoken,language,terms=False),spoken)

if __name__=='__main__':unittest.main()
