"""Public-inventory coverage, independently written math expectations, and protected inputs."""
import json,sys,time,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'perception'))
from plugins.moss_tts_runtime.text_normalization import integer_words,normalize_text
from plugins.moss_tts_runtime import zh_math
from plugins.moss_tts_runtime.lexicon import load_lexicon,rewrite_terms

class ChineseMathTests(unittest.TestCase):
    def test_independent_category_fixtures_and_idempotence(self):
        cases=json.loads((ROOT/'tools/fixtures/tts_zh_frontend_cases.json').read_text())
        self.assertGreaterEqual(len(cases),70);self.assertGreaterEqual(len({c['category'] for c in cases}),9)
        for case in cases:
            with self.subTest(id=case['id'],raw=case['input']):
                value=normalize_text(case['input'],'zh',terms=False)
                self.assertEqual(value,case['expected'])
                self.assertEqual(normalize_text(value,'zh',terms=False),value)
    def test_dates_code_paths_versions_and_ambiguous_notation(self):
        for raw in ('y=2x-3','x = up','config = value','x = 0004','file/A = 4','x = 1.2.3',
                    '`y = 2x - 3; 5V`','```python\ny = 2*x - 3\n```',
                    'https://host/a²?y=4','x = 4 /path','x = 2_000',
                    'Python3.12 v1.2.3 Orin-8GB TTS-100M','2026-10-28 14:25 ID: 0012-34 2/3',
                    'RX5V 5V.bin board/5V.bin',r'C:\models\5V\weights',
                    '公式2026/10/28，时间12:05，版本1.2.3。','计算1,234.5 + 2',
                    r'公式 $\unknown{x}$','公式x = (y + 1','公式x = 2foo',
                    '公式x = ∑y','数学 G P T','ϕx = 3','∅ + x = 3'):
            with self.subTest(raw=raw):self.assertEqual(normalize_text(raw,'zh',terms=False),raw)
        for raw in ('计算机进行复杂模拟。2007年，新的范式出现。','数学发展史：2007年。','在计算机上设置x=4。'):
            self.assertEqual(normalize_text(raw,'zh',terms=False),raw)
    def test_si_symbol_case_and_complete_public_unit_inventory(self):
        self.assertNotEqual(zh_math.UNITS['mW'],zh_math.UNITS['MW'])
        self.assertNotEqual(zh_math.UNITS['mV'],zh_math.UNITS['MV'])
        self.assertGreaterEqual(len(zh_math.UNITS),150)
        self.assertNotIn('ds',zh_math.UNITS);self.assertNotIn('ω',zh_math.UNITS)
        for symbol in zh_math.UNITS:
            raw='数量5'+symbol
            self.assertNotEqual(normalize_text(raw,'zh',terms=False),raw,symbol)
    def test_entire_public_glossary_has_runtime_entries(self):
        catalog=json.loads((ROOT/'docs/tts_frontend_resources/terminology_catalog.json').read_text())
        self.assertEqual(catalog['source_terms'],148);self.assertEqual(len({c['category'] for c in catalog['entries']}),12)
        names={e['term'].casefold() for e in load_lexicon() if e['language']=='zh'}
        for record in catalog['entries']:
            for term in record['terms']:
                if term not in ('Top','AI/LLM'):self.assertIn(term.casefold(),names)
        self.assertEqual(rewrite_terms('GPT BERT CLIP RAG','zh',include_trials=False),'G P T Bert clip rag')
    def test_parser_limits_and_no_eval(self):
        for raw in ('公式'+('('*50)+'x'+(')'*50),'公式'+('x+'*500)+'x',"公式 __import__('os').system('false')"):
            started=time.monotonic();self.assertEqual(normalize_text(raw,'zh',terms=False),raw)
            self.assertLess(time.monotonic()-started,2)
        for n,words in [(10010,'一万零一十'),(100000000,'一亿'),(100000001,'一亿零一'),(10**12,'一万亿')]:self.assertEqual(integer_words(n,'zh'),words)
    def test_other_languages_have_separate_math_frontends(self):
        self.assertEqual(normalize_text('x = (y+1)^2','en',terms=False),'X equals open parenthesis Y plus one close parenthesis squared')
        for language in ('en','de','ja','fr','ru','es'):
            self.assertNotEqual(normalize_text('3mW 4MW',language,terms=False),'3mW 4MW')
        self.assertEqual(normalize_text('3mW 4MW','th',terms=False),'3mW 4MW')

if __name__=='__main__':unittest.main()
