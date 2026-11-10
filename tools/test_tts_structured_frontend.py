"""Entity boundaries survive the full frontend, not just the entity helper."""
import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'perception'))
from plugins.tts_text_frontend import frontend_segments,prepare_text,supertonic_text
from plugins.tts_entities import email_reading,spell_digits

LANGUAGES=('zh','en','ja','de','fr','ru','es')


class Tests(unittest.TestCase):
    def test_raw_partition_and_email_digit_ownership(self):
        raw='Python 3.12; ab_07+q-z@xy.org; x = 3²; 12V.'
        for language in LANGUAGES:
            with self.subTest(language=language):
                parts=frontend_segments(raw,language)
                self.assertEqual(''.join(p.raw for p in parts),raw)
                for p in parts:self.assertEqual(raw[p.start:p.end],p.raw)
                self.assertEqual([p.kind for p in parts].count('email'),1)
                self.assertIn(email_reading('ab_07+q-z@xy.org',language),prepare_text(raw,language))
                self.assertEqual(parts[0].spoken,'Python 3.12')
                self.assertNotIn('3²',prepare_text(raw,language))

    def test_full_frontend_protects_versions_urls_and_code(self):
        raw='Python 3.12; version 2.7; `x = 3²; GPU 23℃`; https://host/a²?q=3.14'
        for language in LANGUAGES:
            self.assertEqual(prepare_text(raw,language,terms=True),raw)

    def test_identifiers_are_not_numbers_or_formulas(self):
        labels={'zh':'验证码：','en':'verification code: ','ja':'認証コード ',
                'de':'Bestätigungscode: ','fr':'code de vérification ',
                'ru':'проверочный код ','es':'código de verificación '}
        for language,label in labels.items():
            parts=frontend_segments(label+'001203; x = 4²; ab_07@example.org',language)
            entity=next(p for p in parts if p.kind=='identifier')
            self.assertEqual(entity.spoken,label+spell_digits('001203',language))
            self.assertIn(email_reading('ab_07@example.org',language),''.join(p.spoken for p in parts))

    def test_quantity_and_formula_are_parsed_before_prose_decimals(self):
        self.assertEqual(prepare_text('x = 2.5; 1.2A','en'),
                         'X equals two point five and one point two amperes')
        self.assertEqual(prepare_text('Score 4.58.','en'),'Score four point five eight.')
        self.assertEqual(prepare_text('89.6% 23.5℃','zh'),'百分之八十九点六 二十三点五摄氏度')

    def test_ordinary_text_and_thai_fallback(self):
        for language in LANGUAGES+('th',):
            self.assertEqual(prepare_text('Hello world!',language),'Hello world!')
        raw='สวัสดี 3.14 ab_07@example.org x²'
        self.assertEqual(prepare_text(raw,'th'),raw)
        with self.assertRaises(TypeError):prepare_text(None,'en')
        self.assertEqual(frontend_segments('','en'),[])

    def test_standalone_identifier_trace_keeps_original_offsets(self):
        raw='００１２０３'
        parts=frontend_segments(raw,'en')
        self.assertEqual(len(parts),1)
        self.assertEqual(parts[0].raw,raw)
        self.assertEqual(parts[0].end,len(raw))
        self.assertEqual(parts[0].spoken,'zero zero one two zero three')
        self.assertEqual(prepare_text('12345','th'),'12345')


if __name__=='__main__':unittest.main()
