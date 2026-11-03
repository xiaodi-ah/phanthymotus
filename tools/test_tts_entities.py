"""Native identifier spellout and conservative boundaries, independent of board examples."""
import re,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'perception'))
from plugins.tts_entities import spoken_entities,spell_digits,digit_words,email_reading,_SYMBOLS
from plugins.tts_text_frontend import supertonic_text

LANGS=('zh','en','ja','de','fr','ru','es')


class Tests(unittest.TestCase):
    def test_identifiers_preserve_every_digit_in_every_language(self):
        prefixes={'zh':'验证码是：','en':'verification code: ','ja':'認証コード ','de':'Bestätigungscode: ',
                  'fr':'code de vérification fictif ','ru':'проверочный код ','es':'código de verificación ficticio '}
        for language in LANGS:
            raw=prefixes[language]+'009314'
            self.assertEqual(spoken_entities(raw,language),prefixes[language]+spell_digits('009314',language))
            digits=digit_words(language)
            pattern=re.compile('|'.join(re.escape(w) for w in sorted(digits,key=len,reverse=True)))
            for number in ['009314','87300210468829','100000000001']:
                recovered=''.join(str(digits.index(w)) for w in pattern.findall(spell_digits(number,language)))
                self.assertEqual(recovered,number)

    def test_email_preserves_characters_and_symbols(self):
        for language in LANGS:
            raw='ab_07+q-z@xy.org'
            actual=spoken_entities(raw,language)
            self.assertEqual(actual,email_reading(raw,language))
            for symbol in '_+-@.':self.assertIn(_SYMBOLS[language][symbol],actual)
            self.assertIn(digit_words(language)[0],actual);self.assertIn(digit_words(language)[7],actual)
            self.assertEqual(spoken_entities(actual,language),actual)
        for language,prefix in [('zh','邮箱'),('ja','メール')]:
            self.assertEqual(spoken_entities(prefix+'ab_07@example.org',language),prefix+email_reading('ab_07@example.org',language))
        self.assertEqual(email_reading('service_01@example.com','en'),'service underscore zero one at example dot com')

    def test_versions_dates_ips_paths_and_quantities_are_not_ids(self):
        for language in LANGS:
            for raw in ['Python3.12 v1.2.3 Model7.5','Python 3.12 version 2.7 版本3.6 バージョン4.2 версия 1.3','2027-04-18 14:25','192.168.3.14',
                        'https://host/3.14?email=a@b.com','`verification code 009314`',
                        '```text\na@b.com 3.14\n```','There are 1000000 users.',
                        '1,234.5 1.234,5','file/3.14.bin',r'C:\models\3.14\weights']:
                with self.subTest(language=language,raw=raw):self.assertEqual(spoken_entities(raw,language),raw)

    def test_plain_decimal_and_terminal_period(self):
        self.assertEqual(spoken_entities('Score 4.58.','en'),'Score four point five eight.')
        self.assertEqual(spoken_entities('增加3.6个百分点。','zh'),'增加三点六个百分点。')
        self.assertEqual(spoken_entities('3,6 points','fr'),'trois virgule six points')
        self.assertEqual(spoken_entities('3,600','fr'),'3,600')

    def test_numeric_units_and_formulas_keep_existing_grammar(self):
        from plugins.moss_tts_runtime.text_normalization import normalize_text
        self.assertEqual(supertonic_text('1.2A','en'),'one point two amperes')
        self.assertIn('squared',supertonic_text('a² + b² = c²','en'))
        self.assertIn('pour cent',supertonic_text('89.6%','fr'))
        raw='89.6% 1.2A 23.5℃'
        self.assertEqual(spoken_entities(raw,'zh'),raw)
        self.assertEqual(normalize_text(spoken_entities(raw,'zh'),'zh'),normalize_text(raw,'zh'))

    def test_serial_hyphen_and_phone_groups(self):
        self.assertEqual(spoken_entities('order number 0072-84','en'),
                         'order number '+spell_digits('0072','en')+' hyphen '+spell_digits('84','en'))
        self.assertEqual(spoken_entities('phone number 139-0012-4758','en'),
                         'phone number '+spell_digits('13900124758','en',True))
        self.assertEqual(spoken_entities('ordinary 009314','en'),'ordinary 009314')
        self.assertEqual(spoken_entities('verification code 12--34','en'),'verification code 12--34')
        self.assertEqual(spoken_entities('a%tag@example.com','en'),'a%tag@example.com')
        self.assertEqual(spoken_entities('联系电话13900124758，以及邮箱','zh'),
                         '联系电话一三九，零零一二，四七五八，以及邮箱')

    def test_other_languages_and_ordinary_prose_are_unchanged(self):
        self.assertEqual(spoken_entities('3.6 a@b.com','th'),'3.6 a@b.com')
        for language,text in [('zh','哈密管理计算资源。'),('en','Please close the window.'),('fr','Le robot est prêt.')]:
            self.assertEqual(spoken_entities(text,language),text)


if __name__=='__main__':unittest.main()
