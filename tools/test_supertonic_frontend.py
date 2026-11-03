"""Math semantics survive native normalization; trial terms and ordinary text remain bounded."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'perception'))
from plugins.tts_text_frontend import supertonic_text,spoken_identifier,standalone_digits
from plugins.supertonic_runtime.vendor import UnicodeProcessor


class Tests(unittest.TestCase):
    def test_powers_are_spoken_before_nfkd(self):
        processor=UnicodeProcessor.__new__(UnicodeProcessor)
        for lang,word in [('en','squared'),('ja','二乗'),('de','Quadrat'),('fr','carré'),('ru','квадрате'),('es','cuadrado')]:
            result=supertonic_text('a² + b² = c²; S = πr²',lang)
            self.assertIn(word,result)
            self.assertNotIn('a2',processor._preprocess_text(result,lang))
    def test_unapproved_lexicon_not_enabled_and_prose_preserved(self):
        for lang,text in [('en','Please close the window.'),('de','Die Tür ist geschlossen.'),('fr','Le robot est prêt.'),
                          ('ru','Сегодня хорошая погода.'),('ja','今日は晴れています。'),('es','La puerta está cerrada.')]:
            self.assertEqual(supertonic_text(text,lang),text)
        self.assertEqual(supertonic_text('NVIDIA ModelHub GPU','en'),'NVIDIA ModelHub GPU')
    def test_identifiers_keep_leading_zero_and_quantity_context(self):
        self.assertEqual(spoken_identifier('005917','en'),'zero zero five nine one seven')
        self.assertEqual(spoken_identifier('１２３４５','zh'),'一二三四五')
        self.assertEqual(spoken_identifier('13800138000','zh'),'一三八，零零一三，八零零零。')
        self.assertEqual(spoken_identifier('There are 1000000 users.','en'),'There are 1000000 users.')
        self.assertIsNone(standalone_digits('3.14'))
        self.assertIsNone(standalone_digits('x2'))


if __name__=='__main__':unittest.main()
