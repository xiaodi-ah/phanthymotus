"""Auto routing must use language cues, preserve explicit choices and avoid name-only guesses."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'perception'))
from plugins.moss_tts_runtime import language_for

class LanguageTests(unittest.TestCase):
    def test_clear_latin_sentences(self):
        cases={
            'de':['Guten Morgen. Bitte schließen Sie das Fenster.','Die Temperatur ist 23°C und der Akku ist zu 85% geladen.','Das Gerät ist bereit.'],
            'fr':['Bonjour, bienvenue dans notre service vocal.','La température de la pièce est de 23°C et la batterie est chargée à 85%.','Le robot utilise NVIDIA Jetson pour traiter les données.'],
            'es':['Buenos días. Por favor, cierra la ventana.','La temperatura de la habitación es de 23°C y la batería tiene un 85% de carga.','El robot está listo.'],
            'en':['The temperature is 23°C and the battery is at 85%.','Please play Die Hard for me.','This film is called Bonjour and the actor is French.']}
        for code,texts in cases.items():
            for text in texts:
                with self.subTest(text=text):self.assertEqual(language_for(text),code)
    def test_ambiguous_names_numbers_and_cues_in_protected_values(self):
        for text in ['NVIDIA Jetson GPU 85%.','23°C','Bonjour','robot','GPU https://example.com/le/les/des/du','GPU `el está`','GPU le.les.des@example.com']:
            with self.subTest(text=text):self.assertEqual(language_for(text),'en')
    def test_explicit_language_wins(self):
        for code in ('zh','ja','ru','en','es','fr','th','de'):
            self.assertEqual(language_for('Le robot est prêt.',code),code)
        self.assertEqual(language_for('Hello.','DE_de'),'de')
        with self.assertRaises(ValueError):language_for('Hello.','xx')
    def test_scripts_still_take_priority_over_english_terms(self):
        for code,text in [('zh','请检查 NVIDIA Jetson。'),('ja','NVIDIA Jetsonを確認してください。'),('ru','Проверьте NVIDIA Jetson.'),('th','สวัสดี NVIDIA Jetson')]:
            self.assertEqual(language_for(text),code)

if __name__=='__main__':unittest.main()
