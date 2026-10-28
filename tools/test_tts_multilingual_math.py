"""Native spellout, mathematical scope and conservative prose boundaries."""
import sys,unittest,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'perception'))
from plugins.moss_tts_runtime.text_normalization import normalize_text
from plugins.moss_tts_runtime import multilingual_math as math,icu_numbers
LANGUAGES=('en','ja','de','fr','ru','es')

class MultilingualMathTests(unittest.TestCase):
 def test_independent_equations(self):
  expected={'en':'The formula is Y equals two times X minus three.','de':'Die Formel lautet Y gleich zwei mal X minus drei.','fr':'La formule est Y égale deux fois X moins trois.','es':'La fórmula es ye igual a dos por equis menos tres.','ru':'Формула: игрек равно два умножить на икс минус три.','ja':'数式はワイイコール二かけるエックスマイナス三。'}
  inputs={'en':'The formula is y = 2x - 3.','de':'Die Formel lautet y = 2x - 3.','fr':'La formule est y = 2x - 3.','es':'La fórmula es y = 2x - 3.','ru':'Формула: y = 2x - 3.','ja':'数式はy = 2x - 3。'}
  # Literal mathematical variable names follow each locale's letter table.
  expected['de']='Die Formel lautet Ypsilon gleich zwei mal Iks minus drei.'
  expected['fr']='La formule est i-grec égale deux fois ixe moins trois.'
  for language in LANGUAGES:
   with self.subTest(language=language):self.assertEqual(normalize_text(inputs[language],language,terms=False),expected[language])
 def test_decimal_and_grammatical_unit_forms(self):
  cases=[('en','23.5°C; 1 s; 2 s','twenty-three point five degrees Celsius; one second; two seconds'),('de','23,5°C; 1 s; 2 s','dreiundzwanzig Komma fünf Grad Celsius; eine Sekunde; zwei Sekunden'),('fr','0,5m; 1 s; 2 s','zéro virgule cinq mètre; une seconde; deux secondes'),('es','21m; 1 s; 2 s','veintiún metros; un segundo; dos segundos'),('ru','1 s; 2 s; 5 s; 21 s; 0,5m','одна секунда; две секунды; пять секунд; двадцать одна секунда; ноль целых пять десятых метра'),('ja','1s、2m、23℃','一秒、二 メートル、摂氏二十三度')]
  for language,raw,expected in cases:
   with self.subTest(language=language):self.assertEqual(normalize_text(raw,language,terms=False),expected)
 def test_russian_math_letters_differ_from_english_initialisms(self):
  self.assertEqual(normalize_text('x = y','ru',terms=False),'икс равно игрек')
  self.assertEqual(math.DATA['ru']['letters']['X'],'экс')
  self.assertEqual(math.DATA['ru']['math_letters']['X'],'икс')
 def test_numbers_and_fractional_genders(self):
  for language,expected in [('en','two thousand seven'),('de','zweitausendsieben'),('fr','deux mille sept'),('es','dos mil siete'),('ru','две тысячи семь'),('ja','二千七')]:self.assertEqual(icu_numbers.integer(2007,language),expected)
  self.assertEqual(math.speech('ru').number('1.01'),'одна целая одна сотая')
  self.assertEqual(math.speech('ru').number('2.21'),'две целых двадцать одна сотая')
  self.assertEqual(math.speech('ru').number('0.11'),'ноль целых одиннадцать сотых')
  self.assertEqual(math.speech('en').number('1.00'),'one point zero zero')
 def test_nested_scope_functions_and_latex(self):
  self.assertEqual(normalize_text(r'\frac{a+1}{b-2}','en',terms=False),'open parenthesis A plus one close parenthesis over open parenthesis B minus two close parenthesis')
  for language in LANGUAGES:
   for raw in ('x = (y+1)^2',r'\frac{a+1}{b-2}','sqrt(9)','log_2(8)','max(1,2)','x = 1.2e-3','1.2e-3A','x₁ + x₂ = 3','x = |y|','x = ∛27','x = 5!','x = ½','x ≥ 3'):
    with self.subTest(language=language,raw=raw):
     actual=normalize_text(raw,language,terms=False);self.assertNotEqual(actual,raw);self.assertEqual(normalize_text(actual,language,terms=False),actual)
  self.assertEqual(normalize_text('max(1,5;2,5)','de',terms=False),'Maximum von eins Komma fünf und zwei Komma fünf')
  self.assertEqual(normalize_text('1/2','ja',terms=False),'1/2')
  self.assertEqual(normalize_text('$1/2$','ja',terms=False),'二分の一')
 def test_entire_sourced_unit_inventory(self):
  for language in LANGUAGES:
   self.assertGreaterEqual(len(math.DATA[language]['units']),60)
   for symbol in math.DATA[language]['units']:
    raw='5'+symbol
    with self.subTest(language=language,symbol=symbol):self.assertNotEqual(normalize_text(raw,language,terms=False),raw)
   self.assertNotEqual(normalize_text('3mW',language,terms=False),normalize_text('3MW',language,terms=False))
 def test_protected_inputs_and_whole_failed_candidates(self):
  inputs=('y=2x-3','x = up','x = bad + 2+3','x = 2foo + 2+3','config = value','x = 0004','file/A = 4','x = 1.2.3',r'`y = 2x - 3; 5V`','```python\ny = 2*x - 3\n```','https://host/a²?y=4','x = 4 /path','x = 2_000','Python3.12 v1.2.3 Orin-8GB TTS-100M','2026-10-28 14:25 ID: 0012-34 2/3','RX5V 5V.bin board/5V.bin',r'C:\models\5V\weights',r'$\unknown{x}$','x = (y + 1','x = 2foo','x = ∑y + 2','x = ∫x+2','1,234.5m 1.234,5m 001m','23ABC 85%test','2/3','2007')
  for language in LANGUAGES:
   for raw in inputs:
    with self.subTest(language=language,raw=raw):self.assertEqual(normalize_text(raw,language,terms=False),raw)
   raw='('*50+'x'+')'*50;started=time.monotonic();self.assertEqual(normalize_text(raw,language,terms=False),raw);self.assertLess(time.monotonic()-started,1)
 def test_adjacent_mathematical_sentences(self):
  for language in LANGUAGES:
   raw='x = (y+1)^2. sqrt(9) = 3.'
   self.assertNotIn('sqrt',normalize_text(raw,language,terms=False))
   self.assertNotIn('^',normalize_text(raw,language,terms=False))
 def test_thai_unchanged(self):
  raw='x = (y+1)^2; GPU, GPT, 85% 23℃';self.assertEqual(normalize_text(raw,'th'),raw)

if __name__=='__main__':unittest.main()
