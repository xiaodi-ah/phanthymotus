"""Protect semantics while rendering explicit quantities and selected acronyms."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'perception'))
from plugins.moss_tts_runtime.text_normalization import integer_words,normalize_text

assert normalize_text('23.5°C; -1 m; 85%; 2GB.','en')=='twenty three point five degrees Celsius; minus one meter; eighty five percent; two gigabytes.'
assert normalize_text('23,5°C; 1 s; -2 m; 85%.','de')=='dreiundzwanzig Komma fünf Grad Celsius; eine Sekunde; minus zwei Meter; fünfundachtzig Prozent.'
assert normalize_text('温度23℃、残量85%、距離1.2m。','ja')=='温度にじゅうさんど、残量はちじゅうごパーセント、距離いちてんにメートル。'
assert normalize_text('GPU uses NVIDIA Jetson.','en')=='G P U uses NVIDIA Jetson.'
assert normalize_text('NVIDIA Jetson と GPU。','ja')=='エヌビディア ジェットソン と ジーピーユー。'
assert normalize_text('GPU: 23°C','en',terms=False)=='GPU: twenty three degrees Celsius'
assert normalize_text('1.0 s','en')=='one point zero seconds'
assert normalize_text('1.234m','de')=='1.234m'
assert normalize_text('1,234m','en')=='1,234m'
assert normalize_text('1,234m','ja')=='1,234m'
assert normalize_text('NVIDIA Jetson.','ja')=='エヌビディア ジェットソン.'
assert normalize_text('23°C.','en')=='twenty three degrees Celsius.'
assert normalize_text('1m/s','en')=='one meter per second'
assert integer_words(308000,'ja')=='さんじゅうまんはっせん'
assert integer_words(100001,'de')=='einhunderttausendeins'
for language in ('en','de','ja'):
    for n in range(1000):assert integer_words(n,language)
    for n in (1000,1001,10000,10001,999999):assert integer_words(n,language)
    for text in ('Python3.12 v1.2.3 Orin-8GB TTS-100M model.onnx','2026-10-19 +1-202-555-0100','https://nvidia.com/23m?a=85% contact@nvidia.com','`GPU: 23°C`','1,234.5m 1.234,5m 001m 1234567m','23ABC 85%test','23m.onnx 2GB.bin models/GPU/v1 /data/23m/weights',r'C:\models\23m\GPU'):
        assert normalize_text(text,language)==text,(language,text,normalize_text(text,language))
    for text in ('-23°C 85%','GPU: 2GB','温度23℃'):
        result=normalize_text(text,language);assert normalize_text(result,language)==result
assert normalize_text('温度23℃、85%','zh')=='温度23℃、85%'
assert normalize_text('23°C NVIDIA Jetson','fr')=='23°C NVIDIA Jetson'
try:integer_words(1000000,'en')
except ValueError:pass
else:raise AssertionError('Out-of-range quantity accepted')
print('PASS EN/DE/JA quantities, decimals, units, terminology, idempotence, protected identifiers/URLs, and unsupported languages')
