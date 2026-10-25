"""Protect semantics while rendering explicit quantities and selected acronyms."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'perception'))
from plugins.moss_tts_runtime.text_normalization import integer_words,normalize_text

assert normalize_text('23.5°C; -1 m; 85%; 2GB.','en')=='twenty three point five degrees Celsius; minus one meter; eighty five percent; two gigabytes.'
assert normalize_text('23,5°C; 1 s; -2 m; 85%.','de')=='dreiundzwanzig Komma fünf Grad Celsius; eine Sekunde; minus zwei Meter; fünfundachtzig Prozent.'
assert normalize_text('温度23℃、残量85%、距離1.2m。','ja')=='温度にじゅうさんど、残量はちじゅうごパーセント、距離いちてんにメートル。'
assert normalize_text('GPU uses NVIDIA Jetson.','en')=='GPU uses NVIDIA Jetson.'
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
for language in ('en','de','ja','fr','ru'):
    if language in ('fr','ru'):
        for n in range(101):assert integer_words(n,language)
    else:
        for n in range(1000):assert integer_words(n,language)
        for n in (1000,1001,10000,10001,999999):assert integer_words(n,language)
    for text in ('Python3.12 v1.2.3 Orin-8GB TTS-100M model.onnx','2026-10-19 +1-202-555-0100','https://nvidia.com/23m?a=85% contact@nvidia.com','`GPU: 23°C`','1,234.5m 1.234,5m 001m 1234567m','23ABC 85%test','23m.onnx 2GB.bin models/GPU/v1 /data/23m/weights',r'C:\models\23m\GPU'):
        assert normalize_text(text,language)==text,(language,text,normalize_text(text,language))
    for text in ('-23°C 85%','GPU: 2GB','温度23℃'):
        result=normalize_text(text,language);assert normalize_text(result,language)==result
assert normalize_text('温度23℃、85%','zh',terms=False)=='温度23℃、85%'
assert normalize_text('温度23℃、85%','zh')=='温度二十三摄氏度、百分之八十五'
assert normalize_text('Qwen3 和 Qwen 2.5 使用 GPU、CNCF、个人IP。','zh')=='千问三 和 千问二点五 使用 GPU、CNCF、个人I P。'
assert normalize_text('Qwen3 and Qwen 2.5 use GPU, CNCF and API.','en')=='Qwen3 and Qwen 2.5 use GPU, CNCF and API.'
assert normalize_text('HAMi、ModelHubXC、EngineX 和 OpenMLDB。','zh')=='哈米、ModelHubXC、EngineX 和 OpenMLDB。'
assert normalize_text('US and IT use IP, but send us it.','en')=='US and IT use IP, but send us it.'
assert normalize_text('订单ID：008492610。验证码5690有效。','zh')=='订单I D：零零八四九二六一零。验证码五六九零有效。'
assert normalize_text('Order ID: 008492610.','en')=='Order I D: zero zero eight four nine two six one zero.'
assert normalize_text('-0.5℃ 85% 8.5GB','zh')=='负零点五摄氏度 百分之八十五 八点五吉字节'
for number,spoken in [(0,'零'),(10,'十'),(11,'十一'),(100,'一百'),(101,'一百零一'),(1001,'一千零一'),(1010,'一千零一十'),(10000,'一万'),(10001,'一万零一'),(10010,'一万零一十'),(100010,'十万零一十'),(999999,'九十九万九千九百九十九')]:
    assert integer_words(number,'zh')==spoken,(number,integer_words(number,'zh'))
for language in ('zh','en'):
    for text in ('https://example.com/API/Qwen3?ID=0085 contact@ModelHubXC.com','`HAMi Qwen3 GPU: 23℃`','Qwen3-8B Qwen3_5 Qwen03 Qwen3.json model/Qwen3/API',r'C:\models\Qwen3\GPU','ModelHubXC_cache myModelHubXC','2026-10-23 14:25 2/3 lr=1e-4 y=2x-3'):
        assert normalize_text(text,language)==text,(language,text,normalize_text(text,language))
    assert normalize_text('ID: 192.168.0.1 ID: 0012-34',language)=='I D: 192.168.0.1 I D: 0012-34'
    for text in ('Qwen3 ModelHubXC GPU 85% 8.5GB','Order ID: 008492610.','订单ID：008492610。'):
        result=normalize_text(text,language);assert normalize_text(result,language)==result,(language,result)
    for text in ('HAMi Qwen3 ModelHubXC EngineX GPU CNCF IP','Order ID: 008492610.'):
        assert normalize_text(text,language,terms=False)==text
for language in ('es','th'):
    assert normalize_text('HAMi Qwen3 ModelHubXC GPU 85% ID: 008492610',language)=='HAMi Qwen3 ModelHubXC GPU 85% ID: 008492610'
assert normalize_text('23°C NVIDIA Jetson','fr',terms=False)=='23°C NVIDIA Jetson'
assert normalize_text('23°C, 85%.','fr')=='vingt-trois degrés Celsius, quatre-vingt-cinq pour cent.'
assert normalize_text('23°C, 85%.','ru')=='двадцать три градуса Цельсия, восемьдесят пять процентов.'
for number,word in [(0,'zéro'),(17,'dix-sept'),(21,'vingt-et-un'),(71,'soixante-et-onze'),(80,'quatre-vingts'),(81,'quatre-vingt-un'),(91,'quatre-vingt-onze'),(100,'cent')]:
    assert integer_words(number,'fr')==word
for number,unit in [(1,'процент'),(2,'процента'),(5,'процентов'),(11,'процентов'),(21,'процент'),(22,'процента'),(100,'процентов')]:
    assert normalize_text(str(number)+'%','ru').endswith(' '+unit)
assert normalize_text('-0,5°C; 1°C; 2°C','fr')=='moins zéro virgule cinq degré Celsius; un degré Celsius; deux degrés Celsius'
for language in ('fr','ru'):
    assert normalize_text('1000°C 123% 2GB Qwen3 ID: 0012',language)=='1000°C 123% 2GB Qwen3 ID: 0012'
    assert normalize_text('23°C 85%',language,terms=False)=='23°C 85%'
assert normalize_text('23.5°C','ru')=='23.5°C'
try:integer_words(1000000,'en')
except ValueError:pass
else:raise AssertionError('Out-of-range quantity accepted')
print('PASS language-specific terms/quantities/IDs, idempotence, protected identifiers/URLs/code, and unchanged quantity-only/other-language modes')
