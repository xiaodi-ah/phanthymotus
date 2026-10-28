"""Chinese compatibility and identifier protection; foreign math has its own suite."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'perception'))
from plugins.moss_tts_runtime.text_normalization import integer_words,normalize_text

assert normalize_text('温度23℃、85%','zh',terms=False)=='温度二十三摄氏度、百分之八十五'
assert normalize_text('温度23℃、85%','zh')=='温度二十三摄氏度、百分之八十五'
assert normalize_text('Qwen3 和 Qwen 2.5 使用 GPU、CNCF、个人IP。','zh')=='千问三 和 千问二点五 使用 G P U、C N C F、个人I P。'
assert normalize_text('Qwen3 and Qwen 2.5 use GPU, CNCF and API.','en',terms=False)=='Qwen3 and Qwen 2.5 use GPU, CNCF and API.'
assert normalize_text('HAMi、ModelHubXC、EngineX 和 OpenMLDB。','zh')=='哈密、ModelHubXC、EngineX 和 OpenMLDB。'
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
        assert normalize_text(text,language,terms=False)==(text.replace('HAMi','哈密').replace('GPU','G P U').replace('CNCF','C N C F') if language=='zh' else text)
assert normalize_text('HAMi Qwen3 GPU 85% ID: 008492610','th')=='HAMi Qwen3 GPU 85% ID: 008492610'
for language in ('en','ja','de','fr','ru','es'):
    for text in ('Python3.12 v1.2.3 Orin-8GB TTS-100M model.onnx','2026-10-19 +1-202-555-0100','https://nvidia.com/23m?a=85% contact@nvidia.com','`GPU: 23°C`','1,234.5m 1.234,5m 001m','23ABC 85%test','23m.onnx 2GB.bin models/GPU/v1 /data/23m/weights',r'C:\models\23m\GPU'):
        assert normalize_text(text,language,terms=False)==text,(language,text,normalize_text(text,language,terms=False))
    assert integer_words(1000000,language)
    try:integer_words(10**16,language)
    except ValueError:pass
    else:raise AssertionError('Out-of-range quantity accepted')
print('PASS Chinese compatibility, protected identifiers, number bounds and Thai deferral')
