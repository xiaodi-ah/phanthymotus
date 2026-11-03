"""Conservative spoken forms for prose decimals, explicit identifiers and email."""
import re
from functools import lru_cache
from .moss_tts_runtime import multilingual_math,zh_math
from .moss_tts_runtime.text_normalization import integer_words

_CUES={
    'zh':r'验证码|校验码|认证码|订单编号|订单号|序列号|编号|手机号码|手机号|电话号码|联系电话',
    'en':r'verification code|confirmation code|security code|order (?:number|ID)|serial number|phone number|telephone number|mobile number',
    'ja':r'認証コード|確認コード|検証コード|注文番号|注文ID|シリアル番号|電話番号|携帯番号',
    'de':r'Bestätigungscode|Verifizierungscode|Prüfcode|Bestellnummer|Auftragsnummer|Seriennummer|Telefonnummer|Mobilnummer',
    'fr':r'code de (?:vérification|confirmation|sécurité)|numéro de (?:commande|série|téléphone)',
    'ru':r'проверочный код|код (?:подтверждения|проверки)|номер (?:заказа|телефона)|серийный номер',
    'es':r'código de (?:verificación|confirmación|seguridad)|número de (?:pedido|serie|teléfono)',
}
_SYMBOLS={
    'zh':{'@':'艾特','.':'点','_':'下划线','-':'连字符','+':'加号'},
    'en':{'@':'at','.':'dot','_':'underscore','-':'hyphen','+':'plus'},
    'ja':{'@':'アットマーク','.':'ドット','_':'アンダースコア','-':'ハイフン','+':'プラス'},
    'de':{'@':'ät','.':'Punkt','_':'Unterstrich','-':'Bindestrich','+':'plus'},
    'fr':{'@':'arobase','.':'point','_':'tiret bas','-':'tiret','+':'plus'},
    'ru':{'@':'собака','.':'точка','_':'нижнее подчёркивание','-':'дефис','+':'плюс'},
    'es':{'@':'arroba','.':'punto','_':'guion bajo','-':'guion','+':'más'},
}
_EMAIL=re.compile(r"""(?<![A-Za-z0-9_.!#$%&'*+/=?^`{|}~\-])[A-Za-z0-9_+\-]+(?:\.[A-Za-z0-9_+\-]+)*@[A-Za-z0-9]+(?:[\-][A-Za-z0-9]+)*(?:\.[A-Za-z0-9]+(?:[\-][A-Za-z0-9]+)*)*\.[A-Za-z]{2,24}(?![A-Za-z0-9_-]|\.[A-Za-z0-9])""")
_PROTECTED=re.compile(r'```[\s\S]*?```|`[^`]*`|https?://[^\s]+|'
    r'(?i:\b(?:version|release|Python|CUDA|TensorRT|PyTorch|torch|JetPack|versión|версия|версии)\s*[:=]?\s*v?[0-9]+(?:\.[0-9]+){1,3})|'
    r'(?:版本|バージョン)\s*v?[0-9]+(?:\.[0-9]+){1,3}')
_DECIMAL=re.compile(r'(?<![A-Za-z0-9_.,/\\-])[+−-]?(?:0|[1-9][0-9]{0,14})[.,][0-9]{1,8}(?![A-Za-z0-9_,/\\-]|\.[A-Za-z0-9_])')


@lru_cache(maxsize=7)
def digit_words(language):
    if language=='zh':return tuple('零一二三四五六七八九')
    if language=='ja':return ('ゼロ','イチ','ニ','サン','ヨン','ゴ','ロク','ナナ','ハチ','キュウ')
    return tuple(integer_words(n,language) for n in range(10))


def spell_digits(digits,language,phone=False):
    if not digits.isascii() or not digits.isdigit() or len(digits)>32:
        raise ValueError('Expected at most 32 ASCII identifier digits')
    words=digit_words(language)
    if phone and len(digits)==11:
        groups=[digits[:3],digits[3:7],digits[7:]]
    else:
        width=4 if len(digits)>8 else 3
        groups=[digits[i:i+width] for i in range(0,len(digits),width)]
    join='' if language=='zh' else ' '
    separator='，' if language=='zh' else '、' if language=='ja' else ', '
    return separator.join(join.join(words[int(c)] for c in group) for group in groups)


@lru_cache(maxsize=7)
def identifier_pattern(language):
    return re.compile(r'(?<![A-Za-z])(?P<cue>'+_CUES[language]+r')'
        r'(?P<space>[ \t]*(?:(?:fictif|ficticio|ficticia|is|ist|est|es|为|是|は)[ \t]*)?[:：#]?[ \t]*)'
        r'(?P<value>\+?[0-9][0-9 -]{0,46}[0-9])(?![A-Za-z0-9_/-]|\.[A-Za-z0-9_])',re.I)


def email_reading(address,language):
    # Keep lowercase word runs natural; spell short codes, digits and every separator.
    letters=multilingual_math.DATA['en' if language=='zh' else language]['letters']
    tokens=[]
    for token in re.findall(r'[A-Za-z]+|[0-9]+|.',address):
        if token.isalpha() and token.islower() and len(token)>=3:
            tokens.append(token)
            continue
        for char in token:
            if char.isdigit():tokens.append(digit_words(language)[int(char)])
            elif char.isalpha():tokens.append(letters[char.upper()])
            else:tokens.append(_SYMBOLS[language][char])
    return ' '.join(tokens)


def spoken_entities(text,language):
    if language not in _CUES:return text
    def prose(value):
        def identifier(match):
            raw=match['value'];digits=re.sub(r'[^0-9]','',raw)
            if len(digits)>32 or re.search(r'-\s*-',raw):return match[0]
            phone=bool(re.search(r'phone|telephone|mobile|手机|电话|携帯|Telefon|Mobil|téléphone|телефон|teléfono',match['cue'],re.I))
            prefix=_SYMBOLS[language]['+']+' ' if raw.startswith('+') else ''
            if '-' in raw and not phone:
                reading=(' '+_SYMBOLS[language]['-']+' ').join(spell_digits(part.replace(' ',''),language) for part in raw.lstrip('+').split('-'))
            else:reading=spell_digits(digits,language,phone)
            return match['cue']+match['space']+prefix+reading
        value=identifier_pattern(language).sub(identifier,value)
        def decimal(match):
            raw=match[0];whole,tail=re.split('[.,]',raw)
            if language=='zh' and zh_math._UNIT_RE.match(match.string[match.end():].lstrip()):return raw
            if ',' in raw and (language in ('en','ja','zh') or len(tail)==3):return raw
            if '.' in raw and language in ('de','es') and len(tail)==3:return raw
            return zh_math.number(raw) if language=='zh' else multilingual_math.speech(language).number(raw)
        return _DECIMAL.sub(decimal,value)
    def segment(value):
        pieces=[];start=0
        for match in _EMAIL.finditer(value):
            pieces.extend([prose(value[start:match.start()]),email_reading(match[0],language)])
            start=match.end()
        pieces.append(prose(value[start:]));return ''.join(pieces)
    pieces=[];start=0
    for match in _PROTECTED.finditer(text):
        pieces.extend([segment(text[start:match.start()]),match[0]]);start=match.end()
    pieces.append(segment(text[start:]));return ''.join(pieces)
