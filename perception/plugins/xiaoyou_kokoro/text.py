"""Prepare written entities before duration-based splitting loses their context."""
import re
import unicodedata

from plugins.vits2_tts_trt.frontend.math_units import _english_quantity
from plugins.vits2_tts_trt.frontend.special_text import (
    fold_numeric_width, normalize_special_text, read_numeric_literal,
    verbalize_chinese_number,
)


def _group_digits(spoken, language):
    def grouped(match):
        values = list(match.group()) if language == 'ZH' else match.group().split()
        first = 3 if len(values) == 11 else 4
        groups = [values[:first]] + [values[i:i + 4] for i in range(first, len(values), 4)]
        joiner = '' if language == 'ZH' else ' '
        return ', '.join(joiner.join(group) for group in groups)

    digits = 'zero|one|two|three|four|five|six|seven|eight|nine'
    pattern = (r'[零一二三四五六七八九]{10,}' if language == 'ZH' else
               rf'\b(?:{digits})(?:\s+(?:{digits})){{9,}}\b')
    return re.sub(pattern, grouped, spoken)


def prepare_text(text, language=None):
    # NFC/width folding retains superscripts and Celsius signs for the math reader.
    text = fold_numeric_width(unicodedata.normalize('NFC', text)).strip()
    zh = language == 'zh' or language != 'en' and bool(re.search(r'[\u3400-\u9fff]', text))
    language = 'ZH' if zh else 'EN'
    digits = re.sub(r'\s+', '', text)
    if re.fullmatch(r'[0-9]+', digits) and (len(digits) >= 5 or digits.startswith('0') and len(digits) > 1):
        return _group_digits(read_numeric_literal(digits, language), language)

    def number(value, selected_language):
        value = value.replace(',', '')
        return (verbalize_chinese_number(value) if selected_language == 'ZH'
                else _english_quantity(value).replace('-', ' '))

    def plain(fragment):
        def address(match):
            value = match.group()
            return (read_numeric_literal(value, language)
                    if all(int(part) <= 255 for part in value.split('.')) else value)

        fragment = re.sub(r'(?<![A-Za-z0-9_.])[0-9]{1,3}(?:\.[0-9]{1,3}){3}(?![A-Za-z0-9_]|\.[A-Za-z0-9])', address, fragment)
        fragment = re.sub(r'(?<![A-Za-z0-9_.])[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?(?![A-Za-z0-9_]|\.[0-9])',
                          lambda m: number(m.group(), language), fragment)
        # Unclaimed symbols still carry meaning, e.g. "AI Agent＋垂直世界模型".
        symbols = {'+': ('加', 'plus'), '=': ('等于', 'equals'),
                   '±': ('正负', 'plus or minus'), 'π': ('派', 'pi'),
                   '@': ('艾特', 'at'), '°': ('度', 'degrees')}
        return ''.join(' ' + symbols[c][not zh] + ' ' if c in symbols else c for c in fragment)

    spoken = normalize_special_text(text, plain, number, language=language)
    return _group_digits(re.sub(r'\s+', ' ', spoken).strip(), language)
