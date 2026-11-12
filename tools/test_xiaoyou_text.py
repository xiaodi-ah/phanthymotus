"""Regressions for board failures; uses the shipped generic upstream TN helpers."""
import contextlib
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'perception'))
from plugins.xiaoyou_kokoro.text import prepare_text
from plugins.xiaoyou_kokoro.runtime import NativeKokoro, split_once
from plugins.xiaoyou_kokoro.frontend import Frontend
from plugins.xiaoyou_kokoro.adapter import XiaoyouKokoroAdapter


class TextTests(unittest.TestCase):
    def test_warmup_pays_the_english_number_reader_initialization(self):
        from plugins.vits2_tts_trt.frontend.math_units import _english_cardinals
        _english_cardinals.cache_clear()
        adapter = XiaoyouKokoroAdapter.__new__(XiaoyouKokoroAdapter)
        adapter.synthesize = lambda text: prepare_text(text).encode()
        self.assertGreater(adapter.warmup(), 0)
        self.assertEqual(_english_cardinals.cache_info().currsize, 1)
        misses = _english_cardinals.cache_info().misses
        prepare_text('There are 2,000 models.')
        self.assertEqual(_english_cardinals.cache_info().misses, misses)

    def test_formulas_keep_exponents_and_implicit_multiplication(self):
        self.assertEqual(prepare_text('a² + b² = c²'), 'a squared plus b squared equals c squared')
        self.assertEqual(prepare_text('y = 2x - 3'), 'y equals two times x minus three')
        self.assertIn('S等于派乘以r的平方', prepare_text('公式 S = πr²。').replace(' ', ''))

    def test_measurements_and_decimal_precision(self):
        zh = prepare_text('输出5V、电流1.2A、功率6W，允许误差±0.5℃。')
        for expected in ['五伏特', '一点二安培', '六瓦特', '正负零点五摄氏度']:
            self.assertIn(expected, zh.replace(' ', ''))
        self.assertIn('plus or minus zero point five degrees Celsius', prepare_text('Error ±0.5℃.'))
        self.assertIn('zero point zero five eight', prepare_text('CER is 0.058.'))

    def test_identifiers_keep_leading_zeros_and_every_digit(self):
        self.assertEqual(prepare_text('18610223345'), 'one eight six, one zero two two, three three four five')
        zh = prepare_text('验证码005917，订单编号20260810009173。').replace(' ', '').replace(',', '')
        self.assertIn('零零五九一七', zh)
        self.assertIn('二零二六零八一零零零九一七三', zh)
        en = prepare_text('verification code 005917, order number 20260810009173.').replace(',', '')
        self.assertIn('zero zero five nine one seven', en)
        self.assertIn('two zero two six zero eight one zero zero zero nine one seven three', en)

    def test_addresses_are_literal_and_do_not_become_quantities(self):
        zh = prepare_text('邮箱 service_01@example.com，地址192.0.2.10。')
        for token in ['下划线', '零 一', '艾特', '点', '一九二点零点二点一零']:
            self.assertIn(token.replace(' ', ''), zh.replace(' ', ''))
        en = prepare_text('Email service_01@example.com at 192.0.2.10.')
        self.assertIn('underscore zero one at', en)
        self.assertIn('one nine two dot zero dot two dot one zero', en)

    def test_grouped_quantities_are_expanded_before_chunking(self):
        text = prepare_text('There are 2,000,000 models and 200,000 cards.')
        self.assertIn('two million', text)
        self.assertIn('two hundred thousand', text)
        self.assertIn('二百万', prepare_text('共有2,000,000个模型。'))
        raw = 'There are 2,000,000 models across the world.'
        parts = split_once(raw)
        self.assertTrue(any('2,000,000' in p for p in parts))
        self.assertEqual(''.join(raw.split()), ''.join(''.join(parts).split()))

    def test_width_symbols_accents_and_ordinary_text(self):
        self.assertIn('加', prepare_text('AI Agent＋垂直世界模型'))
        self.assertNotIn('+', prepare_text('AI Agent＋垂直世界模型'))
        for text in ["L'Oréal and mise-en-scène.", 'Hello, I am Xiaoyou.']:
            self.assertEqual(prepare_text(text), text)
        with self.assertRaises(ValueError):
            split_once('é' * 800)

    def test_math_letters_are_explicit_without_changing_ordinary_articles(self):
        frontend = Frontend.__new__(Frontend)
        frontend.en = Mock(return_value=('phones', []))
        frontend.en.lexicon.get_NNP.return_value = ('ˈA', 3)
        self.assertEqual(frontend._english('a'), 'ˈA')
        frontend._english('a squared plus b squared')
        self.assertIn('[a](/ˈA/)', frontend.en.call_args.args[0])
        frontend._english('I need a cup.')
        self.assertEqual(frontend.en.call_args.args[0], 'I need a cup.')

    def test_runtime_prepares_once_before_recursive_duration_splits(self):
        runtime = NativeKokoro.__new__(NativeKokoro)
        runtime.lock = threading.Lock()
        runtime.torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
        runtime.max_frames = 200
        runtime.frontend = SimpleNamespace(
            prepare=Mock(side_effect=prepare_text),
            phonemize=lambda text, language: ('', list(text)))
        runtime._encode = lambda ids: [SimpleNamespace(sum=lambda: SimpleNamespace(item=lambda: len(ids) * 10))]
        runtime._decode = lambda cache: b'audio'
        text = 'The verification code is 005917 and the order number is 20260810009173.'
        segments = list(runtime.generate(text))
        self.assertGreater(len(segments), 1)
        runtime.frontend.prepare.assert_called_once_with(text)
        spoken = ' '.join(part for part, _ in segments)
        self.assertEqual(spoken.split(), prepare_text(text).split())
        self.assertNotIn('005917', spoken)

    def test_duration_floor_skips_impossible_encoder_calls_without_losing_text(self):
        runtime = NativeKokoro.__new__(NativeKokoro)
        runtime.max_frames = 20
        runtime.frontend = SimpleNamespace(phonemize=lambda text, language: ('', list(text)))
        calls = []
        def encode(ids):
            self.assertLessEqual(len(ids) + 2, runtime.max_frames)
            calls.append(ids)
            return [SimpleNamespace(sum=lambda: SimpleNamespace(item=lambda: len(ids) + 2))]
        runtime._encode = encode
        runtime._decode = lambda cache: b'audio'
        text = 'one two three four five six seven eight nine ten'
        chunks = list(runtime._generate(text))
        self.assertTrue(calls)
        self.assertEqual(' '.join(piece for piece, _ in chunks).split(), text.split())


if __name__ == '__main__':
    unittest.main()
