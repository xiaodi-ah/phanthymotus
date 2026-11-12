"""Join pauses shrink without removing speech or delaying the initial prebuffer."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'perception'))
from plugins.xiaoyou_kokoro import runtime
from plugins.xiaoyou_kokoro.adapter import XiaoyouKokoroAdapter
from utils.resample import downsample_24k_to_16k


def fixture():
    return np.concatenate([np.zeros(7200), np.full(24000, .1), np.zeros(9600)]).astype(np.float32)


def silence(wave):
    active = np.flatnonzero(np.abs(wave) > 1e-4)
    return int(active[0]), wave.size - int(active[-1]) - 1


class SeamTests(unittest.TestCase):
    def test_runtime_shortens_internal_gaps_without_dropping_speech(self):
        native = runtime.NativeKokoro.__new__(runtime.NativeKokoro)
        native.max_frames = 200
        native.frontend = SimpleNamespace(phonemize=lambda text, language: ('', list(text)))
        native._encode = lambda ids: [SimpleNamespace(sum=lambda: SimpleNamespace(item=lambda: len(ids) * 10))]
        original = fixture()
        native._decode = lambda cache: original
        parts = list(native._generate('one two three four five six seven eight nine ten'))
        self.assertGreater(len(parts), 1)
        self.assertEqual(silence(parts[0][1])[0], 7200)
        self.assertEqual(silence(parts[-1][1])[1], 9600)
        for _, wave in parts:
            np.testing.assert_array_equal(wave[wave > .01], original[original > .01])
        for (_, left), (_, right) in zip(parts, parts[1:]):
            self.assertLessEqual((silence(left)[1] + silence(right)[0]) / 24000, .125)

    def test_punctuation_keeps_a_natural_pause_and_outer_edges(self):
        original = fixture()
        for text, maximum in [('word', .12), ('word,', .18), ('句子。', .30), ('"Sentence."', .30)]:
            left = runtime.trim_join_silence(original, text, True, False)
            right = runtime.trim_join_silence(original, 'next', False, True)
            gap = (silence(left)[1] + silence(right)[0]) / 24000
            self.assertGreaterEqual(gap, maximum - .005)
            self.assertLessEqual(gap, maximum + .005)
            self.assertEqual(silence(left)[0], 7200)
            self.assertEqual(silence(right)[1], 9600)

    def test_single_chunk_and_silent_chunks_are_unchanged(self):
        for original in [fixture(), np.zeros(24000, dtype=np.float32)]:
            self.assertIs(runtime.trim_join_silence(original, 'hello', True, True), original)
        quiet = np.zeros(24000, dtype=np.float32)
        self.assertIs(runtime.trim_join_silence(quiet, 'hello', False, False), quiet)

    def test_quiet_speech_and_internal_pauses_are_preserved(self):
        original = fixture()
        original[2000:3000] = .0002
        original[35000:36000] = -.0002
        original[16000:17000] = 0
        trimmed = runtime.trim_join_silence(original, 'hello', False, False)
        np.testing.assert_array_equal(trimmed[trimmed != 0], original[original != 0])
        first_loud = int(np.flatnonzero(trimmed > .01)[0])
        np.testing.assert_array_equal(trimmed[first_loud + 8800:first_loud + 9800], np.zeros(1000))

    def test_initial_half_second_is_available_without_lookahead(self):
        original = np.zeros(24000, dtype=np.float32)
        original[1000:2000] = .1
        result = runtime.trim_join_silence(original, 'hello', True, False)
        self.assertGreaterEqual(len(result), 12000)
        np.testing.assert_array_equal(result[:12000], original[:12000])
        short = original[:10000]
        np.testing.assert_array_equal(runtime.trim_join_silence(short, 'hello', True, False), short)

    def test_outer_segments_keep_context_and_pcm_frame_continuity(self):
        waves = [np.full(2600, .1, dtype=np.float32), np.full(1700, .2, dtype=np.float32)]
        with patch('plugins.xiaoyou_kokoro.adapter.NativeKokoro') as native:
            native.return_value.generate.side_effect = [iter([('一', waves[0])]), iter([('二', waves[1])])]
            chunks = list(XiaoyouKokoroAdapter('/unused').synthesize_segments(['一', '二']))
        self.assertEqual(b''.join(chunks), b''.join(downsample_24k_to_16k(w) for w in waves))
        self.assertTrue(all(len(c) == 3200 for c in chunks[:-1]))
        calls = native.return_value.generate.call_args_list
        self.assertEqual(calls[0].kwargs, dict(first=True, last=False))
        self.assertEqual(calls[1].kwargs, dict(first=False, last=True))

    def test_first_chunk_is_shorter_but_indivisible_words_keep_the_hard_limit(self):
        native = runtime.NativeKokoro.__new__(runtime.NativeKokoro)
        native.max_frames = 200
        native.frontend = SimpleNamespace(phonemize=lambda text, language: ('', list(text)))
        native._encode = lambda ids: [SimpleNamespace(sum=lambda: SimpleNamespace(item=lambda: len(ids) * 10))]
        decoded = []
        def decode(cache):
            frames = cache[-1].sum().item()
            self.assertLessEqual(frames, 200)
            decoded.append(frames)
            return np.ones(frames * 600, dtype=np.float32)
        native._decode = decode
        text = 'one two three four five six seven eight nine ten'
        parts = list(native._generate(text))
        self.assertEqual(' '.join(t for t, _ in parts).split(), text.split())
        self.assertLessEqual(decoded[0], 80)
        self.assertTrue(any(n > 80 for n in decoded[1:]))
        decoded.clear()
        self.assertEqual(len(list(native._generate('abcdefghijkl'))), 1)
        self.assertEqual(decoded, [120])
        with self.assertRaises(ValueError):
            list(native._generate('x' * 25))


if __name__ == '__main__':
    unittest.main()
