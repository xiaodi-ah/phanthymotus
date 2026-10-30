"""Route selection, backend isolation and the fixed 16 kHz delivery contract."""
import sys
import threading
import unittest
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'perception'))
from plugins.routed_tts_runtime import RoutedAdapter, route_language
from plugins.supertonic_runtime import SupertonicAdapter, pcm16


class RoutingTests(unittest.TestCase):
    def test_plugin_schema_dispatch_and_instance_isolation(self):
        import ast
        source = Path(__file__).resolve().parents[1] / 'perception/plugins/tts.py'
        assignment = next(n for n in ast.parse(source.read_text()).body
                          if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'TOOLS' for t in n.targets))
        schema = ast.literal_eval(assignment.value)
        class Node:
            def __init__(self, topic, adapter, **kwargs):
                self._adapter = adapter; self._input_topic = topic
        with patch.dict(sys.modules, {'plugins.tts': SimpleNamespace(TOOLS=schema, _TTSNode=Node)}):
            plugin = importlib.import_module('plugins.moss_tts')
        added = []
        default = plugin.TTSPlugin({}, SimpleNamespace(add_node=added.append))
        routed = plugin.TTSPlugin({'routing': 'moss_supertonic'}, SimpleNamespace(add_node=added.append))
        self.assertIn('th', default.get_tools()[0]['inputSchema']['properties']['language']['enum'])
        self.assertIn('th', routed.get_tools()[0]['inputSchema']['properties']['language']['enum'])
        with patch('plugins.routed_tts_runtime.RoutedAdapter', side_effect=lambda *a, **kw: object()) as factory:
            one, two = routed._node('one'), routed._node('two')
            self.assertIs(routed._node('one'), one)
            self.assertIsNot(one._adapter, two._adapter)
            self.assertEqual(factory.call_count, 2)
        self.assertEqual(routed._language('สวัสดี', 'auto'), 'th')
        self.assertEqual(len(added), 2)

    def test_explicit_default_and_scripts(self):
        for code, text in [('zh', '请检查 NVIDIA GPU。'), ('ja', 'GPUを確認してください。'),
                           ('ru', 'Проверьте GPU.'), ('de', 'Der Roboter ist bereit.'),
                           ('fr', 'Le robot est prêt.'), ('es', 'El robot está listo.'), ('en', 'Hello.')]:
            self.assertEqual(route_language(text), code)
        self.assertEqual(route_language('你好。', 'en'), 'en')
        self.assertEqual(route_language('Hello.', 'ZH_CN', 'fr'), 'zh')
        self.assertEqual(route_language('ModelHub', 'auto', 'fr'), 'fr')
        self.assertEqual(route_language('ModelHub'), 'en')

    def test_thai_routes_to_moss_and_unknown_is_rejected(self):
        self.assertEqual(route_language('สวัสดีครับ'), 'th')
        self.assertEqual(route_language('Hello', 'th'), 'th')
        for text, code in [('Hello', 'xx')]:
            with self.assertRaises(ValueError): route_language(text, code)

    def test_routes_reused_and_warmup_hits_both_backends(self):
        seen = []
        def backend(name):
            def stream(text, language):
                seen.append((name, text, language))
                yield name.encode()
            return SimpleNamespace(synthesize_stream_language=stream)
        adapter = RoutedAdapter.__new__(RoutedAdapter)
        adapter.language = 'auto'; adapter.moss = backend('moss'); adapter.supertonic = backend('supertonic')
        self.assertEqual(list(adapter.warmup()), [b'moss', b'supertonic'])
        for text, code in [('你好。', 'auto'), ('Hello.', 'auto'), ('Bonjour.', 'fr'), ('再见。', 'auto')]:
            adapter.synthesize(text, code)
        self.assertEqual([x[0] for x in seen], ['moss', 'supertonic', 'moss', 'supertonic', 'supertonic', 'moss'])
        self.assertEqual(adapter.synthesize('สวัสดี', 'auto'), b'moss')
        with self.assertRaises(ValueError): adapter.synthesize(' ')
        self.assertEqual(len(seen), 7)

    def test_native_rng_matches_baseline_and_is_instance_local(self):
        seen = []
        adapter = SupertonicAdapter.__new__(SupertonicAdapter)
        adapter.seed = 42; adapter.lock = threading.Lock(); adapter.style = object()
        model = SimpleNamespace(text_processor=lambda *a: (np.zeros((1, 10)), None))
        def infer(text, language, style, steps, speed):
            self.assertEqual((steps, speed), (8, 1.05))
            value = model.rng.randn(100).astype(np.float32)
            seen.append(value)
            return value.reshape(1, -1), None
        model._infer = infer; adapter.model = model
        np.random.seed(987)
        state = np.random.get_state()
        a = list(adapter.native_blocks('Hello.', 'en'))[0]
        self.assertTrue(np.array_equal(a, np.random.RandomState(42).randn(100).astype(np.float32)))
        self.assertTrue(np.array_equal(state[1], np.random.get_state()[1]))
        self.assertTrue(np.array_equal(a, list(adapter.native_blocks('Hello.', 'en'))[0]))

    def test_resampling_and_pause_are_real_audio_not_early_placeholder(self):
        adapter = SupertonicAdapter.__new__(SupertonicAdapter)
        audio = (.1 * np.sin(2 * np.pi * 440 * np.arange(44100) / 44100)).astype(np.float32)
        adapter.native_blocks = lambda *args: (x for x in (audio, audio))
        pieces = list(adapter.synthesize_stream_language('test', 'en'))
        self.assertEqual([len(p) for p in pieces], [32000, 9600, 32000])
        self.assertEqual(pieces[0], pcm16(resample_poly(audio, 160, 441)))
        self.assertGreater(np.max(np.abs(np.frombuffer(pieces[0], '<i2'))), 100)
        for invalid in ([], [np.nan], [[0.1]]):
            with self.assertRaises(ValueError): pcm16(invalid)


if __name__ == '__main__': unittest.main()
