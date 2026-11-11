"""CPU contract checks; run with the same Python environment as perception."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'perception'))
import numpy as np
from plugins.xiaoyou_kokoro.runtime import split_once, verify_assets
from plugins.xiaoyou_kokoro.adapter import XiaoyouKokoroAdapter
from utils.resample import downsample_24k_to_16k


class ContractTests(unittest.TestCase):
    def test_split_preserves_content_and_latin_tokens(self):
        for text in ['欢迎来访，今天的 meeting 已安排好。请跟我来。',
                     'Please send 3.14 dollars to xiaoyou@example.com today.',
                     '欢迎光临这里是小优我们一起开始今天的工作吧']:
            parts = split_once(text)
            self.assertEqual(''.join(text.split()), ''.join(''.join(parts).split()))
            self.assertTrue(all(0 < len(p) < len(text) for p in parts))
        a, b = split_once('Please send 3.14 dollars to xiaoyou@example.com today.')
        self.assertIn('3.14', [p for s in (a, b) for p in s.split()])
        self.assertTrue(any('xiaoyou@example.com' in p for p in (a, b)))
        with self.assertRaises(ValueError):
            split_once('a' * 800)

    def test_assets_reject_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = []
            for name in ['kokoro.pth', 'config.json', 'A.pt']:
                data = name.encode()
                (root / name).write_bytes(data)
                entries.append({'path': name, 'bytes': len(data),
                                'sha256': hashlib.sha256(data).hexdigest()})
            manifest = {'format': 'kokoro-native-gpu-v1', 'precision': 'FP32', 'files': entries}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(verify_assets(root), manifest)
            (root / 'A.pt').write_bytes(b'B.pt')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                verify_assets(root)
            manifest['files'][0]['path'] = '../escape'
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'asset set'):
                verify_assets(root)

    def test_pcm_frames_preserve_all_samples_across_segments(self):
        waves = [np.sin(np.arange(n, dtype=np.float32) * .1) * .2 for n in [2600, 1700]]
        expected = b''.join(downsample_24k_to_16k(w) for w in waves)
        with patch('plugins.xiaoyou_kokoro.adapter.NativeKokoro') as native:
            native.return_value.generate.return_value = iter(zip(['一', '二'], waves))
            chunks = list(XiaoyouKokoroAdapter('/unused').synthesize_stream('一二'))
        self.assertEqual(b''.join(chunks), expected)
        self.assertTrue(all(len(c) == 3200 for c in chunks[:-1]))
        self.assertTrue(0 < len(chunks[-1]) <= 3200 and len(chunks[-1]) % 2 == 0)


if __name__ == '__main__':
    unittest.main()
