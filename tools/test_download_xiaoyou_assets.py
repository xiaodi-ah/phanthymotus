import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest

from download_xiaoyou_assets import download


class DownloadTests(unittest.TestCase):
    def test_download_checks_hash_size_and_destination(self):
        data = b'fixed model bytes'

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                entry = {'path': 'model.pth', 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                manifest = {'base_url': 'http://127.0.0.1:' + str(server.server_port), 'files': [entry]}
                download(manifest, root)
                self.assertEqual((root / 'model.pth').read_bytes(), data)
                self.assertEqual(json.loads((root / 'manifest.json').read_text()), manifest)
                download(manifest, root)
                entry['sha256'] = '0' * 64
                with self.assertRaisesRegex(RuntimeError, 'SHA mismatch'):
                    download(manifest, root)
                self.assertEqual((root / 'model.pth').read_bytes(), data)
                entry['bytes'] = len(data) - 1
                with self.assertRaisesRegex(RuntimeError, 'exceeds'):
                    download(manifest, root)
                entry['path'] = '../escape'
                with self.assertRaises(ValueError):
                    download(manifest, root)
                entry['path'] = 'link/model.pth'
                (root / 'link').symlink_to(root.parent, target_is_directory=True)
                with self.assertRaisesRegex(ValueError, 'escapes'):
                    download(manifest, root)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
