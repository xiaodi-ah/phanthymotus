"""ZipVoice voice-data filenames must be URL-encoded and still SHA-verified."""
import hashlib,io,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from download_moss_assets import download
class Tests(unittest.TestCase):
 def test_filename_spaces_and_percent_are_encoded(self):
  content=b'voice data';calls=[]
  class Opener:
   def open(self,url,timeout):calls.append(url);return io.BytesIO(content)
  manifest=dict(base_url='http://example.invalid/models',files=[dict(path='voices/!v/Mr serious%1',bytes=len(content),sha256=hashlib.sha256(content).hexdigest())])
  with tempfile.TemporaryDirectory() as folder,patch('download_moss_assets.urllib.request.build_opener',return_value=Opener()):
   download(manifest,Path(folder));download(manifest,Path(folder))
   self.assertEqual(calls,['http://example.invalid/models/voices/%21v/Mr%20serious%251'])
   self.assertEqual((Path(folder)/'voices/!v/Mr serious%1').read_bytes(),content)
 def test_invalid_paths_and_bad_sha_are_rejected(self):
  for path in ['../escape','/absolute']:
   with tempfile.TemporaryDirectory() as folder,self.assertRaises(ValueError):download(dict(base_url='http://example.invalid',files=[dict(path=path,bytes=1,sha256='0'*64)]),Path(folder))
  class Opener:
   def open(self,*a,**k):return io.BytesIO(b'a')
  with tempfile.TemporaryDirectory() as folder,patch('download_moss_assets.urllib.request.build_opener',return_value=Opener()),self.assertRaises(RuntimeError):
   download(dict(base_url='http://example.invalid',files=[dict(path='voice',bytes=1,sha256='0'*64)]),Path(folder))
if __name__=='__main__':unittest.main()
