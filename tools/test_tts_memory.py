"""Native imports stay optional; allocation headroom and engine-file lifetime are bounded."""
import importlib,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'perception'))
class Tests(unittest.TestCase):
 def test_helpers_do_not_load_onnx_runtime(self):
  script='''import sys,json
sys.path.insert(0,sys.argv[1])
from plugins.moss_tts_runtime import ort_cpu_runtime as moss
from plugins.supertonic_runtime import vendor as supertonic
assert 'onnxruntime' not in sys.modules
assert moss._resolve_ort_providers('cpu')==['CPUExecutionProvider']
assert 'onnxruntime' not in sys.modules
class Ort:
 @staticmethod
 def InferenceSession(*args,**kwargs):return (args,kwargs)
sys.modules['onnxruntime']=Ort
assert supertonic.load_onnx('model.onnx',None,['CPUExecutionProvider'])[0]==('model.onnx',)
'''
  subprocess.run([sys.executable,'-B','-c',script,str(ROOT/'perception')],check=True)
 def session_module(self):
  with patch.dict(sys.modules,{'tensorrt':SimpleNamespace()}):
   return importlib.import_module('plugins.moss_tts_runtime.trt_session')
 def test_kv_growth_has_bounded_headroom_and_reuses_buffer(self):
  cls=self.session_module().TrtSession;s=cls.__new__(cls);s.buffers={}
  a=s.buffer('kv',1_000_000,torch.float32,'cpu');pointer=a.data_ptr()
  self.assertEqual(s.buffer('kv',999_999,torch.float32,'cpu').data_ptr(),pointer)
  s.buffer('kv',1_000_001,torch.float32,'cpu')
  self.assertLessEqual(s.buffers['kv'].numel(),1_000_001+65536//4)
  self.assertLessEqual(s.buffers['kv'].numel(),1_250_000)
 def test_engine_mapping_closed_after_deserialization(self):
  load=self.session_module().load_engine;seen=[]
  def deserialize(data):seen.append(data);self.assertEqual(bytes(data),b'engine');return 'loaded'
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'test.plan';path.write_bytes(b'engine')
   self.assertEqual(load(SimpleNamespace(deserialize_cuda_engine=deserialize),path),'loaded')
  self.assertTrue(seen[0].closed)
 def test_request_buffers_are_cleared_after_sync(self):
  cls=self.session_module().TrtSession;s=cls.__new__(cls);events=[]
  s.stream=SimpleNamespace(synchronize=lambda:events.append(len(s.buffers)));s.buffers={'x':object()};s.addresses={'x':12};s.input_shapes={'x':(1,)}
  s.release_temporary();self.assertEqual(events,[1]);self.assertFalse(s.buffers or s.addresses or s.input_shapes)
 def test_idle_codec_releases_graph_and_state_but_keeps_weights(self):
  with patch.dict(sys.modules,{'tensorrt':SimpleNamespace()}):
   cls=importlib.import_module('plugins.moss_tts_runtime.trt_codec_session').CodecGraphSession
  s=cls.__new__(cls);events=[];s.engine=object();weights=s.engine
  s.stream=SimpleNamespace(synchronize=lambda:events.append(True));s.graph=object();s.context=object();context=s.context;s.buffers={'kv':object()};s.host={'audio':object()}
  s.release_temporary();self.assertEqual(events,[True]);self.assertIs(s.engine,weights)
  self.assertIsNone(s.graph);self.assertIs(s.context,context);self.assertFalse(s.buffers or s.host)
if __name__=='__main__':unittest.main()
