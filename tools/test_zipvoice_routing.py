"""Original chunking, cancellation, routing and independent MCP model ownership."""
import ast,importlib,json,sys,tempfile,threading,unittest,wave
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
from scipy.signal import resample_poly
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'perception'))
from plugins.zipvoice_tts_runtime import ZipVoiceAdapter,ZipSupertonicAdapter,route_language

class Tests(unittest.TestCase):
 def make_adapter(self,folder,generate=None):
  root=Path(folder)
  with wave.open(str(root/'reference-24k.wav'),'wb') as w:w.setparams((1,2,24000,0,'NONE',''));w.writeframes(np.ones(240,dtype='<i2').tobytes())
  (root/'reference.json').write_text(json.dumps(dict(text='正式中文参考。')))
  self.seen=[]
  def default_generate(text,config,callback):
   self.seen.append(text);chunks=[np.linspace(-.1,.1,240,dtype=np.float32)]*2
   for i,a in enumerate(chunks):
    if not callback(a,(i+1)/2):break
   return NS(samples=np.concatenate(chunks))
  engine=NS(sample_rate=24000,generate=generate or default_generate)
  fake=NS(__version__='1.13.6+cuda',OfflineTtsConfig=lambda **kw:NS(**kw,validate=lambda:True),OfflineTtsModelConfig=lambda **kw:NS(**kw),OfflineTtsZipvoiceModelConfig=lambda **kw:NS(**kw),OfflineTts=lambda cfg:engine,GenerationConfig=NS)
  with patch.dict(sys.modules,{'sherpa_onnx':fake}):return ZipVoiceAdapter(root)
 def test_original_policy_and_audio_contract(self):
  with tempfile.TemporaryDirectory() as d:
   a=self.make_adapter(d)
   self.assertEqual(a.config.extra,{'min_char_in_sentence':'30','max_char_in_sentence':'80'})
   self.assertEqual((a.config.num_steps,a.config.speed,a.config.silence_scale),(8,1.,1.))
   pieces=list(a.synthesize_stream_language('编号005917。','zh'))
   self.assertEqual([len(x) for x in pieces],[320,320]);self.assertEqual(self.seen[0],'编号零零五，九一七。')
   signal=np.linspace(-.1,.1,240,dtype=np.float32)
   self.assertEqual(pieces[0],(np.clip(resample_poly(signal,2,3),-1,1)*32767).astype('<i2').tobytes())
 def test_close_cancels_a_full_callback_queue(self):
  stopped=threading.Event()
  def generate(text,config,callback):
   for i in range(100):
    if not callback(np.ones(240,dtype=np.float32)*.1,(i+1)/100):stopped.set();break
   return NS(samples=[])
  with tempfile.TemporaryDirectory() as d:
   a=self.make_adapter(d,generate);stream=a.synthesize_stream_language('测试。','zh');next(stream);stream.close()
   self.assertTrue(stopped.is_set());self.assertFalse(a.lock.locked())
 def test_incomplete_or_invalid_audio_is_not_silent_success(self):
  def incomplete(text,config,callback):callback(np.ones(240,dtype=np.float32)*.1,.5);return NS(samples=np.ones(240))
  def invalid(text,config,callback):callback(np.array([np.nan]),1.);return NS(samples=[])
  with tempfile.TemporaryDirectory() as d:
   for fn in [incomplete,invalid]:
    with self.assertRaises(RuntimeError):list(self.make_adapter(d,fn).synthesize_stream_language('测试。','zh'))
 def test_route_coverage_and_explicit_language(self):
  for code,text in [('zh','你好。'),('en','Hello.'),('ja','こんにちは。'),('ru','Привет.')]:self.assertEqual(route_language(text),code)
  self.assertEqual(route_language('005917'),'zh');self.assertEqual(route_language('Bonjour.','fr'),'fr')
  self.assertEqual(route_language('Hello.','auto','de'),'de')
  for code in ['th','xx']:
   with self.assertRaises(ValueError):route_language('test',code)
 def test_seven_language_backend_ownership_and_warmup(self):
  seen=[]
  def backend(name):
   def stream(text,language):seen.append((name,text,language));yield name.encode()
   return NS(synthesize_stream_language=stream,validate_text=lambda *a:None)
  with patch('plugins.zipvoice_tts_runtime.ZipVoiceAdapter',side_effect=lambda *a:backend('zip')),patch('plugins.zipvoice_tts_runtime.SupertonicAdapter',side_effect=lambda *a,**k:backend('super')):
   a=ZipSupertonicAdapter('zip','super');b=ZipSupertonicAdapter('zip','super')
   self.assertIsNot(a.zipvoice,b.zipvoice);self.assertIsNot(a.supertonic,b.supertonic)
   self.assertEqual(list(a.warmup()),[b'zip',b'super'])
   self.assertEqual(a.synthesize('你好。'),b'zip');self.assertEqual(a.synthesize('Hello.'),b'super')
   self.assertEqual([x[0] for x in seen],['zip','super','zip','super'])
 def test_plugin_schema_factory_and_language_validation(self):
  tree=ast.parse((R/'perception/plugins/tts.py').read_text());assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TOOLS' for t in n.targets))
  class Node:
   def __init__(self,topic,adapter,**kw):self._adapter=adapter;self._input_topic=topic
  with patch.dict(sys.modules,{'plugins.tts':NS(TOOLS=ast.literal_eval(assignment.value),_TTSNode=Node)}):plugin=importlib.import_module('plugins.moss_tts')
  with patch.object(plugin,'_TTSNode',Node),patch('plugins.zipvoice_tts_runtime.ZipSupertonicAdapter',side_effect=lambda *a,**k:object()) as factory:
   p=plugin.TTSPlugin({'routing':'zipvoice_supertonic'},NS(add_node=lambda n:None));tools=p.get_tools()
   self.assertEqual(set(tools[0]['inputSchema']['properties']['language']['enum']),{'auto','zh','en','ja','de','fr','ru','es'})
   self.assertNotIn('th',p.dispatch('tts',{'action':'info'})['languages'])
   self.assertIsNot(p._node('one')._adapter,p._node('two')._adapter);self.assertEqual(factory.call_count,2)
   with self.assertRaises(ValueError):p._language('สวัสดี','auto')

if __name__=='__main__':unittest.main()
