"""Check MCP model lifetime and routing without ROS or GPU dependencies."""
import ast
import json
import tempfile
import time
import hashlib
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

source=Path(__file__).resolve().parents[1]/'perception/plugins/moss_tts.py'
plugin_class=next(node for node in ast.parse(source.read_text()).body if isinstance(node,ast.ClassDef) and node.name=='TTSPlugin')

class Adapter:
    created=[]
    def __init__(self,*args,language='auto',**kwargs):
        self.language=language;self.created.append(self)

class Node:
    def __init__(self,topic,adapter,**kwargs):
        self._input_topic=topic or '';self._adapter=adapter;self.state='idle';self._worker_thread=None;self.destroyed=False;self.queued=[]
    def start(self):self.state='running';return self._status_dict()
    def stop(self):self.state='idle';return {'state':'idle'}
    def destroy_node(self):self.destroyed=True
    def _status_dict(self):return {'state':self.state,'topic_in':[],'topic_out':[]}
    def enqueue(self,text,**kwargs):self.queued.append(text)

class Executor:
    def __init__(self):self.nodes=[]
    def add_node(self,node):self.nodes.append(node)
    def remove_node(self,node):self.nodes.remove(node)

class LifecycleTests(unittest.TestCase):
    def setUp(self):
        Adapter.created=[]
        namespace={'MossAdapter':Adapter,'_TTSNode':Node,'hashlib':hashlib,'threading':threading,'LANGUAGES':('zh','en'),'language_for':lambda text,language='auto':language}
        exec(compile(ast.Module(body=[plugin_class],type_ignores=[]),str(source),'exec'),namespace)
        self.executor=Executor();self.plugin=namespace['TTSPlugin']({},self.executor)
    def call(self,action,key='eval',**args):
        return self.plugin.dispatch('tts',dict(action=action,instance_id=key,**args))
    def test_repeated_cases_reuse_one_independent_model(self):
        for _ in range(20):
            self.assertEqual(self.call('start',input_topic='/eval/text')['state'],'running')
            self.assertEqual(self.call('speak',text='hello')['status'],'queued')
            self.assertEqual(self.call('stop')['state'],'idle')
        self.assertEqual(len(Adapter.created),1,'Each stop/start loaded another full model')
        self.assertEqual(len(self.executor.nodes),1)
        self.assertEqual(self.call('info')['state'],'idle')
    def test_three_instances_never_share_a_model(self):
        for _ in range(5):
            for key in ('a','b','c'):
                self.call('start',key,input_topic='/'+key)
            self.assertEqual(len({id(node._adapter) for node in self.plugin.nodes.values()}),3)
            self.plugin.dispatch('tts',{'action':'stop'})
        self.assertEqual(len(Adapter.created),3)
    def test_stopped_instance_can_change_topic_without_reloading(self):
        self.call('start',input_topic='/first');previous=self.plugin.nodes['eval']
        with self.assertRaises(ValueError):self.call('start',input_topic='/second')
        self.call('stop');self.call('start',input_topic='/second')
        self.assertEqual(self.plugin.nodes['eval']._input_topic,'/second')
        self.assertIs(self.plugin.nodes['eval']._adapter,previous._adapter)
        self.assertTrue(previous.destroyed)
        self.assertEqual(len(Adapter.created),1)
    def test_config_updates_retained_model(self):
        self.call('start',input_topic='/eval');self.call('stop')
        self.call('config',language='en');self.call('start',input_topic='/eval')
        self.assertEqual(self.plugin.nodes['eval']._adapter.language,'en')
        self.assertEqual(len(Adapter.created),1)
    def test_unkeyed_speak_reuses_sole_instance_and_rejects_ambiguity(self):
        self.call('start',input_topic='/eval')
        self.plugin.dispatch('tts',{'action':'speak','text':'hello'})
        self.assertEqual(self.plugin.nodes['eval'].queued,['hello'])
        self.call('start','other',input_topic='/other')
        with self.assertRaises(ValueError):self.plugin.dispatch('tts',{'action':'speak','text':'hello'})
        self.assertEqual(len(Adapter.created),2)
    def test_stop_preserves_node_while_worker_finishes(self):
        self.call('start',input_topic='/eval')
        node=self.plugin.nodes['eval'];node._worker_thread=SimpleNamespace(is_alive=lambda:True)
        self.assertEqual(self.call('stop'),{'state':'stopping'})
        self.assertIs(self.plugin.nodes['eval'],node)

class ReleaseTests(unittest.TestCase):
    def setUp(self):
        path=Path(__file__).with_name('test_moss_mcp.py')
        function=next(node for node in ast.parse(path.read_text()).body if isinstance(node,ast.FunctionDef) and node.name=='wait_for_test_release')
        namespace={'Path':Path,'json':json,'time':time}
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),namespace)
        self.release=namespace['wait_for_test_release']
        self.folder=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]/'tmp')
        self.addCleanup(self.folder.cleanup)
        self.path=Path(self.folder.name)
    def test_loaded_model_waits_for_explicit_release(self):
        marker=self.path/'start';marker.touch()
        self.release(self.path,SimpleNamespace(pid=123,poll=lambda:None),str(marker))
        self.assertEqual(json.loads((self.path/'ready.json').read_text()),{'framework_pid':123,'ready':True})
    def test_release_timeout_fails_instead_of_starting(self):
        with self.assertRaises(TimeoutError):self.release(self.path,SimpleNamespace(pid=123,poll=lambda:None),str(self.path/'missing'),timeout=0)
    def test_exited_framework_is_detected_before_release(self):
        with self.assertRaises(RuntimeError):self.release(self.path,SimpleNamespace(pid=123,poll=lambda:1),str(self.path/'missing'))

if __name__=='__main__':unittest.main()
