"""Exercise the real audio worker with virtual time and DDS publishers."""
import ast
import logging
import queue
import threading
import unittest
from pathlib import Path
from types import MethodType,SimpleNamespace
from unittest.mock import patch

source=Path(__file__).resolve().parents[1]/'perception/plugins/tts.py'
node_class=next(item for item in ast.parse(source.read_text()).body if isinstance(item,ast.ClassDef) and item.name=='_TTSNode')
methods=[item for item in node_class.body if isinstance(item,ast.FunctionDef) and item.name in ('_worker','_wait_for_audio_subscriber')]
namespace={'queue':queue,'CHUNK_BYTES':3200,'SAMPLE_RATE':16000,'log':logging.getLogger('delivery-test')}
exec(compile(ast.Module(body=methods,type_ignores=[]),str(source),'exec'),namespace)

class Finished(BaseException):pass

class Clock:
    def __init__(self):self.now=0.0
    def monotonic(self):return self.now
    def time(self):return 1000+self.now
    def sleep(self,seconds):self.now+=seconds

class Event:
    def __init__(self,clock,cancel_at=float('inf')):self.clock=clock;self.cancel_at=cancel_at
    def is_set(self):return self.clock.now>=self.cancel_at
    def wait(self,seconds):self.clock.sleep(seconds);return self.is_set()

class Message:
    def __init__(self):self.header=SimpleNamespace(stamp=None);self.data=[];self.format=''

class Input:
    def __init__(self):self.used=False
    def get(self,**kwargs):
        if self.used:raise Finished()
        self.used=True;return ('sample','trace','en')

class DeliveryTests(unittest.TestCase):
    def exercise(self,optimized,*,ready_at=0.0,total_bytes=2560*60):
        clock=Clock();packets=[];received=[];spans=[]
        payload=bytes(index%251 for index in range(total_bytes))
        def generate(text,language):
            for index in range(0,len(payload),2560):
                clock.sleep(.025);yield payload[index:index+2560]
        def publish(message):
            packets.append((clock.now,bytes(message.data)))
            if clock.now>=ready_at:received.append(bytes(message.data))
        node=SimpleNamespace(_adapter=SimpleNamespace(fast_delivery=optimized,synthesize_stream_language=generate),_stop_event=Event(clock),_text_queue=Input(),_pub=SimpleNamespace(get_subscription_count=lambda:int(clock.now>=ready_at),publish=publish),_perf_pub=SimpleNamespace(publish=spans.append),get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(to_msg=lambda:None)))
        node._wait_for_audio_subscriber=MethodType(namespace['_wait_for_audio_subscriber'],node)
        with patch.dict('sys.modules',{'time':clock,'audio_msgs.msg':SimpleNamespace(AudioChunk=Message)}),patch.dict(namespace,{'String':Message}):
            with self.assertRaises(Finished):namespace['_worker'](node)
        return SimpleNamespace(payload=payload,packets=packets,received=b''.join(received),spans=spans,clock=clock)
    def test_faster_delivery_preserves_every_pcm_byte(self):
        legacy=self.exercise(False);optimized=self.exercise(True)
        self.assertEqual(legacy.received,legacy.payload)
        self.assertEqual(optimized.received,legacy.payload)
        self.assertLess(optimized.packets[-1][0],legacy.packets[-1][0])
        self.assertEqual(len(optimized.packets),len(legacy.packets))
    def test_late_subscriber_reproduces_loss_and_gate_prevents_it(self):
        legacy=self.exercise(False,ready_at=.4);optimized=self.exercise(True,ready_at=.4)
        self.assertNotEqual(legacy.received,legacy.payload)
        self.assertEqual(optimized.received,optimized.payload)
        self.assertGreaterEqual(optimized.packets[0][0],.5)
    def test_sub_frame_audio_waits_and_is_not_dropped(self):
        optimized=self.exercise(True,ready_at=.2,total_bytes=400)
        self.assertEqual(optimized.received,optimized.payload)
        self.assertEqual(len(optimized.packets),1)
    def test_legacy_mode_keeps_three_frame_prebuffer(self):
        # The existing 100ms interval and three-frame buffering remain selectable.
        legacy=self.exercise(False)
        self.assertEqual(len(legacy.packets[0][1]),3200)
        self.assertEqual(legacy.packets[0][0],legacy.packets[2][0])
    def test_gate_requires_continuous_matching(self):
        clock=Clock()
        node=SimpleNamespace(_stop_event=Event(clock),_pub=SimpleNamespace(get_subscription_count=lambda:int(.1<=clock.now<.15 or clock.now>=.3)))
        with patch.dict('sys.modules',{'time':clock}):wait=namespace['_wait_for_audio_subscriber'](node)
        self.assertGreaterEqual(wait,.4)
    def test_missing_subscriber_times_out_without_publishing(self):
        clock=Clock();node=SimpleNamespace(_stop_event=Event(clock),_pub=SimpleNamespace(get_subscription_count=lambda:0))
        with patch.dict('sys.modules',{'time':clock}):
            with self.assertRaisesRegex(RuntimeError,'No stable'):namespace['_wait_for_audio_subscriber'](node,timeout=.2)
    def test_stop_interrupts_subscriber_wait(self):
        clock=Clock();node=SimpleNamespace(_stop_event=Event(clock,cancel_at=.05),_pub=SimpleNamespace(get_subscription_count=lambda:0))
        with patch.dict('sys.modules',{'time':clock}):
            with self.assertRaisesRegex(RuntimeError,'stopped'):namespace['_wait_for_audio_subscriber'](node)
        self.assertLess(clock.now,.1)

if __name__=='__main__':unittest.main()
