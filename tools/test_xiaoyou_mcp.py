"""Real MCP to ROS probe for one or three independent E inference containers."""
from functools import partial
import argparse,concurrent.futures,json,os,threading,time,uuid,wave
from pathlib import Path
import urllib.request
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile,ReliabilityPolicy,HistoryPolicy
from audio_msgs.msg import AudioChunk
from std_msgs.msg import String

SAMPLES=[('zh','你好，我是小优，很高兴见到你。'),('en','Hello, I am Xiaoyou. It is nice to meet you.'),('mixed','今天的 meeting 已经安排好了，请点击 Start 按钮开始。'),('long','欢迎来到这里，我会帮你安排今天的工作。'*8)]
EOF=b'\x01\x00\xff\xff\x01\x00\xff\xff'

def main():
 p=argparse.ArgumentParser();p.add_argument('--ports',required=True);p.add_argument('--output',required=True);p.add_argument('--rounds',type=int,default=3);p.add_argument('--late-subscriber-ms',type=int,default=0);a=p.parse_args()
 ports=[int(x) for x in a.ports.split(',')];assert len(ports) in (1,3) and len(set(ports))==len(ports) and 1<=a.rounds<=20
 folder=Path(a.output);folder.mkdir(parents=True,exist_ok=True)
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 def rpc(port,method,params):
  body=json.dumps({'jsonrpc':'2.0','id':uuid.uuid4().hex,'method':method,'params':params}).encode()
  request=urllib.request.Request(f'http://127.0.0.1:{port}/mcp',data=body,headers={'Content-Type':'application/json'})
  with opener.open(request,timeout=120) as response:r=json.load(response)
  if 'error' in r:raise RuntimeError(r['error'])
  return r['result']
 def call(port,action,**kwargs):
  r=rpc(port,'tools/call',{'name':'tts','arguments':dict(action=action,**kwargs)})
  value=json.loads(r['content'][0]['text'])
  if value.get('status')=='error' or value.get('state')=='error':raise RuntimeError(value)
  return value
 rclpy.init();node=Node('xiaoyou_probe_'+uuid.uuid4().hex[:8]);executor=MultiThreadedExecutor(num_threads=3);executor.add_node(node)
 qos=QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,history=HistoryPolicy.KEEP_LAST,depth=200)
 lock=threading.Lock();active={};rows=[];spans={};traces={};subscriptions={};errors=[]
 def audio(index,msg):
  with lock:
   trace=active.get(index)
   if trace is None:return
   state=traces[trace];data=bytes(msg.data)
   if msg.format!='audio/pcm-16k':errors.append('Wrong format '+msg.format)
   if data==EOF:state['eof']=True;return
   if not data or len(data)%2 or len(data)>3200:errors.append('Invalid PCM frame size')
   state['pcm'].extend(data);state['arrivals'].append(time.perf_counter())
 def perf(msg):
  value=json.loads(msg.data)
  if 'pcm_bytes' in value.get('meta',{}):
   with lock:spans[value.get('trace_id')]=value
 perf_sub=node.create_subscription(String,'/perception/perf_spans',perf,qos)
 spin=threading.Thread(target=executor.spin,daemon=True)
 topics=['/xiaoyou_e_trial/u'+uuid.uuid4().hex+'/text' for _ in ports]
 try:
  for index,port in enumerate(ports):
   tools=rpc(port,'tools/list',{});assert any(t['name']=='tts' for t in tools['tools'])
   deadline=time.monotonic()+180
   while True:
    info=call(port,'info')
    if info.get('state')!='loading':break
    if time.monotonic()>deadline:raise TimeoutError('Model startup timed out')
    time.sleep(.2)
   assert info.get('engine')=='kokoro-xiaoyou-zh-en' and not info.get('error'),info
   if not a.late_subscriber_ms:subscriptions[index]=node.create_subscription(AudioChunk,topics[index]+'/tts',partial(audio,index),qos)
   configured=call(port,'config',speed=1.0,language='auto');assert configured.get('status')!='error'
   started=call(port,'start',input_topic=topics[index]);assert started['state']=='running',started
  spin.start()
  executor.wake()
  def measure(index,round_id,case):
   name,text=case;trace=uuid.uuid4().hex
   with lock:active[index]=trace;traces[trace]={'pcm':bytearray(),'arrivals':[],'eof':False}
   if a.late_subscriber_ms and index in subscriptions:node.destroy_subscription(subscriptions.pop(index))
   start=time.perf_counter();result=call(ports[index],'speak',text=text,_trace_id=trace);assert result.get('status')=='queued',result
   if a.late_subscriber_ms:
    time.sleep(a.late_subscriber_ms/1000)
    subscriptions[index]=node.create_subscription(AudioChunk,topics[index]+'/tts',partial(audio,index),qos)
   deadline=time.monotonic()+120
   progress_at=0
   while True:
    with lock:done=traces[trace]['eof'] and trace in spans
    if done:break
    if time.monotonic()-progress_at>1:
     with lock:progress={'trace':trace,'case':name,'pcm_bytes':len(traces[trace]['pcm']),'eof':traces[trace]['eof'],'perf_received':trace in spans,'perf_traces':list(spans),'spin_alive':spin.is_alive(),'audio_publishers':node.count_publishers(topics[index]+'/tts'),'perf_publishers':node.count_publishers('/perception/perf_spans')}
     (folder/'progress.json').write_text(json.dumps(progress));progress_at=time.monotonic()
    if time.monotonic()>deadline:raise TimeoutError('Missing EOF/completion: '+name)
    time.sleep(.01)
   with lock:
    state=traces[trace];pcm=bytes(state['pcm']);arrivals=list(state['arrivals']);expected=spans[trace]['meta']['pcm_bytes'];active.pop(index)
   assert not errors,errors
   assert pcm and len(pcm)==expected and arrivals,(len(pcm),expected)
   samples=np.frombuffer(pcm,dtype='<i2');assert np.max(np.abs(samples.astype(np.int32)))>100
   seconds=len(pcm)/32000;ttft=arrivals[0]-start;rtf=(arrivals[-1]-start)/seconds
   assert ttft<60
   path=folder/f'{index}-r{round_id}-{name}.wav'
   with wave.open(str(path),'wb') as w:w.setparams((1,2,16000,0,'NONE','not compressed'));w.writeframes(pcm)
   row=dict(instance=index,round=round_id,case=name,text=text,pcm_bytes=len(pcm),seconds=seconds,first_pcm_seconds=ttft,rtf_to_last_pcm=rtf,pcm_complete=True)
   with lock:rows.append(row)
  for round_id in range(a.rounds):
   for case in SAMPLES:
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(ports)) as pool:
     tasks=[pool.submit(measure,i,round_id,case) for i in range(len(ports))]
     for task in tasks:task.result()
  for i,port in enumerate(ports):
   stopped=call(port,'stop');assert stopped['state']=='idle',stopped
   restarted=call(port,'start',input_topic=topics[i]);assert restarted['state']=='running',restarted
   call(port,'stop')
  result=dict(passed=True,transport='real MCP to ROS AudioChunk',containers=len(ports),requests=len(rows),rows=rows,all_pcm_complete=True,late_subscriber_ms=a.late_subscriber_ms,all_rtf_below_one=all(r['rtf_to_last_pcm']<1 for r in rows))
  result['passed']=result['all_rtf_below_one']
  (folder/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False));assert result['passed'],'Transport RTF must remain below one'
 finally:
  executor.shutdown(timeout_sec=3);node.destroy_node();rclpy.shutdown()
if __name__=='__main__':main()
