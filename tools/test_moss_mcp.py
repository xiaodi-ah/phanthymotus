"""Exercise the actual framework MCP -> independent models -> ROS AudioChunk path."""
import argparse,concurrent.futures,json,os,subprocess,sys,threading,time,uuid,wave
from pathlib import Path
import urllib.request
import numpy as np
SAMPLES=[('zh','你好，欢迎使用多语言语音服务。'),('ja','こんにちは。多言語音声サービスへようこそ。'),('ru','Здравствуйте! Добро пожаловать в многоязычный голосовой сервис.'),('en','Hello, and welcome to our multilingual voice service.'),('es','Hola, bienvenido a nuestro servicio de voz multilingüe.'),('fr','Bonjour, bienvenue dans notre service vocal multilingue.'),('th','สวัสดี ยินดีต้อนรับสู่บริการเสียงหลายภาษา'),('de','Hallo, willkommen bei unserem mehrsprachigen Sprachdienst.')]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--triple',action='store_true');args=parser.parse_args()
    folder=Path(args.output);folder.mkdir(parents=True,exist_ok=True)
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.qos import QoSProfile,ReliabilityPolicy,HistoryPolicy,DurabilityPolicy
    from audio_msgs.msg import AudioChunk
    from std_msgs.msg import String
    port=int(os.environ.get('MCP_PORT',16873));ws_port=int(os.environ.get('WS_PORT',16874))
    assert port!=15720 and ws_port!=15721,'This test must verify environment port overrides'
    env=dict(os.environ,MCP_PORT=str(port),WS_PORT=str(ws_port),TTS_PLUGIN='moss_tts')
    log=(folder/'framework.log').open('w')
    server=subprocess.Popen(['/bin/bash','-c','source /opt/ros/humble/install/setup.bash && source /ros_ws/install/setup.bash && python3 -B /work/main.py'],env=env,stdout=log,stderr=subprocess.STDOUT)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def rpc(method,params):
        body=json.dumps({'jsonrpc':'2.0','id':uuid.uuid4().hex,'method':method,'params':params}).encode()
        req=urllib.request.Request(f'http://127.0.0.1:{port}/mcp',data=body,headers={'Content-Type':'application/json'})
        with opener.open(req,timeout=120) as r:response=json.load(r)
        if 'error' in response:raise RuntimeError(response['error'])
        return response['result']
    def call(action,**values):
        result=rpc('tools/call',{'name':'tts','arguments':{'action':action,**values}})
        return json.loads(result['content'][0]['text'])
    rclpy.init();node=Node('moss_submission_probe');executor=MultiThreadedExecutor(num_threads=2);executor.add_node(node)
    qos=QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,history=HistoryPolicy.KEEP_LAST,depth=200,durability=DurabilityPolicy.VOLATILE)
    lock=threading.Lock();buffers={};events={};done={};rows=[];active={};subs=[]
    def audio(instance,msg):
        if msg.format!='audio/pcm-16k':raise RuntimeError('Wrong audio format: '+msg.format)
        stamp=time.perf_counter()
        with lock:
            trace=active.get(instance)
            if trace:
                buffers[trace].extend(bytes(msg.data));events[trace].append(stamp)
    def perf(msg):
        payload=json.loads(msg.data)
        if payload.get('span')=='tts_playback':
            with lock:done[payload.get('trace_id')]=payload
    subs.append(node.create_subscription(String,'/perception/perf_spans',perf,qos))
    spin=threading.Thread(target=executor.spin,daemon=True);spin.start()
    try:
        deadline=time.monotonic()+30
        while True:
            if server.poll() is not None:raise RuntimeError('Framework exited during startup; see framework.log')
            try:tools=rpc('tools/list',{});break
            except Exception:
                if time.monotonic()>deadline:raise
                time.sleep(.2)
        assert any(t['name']=='tts' for t in tools['tools'])
        count=3 if args.triple else 1
        for i in range(count):
            subs.append(node.create_subscription(AudioChunk,f'/moss_trial/instance{i}/text/tts',lambda msg,i=i:audio(i,msg),qos))
            result=call('start',instance_id=f'trial{i}',input_topic=f'/moss_trial/instance{i}/text')
            assert result['state']=='running',result
        status=call('info');assert status['independent_models']==count and status['state']=='running',status
        time.sleep(.5)
        def measure(instance,label,text,language):
            trace=uuid.uuid4().hex
            with lock:buffers[trace]=bytearray();events[trace]=[];active[instance]=trace
            start=time.perf_counter();queued=call('speak',instance_id=f'trial{instance}',text=text,language=language,_trace_id=trace)
            assert queued['status']=='queued',queued
            deadline=time.monotonic()+90
            while True:
                with lock:finished=trace in done
                if finished:break
                if server.poll() is not None:raise RuntimeError('Framework exited during synthesis')
                if time.monotonic()>deadline:raise TimeoutError('No TTS completion for '+label)
                time.sleep(.01)
            time.sleep(.05)
            with lock:pcm=bytes(buffers[trace]);arrivals=list(events[trace]);active.pop(instance,None)
            assert pcm and len(pcm)%2==0 and arrivals,'Missing/invalid streamed PCM'
            assert len(pcm)==done[trace]['meta']['pcm_bytes'],'DDS dropped audio bytes'
            samples=np.frombuffer(pcm,dtype='<i2');assert np.any(samples),'Silent PCM'
            duration=len(pcm)/32000
            elapsed=arrivals[-1]-start
            row={'instance':instance,'label':label,'language':language,'text':text,'audio_seconds':duration,'pcm_bytes':len(pcm),'chunks':len(arrivals),'first_pcm_seconds':arrivals[0]-start,'last_pcm_seconds':elapsed,'rtf_to_last_pcm':elapsed/duration,'rms':float(np.sqrt(np.mean((samples.astype(np.float64)/32768)**2)))}
            with wave.open(str(folder/(f'{instance}-{label}.wav')),'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(pcm)
            with lock:
                rows.append(row);(folder/'results.json').write_text(json.dumps({'commit':os.environ.get('TTS_COMMIT_ID'),'independent_models':count,'transport':'real MCP/ROS; RTF through last received PCM, not player completion','results':rows},ensure_ascii=False,indent=2))
            print('RESULT='+json.dumps(row,ensure_ascii=False),flush=True)
        if args.triple:
            for round_index in range(5):
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                    jobs=[]
                    for i in range(3):
                        code,text=[SAMPLES[0],SAMPLES[3],SAMPLES[1]][(i+round_index)%3]
                        jobs.append(pool.submit(measure,i,f'round{round_index}-{code}',text,code))
                    for job in jobs:job.result()
        else:
            for code,text in SAMPLES:measure(0,code,text,code)
        call('stop')
        assert len(rows)==(15 if args.triple else 8)
        summary={'completed':True,'requests':len(rows),'independent_models':count,'first_pcm_range':[min(r['first_pcm_seconds'] for r in rows),max(r['first_pcm_seconds'] for r in rows)],'rtf_range':[min(r['rtf_to_last_pcm'] for r in rows),max(r['rtf_to_last_pcm'] for r in rows)],'all_rtf_below_1':all(r['rtf_to_last_pcm']<1 for r in rows),'all_first_pcm_below_1s':all(r['first_pcm_seconds']<1 for r in rows),'recognized_first_word_not_checked':True,'container_count':1}
        (folder/'summary.json').write_text(json.dumps(summary,indent=2));print('SUMMARY='+json.dumps(summary),flush=True)
    finally:
        executor.shutdown(timeout_sec=3);node.destroy_node();rclpy.shutdown()
        if server.poll() is None:
            server.terminate()
            try:server.wait(timeout=10)
            except subprocess.TimeoutExpired:server.kill();server.wait()
        log.close()
if __name__=='__main__':main()
