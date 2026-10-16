"""Exercise the actual framework MCP -> independent models -> ROS AudioChunk path."""
import argparse,concurrent.futures,json,os,subprocess,sys,threading,time,uuid,wave
from pathlib import Path
import urllib.request
import numpy as np
SAMPLES=[('zh','你好，欢迎使用多语言语音服务。'),('ja','こんにちは。多言語音声サービスへようこそ。'),('ru','Здравствуйте! Добро пожаловать в многоязычный голосовой сервис.'),('en','Hello, and welcome to our multilingual voice service.'),('es','Hola, bienvenido a nuestro servicio de voz multilingüe.'),('fr','Bonjour, bienvenue dans notre service vocal multilingue.'),('th','สวัสดี ยินดีต้อนรับสู่บริการเสียงหลายภาษา'),('de','Hallo, willkommen bei unserem mehrsprachigen Sprachdienst.')]

def wait_for_test_release(folder,server,start_file,timeout=180):
    if not start_file:return
    (folder/'ready.json').write_text(json.dumps({'framework_pid':server.pid,'ready':True}))
    deadline=time.monotonic()+timeout
    while not Path(start_file).is_file():
        if server.poll() is not None:raise RuntimeError('Framework exited before test release')
        if time.monotonic()>=deadline:raise TimeoutError('Waiting for other test models timed out')
        time.sleep(.1)


def main():
    global SAMPLES
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--triple',action='store_true');parser.add_argument('--rounds',type=int,default=5);parser.add_argument('--languages',default='zh,ja,ru,en,es,fr,th,de');parser.add_argument('--samples-file');parser.add_argument('--restart-each-case',action='store_true');parser.add_argument('--recreate-subscriber',action='store_true');parser.add_argument('--subscriber-delay-ms',type=int,default=0);parser.add_argument('--record-audio-loss',action='store_true');args=parser.parse_args()
    if args.samples_file:
        samples=json.loads(Path(args.samples_file).read_text())
        if not isinstance(samples,dict) or not samples or any(code not in dict(SAMPLES) or not isinstance(text,str) or not text.strip() for code,text in samples.items()):raise ValueError('Invalid sample file')
        SAMPLES=list(samples.items())
    if not 1<=args.rounds<=100:raise ValueError('rounds must be 1..100')
    if not 0<=args.subscriber_delay_ms<=500 or (args.subscriber_delay_ms and not args.recreate_subscriber):raise ValueError('Subscriber delay requires recreation and must be 0..500 ms')
    selected=args.languages.split(',')
    if not selected or len(set(selected))!=len(selected) or any(code not in dict(SAMPLES) for code in selected):raise ValueError('Invalid language selection')
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
    lock=threading.Lock();buffers={};events={};done={};rows=[];active={};subs=[];audio_subs={}
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
            if not args.recreate_subscriber:
                audio_subs[i]=node.create_subscription(AudioChunk,f'/moss_trial/instance{i}/text/tts',lambda msg,i=i:audio(i,msg),qos)
            result=call('start',instance_id=f'trial{i}',input_topic=f'/moss_trial/instance{i}/text')
            assert result['state']=='running',result
        status=call('info');assert status['independent_models']==count and status['state']=='running',status
        wait_for_test_release(folder,server,os.environ.get('MOSS_TEST_START_FILE'))
        time.sleep(.5)
        def measure(instance,label,text,language):
            if args.restart_each_case:
                status=call('start',instance_id=f'trial{instance}',input_topic=f'/moss_trial/instance{instance}/text')
                assert status['state']=='running',status
            if args.recreate_subscriber and instance in audio_subs:
                node.destroy_subscription(audio_subs.pop(instance))
            if args.recreate_subscriber and not args.subscriber_delay_ms:
                audio_subs[instance]=node.create_subscription(AudioChunk,f'/moss_trial/instance{instance}/text/tts',lambda msg:audio(instance,msg),qos)
            trace=uuid.uuid4().hex
            with lock:buffers[trace]=bytearray();events[trace]=[];active[instance]=trace
            target={'instance_id':f'trial{instance}'} if count>1 else {}
            start=time.perf_counter();queued=call('speak',**target,text=text,language=language,_trace_id=trace)
            assert queued['status']=='queued',queued
            if args.recreate_subscriber and args.subscriber_delay_ms:
                time.sleep(args.subscriber_delay_ms/1000)
                audio_subs[instance]=node.create_subscription(AudioChunk,f'/moss_trial/instance{instance}/text/tts',lambda msg:audio(instance,msg),qos)
            deadline=time.monotonic()+180
            while True:
                with lock:finished=trace in done
                if finished:break
                if server.poll() is not None:raise RuntimeError('Framework exited during synthesis')
                if time.monotonic()>deadline:raise TimeoutError('No TTS completion for '+label)
                time.sleep(.01)
            time.sleep(.05)
            with lock:pcm=bytes(buffers[trace]);arrivals=list(events[trace]);active.pop(instance,None)
            assert pcm and len(pcm)%2==0 and arrivals,'Missing/invalid streamed PCM'
            expected_bytes=done[trace]['meta']['pcm_bytes']
            complete=len(pcm)==expected_bytes
            samples=np.frombuffer(pcm,dtype='<i2');assert np.any(samples),'Silent PCM'
            duration=len(pcm)/32000
            elapsed=arrivals[-1]-start
            row={'instance':instance,'label':label,'language':language,'text':text,'audio_seconds':duration,'pcm_bytes':len(pcm),'chunks':len(arrivals),'first_pcm_seconds':arrivals[0]-start,'last_pcm_seconds':elapsed,'rtf_to_last_pcm':elapsed/duration,'rms':float(np.sqrt(np.mean((samples.astype(np.float64)/32768)**2)))}
            row.update(expected_pcm_bytes=expected_bytes,pcm_complete=complete,delivery=os.environ.get('MOSS_TTS_DELIVERY','optimized'),subscriber_wait_seconds=done[trace]['meta'].get('subscriber_wait_seconds'))
            if args.restart_each_case:
                stopped=call('stop',instance_id=f'trial{instance}')
                assert stopped['state']=='idle',stopped
                assert call('info')['independent_models']==count,'Per-case stop discarded a loaded model'
                memory=Path(f'/proc/{server.pid}/status').read_text().splitlines()
                row['framework_memory_kib']={line.split(':')[0]:int(line.split()[1]) for line in memory if line.startswith(('VmRSS:','VmHWM:'))}
            with wave.open(str(folder/(f'{instance}-{label}.wav')),'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(pcm)
            with lock:
                rows.append(row);(folder/'results.json').write_text(json.dumps({'commit':os.environ.get('TTS_COMMIT_ID'),'independent_models':count,'transport':'real MCP/ROS; RTF through last received PCM, not player completion','results':rows},ensure_ascii=False,indent=2))
            print('RESULT='+json.dumps(row,ensure_ascii=False),flush=True)
            assert complete or args.record_audio_loss,'DDS dropped audio bytes'
        if args.triple:
            triple_samples=[row for row in SAMPLES if row[0] in selected]
            for round_index in range(args.rounds):
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                    jobs=[]
                    for i in range(3):
                        code,text=triple_samples[(i+round_index)%len(triple_samples)]
                        jobs.append(pool.submit(measure,i,f'round{round_index}-{code}',text,code))
                    for job in jobs:job.result()
        else:
            for round_index in range(args.rounds if args.restart_each_case else 1):
                for code,text in SAMPLES:
                    if code in selected:measure(0,f'round{round_index}-{code}' if args.restart_each_case else code,text,code)
        assert call('info')['independent_models']==count,'Speak loaded an unexpected additional model'
        call('stop')
        assert len(rows)==(3*args.rounds if args.triple else len(selected)*(args.rounds if args.restart_each_case else 1))
        summary={'completed':True,'requests':len(rows),'independent_models':count,'first_pcm_range':[min(r['first_pcm_seconds'] for r in rows),max(r['first_pcm_seconds'] for r in rows)],'rtf_range':[min(r['rtf_to_last_pcm'] for r in rows),max(r['rtf_to_last_pcm'] for r in rows)],'all_rtf_below_1':all(r['rtf_to_last_pcm']<1 for r in rows),'all_first_pcm_below_1s':all(r['first_pcm_seconds']<1 for r in rows),'recognized_first_word_not_checked':True,'container_count':1}
        summary.update(restart_each_case=args.restart_each_case,recreate_subscriber=args.recreate_subscriber,subscriber_delay_ms=args.subscriber_delay_ms,delivery=os.environ.get('MOSS_TTS_DELIVERY','optimized'),all_pcm_complete=all(row['pcm_complete'] for row in rows),audio_loss_requests=sum(not row['pcm_complete'] for row in rows))
        (folder/'summary.json').write_text(json.dumps(summary,indent=2));print('SUMMARY='+json.dumps(summary),flush=True)
    finally:
        executor.shutdown(timeout_sec=3);node.destroy_node();rclpy.shutdown()
        if server.poll() is None:
            server.terminate()
            try:server.wait(timeout=10)
            except subprocess.TimeoutExpired:server.kill();server.wait()
        log.close()
if __name__=='__main__':main()
