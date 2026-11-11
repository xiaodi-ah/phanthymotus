"""The native engine must preserve every PCM byte for a late DDS subscriber."""
import json
import threading
import time
import plugins.tts as tts

def test_late_subscriber_receives_complete_audio_faster_than_playback(monkeypatch):
    payload=b'\x02\x00'*1600
    class Adapter:
        dry_run_text='probe'
        requires_subscriber_wait=True
        frame_interval_s=.05
        def synthesize_stream(self,text):
            yield from [payload]*40
    monkeypatch.setattr(tts,'fire_hook',lambda *a,**k:None)
    node=tts._TTSNode('/native_delivery',Adapter(),'native_delivery')
    node._pub.subscription_count=0
    events=[];done=threading.Event();publish=node._pub.publish
    def record(msg):
        events.append((time.monotonic(),bytes(msg.data)))
        publish(msg)
        if bytes(msg.data)==tts.AUDIO_EOF_MAGIC:done.set()
    node._pub.publish=record
    assert node.start()['state']=='running'
    start=time.monotonic()
    timer=threading.Timer(.15,lambda:setattr(node._pub,'subscription_count',1));timer.start()
    try:
        node.enqueue('hello',trace_id='native-delivery')
        assert done.wait(8),'No EOF'
        node.stop()
        audio=[(stamp,data) for stamp,data in events if data!=tts.AUDIO_EOF_MAGIC]
        assert b''.join(data for _,data in audio)==payload*40
        assert audio[0][0]-start>=.60,'First frame was sent before stable subscription'
        assert (audio[-1][0]-start)/4<.9,'Transport pacing forced RTF over one'
        spans=[json.loads(value) for value in node._perf_pub.messages]
        assert any(s.get('trace_id')=='native-delivery' and s['meta'].get('pcm_bytes')==len(payload)*40 for s in spans)
    finally:
        timer.cancel();node.stop();node.destroy_node()
