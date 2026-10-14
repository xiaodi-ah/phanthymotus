"""On-device MOSS inference, using our Orin frame-fusion engines."""
import ctypes,gc,json,queue,threading
from pathlib import Path
from types import MethodType
import numpy as np
from .resample import StreamingPCM
from .text_chunks import text_chunks

LANGUAGES=('zh','ja','ru','en','es','fr','th','de')
VOICES={'zh':'Zhiming','ja':'Soyo','ru':'Adam','en':'Adam','es':'Ava','fr':'Bella','th':'Ava','de':'Nathan'}

def language_for(text,language='auto'):
    language=str(language or 'auto').lower().replace('_','-').split('-')[0]
    if language!='auto':
        if language not in LANGUAGES:raise ValueError('Unsupported TTS language: '+language)
        return language
    if any('\u3040'<=c<='\u30ff' for c in text):return 'ja'
    if any('\u0e00'<=c<='\u0e7f' for c in text):return 'th'
    if any('\u0400'<=c<='\u052f' for c in text):return 'ru'
    if any('\u3400'<=c<='\u9fff' for c in text):return 'zh'
    # shortcut: Latin auto uses the English reference voice; supply language for ES/FR/DE.
    return 'en'

class MossAdapter:
    dry_run_text='你好。'
    def __init__(self,model_dir,cuda_graph=True,language='auto',seed=42):
        import torch,sentencepiece as spm,tensorrt as trt
        from . import ort_cpu_runtime as vendor
        from .trt_session import TrtSession
        from .trt_fused_session import FusedSession
        from .trt_codec_session import CodecGraphSession
        if trt.__version__!='10.4.0':raise RuntimeError('These Orin engines require TensorRT 10.4.0; got '+trt.__version__)
        if not torch.cuda.is_available():raise RuntimeError('Jetson CUDA is required')
        torch.set_num_threads(1);root=Path(model_dir)
        vendor._extract_last_hidden=lambda value:value if value.ndim==2 else value[:,-1,:]
        runtime=vendor.OrtCpuRuntime.__new__(vendor.OrtCpuRuntime)
        runtime.manifest=json.loads((root/'browser_manifest.json').read_text())
        runtime.tts_meta=json.loads((root/'tts_meta.json').read_text())
        defaults=runtime.manifest['generation_defaults']
        defaults.update(max_new_frames=375,sample_mode=vendor.SAMPLE_MODE_FIXED,do_sample=True,text_temperature=1.5,text_top_p=1.,text_top_k=50,audio_temperature=1.7,audio_top_p=.8,audio_top_k=25,audio_repetition_penalty=1.)
        runtime.sessions={'prefill':TrtSession(root/'prefill.plan'),'decode':TrtSession(root/'decode.plan'),'local_fixed_sampled_frame':FusedSession(root/'frame.plan',cuda_graph=cuda_graph)}
        def fused_frame(self,global_hidden,*,previous_token_sets_by_channel):
            uniform=np.clip(self.rng.random((1,17)),0,.99999994).astype(np.float32)
            continuing,frame=self.sessions['local_fixed_sampled_frame'].run(['should_continue','frame_token_ids'],{'global_hidden':global_hidden,'random_u':uniform})
            return bool(continuing[0,0]),frame.reshape(-1).tolist()
        runtime.run_local_fixed_sampled_frame=MethodType(fused_frame,runtime)
        codec_meta=json.loads((root/'codec_meta.json').read_text())
        self.codec=vendor.CodecStreamingDecodeSession(codec_meta,CodecGraphSession(root/'codec.plan',codec_meta) if cuda_graph else TrtSession(root/'codec.plan'))
        self.tokenizer=spm.SentencePieceProcessor(model_file=str(root/'tokenizer.model'))
        self.voices={v['voice']:v['prompt_audio_codes'] for v in runtime.manifest['builtin_voices']}
        self.runtime=runtime;self.language=language;self.seed=int(seed);self.lock=threading.Lock()
        gc.collect();ctypes.CDLL('libc.so.6').malloc_trim(0)
    def reset_codec(self):
        import torch
        if all(isinstance(value,torch.Tensor) for value in self.codec.state_feeds.values()):
            positions={str(spec['cached_positions_input_name']) for spec in self.codec.attention_specs}
            # Queue reset on the codec stream before it reads its recurrent state.
            with torch.cuda.stream(self.codec.session.stream):
                for name,value in self.codec.state_feeds.items():value.fill_(-1 if name in positions else 0)
        else:self.codec.reset()
    def synthesize(self,text,language=None):return b''.join(self.synthesize_stream_language(text,language))
    def synthesize_stream(self,text):yield from self.synthesize_stream_language(text,self.language)
    def synthesize_stream_language(self,text,language=None):
        if not isinstance(text,str) or not text.strip():raise ValueError('text must be a nonempty string')
        code=language_for(text,language if language not in (None,'auto') else self.language)
        channel=queue.Queue(maxsize=8);cancel=threading.Event();end=object()
        def put(value):
            while not cancel.is_set():
                try:channel.put(value,timeout=.1);return True
                except queue.Full:pass
            return False
        def produce():
            try:
                with self.lock:
                    for chunk in text_chunks(text,self.tokenizer):
                        if cancel.is_set():return
                        self.reset_codec();self.runtime.rng=np.random.default_rng(self.seed);converter=StreamingPCM()
                        request=self.runtime.build_voice_clone_request_rows(self.voices[VOICES[code]],self.tokenizer.encode(chunk,out_type=int))
                        prompt=len(request['inputIds'])
                        if prompt>512:raise ValueError('Text exceeds certified prefill profile (512 rows); use shorter utterances')
                        budget=min(375,768-prompt)
                        if budget<=0:raise ValueError('No certified KV capacity for this request')
                        self.runtime.manifest['generation_defaults']['max_new_frames']=budget
                        def on_frame(all_frames,index,frame):
                            if cancel.is_set():raise RuntimeError('Synthesis cancelled')
                            audio,length=self.codec.run_frames([frame]);pcm=converter.process(audio[0,:,:length])
                            if not put(pcm):raise RuntimeError('Synthesis cancelled')
                        frames=self.runtime.generate_audio_frames(request,on_frame=on_frame)
                        if not frames:raise RuntimeError('Model produced no audio')
                        if len(frames)>=budget:raise RuntimeError('Frame/KV limit reached; audio may be incomplete')
            except Exception as error:
                if not cancel.is_set():put(error)
            finally:put(end)
        worker=threading.Thread(target=produce,daemon=True,name='moss-inference');worker.start()
        try:
            while True:
                value=channel.get(timeout=120)
                if value is end:break
                if isinstance(value,Exception):raise value
                yield value
        finally:
            cancel.set();worker.join(timeout=5)
