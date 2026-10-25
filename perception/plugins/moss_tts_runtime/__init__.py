"""On-device MOSS inference, using our Orin frame-fusion engines."""
import ctypes,gc,json,logging,os,queue,re,threading,unicodedata
from pathlib import Path
from types import MethodType
import numpy as np
from .resample import StreamingPCM,SentenceEnding
from .text_chunks import text_chunks
from .text_normalization import normalize_text

LANGUAGES=('zh','ja','ru','en','es','fr','th','de')
VOICES={'zh':'Zhiming','ja':'Saki','ru':'Adam','en':'Adam','es':'Ava','fr':'Bella','th':'Ava','de':'Adam'}
_LATIN_CUES={
    'en':set('the this that these those is are was were and with for from please hello welcome your you of to'.split()),
    'de':set('der die das den dem des ein eine einer einen ist sind und nicht bitte guten morgen willkommen sie auf für zum zur mit wird werden'.split()),
    'fr':set('le les des du une est sont et dans pour avec vous votre veuillez bonjour bienvenue merci sur au aux il cette ces'.split()),
    'es':set('el los las del una es son está están por para con usted favor buenos días hola bienvenido bienvenida gracias sobre al este esta y tiene tienen cuando hacia'.split())}

def language_for(text,language='auto'):
    language=str(language or 'auto').lower().replace('_','-').split('-')[0]
    if language!='auto':
        if language not in LANGUAGES:raise ValueError('Unsupported TTS language: '+language)
        return language
    if any('\u3040'<=c<='\u30ff' for c in text):return 'ja'
    if any('\u0e00'<=c<='\u0e7f' for c in text):return 'th'
    if any('\u0400'<=c<='\u052f' for c in text):return 'ru'
    if any('\u3400'<=c<='\u9fff' for c in text):return 'zh'
    value=re.sub(r'https?://\S+|\S+@\S+|`[^`]*`',' ',text)
    words=set(re.findall(r'[^\W\d_]+',unicodedata.normalize('NFC',value).casefold()))
    scores=sorted(((len(words&cues),code) for code,cues in _LATIN_CUES.items()),reverse=True)
    if scores[0][0]>=2 and scores[0][0]-scores[1][0]>=2:return scores[0][1]
    # shortcut: lexical cues handle clear Latin text; use explicit language for short/mixed ambiguity.
    return 'en'

class MossAdapter:
    dry_run_text='你好。'
    def __init__(self,model_dir,cuda_graph=True,language='auto',seed=42):
        delivery=os.environ.get('MOSS_TTS_DELIVERY','optimized')
        if delivery not in ('legacy','optimized'):raise ValueError('MOSS_TTS_DELIVERY must be legacy or optimized')
        self.fast_delivery=delivery=='optimized'
        self.stop_mode=os.environ.get('MOSS_TTS_STOP_MODE','guarded')
        if self.stop_mode not in ('sample','greedy','guarded'):raise ValueError('MOSS_TTS_STOP_MODE must be sample, greedy or guarded')
        self.text_frontend=os.environ.get('MOSS_TTS_TEXT_FRONTEND','numbers')
        if self.text_frontend not in ('off','numbers','numbers_terms'):raise ValueError('MOSS_TTS_TEXT_FRONTEND must be off, numbers or numbers_terms')
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
        frame_session=runtime.sessions['local_fixed_sampled_frame']
        if 'repetition_seen_mask' in frame_session.inputs:
            sampling=json.loads((root/'frame_sampling.json').read_text())
            expected=dict(text_temperature=1.,audio_temperature=.8,audio_top_k=25,audio_top_p=.95,audio_repetition_penalty=1.2,repetition_history=True)
            if sampling!=expected:raise RuntimeError('Unsupported Nano frame sampling metadata')
            defaults.update({key:value for key,value in sampling.items() if key!='repetition_history'})
        def fused_frame(self,global_hidden,*,previous_token_sets_by_channel):
            uniform=np.clip(self.rng.random((1,17)),0,.99999994).astype(np.float32)
            original_u=float(uniform[0,0]);session=self.sessions['local_fixed_sampled_frame']
            feeds={'global_hidden':global_hidden,'random_u':uniform}
            if 'repetition_seen_mask' in getattr(session,'inputs',()):
                if len(previous_token_sets_by_channel)!=16:raise ValueError('Expected 16 repetition histories')
                mask=np.zeros((1,16,1024),np.int32)
                for channel,seen in enumerate(previous_token_sets_by_channel):
                    ids=np.asarray(list(seen),dtype=np.int32)
                    if np.any((ids<0)|(ids>=1024)):raise ValueError('Invalid audio token in repetition history')
                    mask[0,channel,ids]=1
                feeds['repetition_seen_mask']=mask
            if self.greedy_stop:
                # Force the slot embedding inside the graph; decide EOS from its exported scores.
                uniform[:,0]=0
            continuing,frame,scores=session.run(['should_continue','frame_token_ids','text_scores'],feeds)
            if scores.shape!=(1,2) or not np.isfinite(scores).all():raise RuntimeError('Invalid continuation/EOS scores')
            probabilities=np.exp(scores[0].astype(np.float64)-scores.max());probabilities/=probabilities.sum()
            selected=bool(scores[0,0]>=scores[0,1]) if self.greedy_stop else bool(continuing[0,0])
            guarded=getattr(self,'guard_stop',False) and not self.greedy_stop
            overridden=guarded and not selected and probabilities[0]>=.99
            if overridden:
                # Recompute under the continuation embedding; EOS-conditioned audio cannot be reused.
                uniform[:,0]=0
                frame,=session.run(['frame_token_ids'],feeds)
                selected=True
            self.last_stop=dict(continue_probability=float(probabilities[0]),random_u=original_u,continuing=selected,mode='greedy' if self.greedy_stop else 'guarded' if guarded else 'sample',rare_eos_overridden=bool(overridden))
            return selected,frame.reshape(-1).tolist()
        runtime.greedy_stop=False;runtime.guard_stop=False
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
        if self.text_frontend!='off':text=normalize_text(text,code,terms=self.text_frontend=='numbers_terms')
        channel=queue.Queue(maxsize=8);cancel=threading.Event();end=object()
        def put(value):
            while not cancel.is_set():
                try:channel.put(value,timeout=.1);return True
                except queue.Full:pass
            return False
        def produce():
            try:
                with self.lock:
                    # Reset request stopping policy under the model lock.
                    self.runtime.greedy_stop=self.stop_mode in ('greedy','guarded') and code=='en'
                    self.runtime.guard_stop=self.stop_mode=='guarded' and code!='en'
                    for chunk in text_chunks(text,self.tokenizer):
                        if cancel.is_set():return
                        self.reset_codec();self.runtime.rng=np.random.default_rng(self.seed);converter=StreamingPCM();ending=SentenceEnding()
                        request=self.runtime.build_voice_clone_request_rows(self.voices[VOICES[code]],self.tokenizer.encode(chunk,out_type=int))
                        prompt=len(request['inputIds'])
                        if prompt>512:raise ValueError('Text exceeds certified prefill profile (512 rows); use shorter utterances')
                        budget=min(375,768-prompt)
                        if budget<=0:raise ValueError('No certified KV capacity for this request')
                        self.runtime.manifest['generation_defaults']['max_new_frames']=budget
                        def on_frame(all_frames,index,frame):
                            if cancel.is_set():raise RuntimeError('Synthesis cancelled')
                            audio,length=self.codec.run_frames([frame]);pcm=ending.process(converter.process(audio[0,:,:length]))
                            if pcm and not put(pcm):raise RuntimeError('Synthesis cancelled')
                        frames=self.runtime.generate_audio_frames(request,on_frame=on_frame)
                        logging.getLogger(__name__).info('MOSS chunk: tokens=%s frames=%s budget=%s stop=%s',len(self.tokenizer.encode(chunk,out_type=int)),len(frames),budget,getattr(self.runtime,'last_stop',None))
                        if not frames:raise RuntimeError('Model produced no audio')
                        if len(frames)>=budget:raise RuntimeError('Frame/KV limit reached; audio may be incomplete')
                        pcm=ending.finish()
                        if pcm and not put(pcm):raise RuntimeError('Synthesis cancelled')
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
