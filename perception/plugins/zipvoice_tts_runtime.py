"""ZipVoice Chinese and Supertonic six-language routing, with independent state."""
import ctypes
import gc
import json
import queue
import threading
import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from .routed_tts_runtime import route_language as detect_language
from .supertonic_runtime import SupertonicAdapter, LANGUAGES as FOREIGN_LANGUAGES
from .tts_text_frontend import prepare_text

LANGUAGES = ('zh', *FOREIGN_LANGUAGES)


def route_language(text, language='auto', default='auto'):
    code = detect_language(text, language, default)
    if code not in LANGUAGES:
        raise ValueError('Language not covered by ZipVoice/Supertonic: ' + code)
    return code


class ZipVoiceAdapter:
    def __init__(self, model_dir):
        import sherpa_onnx
        if sherpa_onnx.__version__ != '1.13.6+cuda':
            raise RuntimeError('ZipVoice requires the validated sherpa-onnx 1.13.6+cuda')
        root = Path(model_dir)
        config = sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
            zipvoice=sherpa_onnx.OfflineTtsZipvoiceModelConfig(
                tokens=str(root/'tokens.txt'), encoder=str(root/'encoder.onnx'),
                decoder=str(root/'decoder.onnx'), vocoder=str(root/'vocos_24khz.onnx'),
                lexicon=str(root/'lexicon.txt'), data_dir=str(root/'espeak-ng-data'),
                feat_scale=.1, t_shift=.5, target_rms=.1, guidance_scale=3.),
            num_threads=2, provider='cuda:'+str(root/'ort-lean.conf')))
        if not config.validate():
            raise ValueError('Invalid ZipVoice model assets')
        self.tts = sherpa_onnx.OfflineTts(config)
        if self.tts.sample_rate != 24000:
            raise ValueError('Expected ZipVoice 24 kHz output')
        with wave.open(str(root/'reference-24k.wav')) as source:
            if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (24000, 1, 2):
                raise ValueError('Expected mono PCM16 Junhao reference at 24 kHz')
            samples = np.frombuffer(source.readframes(source.getnframes()), '<i2').astype(np.float32)/32768
        self.config = sherpa_onnx.GenerationConfig()
        self.config.reference_audio = samples
        self.config.reference_sample_rate = 24000
        self.config.reference_text = json.loads((root/'reference.json').read_text())['text']
        self.config.num_steps = 8
        self.config.speed = 1.
        self.config.silence_scale = 1.
        self.config.extra = {'min_char_in_sentence': '30', 'max_char_in_sentence': '80'}
        self.lock = threading.Lock()

    def synthesize_stream_language(self, text, language):
        if language != 'zh' or not isinstance(text, str) or not text.strip():
            raise ValueError('ZipVoice route requires nonempty Chinese text')
        prepared = prepare_text(text, 'zh', terms=True)
        with self.lock:
            messages = queue.Queue(maxsize=2)
            cancelled = threading.Event()
            progress = []
            frames = 0
            def send(kind, value):
                while not cancelled.is_set():
                    try:
                        messages.put((kind, value), timeout=.1)
                        return True
                    except queue.Full:
                        continue
                return False
            def callback(samples, fraction):
                nonlocal frames
                if cancelled.is_set():
                    return 0
                samples = np.asarray(samples, dtype=np.float32)
                if not samples.size or not np.isfinite(samples).all():
                    raise RuntimeError('ZipVoice returned invalid audio')
                progress.append(float(fraction)); frames += len(samples)
                audio = resample_poly(samples, 2, 3)
                pcm = (np.clip(audio, -1, 1)*32767).astype('<i2').tobytes()
                # v1.13.6 C++ continues on true, despite its Python docstring.
                return int(send('pcm', pcm))
            def generate():
                try:
                    audio = self.tts.generate(prepared, self.config, callback=callback)
                    if not cancelled.is_set():
                        if not progress or progress[-1] != 1. or frames != len(audio.samples):
                            raise RuntimeError('ZipVoice generation ended before all chunks completed')
                        send('done', None)
                except BaseException as error:
                    send('error', error)
            worker = threading.Thread(target=generate, name='zipvoice-generate')
            worker.start()
            try:
                while True:
                    kind, value = messages.get(timeout=600)
                    if kind == 'done':
                        break
                    if kind == 'error':
                        raise RuntimeError('ZipVoice synthesis failed') from value
                    yield value
            finally:
                cancelled.set()
                worker.join()
                gc.collect()
                ctypes.CDLL('libc.so.6').malloc_trim(0)


class ZipSupertonicAdapter:
    fast_delivery = True

    def __init__(self, model_dir, supertonic_model_dir, language='auto', seed=42):
        route_language('test', language)
        self.language = language
        self.zipvoice = ZipVoiceAdapter(model_dir)
        self.supertonic = SupertonicAdapter(supertonic_model_dir, seed=seed)

    def warmup(self):
        yield from self.zipvoice.synthesize_stream_language('你好，请确认设备已经连接。', 'zh')
        yield from self.supertonic.synthesize_stream_language('Hello.', 'en')

    def synthesize_stream_language(self, text, language=None):
        if not isinstance(text, str) or not text.strip():
            raise ValueError('text must be a nonempty string')
        code = route_language(text, language, self.language)
        backend = self.zipvoice if code == 'zh' else self.supertonic
        yield from backend.synthesize_stream_language(text, code)

    def validate_text(self, text, language=None):
        code = route_language(text, language, self.language)
        if code != 'zh':
            self.supertonic.validate_text(text, code)

    def synthesize(self, text, language=None):
        return b''.join(self.synthesize_stream_language(text, language))
