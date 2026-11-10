"""Supertonic 3 Orin TensorRT adapter; native blocks -> the existing 16 kHz PCM contract."""
import logging
import threading
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

LANGUAGES = ('en', 'ja', 'de', 'fr', 'ru', 'es')


class DurationLimitError(ValueError):
    """Split the text before allocating a large predicted latent/audio buffer."""


def pcm16(audio):
    audio = np.asarray(audio)
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError('Expected nonempty finite mono audio')
    return np.rint(np.clip(audio, -1, 32767 / 32768) * 32768).astype('<i2').tobytes()


class SupertonicAdapter:
    def __init__(self, model_dir, seed=42):
        import torch
        import tensorrt as trt
        from . import vendor
        from ..moss_tts_runtime.trt_session import TrtSession

        if trt.__version__ != '10.4.0' or not torch.cuda.is_available():
            raise RuntimeError('Supertonic engines require Jetson TensorRT 10.4.0 and CUDA')
        torch.set_num_threads(1)
        # The four stages run serially under native_blocks' lock; run() synchronizes
        # each stream. Share shape-sized scratch instead of four profile maxima.
        workspace = {}

        class Session(TrtSession):
            def __init__(self, name):
                super().__init__(Path(model_dir) / (name + '.plan'), workspace=workspace)
                self.cpu_output = name in ('duration_predictor', 'vocoder')

            def run(self, requested, feeds):
                values = super().run(requested, feeds)
                return [v.cpu().numpy().copy() for v in values] if self.cpu_output else values

        class Runtime(vendor.TextToSpeech):
            def sample_noisy_latent(self, duration):
                if not np.isfinite(duration).all() or np.any(duration <= 0):
                    raise RuntimeError('Invalid predicted speech duration')
                size = self.base_chunk_size * self.chunk_compress_factor
                length = int((duration.max() * self.sample_rate + size - 1) / size)
                if not 1 <= length <= 192:
                    raise DurationLimitError('Predicted duration exceeds the bounded inference block')
                lengths = (duration * self.sample_rate).astype(np.int64)
                mask = vendor.get_latent_mask(lengths, self.base_chunk_size, self.chunk_compress_factor)
                latent = self.rng.randn(len(duration), self.ldim * self.chunk_compress_factor, length).astype(np.float32)
                return latent * mask, mask

        root = Path(model_dir)
        self.model = Runtime(vendor.load_cfgs(str(root)), vendor.load_text_processor(str(root)),
                             *[Session(n) for n in ('duration_predictor', 'text_encoder', 'vector_estimator', 'vocoder')])
        if self.model.sample_rate != 44100:
            raise ValueError('Expected Supertonic 3 native sample rate 44100')
        self.style = vendor.load_voice_style([str(root / 'F1.json')])
        self.seed = int(seed)
        self.lock = threading.Lock()
        self.sessions = [self.model.dp_ort, self.model.text_enc_ort,
                         self.model.vector_est_ort, self.model.vocoder_ort]
        self.workspace = workspace

    def release_temporary(self):
        import ctypes
        import gc
        import torch
        for session in self.sessions:
            session.release_temporary()
        self.workspace.clear()
        gc.collect()
        torch.cuda.empty_cache()
        ctypes.CDLL('libc.so.6').malloc_trim(0)

    def validate_text(self, text, language):
        from .text_chunks import text_chunks
        from ..tts_text_frontend import supertonic_text
        list(text_chunks(supertonic_text(text,language), language, self.model.text_processor))

    def native_blocks(self, text, language):
        from .text_chunks import text_chunks, split_point, continuation_text
        if not isinstance(text, str) or not text.strip():
            raise ValueError('text must be a nonempty string')
        if language not in LANGUAGES:
            raise ValueError('Unsupported Supertonic route: ' + str(language))
        from ..tts_text_frontend import supertonic_text
        prepared=supertonic_text(text,language)
        with self.lock:
            # Legacy RandomState matches the validated np.random.seed(42) baseline,
            # while keeping each instance independent of all other requests.
            self.model.rng = np.random.RandomState(self.seed)
            def infer(chunk,final):
                rng = self.model.rng.get_state()
                try:
                    wav, _ = self.model._infer([continuation_text(chunk,final)], [language], self.style, 8, 1.05)
                except DurationLimitError:
                    self.model.rng.set_state(rng)
                    if len(chunk) <= 8:
                        raise
                    cut = split_point(chunk, len(chunk) // 2, language)
                    yield from infer(chunk[:cut].strip(),False)
                    yield from infer(chunk[cut:].strip(),final)
                    return
                audio = wav.reshape(-1)
                if not audio.size or not np.isfinite(audio).all():
                    raise RuntimeError('Supertonic returned invalid audio')
                logging.getLogger(__name__).info('Supertonic block: language=%s chars=%s samples=%s', language, len(chunk), len(audio))
                yield audio
            try:
                chunks=iter(text_chunks(prepared, language, self.model.text_processor))
                current=next(chunks,None)
                for following in chunks:
                    yield from infer(current,False)
                    current=following
                if current is not None:yield from infer(current,True)
            finally:
                self.release_temporary()

    def synthesize_stream_language(self, text, language):
        from .audio_join import INTER_BLOCK_PAUSE_SECONDS, trim_zero_padding
        blocks = self.native_blocks(text, language)
        try:
            for index, audio in enumerate(blocks):
                if index:
                    yield bytes(round(16000 * INTER_BLOCK_PAUSE_SECONDS) * 2)
                yield trim_zero_padding(pcm16(resample_poly(audio, 160, 441)))
        finally:
            blocks.close()
