"""24 kHz trained voice to the platform's 16 kHz mono PCM16 stream."""
import numpy as np
from utils.resample import downsample_24k_to_16k
from .runtime import NativeKokoro


class XiaoyouKokoroAdapter:
    dry_run_text = '你好。'
    requires_subscriber_wait = True
    frame_interval_s = .05  # Faster delivery keeps DDS settling from forcing transport RTF above one.

    def __init__(self, directory, speed=1., on_status=None):
        if on_status:
            on_status('Loading Xiaoyou FP32 CUDA model')
        self.runtime = NativeKokoro(directory, speed=speed)

    def synthesize_stream(self, text):
        pending = b''
        for _, wave in self.runtime.generate(text):
            if not np.isfinite(wave).all():
                raise RuntimeError('Invalid native audio')
            pcm = downsample_24k_to_16k(wave)
            pending += pcm
            while len(pending) >= 3200:
                yield pending[:3200]
                pending = pending[3200:]
        if pending:
            yield pending

    def synthesize(self, text):
        return b''.join(self.synthesize_stream(text))

    def warmup(self):
        return len(self.synthesize(self.dry_run_text))

    def set_speed(self, speed):
        self.runtime.set_speed(speed)

    def set_language(self, language):
        if language not in ('auto', 'zh', 'en', 'mixed', 'en-us'):
            raise ValueError('Xiaoyou supports Chinese, English and mixed text')
