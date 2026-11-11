"""Native FP32 CUDA Kokoro; bound predicted duration before allocating the vocoder."""
import hashlib
import json
import math
from pathlib import Path
import re
import threading

import numpy as np

from .frontend import Frontend


ASCII_TOKEN = re.compile(r'[A-Za-z0-9_@:/.-]')


def split_once(text):
    """Keep punctuation and complete Latin/number spans when dividing a long input."""
    center = len(text) / 2
    lo, hi = len(text) * .2, len(text) * .8
    punctuation = [i + 1 for i, c in enumerate(text) if c in ',，;；。！？!?' and lo <= i + 1 <= hi]
    spaces = [i + 1 for i, c in enumerate(text) if c.isspace() and lo <= i + 1 <= hi]
    boundaries = [i for i in range(1, len(text)) if lo <= i <= hi and
                  not (ASCII_TOKEN.fullmatch(text[i - 1]) and ASCII_TOKEN.fullmatch(text[i]))]
    choices = punctuation or spaces or boundaries
    if not choices:
        raise ValueError('Input has a single unsplittable token over the inference budget')
    cut = min(choices, key=lambda i: abs(i - center))
    parts = text[:cut].strip(), text[cut:].strip()
    if not all(parts):
        raise ValueError('Cannot split input without dropping content')
    return parts


def verify_assets(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('format') != 'kokoro-native-gpu-v1' or manifest.get('precision') != 'FP32':
        raise ValueError('Unsupported Xiaoyou model format or precision')
    voice = manifest.get('voice', 'A')
    if voice not in ('A', 'E'):
        raise ValueError('Unsupported trained voice')
    if len(manifest['files']) != 3 or {r['path'] for r in manifest['files']} != {'kokoro.pth', 'config.json', voice + '.pt'}:
        raise ValueError('Incomplete inference asset set')
    if manifest.get('voice_file', voice + '.pt') != voice + '.pt':
        raise ValueError('Voice file does not match the release')
    for entry in manifest['files']:
        path = directory / entry['path']
        if path.stat().st_size != entry['bytes']:
            raise ValueError('Model asset size mismatch: ' + entry['path'])
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
        if h.hexdigest() != entry['sha256']:
            raise ValueError('Model asset SHA256 mismatch: ' + entry['path'])
    return manifest


class NativeKokoro:
    def __init__(self, directory, speed=1., max_frames=200):
        import torch
        from .kokoro.model import KModel
        self.manifest = verify_assets(directory)
        if not torch.cuda.is_available():
            raise RuntimeError('Xiaoyou requires CUDA inference; CPU fallback is disabled')
        if not 20 <= int(max_frames) <= 400:
            raise ValueError('max_frames must be between 20 and 400')
        self.torch = torch
        self.lock = threading.Lock()
        self.max_frames = int(max_frames)
        self.set_speed(speed)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        directory = Path(directory)
        self.model = KModel(repo_id='hexgrad/Kokoro-82M', config=str(directory / 'config.json'),
                            model=str(directory / 'kokoro.pth')).cuda().float().eval()
        voice = torch.load(directory / self.manifest.get('voice_file', self.manifest.get('voice', 'A') + '.pt'), map_location='cpu', weights_only=True)
        if tuple(voice.shape) != (510, 1, 256) or not torch.isfinite(voice).all() or not torch.equal(voice, voice[:1].expand_as(voice)):
            raise ValueError('Invalid fixed voice vector')
        self.style = voice[0].cuda().float()
        if any(p.device.type != 'cuda' or p.dtype != torch.float32 for p in self.model.parameters()):
            raise RuntimeError('Model is not fully FP32 CUDA')
        self.frontend = Frontend(directory / 'config.json')

    def set_speed(self, speed):
        speed = float(speed)
        if not math.isfinite(speed) or not .3 <= speed <= 2:
            raise ValueError('speed must be finite and between 0.3 and 2')
        self.speed = speed

    def _encode(self, ids):
        t, m = self.torch, self.model
        tokens = t.tensor([[0, *ids, 0]], dtype=t.long, device='cuda')
        lengths = t.full((1,), tokens.shape[1], dtype=t.long, device='cuda')
        mask = t.zeros_like(tokens, dtype=t.bool)
        encoded = m.bert_encoder(m.bert(tokens, attention_mask=(~mask).int())).transpose(-1, -2)
        d = m.predictor.text_encoder(encoded, self.style[:, 128:], lengths, mask)
        x, _ = m.predictor.lstm(d)
        durations = (m.predictor.duration_proj(x).sigmoid().sum(-1) / self.speed).round().clamp(min=1).long()[0]
        return tokens, lengths, mask, d, durations

    def _decode(self, cache):
        t, m = self.torch, self.model
        tokens, lengths, mask, d, durations = cache
        frames = int(durations.sum().item())
        if frames > self.max_frames:
            raise ValueError('Duration budget must be checked before decoder allocation')
        indices = t.repeat_interleave(t.arange(tokens.shape[1], device='cuda'), durations)
        alignment = t.zeros((tokens.shape[1], indices.shape[0]), dtype=t.float32, device='cuda')
        alignment[indices, t.arange(indices.shape[0], device='cuda')] = 1
        alignment = alignment[None]
        prosody = d.transpose(-1, -2) @ alignment
        f0, energy = m.predictor.F0Ntrain(prosody, self.style[:, 128:])
        encoded = m.text_encoder(tokens, lengths, mask)
        t.manual_seed(self.manifest['seed'])
        t.cuda.manual_seed_all(self.manifest['seed'])
        audio = m.decoder(encoded @ alignment, f0, energy, self.style[:, :128]).squeeze()
        wave = audio.detach().float().cpu().numpy().reshape(-1)
        if not wave.size or not np.isfinite(wave).all() or wave.size != frames * 600:
            raise RuntimeError('Invalid native waveform or duration')
        return wave

    def _generate(self, text):
        language = 'mixed' if re.search(r'[\u3400-\u9fff]', text) else 'en'
        _, ids = self.frontend(text, language)
        if len(ids) <= 510:
            cache = self._encode(ids)
            if int(cache[-1].sum().item()) <= self.max_frames:
                yield text, self._decode(cache)
                return
            del cache
        for piece in split_once(text):
            yield from self._generate(piece)

    def generate(self, text):
        if not isinstance(text, str) or not text.strip() or len(text) > 16000:
            raise ValueError('text must contain 1 to 16000 characters')
        with self.lock, self.torch.inference_mode():
            yield from self._generate(text.strip())
