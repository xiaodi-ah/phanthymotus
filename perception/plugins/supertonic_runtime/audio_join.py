"""Remove inaudible block padding without altering speech or in-block pauses."""
import numpy as np

INTER_BLOCK_PAUSE_SECONDS = 0.08


def trim_zero_padding(pcm, sample_rate=16000):
    if not pcm or len(pcm) % 2:
        raise ValueError('Expected nonempty PCM16 audio')
    samples = np.frombuffer(pcm, dtype='<i2')
    active = np.flatnonzero(samples)
    if not active.size:
        return pcm
    guard = round(sample_rate * 0.02)
    minimum = round(sample_rate * 0.08)
    start, stop = 0, len(samples)
    if active[0] >= minimum:
        start = max(0, int(active[0]) - guard)
    if len(samples) - active[-1] - 1 >= minimum:
        stop = min(len(samples), int(active[-1]) + 1 + guard)
    return pcm[start * 2:stop * 2]
