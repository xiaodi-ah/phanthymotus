"""Compare the bounded CUDA path with the training model, then save trial WAVs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'perception'))
import numpy as np
import torch
from plugins.xiaoyou_kokoro.adapter import XiaoyouKokoroAdapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--barrier', help='Controller creates this file after all models are ready')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    assert torch.cuda.is_available(), 'CUDA required'
    assert torch.cuda.device_count() == 1, 'Expose only the assigned physical GPU'
    assert 1 <= args.rounds <= 20
    # Leave capacity for the separate E training process on the server.
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(.08 if total > 32 * 1024**3 else .5)
    start = time.monotonic()
    adapter = XiaoyouKokoroAdapter(args.model_dir)
    runtime = adapter.runtime
    report = {'device': torch.cuda.get_device_name(), 'precision': 'FP32',
              'load_seconds': time.monotonic() - start, 'cases': []}
    with torch.inference_mode():
        _, ids = runtime.frontend('你好，我是小优。', 'mixed')
        cache = runtime._encode(ids)
        bounded = runtime._decode(cache)
        torch.manual_seed(runtime.manifest['seed'])
        torch.cuda.manual_seed_all(runtime.manifest['seed'])
        original, durations = runtime.model.forward_with_tokens(cache[0], runtime.style)
        original = original.cpu().numpy().reshape(-1)
        assert np.array_equal(durations.cpu().numpy(), cache[-1].cpu().numpy())
        np.testing.assert_allclose(bounded, original, rtol=1e-5, atol=1e-6)
        report['native_parity_max_abs'] = float(np.max(np.abs(bounded - original)))
    if args.barrier:
        (output / 'ready.json').write_text(json.dumps({'pid': os.getpid(), 'time': time.time()}))
        deadline = time.monotonic() + 180
        while not Path(args.barrier).exists():
            if time.monotonic() > deadline:
                raise TimeoutError('Concurrent inference barrier timed out')
            time.sleep(.1)
    report['inference_started_unix'] = time.time()
    cases = [('zh', '你好，我是小优，很高兴见到你。'),
                       ('en', 'Hello, I am Xiaoyou. It is nice to meet you.'),
                       ('mixed', '今天的 meeting 已经安排好了，请点击 Start 按钮开始。'),
                       ('long', '欢迎来到这里，我会帮你安排今天的工作。' * 8)]
    for case in range(args.rounds * len(cases)):
        name, text = cases[case % len(cases)]
        round_id = case // len(cases) + 1
        start = time.monotonic()
        first = None
        chunks = []
        for chunk in adapter.synthesize_stream(text):
            if first is None:
                first = time.monotonic() - start
            chunks.append(chunk)
        elapsed = time.monotonic() - start
        pcm = b''.join(chunks)
        samples = np.frombuffer(pcm, dtype='<i2')
        assert samples.size and np.max(np.abs(samples.astype(np.int32))) > 100
        assert all(len(c) == 3200 for c in chunks[:-1])
        filename = name + (f'-r{round_id}' if args.rounds > 1 else '') + '.wav'
        with wave.open(str(output / filename), 'wb') as stream:
            stream.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            stream.writeframes(pcm)
        seconds = len(pcm) / 32000
        report['cases'].append({'name': name, 'text': text, 'seconds': seconds, 'round': round_id,
                                'file': filename, 'pcm_sha256': hashlib.sha256(pcm).hexdigest(),
                                'peak_pcm_abs': int(np.max(np.abs(samples.astype(np.int32)))),
                                'generation_seconds': elapsed, 'first_pcm_seconds': first,
                                'rtf': elapsed / seconds, 'chunks': len(chunks)})
    report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    report['max_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['inference_finished_unix'] = time.time()
    report['quality_listening_complete'] = False
    (output / 'checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
