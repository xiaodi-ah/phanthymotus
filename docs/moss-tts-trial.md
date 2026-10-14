# MOSS multilingual TTS trial for JetPack 6.1

This branch runs MOSS-TTS-Nano locally on Jetson Orin. It uses our frame-fused TensorRT engine with FP16 model computation, explicit FP32 sampling and CUDA Graph. It does not call the remote A100 API or use a colleague's optimization commit or engines.

The upstream fork baseline is f11074901ebfff3e542579a30bd3c8efc36db0fc. Only TTS is enabled in this trial configuration. The repository contains no model weights, engine plans or dependency wheels; `perception/moss_assets.json` pins the internal JuiceFS URLs, sizes and SHA256 values. Builds fail on missing or corrupt assets.

## Build and submit

```bash
docker build --network=host -f perception/Dockerfile.jetson -t local/tts-moss-submit:trial .
docker run --rm --runtime nvidia --network host \
  -e TTS_PLUGIN=moss_tts -e MCP_PORT=15720 -e WS_PORT=15721 \
  local/tts-moss-submit:trial
```

Submit this branch's full 40-character commit and `https://github.com/xiaodi-ah/phanthymotus.git`; set `TTS_PLUGIN="moss_tts"` and `JP_VERSION="6"`. These engines were built on Orin sm_87 / JetPack6.1 / TensorRT10.4.0. JP5 is not supported by this trial. The Dockerfile defaults to the same versioned JP6.1 perception base used for our device tests and installs our previously verified ARM64 wheels without replacing NVIDIA torch.

## Contract

The official MCP tool remains `tts`, with actions `start`, `stop`, `speak`, `info`, `config`. Each started `instance_id` owns independent TensorRT engines, KV and codec state. Model construction/capture is serialized; inference across instances may overlap. Start performs a real short synthesis before reporting `running`.

Example arguments:

```json
{"action":"start","instance_id":"tts0","input_topic":"/trial/tts0/text"}
{"action":"speak","instance_id":"tts0","text":"Hello, welcome.","language":"en"}
```

ROS input is a `std_msgs/String` JSON object with `text` and optional `language`/`lang`. Output is `audio_msgs/AudioChunk`, format `audio/pcm-16k`, on `<input_topic>/tts`. The framework and evaluator assemble that PCM into WAV; the topic chunks do not contain repeated WAV headers. Native 48kHz stereo is converted to 16kHz mono PCM16 with a continuous stateful anti-alias filter. The producer queue is bounded to eight native frames and publishing stays at most about one second ahead of the playback clock.

Language labels: zh/ja/ru/en/es/fr/th/de. Explicit language selects the reference voice. Chinese uses Zhiming. Requests stream sentence by sentence, with each sentence limited to75 text tokens; each chunk resets codec state and uses seed42. Decimal points, version/file/domain names and common English abbreviations are preserved. Oversized sentences prefer whitespace/clause boundaries and fall back to character boundaries. Auto detects Chinese, Japanese, Cyrillic and Thai scripts; Latin auto uses the English reference voice, so supply `language` for ES/FR/DE. Thai is best effort and is not an officially supported MOSS language. It is exposed for trial diagnostics, not declared quality-verified.

Shared config currently accepts `language`; speed=1 and the default language-specific speaker only. Language changes apply to future requests without reloading models. The fixed sampling parameters are text temperature1.5; audio temperature1.7, top_k25, top_p0.8, repetition penalty1; seed42. Fixed candidate count and FP32 sampling may change stochastic output relative to the old CPU float64 route.

Certified engine profiles allow at most512 prefill rows and768 cached rows. The runtime bounds generation by available KV capacity and at most375 frames. It reports an error if that bound is reached instead of declaring a truncated utterance successful. The frame/KV bound applies to each chunk. Audio is streamed and concatenated without extra silence or crossfades; paragraph output can exceed30 seconds. Long texts, normalization, formulas and mixed technical names still need broader evaluation.

## Existing evidence and limits

Native in-process eight short samples on Orin: frame fusion RTF0.642–0.675; with CUDA Graph RTF0.579–0.756, first PCM0.095–0.132s. These timings exclude model loading/capture, transport and playback. CUDA Graph on/off token and PCM16 WAV output matched for all eight samples. Independent engine sampling validation covered612 checks plus72 Graph on/off output comparisons.

Three independent model processes previously completed two concurrent rounds/six requests at RTF0.570–0.741, then our shared-host memory protection stopped the own container at519MiB available RAM. This is not a three-instance stability pass. The new framework adapter preserves independent models but shares ordinary process/runtime libraries; final MCP/ROS results are recorded separately for the exact commit.

Only short samples were tested. No full GSB/ASR/listening validation or long-duration stability certification. Thai retains long low-volume audio. First PCM is not recognized first-word latency. A successful build or trial submission does not establish contest admission.

## Verification

`tools/test_moss_submission.py` checks streaming resampler continuity, anti-aliasing, PCM validity and language validation. `tools/test_moss_mcp.py --output /path` tests eight samples through the actual MCP/ROS path; `--triple` starts three independent MCP model instances and sends five concurrent rounds. It checks received PCM byte counts against published completion metadata and writes WAV/results. Use `--samples-file tools/moss_long_text_samples.json --languages zh,en` for the committed long-text fixture; combine with `--triple --rounds 5` for concurrent paragraphs. Request completion timeout is180 seconds for long fixtures. That three-instance test is one framework process/container; it does not prove stability of three separately deployed framework containers.


## Local three-model memory optimization (2026-10-13)

The optimization branch preserves separate engines, execution contexts, KV and codec state per model. Prefill uses one output bank; the recurrent decode and codec retain two banks. Initial GPU input buffers are discarded after recurrence moves to device outputs. The codec resets its existing device state on its own CUDA stream (zeros, positions=-1), avoiding repeated large CPU cache allocations/uploads. Weights, sampling and precision are unchanged.

On the same Orin, seven short language samples (Thai excluded) matched legacy PCM bitwise. The actual MCP/ROS path completed 30 concurrent rounds /90 requests across seven languages; all90 received WAV PCM matched their baseline sample bitwise. No audio-byte loss or memory protection stop. This is three independent models in one framework process/container, not three independently deployed containers.

Single-model Torch buffer allocation after seven cases:178.6→126.4MiB; reserved274→210MiB. These figures exclude TensorRT-owned engine weights/context memory. Shared-host late-test available RAM was about942–960MiB at5-second samples. Single MCP/ROS RTF0.785–0.887; three-model RTF0.883–1.138, median1.020, with48/90 at or above1. First received PCM0.358–0.653s; recognized first-word latency was not checked.

The memory improvements do not establish full contest admission: concurrent RTF remains above1 in some cases; long inputs, hours of stability, three containers, listening/GSB remain unverified. Existing submission YAML remains on37b4fb2e13773c461b53e778fe65c001cbbe12c5. Local artifacts: outputs/memory-ab-20261013-073746/ and outputs/memory-stability-20261013-073743/ in the project workspace. Experimental image:local/tts-moss-submit:mem-opt-20261013-e.


## Concurrent RTF optimization with codec CUDA Graph (2026-10-13)

Profiling three native requests found142 codec calls with5.146s summed wall time and1.208s GPU-event time around TensorRT enqueue; events omit some queue delays and profiling adds overhead, so the difference is not a pure CPU-time measurement. Serializing all GPU stages made native RTF worse (1.50–1.89 in the profiled control) and was rejected. Codec IO has fixed profile shapes, no host/shape IO, and no auxiliary streams.

The new codec adapter binds fixed input/output banks once, captures TensorRT plus recurrent-state device copies in one CUDA Graph, and asynchronously copies audio/length into pinned CPU buffers on the same private stream. One explicit stream synchronization precedes all CPU reads. Recurrent values return the updated input bank so the existing request reset stays valid. Every model retains its own engine/context/graph/buffers. Existing `cuda_graph=true` enables both frame and codec graphs; false uses the earlier codec adapter. Weights, precision and sampling are unchanged.

This follows [NVIDIA CUDA Graph guidance](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/performance/optimization.html) and [PyTorch nonblocking-copy guidance](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html). No new dependencies or engines are required.

Seven native A/B PCM pairs and all90 concurrent received WAV PCM matched the79ae3ba baseline bitwise. Actual MCP/ROS three-model seven-language rotation completed30 rounds /90 requests: RTF0.754–0.908 (median0.822), all90 below1; first PCM0.257–0.534s. The previous separate run had RTF0.883–1.138, with48/90 at or above1; these full-chain comparisons were not measured in the same time window. Single MCP/ROS RTF0.730–0.874. Native optimized single-model RTF0.369–0.495 in the same-run A/B.

Late-test shared-host available RAM stayed around983–1005MiB at5-second samples; no memory protection stop, no audio-byte loss. Artifacts: outputs/codec-ab-20261013-085608/, outputs/codec-stability-20261013-085605/, outputs/concurrent-profile-20261013-084029/ in the project workspace. Experimental image:local/tts-moss-submit:codec-graph-20261013-a. Thai, long inputs, hours of stability and GSB were not checked; first PCM is not recognized first-word latency. Submission YAML remains on37b4fb2e13773c461b53e778fe65c001cbbe12c5 pending platform feedback.
