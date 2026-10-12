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

Language labels: zh/ja/ru/en/es/fr/th/de. Explicit language selects the reference voice. Auto detects Chinese, Japanese, Cyrillic and Thai scripts; Latin auto uses the English reference voice, so supply `language` for ES/FR/DE. Thai is best effort and is not an officially supported MOSS language. It is exposed for trial diagnostics, not declared quality-verified.

Shared config currently accepts `language`; speed=1 and the default language-specific speaker only. Language changes apply to future requests without reloading models. The fixed sampling parameters are text temperature1.5; audio temperature1.7, top_k25, top_p0.8, repetition penalty1; seed42. Fixed candidate count and FP32 sampling may change stochastic output relative to the old CPU float64 route.

Certified engine profiles allow at most512 prefill rows and768 cached rows. The runtime bounds generation by available KV capacity and at most375 frames. It reports an error if that bound is reached instead of declaring a truncated utterance successful. Long texts, normalization, formulas and mixed technical names still need broader evaluation.

## Existing evidence and limits

Native in-process eight short samples on Orin: frame fusion RTF0.642–0.675; with CUDA Graph RTF0.579–0.756, first PCM0.095–0.132s. These timings exclude model loading/capture, transport and playback. CUDA Graph on/off token and PCM16 WAV output matched for all eight samples. Independent engine sampling validation covered612 checks plus72 Graph on/off output comparisons.

Three independent model processes previously completed two concurrent rounds/six requests at RTF0.570–0.741, then our shared-host memory protection stopped the own container at519MiB available RAM. This is not a three-instance stability pass. The new framework adapter preserves independent models but shares ordinary process/runtime libraries; final MCP/ROS results are recorded separately for the exact commit.

Only short samples were tested. No full GSB/ASR/listening validation or long-duration stability certification. Thai retains long low-volume audio. First PCM is not recognized first-word latency. A successful build or trial submission does not establish contest admission.

## Verification

`tools/test_moss_submission.py` checks streaming resampler continuity, anti-aliasing, PCM validity and language validation. `tools/test_moss_mcp.py --output /path` tests eight samples through the actual MCP/ROS path; `--triple` starts three independent MCP model instances and sends five concurrent rounds. It checks received PCM byte counts against published completion metadata and writes WAV/results. That three-instance test is one framework process/container; it does not prove stability of three separately deployed framework containers.
