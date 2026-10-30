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

Submit this branch's full 40-character commit and `https://github.com/xiaodi-ah/phanthymotus.git`; set submission YAML `TTS_PLUGIN="tts"` and `JP_VERSION="6"`. The evaluator uses `TTS_PLUGIN` as the MCP tool name. Reference containers inherit the Dockerfile default `TTS_PLUGIN=moss_tts` to select the MOSS backend; the explicit local run override above applies to that backend selector. These engines were built on Orin sm_87 / JetPack6.1 / TensorRT10.4.0. JP5 is not supported by this trial. The Dockerfile defaults to the same versioned JP6.1 perception base used for our device tests and installs our previously verified ARM64 wheels without replacing NVIDIA torch.

## Contract

The official MCP tool remains `tts`, with actions `start`, `stop`, `speak`, `info`, `config`. Each started `instance_id` owns independent TensorRT engines, KV and codec state. Model construction/capture is serialized; inference across instances may overlap. Start performs a real short synthesis before reporting `running`. Stop halts the instance worker and leaves its independent model loaded; a later start with the same instance ID reuses those engines and CUDA Graphs. A stopped instance can change its input topic without loading another model. Container exit releases the retained models. Unkeyed `speak` reuses the only started instance; if multiple nondefault instances exist, specify `instance_id` or `input_topic` instead of implicitly loading another model.

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

`tools/test_moss_lifecycle.py` checks repeated per-case start/speak/stop, separate instance ownership, stopped-topic changes and routing without ROS or GPU dependencies. `tools/test_moss_submission.py` checks streaming resampler continuity, anti-aliasing, PCM validity and language validation. `tools/test_moss_mcp.py --output /path` tests eight samples through the actual MCP/ROS path; `--triple` starts three independent MCP model instances and sends five concurrent rounds. It checks received PCM byte counts against published completion metadata and writes WAV/results. Use `--samples-file tools/moss_long_text_samples.json --languages zh,en` for the committed long-text fixture; combine with `--triple --rounds 5` for concurrent paragraphs. Request completion timeout is180 seconds for long fixtures. Add `--restart-each-case --rounds 3 --languages zh,en` to exercise repeated per-case start/speak/stop and save framework RSS/high-water memory. That three-instance test is one framework process/container; it does not prove stability of three separately deployed framework containers.


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


## Per-case model reuse trial (2026-10-15)

The platform confirmed OOM on the ef6569e evaluation. Its evaluator starts and stops the same instance after each case. This change retains that instance's independent model across stop/start to avoid rebuilding engines and CUDA Graphs for every case. Other instance IDs still own separate models. Weights, sampling, voices, sentence generation and publishing pace are unchanged.

Nine local regression checks cover model lifetime/routing and the three-container test barrier. On the shared Orin, two separate containers loaded models and completed short warmup synthesis. The controller did not launch the third because measured memory headroom would fall below its safety reserve; it released no concurrent requests. Both own containers were cleaned up without OOM, and the other four services kept the same PIDs and restart counts. Three-container inference, repeated per-case real-device testing and whether this fixes platform OOM remain unverified. This commit is prepared for a platform trial while the resource watcher waits for a larger idle window.


## Delivery optimization trial (2026-10-16)

MOSS now waits for an audio subscriber to remain matched for100ms before publishing the first audio; generation continues on its bounded producer queue during this wait. No stable subscriber within3s or stop cancels publication. First-packet prebuffer is one100ms packet instead of three. Subsequent100ms PCM packets are sent at70ms intervals, retaining the existing lead offset. This changes delivery speed, not the waveform, model computation, sampling or playback sample rate. The receiver may buffer more audio before playback finishes. Set `MOSS_TTS_DELIVERY=legacy` in the inference container to use the previous delivery behavior; default is `optimized`. Other adapters retain their existing behavior.

On the shared Orin, one container/model ran45 real MCP/ROS requests with per-case stop/start and subscription recreation. Same-session normal Chinese/English A/B (12 each): receive RTF0.697–0.743 →0.470–0.551, median0.727 →0.512 (29.5% reduction); first PCM median0.196 →0.186s, maximum0.245 →0.207s. With the subscriber created200ms after speak, legacy lost audio bytes on5/6 requests; optimized received every byte on6/6, first PCM0.318–0.321s. This reproduces a discovery-related loss mechanism locally; it does not prove every platform outlier has the same cause.

Optimized seven-language short tests: first PCM0.161–0.202s, RTF0.478–0.574. Two Chinese/English long texts: first PCM0.180–0.204s, RTF0.684–0.686. All27 optimized requests received full PCM. Complete same-session comparison audio matched bitwise; all seven short and two long WAV PCM also matched the existing ef6569e artifacts bitwise. Host available RAM minimum2476MiB, no OOM/protection stop, other four services kept the same IDs/PIDs/restart counts. Own test container was removed and the resource watcher resumed.

Evidence in project workspace: `outputs/delivery-ab-20261016-150754/verified-summary.json`, WAV/results/framework logs. Tests: `python3 -B tools/test_tts_delivery.py` (7 checks), existing lifecycle9 checks, sentence/resampler/language checks and `git diff --check`. Benchmark options `--recreate-subscriber --subscriber-delay-ms 200` reproduce late discovery; `--record-audio-loss` is diagnostic-only and records incomplete PCM instead of aborting on its first occurrence. Default verification still fails on lost PCM.

This was an ef6569e image with read-only source overlays, not a rebuilt exact-commit image. Three-container optimized validation, platform tail latency and listening/GSB remain unverified. Use this revision for the next platform trial. Its inference/delivery source matches the tested overlays; the submission YAML uses MCP tool name tts and JP_VERSION=6.


## Quality trial (2026-10-20)

This revision changes the Japanese reference from Soyo to Saki and the explicit German reference from Nathan to Adam. Both prompt-code arrays already exist in the pinned browser manifest. Chinese remains Zhiming; the remaining reference mapping is unchanged. No assets, weights, engine plans, dependencies or engine profiles change. Builds and inference reuse the existing Orin / TRT10.4.0 plans; no engine compilation is required for these changes.

`MOSS_TTS_TEXT_FRONTEND=numbers` is now the default. It expands explicit quantities/units in English, German and Japanese before the existing sentence/token-budget splitter. Bare numbers, dates, versions, model identifiers, paths, URLs/email, ambiguous grouping/decimal notation and unsupported languages retain their original text. German case-dependent singular forms and a general date/number grammar are outside this trial. Set `off` to disable, or experimental `numbers_terms` to also rewrite selected acronyms/names. The latter is not the submission default because German term rewrites regressed in screening.

The released revision uses `MOSS_TTS_STOP_MODE=greedy` **for English requests only**; its other resolved languages keep sampled stopping. The newer candidate below adds guarded sampling. Set `sample` to restore v1 stopping for every language. The greedy mode chooses the more probable continue/EOS token while preserving all16 audio-channel sampling draws. The existing fused engine already exports its two text scores. Supplying zero for the text uniform forces its audio path to use the assistant-slot embedding; the host then decides whether to emit that frame from the two scores. All17 random values are still consumed per frame. The language-dependent selection resets under the model lock on every request, so an English request cannot change the next language's policy. This needs no additional TensorRT execution; overriding only the sampled stop flag would accept audio conditioned on EOS.

A100 screening used three seeds42/7/2026: four Japanese targets favored Saki over Soyo (normalized ASR CER25.00% vs42.08%); four German targets favored Adam over Nathan and the tested German reference (WER10.81% vs21.17% /13.06%). Two quantity targets per language favored number expansion, especially English/German. These are small-corpus ASR diagnostics, not native listening or leaderboard GSB scores; reference and frontend screenings were separate, so their combined quality still needs checking. If the evaluator omits explicit language, Latin auto still selects English/Adam and English processing; this revision does not add Latin language identification.

One English target stopped after1.28s in both FP32 and FP16 under seed2026 despite continuation probability above99%. In18 paired A100 stop-policy tests, greedy continuation restored the two truncated outputs, preserved every common-prefix audio token, and caused no frame-cap failure. This establishes one reproducible stopping mechanism, not the exact cause of the saved Orin seed42 omission. Certified prefill/KV limits and the frame cap remain enforced; an extended output can still reach a cap and fail explicitly.

Checks: `tools/test_tts_text_normalization.py`, `tools/test_tts_frontend_adapter.py`, `tools/test_tts_stop_policy.py`, existing submission/lifecycle/delivery tests and `git diff --check`. Detailed local audio/ASR evidence is in project-workspace `outputs/quality-v2-20261019/`; the v1 score table is `outputs/evaluation-0fdcc828-v1-20261019/v1-scorecard.html` / `.csv`. Shared Orin resource-gated validation is queued. This is a platform trial: combined Orin quality/performance, three-container behavior and subjective GSB improvement are not yet verified. The previous `v1` tag is retained for rollback.


## Real-case quality candidate (2026-10-22, not published)

Oversized sentences now prefer clause boundaries with at least20 tokens before the cut and12 after it. This avoids independent two/four-token tails while preserving the75-token ceiling and every non-whitespace input character. Existing short sentences are unchanged. The supplied English case210 now splits its long sentences58+21 and52+25 tokens instead of75+4 and75+2; Russian boundary context also improves. This does not establish a subjective quality gain.

The separate Nano v3 frame graph uses text temperature1.0, audio temperature0.8/top_p0.95/top_k25 and repetition penalty1.2. Its additional static `repetition_seen_mask` input applies a signed penalty to prior audio tokens in each channel before TopK. The adapter builds this mask from the official runtime's per-chunk history. New engines require exact `frame_sampling.json` metadata; released two-input engines keep their original1.5/1.7/0.8/1.0 settings. Editing Python defaults cannot change a compiled graph's sampling constants.

`MOSS_TTS_STOP_MODE=guarded` is the new candidate default: English keeps greedy EOS, while other languages reject sampled EOS only when continue probability is at least0.99. A rejected EOS reruns the stateless fused frame with the continuation embedding and the same16 audio random draws; it never reuses EOS-conditioned audio or advances the RNG again. Set `greedy` to restore the released English-only behavior. This probability threshold is a heuristic that needs broader language testing. The adapter records probabilities, original random draws and overrides, plus per-chunk frame budgets. The earlier all-language greedy experiment hit the Russian frame cap and was rejected.

Five distinct deployed references covering the seven tested languages have conservative denoised candidates, with original files retained. Each original source re-encodes exactly to the released reference codes. Filter padding is flushed to preserve the exact sample count and alignment; candidates are encoded offline with the pinned official FP32 ONNX codec. No reference denoising runs on Jetson. Reference denoising is a quality hypothesis, not an established fix for generated hiss.

Workspace tools `tts_quality_v3.py`, `export_fused_frame.py --nano` and `tts_quality_v3_orin.py` compare the four supplied cases, original/new chunks, official Nano sampling, stopping and cleaned references. Local A100 PyTorch FP32 controls are distinct from target Orin/TensorRT validation. The existing submission YAML and released asset manifest remain unchanged until target validation and listening review.

On2026-10-23 the existing-engine Orin controls completed26 requests: baseline4, revised chunks11, cleaned references11. Only the old English baseline hit its frame cap. Old Chinese cases7/11 and Russian31 reproduced the supplied PCM exactly; English reproduced the supplied96-second prefix and generated another80ms before raising the cap error. Chinese early-stop continue probabilities were0.995318 and0.999514. Revised chunks completed English in64.64s and reduced Russian auxiliary ASR WER from24.11% to14.29%; Chinese chunks were unchanged. Cleaned references introduced a new omission in Chinese11, so the released references are retained.

Guarded EOS then completed19 Orin requests without cap failures or OOM: four supplied cases, seven short language cases and eight matched Chinese sentence controls at seeds7/2026. Only Chinese7/11 triggered an override; the old affected sentence PCM is an exact prefix of the repaired output. ASR recovered both missing endings, with full-case CER9.72%→5.56% and5.97%→1.49%. The other comparisons were PCM-identical; both extra seeds read the Chinese sentences completely. The tested1280MiB container left at least1642MiB host memory and other containers were unchanged. Direct-adapter first PCM was0.055–0.104s and RTF0.388–0.477. These measurements exclude ROS/MCP and do not establish subjective GSB improvement or new three-container qualification. Nano sampling remains a separate, unbuilt target engine candidate; the selected fixes reuse the released engines and references.

## Single-model text frontend candidate (2026-10-23)

The deployment keeps one MOSS model per instance. The text processing design references [PR341](https://github.com/4paradigm/phanthymotus/pull/341), inspected at `f14e38d3d2b596809eaf8d2368323448c1b33562`; it adds no VITS backend, phoneme model, FST/G2P dependency or new TensorRT engine.

The existing `numbers_terms` comparison mode screens Chinese Qwen/HAMi readings, IP/ID spelling, digit-by-digit explicitly labeled IDs, and Chinese quantities. English Qwen/product names and acronyms retain their original text, apart from the explicit ID label; acronym and product-word splitting was rejected after it regressed long-case ASR. URLs, emails, quoted code, paths and longer identifiers are protected; unsupported dates/formulas retain their input rather than guessing. Normalization runs before the existing75-token splitting and certification checks. Bare numbers are not globally treated as identifiers. The default `numbers` mode remains unchanged. New Chinese normalization is enabled only in the term candidate; other languages keep their existing processing. These text rewrites need audio comparisons because VITS phone-level readings do not establish MOSS pronunciation quality.

## Sentence ending, output noise and lexicon (2026-10-23, unpublished)

The adapter now retains12ms/384bytes of16kHz PCM per chunk and fades that final tail to zero after successful EOS. `finish()` emits every retained sample; total duration and all earlier samples are unchanged. Frame-cap failures still raise before finalization. This addresses an abrupt nonzero endpoint, not sustained hiss. No reference, token, codec engine or gain changes are involved. Ten saved requests/25 chunks reproduce the separate fade-only controls exactly. Unit checks cover arbitrary chunking, empty/short input, finalization, invalid PCM and the real threaded adapter; no new dependency is needed.

Workspace `outputs/output-noise-ab-20261023/diagnostics.html` compares original/fade-only/at-most6dB `afftdn` controls, removed audio, native48k and official FP32 same-token final-chunk decoding. The denoising control is offline and is not enabled in production: quiet-interval reduction is small for most recordings, and these statistics cannot identify all generated noise. No samples are trimmed. Auxiliary ASR includes25 exact-PCM cached originals and35 newly transcribed controls without a target prompt; the tested processing introduces no additional edit errors in these samples. This does not establish subjective noise improvement.

The user-facing `comparison.html` is now a two-step Chinese listening guide: one matched sentence for background noise and one for the ending, with only two players visible at a time. Its11.04/13.12-second clips come from the exact corresponding final text chunks of old/new16k recordings, without further processing. The page shows the reference text, listening focus, noise comparison and speech-damage questions, optional last-three-second playback and copyable feedback. It keeps answers while switching groups; refresh clears them, as disclosed on the page. Full recordings and technical records are optional links. Regenerate with workspace `tools/tts_listening_review.py`; `tools/test_tts_listening_review.js` checks the generated navigation/player/feedback logic without browser dependencies. This changes presentation only and records no listening verdict automatically.

Same-token codec differences grow with chunk duration. Chinese7/11 final chunks have overall SNR19.0/13.1dB relative to official FP32, versus47.0dB for the short Chinese control. Their early1-second windows are47.1/50.1dB, falling to7.8/6.0dB around the end. These are reconstruction comparisons, not recording SNR/noise scores. A numerical codec cause needs target precision controls and listening verification; a RoPE FP32/full FP32 TensorRT build probe is separate from the submission engines.

The target precision controls subsequently completed20 same-token replays. Keeping RoPE float calculations inFP32 improved the two Chinese long chunks to50.4/51.0dB, while fullFP32 reached115.6/114.9dB relative to official codec output. All ten fullFP32 controls reached113.5–120.7dB. Build optimization level also differs from the old engine, so this does not uniquely isolate one operator as the only cause. The failed1280MiB build attempts affected only owned containers; a1536MiB isolated-builder run succeeded with at least1238MiB host memory and unchanged other container PIDs. Engine construction needs more memory than runtime.

Full single-model Orin inference with the new fullFP32 codec and12ms fade then completed ten requests (three long Chinese/English cases plus seven language controls). Every generated token matched the original request, all sample counts were preserved and every chunk ended at zero. Direct adapter RTF was0.372–0.431 and first PCM0.047–0.094s, excluding ROS/MCP. The1280MiB runtime container left at least1675MiB host memory; no OOM and no other container changes occurred. Evidence: workspace `outputs/noise-chain-20261023-180617/` and `outputs/codec-precision-20261023-180018/`. The new plans are local candidates; the released download manifest and submission YAML still use the old engine. Three-container performance and subjective quality remain unchecked.

Fresh ASR on all25 fullFP32/fade chunks did not establish a content improvement. Chinese7 raw edit count changed9→11 (EngineX spelling/number transcription), Chinese11 stayed2, English210 changed8→9 (It/You), and seven language controls retained their edit totals. The underlying TTS tokens are identical; small transcript changes do not establish better or worse listening quality. Keep the precision candidate for A/B listening rather than automatically publishing it. The earlier no-additional-edit result applies to the separate fade/afftdn controls only.

On2026-10-25 the user reviewed both guided Chinese excerpts and reported less noise with no perceived speech damage in B, for both background noise and the sentence ending. Retain fullFP32 codec plus12ms fade as the regression candidate. This feedback covers those two excerpts; it does not establish whole-recording/multilingual listening quality or three-container qualification. The workspace record `outputs/output-noise-ab-20261023/listening-feedback-20261025.json` binds the verdicts to the exact audio/codec hashes. `outputs/noise-candidate-B-20261025/candidate.json` pins the tested runtime and asset hashes for subsequent validation; unchanged assets are reused through local links. The submission assets, YAML and default frontend are not promoted by this listening result.

## Other languages and previous truncation checks (2026-10-25)

Fresh target testing is blocked by SSH timeouts from both sandbox and host networking. The prepared15-case plan covers six other languages with daily/long texts plus the three previously failing Chinese/English cases. The current quantity-only frontend preserves all input text, with at most58 tokens per chunk. `tools/tts_b_regression_probe.py` records chunk budgets, EOS/limit reasons, guarded overrides, actual PCM and failed partials, and continues to the following case after a request error. Its simulator check covers those logging/error paths, not hardware inference. The owned-container controller can run it with `--noise-chain --b-regression`, retaining the existing resource gate and host reserve. It has not been uploaded or executed on the target.

Meanwhile12 saved-token controls (daily and long EN/JA/DE/FR/ES/RU) were replayed through the official CPU FP32 codec plus12ms fade, retaining exact durations. Their TTS tokens came from the2026-10-22 Orin capture; no new TTS generation or TensorRT performance measurement occurred. These controls assess the decoder change and retain any omissions already present in the tokens. Twenty-four full-recording ASRs used matched settings and no target prompt. Long-text raw error rates were EN WER0%, DE WER5.3%, ES WER5.0%, JA CER37.2%, FR WER32.3%, RU WER20.8%. All six end phrases matched, but Japanese numbers/extra syllables, French repetitions/missing middle information and Russian omitted clauses/numbers remain ASR concerns. Character/word metrics and spelling/number variants are not cross-language listening scores. No subjective rating is inferred.

The existing actual B recordings of cases7/11/210 were also re-audited:47.04/42.96/64.64seconds, every chunk below its recorded budget, no input text lost during splitting, and the previously missing Chinese phrases and English ending present in ASR. This establishes repairs of those three cases, not a guarantee against all long-text omissions. Workspace `outputs/b-multilingual-20261025/long-regression-audit.json` contains the exact evidence.

`outputs/output-noise-ab-20261023/multilingual.html` offers one A/B pair at a time, six languages and three material types. Its default B short clips are actual2026-10-23 Orin recordings; daily/long-ending controls are clearly labeled offline replays. Full recordings, ASR and the truncation evidence are optional details. Source labels, language/type switching, retained answers, playback exclusivity, clipboard fallback, links,16k PCM format and matched sample counts were checked. The new target regression remains pending; neither this page nor replay results change the released manifest, submission YAML or pronunciation rules.

`perception/plugins/moss_tts_runtime/pronunciation_lexicon.json` is the editable source of truth:79 entries with language, spelling/aliases, reading, status, source and evidence. The17 `trial` readings preserve the previous opt-in rules;58 `pending` entries retain original spelling, and4 `rejected` English acronym expansions remain disabled after observed regressions. A trial is not a native-speaker approval. `numbers_terms` alone enables trial readings; the default `numbers` mode remains unchanged. Qwen uses the existing version-only grammar. The loader validates entries and alias conflicts, matches longest aliases in one pass, and shares URL/email/code/path protections with normalization. Docker copies the JSON with the plugin directory; the diagnostic bundle also includes JSON assets.

To extend the library, add a pending entry with its language/source, choose a native reading, compare fixed-text MOSS audio against the original spelling, record evidence, then change status to trial only when it merits a new opt-in comparison. Keep rejected entries for provenance. Re-run `tools/test_tts_lexicon.py`, `tools/test_tts_text_normalization.py` and `tools/test_tts_frontend_adapter.py`. Text/DSP changes require no ONNX export or TensorRT rebuild; precision changes to the codec do require a new engine and renewed performance validation. Nothing in this section has been committed or submitted.

## Multilingual content controls (2026-10-25, local)

The accepted B codec/fade and production voice assignments remain the baseline. Jetson SSH still times out from both sandbox and approved host networking. No target container was started, stopped or inspected during this work. The unpublished runtime now fixes two text issues:

- Sentence abbreviation checks use whole Unicode words (NFC only for boundary lookup). The ASCII suffix of `plaît`/`réunion` no longer prevents a real sentence break. Original text characters are preserved; English initials, abbreviations and bounded chunks remain covered.
- Latin auto routing uses distinct lexical cues and a two-cue/two-point margin for EN/DE/FR/ES. Explicit language and existing non-Latin script routing take priority. Names, short ambiguous input and ties retain English fallback. All 21 existing explicit-language multilingual inputs resolve correctly without language hints; this is a bounded regression corpus, not general language-ID accuracy.

Local tools `tools/tts_content_probe.py` (outside this production repository) completed 42 syntheses and 47 unprompted per-piece ASR checks: 19 stopping/splitting controls, 13 number/reference candidates and 10 Nano/number holdouts. They use the existing official MOSS PyTorch FP32 model on GPU5 and official CPU FP32 codec with B's sentence fade. These are neither new Orin TensorRT runs nor performance/GSB evidence. Audio, tokens, EOS traces, parameters and completion markers remain in `outputs/content-{repair,candidates,followup}-20261025` in the project root.

Eighteen additional teacher-forced controls replay the saved Orin audio-token histories under local PyTorch FP32 and FP16. Four former 31-frame omissions still have continuation probabilities around .9917–.9953 at their old termination, below the fixed random uniform .995464. This supports the rare-EOS guard; these probabilities are local diagnostics, not recovered original TensorRT scores. Japanese proper-name/number errors and the short final phrase do not share that high-confidence premature-EOS pattern, so the same guard cannot solve all content errors.

French/Russian `numbers_terms` candidates spell only explicit temperatures and percentages through 100. Russian fractional quantities and larger values/other units remain untouched. Cardinal forms were checked against Unicode CLDR [French rules](https://raw.githubusercontent.com/unicode-org/cldr/main/common/rbnf/fr.xml) and [Russian rules](https://raw.githubusercontent.com/unicode-org/cldr/main/common/rbnf/ru.xml). Default `numbers` mode remains unchanged for these languages. French Bella controls recover 23°C/85% and a separate 18°C/72% sample; the negative-temperature sample recovers the minus sign but ASR still questions “Celsius.” This is a useful opt-in candidate, not blanket pronunciation approval. Raw WER against spelled-out targets is inflated when ASR writes numerals; compare actual values/content as well.

Russian number expansion alone recovers temperature but not battery reliably. Nano sampling recovers both in this one case, while Japanese and French controls regress (repetitions or changed numbers). Do not globally enable Nano or replace existing references: the tested Soyo/Mei/Umiri Japanese and two native French references provide no stable improvement. Nano uses different compiled frame sampling; changing Python settings against the released static frame engine would not apply it correctly.

The existing port's `content.html` offers three French number pairs with only two players, explicit local-run provenance and optional ASR. No ZIP was created. The pending target bundle now contains 20 requests (15 original regressions, three Latin-auto inputs and two opt-in number cases), pinned runtime/input hashes and per-case resolved language/frontend/EOS evidence. It is prepared locally, not transferred or queued on Jetson. At the time of these controls, public commit/submission and asset manifest were still 2471bb39/old codec.

## B leaderboard trial (2026-10-25)

This version enables the user-accepted full FP32 codec, 12ms sentence fade, guarded rare-EOS handling, balanced long-sentence splitting and Unicode sentence-boundary fixes. It also adds conservative Latin auto routing. The immutable asset bundle is `moss-tts-trt-jp61-v2-codecfp32-20261025`; all 11 files were fetched through the contest HTTP endpoint and verified against their complete SHA256. The codec is 45,858,660 bytes, SHA256 `9bbfb3d93f99f589e7f6f896db9959bf8b137a8c1efad07240f18f47efa7867b`. The original v2 bundle is preserved. Unchanged files reuse immutable files with hard links; the new codec has its own copy.

The frame engine remains v2 with the existing sampling parameters and voice assignments. `numbers` remains the default frontend, so opt-in lexicon/French/Russian quantity candidates are not activated by this submission. Nano parameter switching is not enabled. The target quality test now checks the guarded submission defaults.

Validation includes 45 frontend/stopping/PCM/delivery tests, 9 lifecycle tests, the existing streaming/chunking and normalization checks, and Git file-size checks. The full FP32 codec and fade already completed 10 single-model Orin requests on October 23; the later text-routing/Unicode changes have local coverage. No final-commit image or new three-container test has run because target SSH remains unavailable. Treat this version as a leaderboard trial: memory, RTF and external quality scores still require the platform run. User listening for languages other than Chinese/English covers noise and audible distortion only; it does not establish content or pronunciation accuracy.


### 2026-10-26 quality candidate

The default `guarded` policy uses the more probable continue/EOS decision for English and Japanese. German retains sampled EOS and the 0.99 rare-EOS guard, but rejects continuation when EOS is more probable. Using fully greedy German stopping caused a different long block to hit its frame limit and was discarded. The explicit legacy `greedy` setting keeps its English-only behavior, and `sample` restores sampled EOS for every language. Audio-channel random draws and the continuation embedding are preserved.

French uses an independently validated local frame engine with text/audio temperatures 1.0/0.8, audio top-p 0.95, top-k 25 and repetition penalty 1.2. The other languages retain the exact v2 local engine and original sampling. A single MOSS instance shares the original global prefill/decode and full FP32 codec; switching languages releases the old local frame engine before loading the selected one under the request lock. Only one local frame engine is resident. The immutable asset bundle adds `frame_fr.plan` and `frame_fr_sampling.json`; older bundles without those files preserve their previous sampling. This is a quality repair, not evidence that all pronunciation or long-text issues are solved.

The pronunciation library has 83 entries. `approved` token/phrase readings apply with the default `numbers` frontend; `trial` entries require `numbers_terms`, and `pending`/`rejected` entries remain disabled. `off` bypasses all normalization. The user selected Chinese HAMi → 哈密 and vGPU → V G P U on 2026-10-26. These approved readings cover Chinese HAMi / HAMi vGPU and Chinese/English vGPU, including capitalization variants. English HAMi pronunciation remains a pending candidate; untested Japanese/German term substitutions are not activated by this change. Approval records the requested reading, not a claim of native listening or leaderboard quality. URLs, email addresses, inline code and path-like identifiers remain protected.

The final target regression completed 39 requests / 263 chunks on one Orin MOSS instance, without frame-cap failures or OOM, under a 1792MiB container cap. The Japanese early-EOS block grew from 2.48s to 10.00s; the German overlong block shortened from 13.12s to 4.24s; French Bonjour shortened from 14.00s to 1.04s. Switching French → English → Chinese → French preserved the French waveform, and all 302 chunk/whole recordings matched prior ASR evidence by WAV SHA, language and normalized text. Eighteen whole recordings remain identical to the original submission baseline. Tests cover 56 adapter/stopping/lexicon/language/lifecycle/fade unit tests plus normalization, text preservation and streaming PCM checks.

Direct-adapter first PCM was 0.047–0.510s and RTF 0.354–0.574, including engine switching and diagnostic capture, excluding loading/ROS/MCP/network/playback. Other shared-host containers were unchanged. Japanese names remain inaccurate; the French digital-human paragraph's raw joined-chunk WER increased from 8.4% to 12.8%, and other French pauses remain. These diagnostics are not native listening or GSB. Artifacts: `outputs/eos-switch-r6-20261026-172022` in the project workspace. The candidate assets have 13 files, all verified through the competition HTTP service. No new complete image build, three-container or platform pass is claimed; the candidate is not committed or pushed yet.

### 2026-10-27 未发布质量候选

英文numbers前端增加限定简单公式读法（equals/minus/squared/pi）及明确温度/百分比的plus or minus；保留复杂表达式、路径与代码。西语guarded模式继续保护阈值为0.95，其余语种0.99；显式sample模式保持原采样。源于真实case250在继续概率0.9696时提前EOS，83→169帧后恢复4个术语，旧token前缀不变；Orin三种子定向测试及另外3条文本均正常结束，ASR无整段编辑数回退。仍有其他列表漏读、俄语长低音量、日语读音、法语回退及英文末尾重复，不能称质量已全面达标。详细本地证据在outputs/feedback12-r1-20261027-214530和outputs/feedback12-es-r2-20261027-215233；不是新平台评分。

### 2026-10-28 质量候选更新（未提交）

默认前端对独立俄语块末逗号使用句号送模，off绕过；英语增加V/A/W明确数量读法；法语guarded模式复用德语bounded EOS，EOS已更可能时不继续。法语标点改写经过回归发现内容回退，最终完全撤回。真实RU35长低幅22.8→1.9秒、FR154相对上一候选6.3→1.1秒，英文222末尾重复在本次ASR不再出现。ASR非母语/GSB，日语与专名等问题仍在，未声称全面通过。详见项目outputs/quality-update-20261028/findings.md及R3/R4/R5/R6原始结果；当前未commit/push/提交。

### Chinese mathematics and terminology frontend, 2026-10-28

The default `numbers` frontend now verbalizes bounded Chinese flat equations before the existing sentence/token-budget splitter. It preserves the square/cube scope and says implicit numeric coefficients explicitly: `y = 2x - 3` → `Y等于二乘以X减三`; `a² + b² = c²` → `A的平方加B的平方等于C的平方`; `S = πr²` → `S等于派乘以R的平方`. Decimal values, `×/÷/*//`, square/cube notation and simple comparisons are supported in this bounded grammar. Chinese explicit quantities are now enabled by default, with added V/A/W → 伏特/安培/瓦特 and ± temperature/percentage/electrical-unit readings. This supersedes earlier notes that Chinese quantities require `numbers_terms`. The `off` frontend still bypasses normalization.

This first grammar requires whitespace on both sides of the equation/comparison operator. Unsupported parentheses, roots, scientific notation and higher powers are retained; unspaced assignments, dates, versions, paths, URLs, emails and inline code are protected. It is not a general LaTeX parser, evaluator or automatic formula simplifier. Variables remain Latin letters in the model text; text expansion alone cannot guarantee their pronunciation.

The pronunciation JSON now covers the 27 AI terms in the translated feedback inventory. Chinese initialism candidates use separated letters; whole-word English names retain their identity, with selected casing/hyphen/spacing candidates. These are `trial` readings, requiring `numbers_terms`, until same-backend audio comparisons establish which ones help. No Chinese semantic translation is silently substituted for an English name. Pending/rejected entries remain disabled; the user's approved HAMi → 哈密 / vGPU → V G P U readings remain active. Changing this plaintext frontend needs no new ONNX export, TensorRT build, weights or second model.

Public implementation references, inspected on 2026-10-28:

- [WeTextProcessing Math](https://github.com/wenet-e2e/WeTextProcessing/blob/master/tn/chinese/rules/math.py), [operator mappings](https://github.com/wenet-e2e/WeTextProcessing/blob/master/tn/chinese/data/math/operator.tsv) and [Whitelist](https://github.com/wenet-e2e/WeTextProcessing/blob/master/tn/chinese/rules/whitelist.py): Apache-2.0, number/operator grammar and phrase substitution. Its inspected math rule does not establish support for our variable/power examples. Our existing lightweight frontend remains in use; no Pynini/FST dependency was added.
- [PaddleSpeech Chinese frontend](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/tts/zh_text_frontend.md) and [number rules](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/paddlespeech/t2s/frontend/zh_normalization/num.py): Apache-2.0, useful reference for fractions, percentages, decimal numbers and the TN/G2P separation. PaddleSpeech was not installed.
- [phrase-pinyin-data](https://github.com/mozillazg/phrase-pinyin-data): Chinese phrase-to-pinyin data, useful for polyphone lookup rather than a complete technical-brand dictionary.
- [CMUdict](https://github.com/cmusphinx/cmudict): public English pronunciation data, useful for checking ordinary English words rather than treating all acronyms as letter sequences. Neither pronunciation resource can directly control Nano through this adapter's plaintext-only input; importing phoneme strings would not create phoneme support.

These projects are references, not evidence that our local AI terminology candidates have authoritative pronunciation approval. No external pronunciation dataset or code was vendored in this change.

Chinese math output uses contiguous Chinese words, without inserting spaces between every operator, number and variable. Letter-hint, homophone-only and forced clause-boundary controls are retained as diagnostic recordings; they are not default pronunciation policies.

### Public Chinese frontend coverage, 2026-10-28

This update supersedes the initial flat-equation scope and the opt-in status of the accepted Chinese terminology candidate. The default `numbers` frontend uses `zh_math.py` and `zh_frontend_data.json`: a bounded parser for arithmetic, parentheses, implicit multiplication, fractions, percentages/per-mille, powers, square/nth roots, scientific notation, simple comparisons/chains, absolute values/factorials, common mathematical functions, Greek symbols, and explicit engineering quantities/ranges. Selected dollar-delimited or escaped LaTeX forms (`frac`, `sqrt`, powers and arithmetic/comparison/Greek commands) are accepted. It does not evaluate expressions. Unsupported syntax is retained; there is no Python `eval`, SymPy, FST, model or network dependency. The parser limits candidates to 512 characters, 128 tokens and 16 nested power/group levels.

| Coverage | Inventory |
|---|---:|
| Independently specified text fixtures, including held-out values | 86 in 10 categories |
| Case-sensitive unit spellings, including area/volume aliases and compound rates | 162 |
| Function spellings | 15 |
| Lower/uppercase Greek symbols | 48 |
| Public AI/LLM glossary records imported in full | 148 in 12 categories |
| Pronunciation-library entries, all languages | 342 |
| Active Chinese token/phrase entries | 272 |

The public inventory covers fundamentals, architecture, training, adaptation, inference/deployment, prompt engineering, retrieval, agents, multimodal, evaluation, safety, and ecosystem/engineering. Full names, explicitly listed initialisms and relevant acronym inventory are kept with their source category and immutable source URL. The previous 27 Chinese technical-term candidates were accepted in the user's listening feedback. New deterministic text readings use the same initialism/word-identity policy, without claiming per-term native pronunciation approval. Other-language trials/rejections and the Qwen version candidate retain their existing status. Existing approved personal names keep their selected readings.

`numbers` enables approved Chinese term policies as well as the math grammar; `numbers_terms` additionally retains the older opt-in trials. `off` bypasses both. Dates, clocks, version strings, paths, longer identifiers, inline/fenced code, ambiguous slash-only fractions and unspaced assignments retain protection. Explicit mathematical context or math delimiters can disambiguate ordinary formulas such as `公式y=2x-3`. Unsupported summation/integral/matrix constructs are not partially interpreted as supported formulas. Scientific notation is spoken rather than evaluated. Source unit data is reviewed rather than lowercased wholesale: mW/MW and mV/MV have distinct readings; the upstream ds, ω and regional tl entries are excluded. Local SI additions and Chinese function/Greek readings are identified separately from upstream data.

Sources and reproducibility:

- WeTextProcessing @ `039007e6b819bd40c4c4a0644f3939da1da47e44`: Apache-2.0 operator/unit data and math/fraction/measure grammar references. The parser code is local.
- AI/LLM glossary @ `abc74818e14884f0e28fe6465c2328c68f9aae53`: MIT, 148 term records across all 12 source pages; source names are imported without substituting definitions for spoken terms.
- CMUdict @ `74790861f652b15e4ac49015a90074ad62a27690`: selected English word pronunciations are metadata, with its redistribution notice. They are not injected as Nano phonemes.
- [NIST SI prefixes](https://www.nist.gov/pml/special-publication-330/sp-330-section-3): reference for symbol case and prefix distinctions; no runtime download.

`docs/tts_frontend_resources/source_manifest.json` records every downloaded source-file SHA256 and revision. `terminology_catalog.json` preserves all 148 source records and their imported forms. `tools/import_tts_frontend_resources.py --sources <local-snapshot-directory>` checks those source hashes and rebuilds the data without network access; repeated imports are byte-idempotent. Keep the source snapshots outside Git. Runtime notices and the MIT/CMU licenses accompany the installed frontend. No model weights, full external dictionary, engine files or files over 1MB are introduced into Git.

Validation: `tools/test_tts_zh_math.py`, `test_tts_text_normalization.py`, `test_tts_lexicon.py`, and `test_tts_frontend_adapter.py` check mathematical scope, all unit spellings, the full imported glossary, idempotence, protections and the actual threaded adapter. The new independently authored Jetson corpus covers every active Chinese term at least once, new math/engineering combinations and unchanged original feedback cases. Audio/ASR results remain diagnostic rather than a leaderboard GSB score or exhaustive human listening approval.

The common grammar also normalizes full-width mathematical characters while preserving superscript meaning, and distinguishes explicit ratio context (`比例1:20`) from clock strings (`时间12:30`). Bare years/numbers are not globally treated as mathematical values; `计算机` is not a math-context trigger. Whole-word FLUX uses its CMUdict reference rather than initialism spelling.

Math context is bounded by sentence boundaries; ratio labels apply to the current clause, so `比例：1:20，时间：12:30` retains the clock while verbalizing the ratio. Context checks do not propagate a ratio label into a following time clause.

### Multilingual mathematics and quality release, 2026-10-28

Relative to published 1.2 (`09256423`), this candidate includes the previously
validated language-specific EOS handling, the French-only Nano frame profile,
Russian fragment punctuation repair, approved Chinese terminology, and the new
Chinese/English/Japanese/German/French/Russian/Spanish mathematics frontend.
Thai behavior is preserved. New foreign initialism spellings remain opt-in
`trial` entries: the default adapter frontend is still `numbers`. Rejected
English GPU/NPU/DCU/MLU expansions stay disabled.

Numbers use the installed ICU spellout library; locale unit tables and math
labels retain their public-source versions, SHA256 records and licenses in
`docs/tts_frontend_resources`. Russian mathematical Latin letters use MathCAT
names separately from English-style initialism readings. The Docker build checks
that the system ICU library exists; ICU70 was verified inside the actual ARM64
inference container. These are text rules, without an additional speech model.

The current runtime exactly matches the final frozen Orin test bundle. The
latest three sequential single-container batches exercised 45 requests: 43
returned normally, while two rejected Japanese candidate forms hit the frame
limit. Normal return does not establish content completeness. The final source
passes 75 frontend/adapter/stopping/lexicon checks; streaming PCM and text-chunk
checks also pass. Source-data rebuilds are byte-identical, and all new repository
resources are below 1MB. The 13 model-download URLs and sizes were checked again
before publication; the five small downloadable files also matched SHA256.

Math A/B ASR diagnostics improved on the small test corpus; they do not predict
leaderboard scores. Japanese mathematics and pronunciation remain weak, with
suspected missing content; Russian/French terms and quantities still need
review. Foreign initialism expansion did not show stable gains and is not
bulk-enabled. No new complete image build, platform run, or three-container
validation is claimed for this final combination. It is a candidate for the
next leaderboard evaluation, with the existing full-FP32 codec and accepted
sentence-fade audio chain retained.

### English reference selection (2026-10-29)

English now uses the pinned built-in Ava reference instead of Adam, following the user's listening preference. This also applies when auto language resolution selects English. Other language mappings are unchanged. Ava's prompt codes already exist in the shipped browser manifest, so no model download, reference encoding or TensorRT rebuild is needed.

The reference-only Orin comparison used six English development sentences at different lengths, fixed seed42 and unchanged frontend/sampling/codec. Adam, Bella and Ava all completed six requests; raw ASR WER was 6.1%, 3.1% and 2.3% respectively. These are small-sample content diagnostics, not a naturalness or leaderboard score. The user preferred Ava in listening. Full evidence remains in the project-local `outputs/voice-natural-20261029/` report and `outputs/voice-natural-20261029-215148/` recordings. The new mapping has not yet received a leaderboard evaluation or a three-container performance retest.

### Japanese detached closing-quote fragments (local quality update)

The adapter now passes the resolved language into text splitting. For Japanese only, a fragment beginning with `」` or `』` and containing fewer than eight text tokens is joined to the preceding block if the combined block remains within 75 tokens. Punctuation and text are retained. This avoids independent synthesis of tiny closing-quote/attribution fragments; ordinary short sentences and other languages keep their existing splitting.

The broader under-12-token experiment restored EOS in three failed development sentences across seeds42/7/2026, but also increased raw ASR edits on one previously normal sentence. The accepted rule is narrower: checking 2,100 development inputs changed only four Japanese cases, whose model chunks exactly match the already tested candidate chunks; the normal regression case stays unchanged. Three cases recovered from frame-limit failures, while a fourth still fails in an earlier long block. This does not establish that all Japanese content or stopping issues are fixed. Local evidence: `outputs/ja-short-dev-20261029/` and `tmp/ja-quote-fix-20261029/checks.json`.

### German and Russian everyday years (local quality update)

Explicit year contexts now use the existing ICU/CLDR spellout rules before model tokenization. Russian distinguishes `в … году` and `… года`, including a compatibility correction for the old ICU locative forty ending. Bare numbers and ambiguous identifiers are preserved. Japanese/French year experiments did not show consistent benefit and are not enabled. See `docs/tts_frontend_resources/multilingual.md` for scope, validation and remaining ASR/listening limitations.
