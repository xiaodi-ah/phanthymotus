# MOSS / Supertonic 路由试验

本提交默认启用 MOSS / Supertonic 路由；模型包已发布并校验，三容器验证尚未通过。
中文与泰语使用 MOSS TensorRT 后端（中文保留 1.3），英文、日文、德文、法文、俄文、西班牙文使用 Supertonic 3 TensorRT。
Supertonic 保留已试听的 F1、8 步、speed=1.05；随机数状态独立于其他实例。

`plugins.tts` 已配置：

```yaml
routing: moss_supertonic
supertonic_model_dir: /models/supertonic-3-trt-jp61
```

插件入口仍为 `TTS_PLUGIN=moss_tts`，现有 `model_dir` 仍指向 MOSS 模型包。
不设置 `routing` 时仍使用原 MOSS 路线。每个 MCP 实例独立拥有两个后端，不共享模型状态；两者均成功预热后才声明运行。

请求显式 `language` 优先，其次是配置语种，最后按现有字符及词汇规则识别。
短词、品牌名称、仅含数字和混合语种可能存在歧义，应显式指定语言。
这不是句内多模型切换：一条请求只选一个后端。泰语走 MOSS；其他未启用语言明确报错，不回退成英语。

## 模型与接口

Supertonic 目录必须包含 `duration_predictor.plan`、`text_encoder.plan`、`vector_estimator.plan`、`vocoder.plan`、`tts.json`、`unicode_indexer.json`、`F1.json`。
当前引擎只在 JP6.1 / TensorRT 10.4.0 上验证，来源为 Supertonic 3 revision `aafc6e32` 的本项目 FP32 TRT 构建结果；不能直接用于其他 TRT 版本或 GPU。
代码来源、许可证及原 helper 校验值见 `perception/plugins/supertonic_runtime/NOTICE.md`。
模型不得提交到 Git。`perception/supertonic_assets.json` 记录不可变 JuiceFS 路径、大小和 SHA256；Dockerfile 使用现有下载器校验安装。8 个资产共 415,938,366 字节，已从比赛 HTTP 服务完整取回验证。

Supertonic 按官方句子块生成，日语目标长度 120 字符、其他语言 300 字符，块间停顿 0.3 秒。
这不是句内增量声码器流式：首 PCM 要等待第一文本块生成完毕。
当前 TRT profile 支持至多 512 个文本 token、512 个 latent 帧；超限明确报错，不截断内容。无标点超长文本仍需后续完善切分。

现有 ROS 音频通道固定为 16 kHz 单声道 PCM16；Supertonic 原生 44.1 kHz 通过多相滤波重采样接入，不能直接改采样率标签。
实机试验同时保存原生和交付版 WAV，便于对照。ASR、听感以及完整 MCP/ROS 时延不能由适配器测试替代。

## 泰语候选（2026-10-30，仅调研）

| 优先级 | 候选 | 适用点与待验证事项 |
| --- | --- | --- |
| 1 | [FastThaiG2P + 泰语微调 Kokoro](https://github.com/awslabs/FastThaiG2P) | 82M、24 kHz、已有 ONNX；带数字、单位和缩写前端。不是原版 Kokoro 直接支持泰语。代码 Apache-2.0，下载具体 checkpoint 前仍需核对权重发布版本及许可。 |
| 2 | [MMS-TTS Thai](https://huggingface.co/facebook/mms-tts-tha) | 官方泰语 VITS checkpoint，可作为轻量基线；[sherpa-onnx 提供 MMS 转换路线](https://k2-fsa.github.io/sherpa/onnx/tts/mms.html)。权重 CC-BY-NC-4.0。 |
| 3 | [Wayu-Paxa-TTS-Edge](https://huggingface.co/wayu-ai/wayu-paxa-tts-edge) | 82M、12 固定音色、24 kHz，支持泰英混合，适合质量对照；权重 CC-BY-NC-4.0，作者定位为研究产物。 |

以上尚未在本项目下载或测试，公开 CPU RTF 不能视为 Jetson 性能。优先用普通泰语、数字单位、泰英术语、长句验证漏读/重复和自然度，再测 Jetson 首 PCM、RTF、内存。固定音色模型不能据此宣称支持参考音频克隆。

## 单实例实机结果

2026-10-30，26 条请求全部完成，两个后端同时驻留；中文前后切换波形相同，18 条外语原生 PCM 与此前单独 Supertonic TRT 实验逐样本相同。自动识别的六条清晰外语句子均选择正确语种，泰语拒绝验证通过。

| 后端 | 请求数 | 首 PCM（秒） | RTF |
| --- | ---: | ---: | ---: |
| MOSS 中文 | 2 | 0.044–0.089 | 0.308–0.433 |
| Supertonic 六语种 | 24 | 0.166–0.747 | 0.037–0.103 |

模型加载约 8.01 秒，以上为预热后推理，不含加载、MCP/ROS、网络和播放器；首 PCM 不是已识别首字。进程峰值 RSS 1126.5 MiB，逐请求采样的 cgroup 最大值 952.4 MiB（非整个运行峰值，两指标口径不同，不相加）。宿主最少可用 3028 MiB，无 OOM；其他四个容器 ID/PID/重启次数未变，自有测试容器已退出并删除。

项目产物 `outputs/routed-tts-20261030/` 包含 results、verification、backend-comparison、交付/原生 WAV、试听 index.html；采集 SHA 校验通过。首次测试因 MOSS 路径未装配失败，失败审计保留，修正后完成以上结果。

本地路由六项、音频交付七项测试通过；原 MOSS 语言/前端/分块回归通过。仍需三容器、完整接口、更多长文本边界和人工听感验证；当前结果不代表可直接提交。

## 中文 1.3 保留与泰语回路由

用户要求泰语暂时交回 MOSS。当前路由为 zh/th → MOSS，其余六语种 → Supertonic；上一节“泰语拒绝”是修改前的历史试验结果。
中文基线 commit 为 `3564266e753226858f6d873cce635f76cacec497`。500 条中文样本在开启/关闭术语的两个模式下，归一化文本和分块与 1.3 完全一致，Zhiming 音色、帧引擎、FP32 codec、降噪及句尾处理保留。此项为前端回归，不冒充 500 条音频重生成。
三容器完整 MCP/ROS 同步请求测试仍待资源；按用户要求先发布本试测提交，不将发布视为三容器验收通过。

## Supertonic 临时显存

首轮完整框架测试在两个容器加载后宿主仅余约 1204 MiB，保护逻辑拒绝启动第三个，未发生 OOM。不能将单实例 RSS 外推为三个容器的系统内存需求。
根据 [TensorRT 内存 API](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/_static/python-api/infer/Core/ExecutionContext.html)，Supertonic 四个串行阶段改用 `create_execution_context_without_device_memory()`，设定所有输入形状后调用 `update_device_memory_size_for_shapes()`，复用该实例内的临时 GPU 缓冲。每个调用已同步，整个请求有实例锁。不同容器/实例不共享此空间，也不共享模型权重或状态。
MOSS 调用不传 workspace，保持原分配与 CUDA Graph 路径。权重、引擎、采样和音色没有变化。修改后 26 条单实例请求的 50 个 WAV 文件与修改前逐字节相同，产物 `outputs/routed-tts-lowmem-20261030/`。

## 三容器结果与提交状态（2026-10-30）

两次实测仅完成两个独立容器的双后端加载与预热，没有启动第三个，也没有执行计划的 72 个同步请求。首次宿主最少可用 1204 MiB；按实际形状复用 Supertonic 临时空间后为 1252 MiB，仍不足以安全增加一个实例。两次由资源预测保护主动停止自有容器，非 OOM。停止后 exit137 是自有容器停止超时后的结束结果，`OOMKilled=false`；四个其他容器 ID/PID 均未变。

当前已安排 MemAvailable≥6400 MiB、CPU<35%、GPU<10% 连续两次后重试；运行时保留1024 MiB宿主内存。日志 `logs/routed-three-container.log`，本地采集 `logs/routed-three-collect-wait6400.log`。资源不足时不启动容器，不停止任何其他任务。

代码、Supertonic不可变模型包及Docker下载逻辑已准备；中文1.3模型包未变，泰语走MOSS。用户随后要求先 commit/push，因此本提交启用 routing 配置并准备更新 submission.yaml；没有代用户创建平台任务。三容器性能、泰语本轮完整链路、最终镜像构建仍未验证，不能标记为提交就绪。汇总 `outputs/routed-submission-preflight-20261030/status.json`。

提交 YAML 中 `TTS_PLUGIN=tts` 是平台 MCP 工具名，保持不变；推理镜像内部 `TTS_PLUGIN=moss_tts` 选择路由插件，二者不是同一用途。
