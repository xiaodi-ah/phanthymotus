# 小优 A 音色：JP6 试提交

分支 `feat/xiaoyou-tts-a` 基于 E 首包优化提交 `28b248df6fea66b086dc610ca34b3c97caced611`。本分支同时加载 A 的训练权重与 `A.pt`：Kokoro-82M、FP32 CUDA、global step 9696。E 版本保留在 `feat/xiaoyou-tts`，项目外的 `submission.yaml` 仍指向 E；A 使用独立的 `submission-a.yaml`。

默认引擎仍为 `kokoro-xiaoyou-zh-en`，平台工具名仍为 `tts`。模型目录为 `/models/kokoro-xiaoyou-a`；构建使用 [A 资产清单](../perception/xiaoyou_a_assets.json)，从 JuiceFS HTTP 下载 `kokoro.pth`（327213067字节）、`A.pt`（523583字节）及配置，逐项校验字节数和 SHA256。权重和音频不进入 Git。基模及推理源码许可与 E 版本相同。

```sh
docker build --network=host --build-arg JP_VERSION=61 -f perception/Dockerfile.jetson -t local/xiaoyou-kokoro-a:jp61 .
```

中文、英文和混读共用 A 模型，输出 ROS AudioChunk：16kHz、PCM16、单声道，3200字节/包（尾包可短），结束发送平台 EOF。模型必须使用 CUDA；保留 E 版本的英文数字预热、实际 DDS 匹配等待、最少时长帧数检查和文本前端修复。stop/start 保留模型；需要 host network、默认接收端 DDS 互通、非root设备组及可执行的 `/tmp`，具体运行约定见 E 说明。

## 本地验证及范围

配置/引擎、交付和时延门槛68项回归通过。JP6 / Orin-8G 上，1536MiB模型容器、默认DDS接收端，两轮真实MCP→ROS验证全部通过：

| 验证方式 | 请求数 | 首PCM（秒） | 含交付RTF |
| --- | ---: | --- | --- |
| 中/英/混读/长句，各3轮 | 12 | 0.402–0.734 | 0.516–0.579 |
| 基础3条及参考榜长句/特殊文本9条，每条重新config/start/speak/stop及DDS话题 | 12 | 0.362–0.756 | 0.505–0.565 |

共24次，13种唯一文本，全部PCM与EOF完整、格式正确、无PCM削波；重复文本的音频逐字节一致，未观察到播放进度缺口。126个镜像内源码文件及实际配置均已核对SHA，确认加载A模型目录及step9696。它们是单容器结果及参考文本，未替代本小优榜的正式30条或三容器准入。完整指标、构建日志、源码与WAV在项目外 `outputs/kokoro-a-submit-20261112/`。

本地试听：`http://127.0.0.1:8768/kokoro-a-submit-20261112/index.html`，需转发服务器16的8768端口；页面展示与实际请求文字对应的13条原始JP6音频。本次未做ASR逐字核验。

JP6 验证镜像 `local/xiaoyou-kokoro-a:jp61-20261112-8db752a5`，ID `sha256:e0580f4a48da993c65480f791a4bde229108e785d242c43e6c14cc3606a0547d`。它从既有JP6缓存镜像构建源码层和A模型层，构建时完成A权重的HTTP下载及完整SHA校验，推理端不挂载源码或模型。构建上下文SHA256：`64a266ccdcdd9a622c2cb9bd618ef582da06dd43c99ec75a7aafbd83bdfab9e1`。缓存基础镜像中保留的E资产未被本配置加载；正式Dockerfile仅下载A资产。

本次是JP6试提交；未重新执行正式Dockerfile的完整基础镜像构建，JP5及三独立GPU容器的1秒门槛仍待验证。共享JP6资源不足以安全启动三个模型，不停止他人服务。性能记录不是榜单评分，未逐条人工听审。代码push不代表已创建平台任务，也无需push Harbor镜像。
