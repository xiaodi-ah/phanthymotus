# 小优 E 音色：JP6 试提交

默认引擎为 `kokoro-xiaoyou-zh-en`，中文、英文及中英混读共用微调后的 Kokoro-82M E 模型（FP32，global step 10244）。模型权重 327,213,067 字节，固定音色 523,583 字节；权重不进入 Git。

基模为 `hexgrad/Kokoro-82M` revision `f3ff3571791e39611d31c381e3a41a3af07b4987`（Apache-2.0）。推理源码及 ALBERT 许可证、来源记录见 `perception/plugins/xiaoyou_kokoro/kokoro/`。离线合成的 A/E 训练音色中，本版本默认且只加载 E；不包含参考项目的 MOSS/Supertonic 方案。

## 构建及接口

```sh
docker build --network=host --build-arg JP_VERSION=61 -f perception/Dockerfile.jetson -t local/xiaoyou-kokoro-e:jp61 .
```

JP6/61 固定 ARM64 平台基础镜像。Dockerfile 从 `http://172.28.4.81:34567/i-wangaodi/tts_xiaoyou/models/` 下载模型及 Python 3.10 前端包，每个文件校验大小和 SHA256，失败即中止。资产由 `perception/xiaoyou_e_assets.json` 与 `perception/xiaoyou_runtime_61.json` 固定；运行时离线，不自动安装或下载。不要用 A 权重替换 E 清单。

MCP 工具名 `tts`；`MCP_PORT`/`WS_PORT` 可分别覆盖默认 15720/15721。输入文本，输出 `<input_topic>/tts` 的 ROS AudioChunk：16kHz、PCM16、单声道，3200 字节/包（尾包可短），结束发送平台 EOF magic。神经网络只使用 CUDA，无 CPU 兜底。stop/start 保留已加载模型。

容器使用 NVIDIA runtime；以非 root 运行时需加入设备的 video/render 组。DDS 使用镜像内仅绑定 loopback 的 UDP 配置，配合 host network；保持镜像设置的 `ROS_LOCALHOST_ONLY=0`，避免 Humble 强制加入 SHM 导致跨容器音频丢失。可使用私有 IPC，无需 host IPC/PID。只读根文件系统需要可写且可执行的 `/tmp`，供 JIT 和 ROS 日志使用。

## 实测与限制

2026-11-11，`10.100.121.16`，Orin-8G / L4T 36.4.3：一个 GPU 推理容器，真实 MCP→ROS 三轮共 12 次请求（中文、英文、混读、长句）全部通过。每条 PCM 字节完整、EOF 正确，stop/start 复用正常。预热后首 PCM 0.894–1.345 秒，包含最后一包交付的 RTF 0.541–0.762。峰值进程 RSS 1435.945 MiB，容器内存上限 1536 MiB；此值不是独立 GPU 显存。跨轮同文 PCM 相同，交付优化前后音频逐样本一致。

对应实测源码包 SHA256 为 `181ed01d73cf255dc726ffc6d1f688400b887ea340dfdf40cf2294ab5d64f17c`，本地实测镜像 ID 为 `sha256:9487765b1288034e1e03bdea05b62b8757c30e88d770aa4397ed2c05a3928170`。镜像未推送 Harbor，平台按提交 commit 自行构建。当前提交中涉及镜像的源文件与该源码包逐字节核对。

本地固定架构 ALBERT 与原训练实现的 4 个长度、4 条完整中文/英文/混读音频比较，最大绝对误差 0。引擎、生命周期及迟订阅交付回归通过；镜像内 `tools/test_xiaoyou_mcp.py` 可接收一个或三个独立容器端口进行真实链路测试。

**这是 JP6 试提交，不是完整硬件验收。** Orin 三独立 GPU 容器并发尚未通过；JP5/Python3.8 前端依赖未齐，当前没有 `xiaoyou_runtime_511.json`，JP5 构建不能完成。提交 YAML 显式设置 `JP_VERSION="6"`。未对实机音频完成 ASR 逐字复核及人工听审，不宣称榜单评分或三容器准入通过。
