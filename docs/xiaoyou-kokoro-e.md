# 小优 E 音色：JP6 试提交

默认引擎为 `kokoro-xiaoyou-zh-en`，中文、英文及中英混读共用微调后的 Kokoro-82M E 模型（FP32，global step 10244）。模型权重 327,213,067 字节，固定音色 523,583 字节；权重不进入 Git。

基模为 `hexgrad/Kokoro-82M` revision `f3ff3571791e39611d31c381e3a41a3af07b4987`（Apache-2.0）。推理源码及 ALBERT 许可证、来源记录见 `perception/plugins/xiaoyou_kokoro/kokoro/`。离线合成的 A/E 训练音色中，本版本默认且只加载 E；不包含参考项目的 MOSS/Supertonic 方案。

## 构建及接口

```sh
docker build --network=host --build-arg JP_VERSION=61 -f perception/Dockerfile.jetson -t local/xiaoyou-kokoro-e:jp61 .
```

JP6/61 固定 ARM64 平台基础镜像。Dockerfile 从 `http://172.28.4.81:34567/i-wangaodi/tts_xiaoyou/models/` 下载模型及 Python 3.10 前端包，每个文件校验大小和 SHA256，失败即中止。资产由 `perception/xiaoyou_e_assets.json` 与 `perception/xiaoyou_runtime_61.json` 固定；运行时离线，不自动安装或下载。不要用 A 权重替换 E 清单。

MCP 工具名 `tts`；`MCP_PORT`/`WS_PORT` 可分别覆盖默认 15720/15721。输入文本，输出 `<input_topic>/tts` 的 ROS AudioChunk：16kHz、PCM16、单声道，3200 字节/包（尾包可短），结束发送平台 EOF magic。神经网络只使用 CUDA，无 CPU 兜底。stop/start 保留已加载模型。

容器使用 NVIDIA runtime；以非 root 运行时需加入设备的 video/render 组。DDS 使用镜像内 UDP 配置，配合 host network；不限制到 loopback 接口，以便与平台默认 DDS 接收端互相发现；保持镜像设置的 `ROS_LOCALHOST_ONLY=0`，避免 Humble 强制加入 SHM 导致跨容器音频丢失。可使用私有 IPC，无需 host IPC/PID。只读根文件系统需要可写且可执行的 `/tmp`，供 JIT 和 ROS 日志使用。

## 实测与限制

首版 2026-11-11，`10.100.121.16`，Orin-8G / L4T 36.4.3：收发端使用相同 DDS 配置，一个 GPU 推理容器，真实 MCP→ROS 三轮共 12 次请求（中文、英文、混读、长句）全部通过。每条 PCM 字节完整、EOF 正确，stop/start 复用正常。预热后首 PCM 0.894–1.345 秒，包含最后一包交付的 RTF 0.541–0.762。峰值进程 RSS 1435.945 MiB，容器内存上限 1536 MiB；此值不是独立 GPU 显存。跨轮同文 PCM 相同，交付优化前后音频逐样本一致。

对应实测源码包 SHA256 为 `181ed01d73cf255dc726ffc6d1f688400b887ea340dfdf40cf2294ab5d64f17c`，本地实测镜像 ID 为 `sha256:9487765b1288034e1e03bdea05b62b8757c30e88d770aa4397ed2c05a3928170`。镜像未推送 Harbor，平台按提交 commit 自行构建。该源码包对应首版提交 `da37bbee`；5803c81 试提交只修改 DDS XML 与发布说明。

本地固定架构 ALBERT 与原训练实现的 4 个长度、4 条完整中文/英文/混读音频比较，最大绝对误差 0。引擎、生命周期及迟订阅交付回归通过；镜像内 `tools/test_xiaoyou_mcp.py` 可接收一个或三个独立容器端口进行真实链路测试。

**这是 JP6 试提交，不是完整硬件验收。** Orin 三独立 GPU 容器并发尚未通过；JP5/Python3.8 前端依赖未齐，当前没有 `xiaoyou_runtime_511.json`，JP5 构建不能完成。提交 YAML 显式设置 `JP_VERSION="6"`。未对实机音频完成 ASR 逐字复核及人工听审，不宣称榜单评分或三容器准入通过。

## 默认 DDS 接收端兼容修复

首版提交的可见评测日志中，已完成的 5 条均为 60 秒首包超时；日志没有最终 30 条汇总。JP6 通信复现确认：原 loopback-only UDP 发布端对默认 DDS 订阅端匹配数为 0、20 包收取 0 包；收发端同配置控制组为 20/20；移除 UDP 接口白名单后对默认订阅端为 20/20，64000 字节逐包一致。平台推理日志仍未取得，不能声称该配置是平台故障的唯一原因。

5803c81 试提交只移除 `perception/deploy/xiaoyou-fastdds.xml` 的接口白名单。完整 GPU 复测须让接收端取消 `FASTRTPS_DEFAULT_PROFILES_FILE`，使用默认 DDS，避免收发双方共享自定义配置掩盖互通问题。

另有参考多语种榜的 59 条原文本地压力测试：52 条成功，7 条失败（加号、公式、邮箱、连续数字串、带重音拉丁词）。本次通信修复未解决这些文本前端限制，不能用通信通过代替内容质量验收。

修复配置后，在同一 JP6 上以默认 DDS 接收端完成真实 MCP→ROS 三轮 12/12 请求，PCM 完整、EOF 和 stop/start 正常。首包 0.922–1.323 秒，包含交付的 RTF 0.541–0.770；峰值进程 RSS 1444.605 MiB。12 个 WAV 的 PCM 与首版实测逐字节一致。本次复测复用了既有 GPU 镜像，仅把本次 XML 只读挂载到镜像原配置位置；未重新构建整镜像。平台应从本次 commit 按 Dockerfile 重新构建，COPY 的 XML 与已测挂载文件 SHA256 相同：`f34e97b6ff24954849748c206ce239066c6a94b40200cf91440751e2aaa92173`。

## 文本前端修复（2026-11-11验证）

复用本仓库 upstream 的通用 `vits2_tts_trt/frontend/special_text.py`、数学/单位与编号读法；这是纯文本处理，不导入 VITS2 推理模型或参考项目 MOSS/Supertonic。JP6 不可变基础镜像已有 inflect 7.4.0，无需新增模型或安装依赖。

先保留上标及单位符号完成整段归一化，再按预测时长分段，分段内只做 G2P。编号保留前导零并逐位读，长编号按 3/4 位分组；邮箱逐字符读出分隔符，IP 逐位读；千分位数量先转成完整数量词，切分也保护小数/千分位内部边界。带重音拉丁词完整送入英文 G2P；公式字母 A 使用字母读音，普通英文冠词仍保留原读法。神经网络 FP32/CUDA、200帧解码预算、音素上限与 E 权重均未调整。

12项离线回归通过。相同参考多语种榜59条原文，原52成功/7失败，最终候选59/59成功，共2331.625秒，无PCM削波。59条均有离线ASR转写；最终微调后54条PCM逐字节相同，复用其已核对转写，5条音频重新识别。手机号、验证码和14位订单号逐位核对通过；英文 a squared、pi times r squared 与 two million/two hundred thousand 读法已由转写核对。品牌和同音字仍需试听，ASR数字/数字词归一化会影响错误率，不能当榜单评分或人工音质结论。

最终源码包在JP6只读挂载复测（沿用原GPU镜像），默认DDS接收端12/12真实MCP→ROS成功；首PCM0.937–4.751秒，含交付RTF0.519–0.772，峰值进程RSS1493.738MiB。完整数据、实际挂载包逐文件SHA、单位回归与两轮结果在项目 `outputs/kokoro-e-frontend-fix-20261111/`；最终59条试听在 `outputs/voice-factory-pilot-20261106/e-board-cases-frontend-final-20261111/`。

前端单独验证时的公开试提交为5803c81；这批文本修复现与下述首包优化一并纳入本次代码提交。上面的耗时是优化前的历史结果；当前验证数据见下节。Orin三容器性能与JP5验收未完成，也未逐条人工听审。

语种歧义：纯数字输入没有语种信息时，runtime沿用原auto规则选择英文逐位读；带中文语境的验证码/编号按中文逐位读。本轮验证了数字完整性，未把纯数字case30宣称为中文朗读验收。

## 平台首包准入反馈（2026-11-12）

`log-evaluation-b2302d6f.zip` 测试5803c81：3实例共90请求，失败0，RTF不达标0；可见实例2的30条TTFT为1.649–2.848秒，平均2.226秒。日志明确TTFT门槛1.0秒、RTF门槛1.0，30条全部被过滤，未进入标注/音质评分。通信和三容器功能已有平台证据，仍未通过性能准入。既有本地MCP测试的passed只验证功能及RTF，不能作为1秒TTFT达标证据；本地前端候选同样有超过1秒的结果。后续首包优化与硬件验收须明确加入该1秒门槛。

## 首包优化（2026-11-12试提交）

在上述文本修复版上，保留 E 权重、FP32 CUDA、分句及全部 PCM：首帧只等待 DDS 实际匹配，不再额外固定等待0.5秒；模型预热同时初始化英文数字读法；每个音素及两端边界至少占一帧，因此最少帧数已超过200帧预算时直接进入原有递归切分，省去必然失败的 encoder 计算。测试探针现在分别记录功能与时延，只有 TTFT≤1秒且 RTF<1 才判 passed。

JP6 / Orin-8G、默认 DDS 接收端、1536MiB 模型容器的真实 MCP→ROS 实测：

| 验证方式 | 请求数 | 首PCM（秒） | 含交付RTF |
| --- | ---: | --- | --- |
| 只读源码挂载：基础3条及参考榜长句/特殊文本9条 | 12 | 0.399–0.832 | 0.506–0.579 |
| 同候选，订阅者晚500ms连接 | 4 | 0.515–0.698 | 0.518–0.609 |
| 烘焙候选的验证镜像：中/英/混读/长句，各3轮 | 12 | 0.347–0.682 | 0.516–0.585 |
| 同验证镜像，每条新建话题及config/start/speak/stop | 4 | 0.358–0.588 | 0.513–0.570 |

32次均完整收到PCM及EOF，未观察到播放进度缺口。12条参考对照WAV与前端修复版逐字节一致；验证镜像的12次及逐条重连的4次也与同文挂载版本一致。最长参考文本批次峰值进程RSS1528.375MiB，三轮验证镜像批次1464.527MiB；均非独立GPU显存指标。

验证镜像仅保存在JP6：`local/xiaoyou-kokoro-e:jp61-ttft-20261112-fe778a43`，ID `sha256:a06dd221490db78e43492d94862f4e10b7af1c950fa0a648fb8765df410c7ec1`。它基于既有固定GPU镜像添加源码层，125个烘焙文件逐一核对SHA；模型、运行依赖及启动方式沿用已验镜像，推理端不再挂载源码。构建上下文SHA256为`60b508aaca01423b347b82347afe9638e13409d1ed9664de2892b28a2a03fdaa`。这是缓存基础镜像的增量验证构建，未重新执行正式`perception/Dockerfile.jetson`；JP6磁盘仅余约1.8GiB，且JP5基础镜像未缓存，未拉取大型镜像或清理他人资源。

证据在项目`outputs/kokoro-e-ttft-opt-20261112/status.json`及同目录源码快照、构建日志、指标和WAV；这些本地产物不进入Git。全部自有测试容器已清理。本次commit包含文本修复、首包优化及回归测试，发布分支为个人fork的`feat/xiaoyou-tts`；平台须从本次完整commit按正式Dockerfile构建，不需要push Harbor镜像。单容器32次达标不能证明三容器并发达标；仍需空闲Orin窗口及本小优榜实际30条文本，正式Dockerfile重建、JP5及人工听审也尚未完成。代码push不代表已创建榜单评测任务。
