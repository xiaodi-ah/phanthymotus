# ZipVoice / Supertonic 原分块候选

本提交将中文切换为 ZipVoice-Distill FP32 ONNX/CUDA，六个外语继续使用 Supertonic3 TensorRT。覆盖 zh/en/ja/de/fr/ru/es；泰语不在此候选覆盖范围。每个 MCP 实例独立持有两个后端；保留既有 MCP、ROS AudioChunk、16k mono PCM16、环境端口协议。

中文固定 Junhao 第二完整句 3.92s，8步、guidance=3、speed=1、t_shift=0.5、feat_scale=0.1、target_rms=0.1。Sherpa 按原策略 min_char_in_sentence=30、max_char_in_sentence=80 分块，silence_scale=1；未启用20/16/12字短块，不额外插入块间静音。既有数学、数字、邮箱和术语结构化前端保持启用。参考文件、转录和图均由 SHA256 清单固定。

基础镜像固定已检查的 JP6.1 digest，内置 sherpa-onnx1.13.6+cuda。构建只下载 ZipVoice 和 Supertonic 资产，无需下载或加载 MOSS 权重。权重/参考来自 `perception/zipvoice_assets.json` 所列不可变 JuiceFS HTTP 路径；Git只存代码、清单及文档。Supertonic仍为F1/8步/speed1.05，CUDA ctypes绑定取代Torch绑定以降低常驻开销；先前六语种预热PCM与Torch绑定逐字节相同，不能外推所有文本。

配置见 `perception/config.yaml`：routing=zipvoice_supertonic；插件仍用TTS_PLUGIN=tts（Docker ENV映射moss_tts实现），JP_VERSION=6。两后端均真实预热成功后才报告running。未覆盖语种明确报错，不回退英语；没有远程模型服务调用。

## 已测结果与限制

2026-11-09 Orin 单ZipVoice、FP32/8步：原分块34字中句首PCM1.255–1.266s、RTF0.222–0.223，短句约0.81s/0.39；冷短句首PCM1.429s，不含加载。20字候选将中句降至约0.98s/0.32，但用户反馈块间静音长，因此保留原策略。这些结果不能证明首音频1s全部准入。12个普通句对照无提示Whisper转写字符编辑均0，长文结尾保留，但数学变量、时/十、品牌词仍待听审。

单ZipVoice试验16次生成成功，RSS高水位729MiB、宿主最低可用1261MiB，无OOM；它不是双模型总显存。此前单进程双模型原分块完成7次预热和15/30正式请求后触发共享宿主内存保护，非容器OOM；其他4容器审计不变。2026-11-09两个后续队列均因资源等待超时未启动，不存在已通过的完整双模型或三容器结果。本提交用于平台试跑，不宣称三容器稳定性通过。

当前ZipVoice为sherpa C++/ORT CUDA运行时，非TensorRT；Supertonic为TensorRT。ZipVoice回调在完整文本块生成后发生，不是帧级流式；输入缩块可能影响韵律。C++分词、参考特征和随机实现不同于之前A100 PyTorch，图SHA相同不等于波形相同。

## 来源

- ZipVoice-Distill encoder/decoder与官方HF revision `889dd609a42f0de9cf9aac335eae42677becb1b9` 逐SHA相同；分发包来自[k2-fsa/sherpa-onnx官方TTS模型release](https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models)。
- ZipVoice代码Apache-2.0，Emilia训练权重按[官方模型卡](https://huggingface.co/k2-fsa/ZipVoice#4-license)为CC-BY-NC；本候选为比赛实验，未声明商业授权。
- [Vocos](https://huggingface.co/charactr/vocos-mel-24khz)为MIT；espeak-ng-data来自同一官方发布包，[eSpeak NG来源与许可](https://github.com/espeak-ng/espeak-ng)。Junhao参考来源/校验记录在模型清单和reference.json。
