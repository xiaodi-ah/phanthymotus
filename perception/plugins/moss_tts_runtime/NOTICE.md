This trial reuses the official OpenMOSS MOSS-TTS-Nano `ort_cpu_runtime.py`, Apache-2.0 (see LICENSE).
Official model revisions: MOSS-TTS-Nano-100M ONNX f52645cb467506d8e18e746ddd59482685b74e58; MOSS-Audio-Tokenizer-Nano ONNX ceff0d0749bfb3fa2d61149794ec6feef0d1e1ae.
The frame fusion, explicit FP32 sampling, TensorRT/CUDA Graph execution and framework integration are our own implementation. No colleague optimization commit/code/engine is used.
Original exports: https://github.com/OpenMOSS/MOSS-TTS-Nano
Model assets and runtime dependency wheels are downloaded from the internal JuiceFS file service; no weights, engine plans or wheels are tracked in Git.
