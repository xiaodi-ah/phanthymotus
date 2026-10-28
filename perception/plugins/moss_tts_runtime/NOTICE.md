This trial reuses the official OpenMOSS MOSS-TTS-Nano `ort_cpu_runtime.py`, Apache-2.0 (see LICENSE).
Official model revisions: MOSS-TTS-Nano-100M ONNX f52645cb467506d8e18e746ddd59482685b74e58; MOSS-Audio-Tokenizer-Nano ONNX ceff0d0749bfb3fa2d61149794ec6feef0d1e1ae.
The frame fusion, explicit FP32 sampling, TensorRT/CUDA Graph execution and framework integration are our own implementation. No colleague optimization commit/code/engine is used.
Original exports: https://github.com/OpenMOSS/MOSS-TTS-Nano
Model assets and runtime dependency wheels are downloaded from the internal JuiceFS file service; no weights, engine plans or wheels are tracked in Git.

Chinese frontend public resources (2026-10-28):
- wetext: https://github.com/wenet-e2e/WeTextProcessing @ 039007e6b819bd40c4c4a0644f3939da1da47e44
- glossary: https://github.com/JingHao-Leon/ai-llm-glossary @ abc74818e14884f0e28fe6465c2328c68f9aae53
- cmudict: https://github.com/cmusphinx/cmudict @ 74790861f652b15e4ac49015a90074ad62a27690
WeTextProcessing operator/unit data: Copyright (c) 2022 Zhendong Peng, Apache-2.0 (LICENSE). Symbol case, SI additions and the bounded expression parser are local adaptations.
AI/LLM glossary term inventory: Copyright (c) 2026 JingHao-Leon, MIT (GLOSSARY_LICENSE).
Selected CMUdict pronunciation data: Copyright (C) 1993-2015 Carnegie Mellon University (CMUDICT_LICENSE). ARPABET is reference metadata, not a supported Nano input.
Pinned source-file SHA256 and the complete 148-item imported inventory: docs/tts_frontend_resources in the source repository. No external code is executed or downloaded at inference time.

The non-Chinese math frontend uses system ICU spellout (ICU 70 validated on x86_64
and the Jetson ARM64 inference container). It does not load another speech model.
`multilingual_frontend_data.json` contains a small subset of CLDR unit patterns,
SRE Greek/function labels, and NeMo Japanese/Russian letter tables. Pinned revisions,
input SHA256 values and explicit authored-policy notes are recorded in
`docs/tts_frontend_resources/multilingual_sources.json`. Included licenses:
`CLDR_LICENSE`, `SRE_LICENSE` and `NEMO_LICENSE`. Locale math scope phrases and
DE/FR/ES initialism spellings are project-authored; they are not an imported full
NeMo or SRE grammar. Thai remains unchanged. New initialism candidates are opt-in;
previously rejected English GPU/NPU/DCU/MLU expansions remain rejected.
