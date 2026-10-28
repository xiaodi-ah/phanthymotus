"""Explicit initialisms; preserve word names and previously rejected pronunciations."""
import json
from pathlib import Path

def update(root):
 p=root/'perception/plugins/moss_tts_runtime/pronunciation_lexicon.json';data=json.loads(p.read_text());entries=data['entries'];existing={(e['language'],e['term'].casefold()):e for e in entries}
 catalog=json.loads((root/'docs/tts_frontend_resources/terminology_catalog.json').read_text())
 initials=set('AGI AI ANN API ASR AWQ BGE BPE CPU C2PA DPO DP FSDP GGML GGUF GLM GPQA GPT GPTQ GQA GRPO GUI HNSW HTTP ICL ID IPO ITL IVF KTO KV LLM LM MCP MHA ML MLA MMLU MQA OCR ORPO PC PEFT PII PP PPL PPO PTQ QAT RL RLAIF RLHF RM RNN RPM RRF SFT SGD TP TPM TPOT TTFT TTS VL VLM VRAM XTTS RTF CNCF NPU DCU MLU GPU'.split())
 # Familiar word names stay words: BERT, CLIP, RAG, FLUX, JSON, CUDA and ONNX.
 locale=json.loads((root/'perception/plugins/moss_tts_runtime/multilingual_frontend_data.json').read_text())
 allterms={t for e in catalog['entries'] for t in e['terms'] if t not in ('Top','AI/LLM')}|initials|{'HAMi','vGPU','TensorRT','NVIDIA','Jetson','ONNX','CUDA','LoRA','BERT','CLIP','RAG','FLUX'}
 records=[]
 for lang in ('en','ja','de','fr','ru','es'):
  letters=locale[lang]['letters'];policies={}
  for term in sorted(allterms):
   old=existing.get((lang,term.casefold()))
   reading=''.join(letters.get(c,c) for c in term) if lang=='ja' else ' '.join(letters.get(c,c) for c in term)
   if term not in initials:reading=term
   if old and (old['status']=='pending' or old['source']=='public-glossary;locale-initialism-policy' and old['status']=='trial') and term in initials:
    old.update(reading=reading,status='trial',source='public-glossary;locale-initialism-policy',evidence='Native-letter candidate; target-device screening required.')
   if old:
    policies[term]={'policy':old['status'],'reading':old['reading'] or term};continue
   policy='trial' if term in initials else 'preserve-word'
   policies[term]={'policy':policy,'reading':reading}
   if policy=='trial':
    entry=dict(term=term,language=lang,reading=reading,status='trial',source='public-glossary;locale-initialism-policy',evidence='Native-letter candidate; target-device screening required.',public_sources=['https://github.com/JingHao-Leon/ai-llm-glossary','https://github.com/NVIDIA/NeMo-text-processing','https://github.com/speech-rule-engine/speech-rule-engine'])
    entries.append(entry);existing[(lang,term.casefold())]=entry
  records.append({'language':lang,'terms':policies})
 # Candidate policies never silently replace full English technical names with a translation.
 out=root/'docs/tts_frontend_resources/multilingual_terminology.json'
 out.write_text(json.dumps({'schema_version':1,'notes':'Word names retain identity. New initialism spellings remain opt-in until device/ASR screening. Existing rejected English GPU/NPU/DCU/MLU spellings are preserved.','languages':records},ensure_ascii=False,indent=2)+'\n')
 p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':update(Path(__file__).resolve().parents[1])
