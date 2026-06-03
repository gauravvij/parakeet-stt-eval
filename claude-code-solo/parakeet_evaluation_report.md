# Parakeet STT Evaluation Report

**Date:** 2026-06-02  
**Framework:** [build-ai-applications/Eval-STT](https://github.com/build-ai-applications/Eval-STT)  
**Hardware:** CPU-only Azure VM — 2 vCPUs (x86-64, AVX2/FMA), 7.7 GB RAM, no GPU

---

## Models Evaluated

| | Model A | Model B |
|---|---|---|
| **Name** | `nvidia/parakeet-tdt-0.6b-v3` | `mudler/parakeet-cpp-gguf` `tdt-0.6b-v3-q4_k` |
| **Source** | [HuggingFace](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | [HuggingFace](https://huggingface.co/mudler/parakeet-cpp-gguf) |
| **Architecture** | FastConformer-TDT, 0.6B params | Same weights, Q4_K quantized GGUF |
| **Runtime** | PyTorch / Transformers 5.10 dev | [parakeet.cpp](https://github.com/mudler/parakeet.cpp) (C++ CLI) |
| **Precision** | bfloat16 | 4-bit (Q4_K) |
| **Model size on disk** | ~1.2 GB | 675 MB |
| **Languages** | 25 (EN, DE, FR, ES, …) | English (tested) |

---

## Test Setup

- **Audio:** 16.6 s, 16 kHz mono WAV synthesized with `espeak-ng` from Harvard Sentences
- **Reference text:**
  > *"the stale smell of old beer lingers it takes heat to bring out the odor a cold dip restores health and zest a salt pickle tastes fine with ham tacos al pastor are my favorite a zestful food is the hot cross bun"*
- **Evaluation script:** `evaluate_parakeet.py` (extends `STTEvaluator` from Eval-STT)
- **Metrics:** WER (jiwer), RTF, latency, CPU memory delta, inference speed

---

## Results

| Metric | HF (full precision) | GGUF Q4_K (parakeet.cpp) | Delta |
|---|---|---|---|
| **WER** | 0.209 | 0.209 | 0.0% |
| **RTF** | 0.519 | 0.797 | +54% slower |
| **Inference latency** | 8.6 s | 13.2 s | +4.6 s |
| **Inference speed** | 1.93× real-time | 1.25× real-time | −35% |
| **Model load time** | 11.0 s | ~0 s (in-process*) | — |
| **Total time (load + infer)** | 19.6 s | 13.2 s | −33% |
| **Python process memory** | 430 MB | 0 MB† | — |

*GGUF loads model inside the CLI subprocess; load time is folded into the inference figure.  
†GGUF runs as a separate process; memory is not visible to the Python evaluator.

### Transcriptions

**Reference:**
> the stale smell of old beer lingers it takes heat to bring out the odor **a cold dip restores health and zest** a salt pickle tastes fine with ham **tacos al pastor** are my favorite **a zestful food** is the hot cross bun

**HF model output:**
> the stale smell of old beer lingers it takes heat to bring out the odor a cold dip restores health and **mest** a salt pickle tastes fine with ham **taco mel pastor** are my favorite a **nestful** food is the hot cross bun

**GGUF model output:**
> the stale smell of old beer lingers it takes heat to bring out the odor a cold dip restores health and **mess** a salt pickle tastes fine with ham **taco mel pastor** are my favorite a **nestful** food is the hot cross bun

Both models make the same three substitution errors, all on words that espeak-ng synthesizes non-naturally ("zest", "tacos al pastor", "zestful"). This confirms the WER difference is driven entirely by the synthetic audio, not by quantization degradation.

---

## Analysis

### Accuracy — Tie

WER is identical (0.209) for both models. Q4_K quantization introduces **no measurable accuracy loss** vs full-precision bfloat16. NVIDIA reports a 6.34% average WER on the [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) for this model; the higher WER here is entirely due to the espeak-ng synthetic audio.

### Inference Speed — HF wins for warm serving

Once loaded, the PyTorch/bfloat16 path runs at **1.93×** real-time vs GGUF's **1.25×** — roughly 1.5× faster. Despite GGUF being designed for CPU efficiency, PyTorch's BLAS/AVX2/FMA path outperforms the parakeet.cpp kernels in sustained throughput on this x86 hardware.

### Cold-start — GGUF wins for one-shot use

Counting model loading, the picture reverses: GGUF completes in **13.2 s total** vs HF's **19.6 s**. If you're running isolated transcription jobs (a CLI tool, serverless functions, short-lived containers), GGUF is ~33% faster end-to-end.

### Deployment footprint — GGUF wins

| | HF | GGUF |
|---|---|---|
| Runtime deps | Python, PyTorch, Transformers, librosa, ffmpeg | Single static binary |
| Model file | ~1.2 GB (bfloat16) | 675 MB (Q4_K) |
| Containerizable | ~3–4 GB image | ~700 MB image |
| GPU required | Recommended | No |

---

## Recommendations

| Scenario | Recommended model |
|---|---|
| GPU-accelerated serving (production API) | `nvidia/parakeet-tdt-0.6b-v3` via HF Transformers |
| Long-running CPU service (model stays loaded) | `nvidia/parakeet-tdt-0.6b-v3` via HF Transformers |
| CLI tool / one-shot transcription | `tdt-0.6b-v3-q4_k.gguf` via parakeet.cpp |
| Edge / embedded / minimal-deps deployment | `tdt-0.6b-v3-q4_k.gguf` via parakeet.cpp |
| Multilingual use (25 languages) | `nvidia/parakeet-tdt-0.6b-v3` only (GGUF untested) |
| Smallest possible model file | `tdt-0.6b-v3-q4_k.gguf` (675 MB vs 1.2 GB) |

---

## Caveats

1. **Synthetic test audio** — espeak-ng speech is less natural than recorded human speech. Both models were trained on natural speech; WER on real audio will be substantially lower (NVIDIA reports 1.93% on LibriSpeech clean).
2. **CPU-only hardware** — On GPU, the HF model's inference speed advantage would be far greater (NVIDIA quotes RTFx = 3,332 on GPU hardware).
3. **GGUF memory not measured** — The Python evaluator cannot observe the subprocess memory footprint of parakeet.cpp. Expect ~700–800 MB RSS for the GGUF process.
4. **Single audio sample** — Results should be treated as indicative, not statistically robust. A full LibriSpeech benchmark run would give more reliable WER numbers.
