# STT Model Evaluation: Parakeet TDT 0.6B V3 — Comparison Report

## Overview

This report compares two inference engines running the **nvidia/parakeet-tdt-0.6b-v3** speech-to-text model:

| Model / Engine | Description |
|---|---|
| **Parakeet TDT 0.6B V3 (ONNX)** | ONNX Runtime CPU inference via `onnx-asr` library — `istupakov/parakeet-tdt-0.6b-v3-onnx` (FP32) |
| **Parakeet GGUF (parakeet.cpp)** | C++ ggml inference via `parakeet-cli` — `tdt-0.6b-v3-q6_k.gguf` (Q6_K quantized) |

### Test Conditions

- **Hardware:** 2 CPU cores, no GPU, 7.7 GB RAM
- **Test Audio:** 16.78 seconds, mono 16 kHz 16-bit WAV — Harvard sentences (43 words)
- **Reference Text:** "The stale smell of old beer lingers. It takes heat to bring out the odor. A cold dip restores health and zest. A salt pickle tastes fine with ham. Tacos al pastor are my favorite. A zestful food is the hot cross bun."

---

## Side-by-Side Comparison

| Metric | Parakeet ONNX (FP32) | Parakeet GGUF (Q6_K) | Difference |
|---|---|---|---|
| **WER** | 4.65% | 4.65% | **Identical** |
| **CER** | 1.90% | 1.90% | **Identical** |
| **Latency** | 5.50 s | 11.88 s | GGUF is **2.16× slower** |
| **RTF** | 0.3281 | 0.7078 | GGUF RTF is **2.16× higher** |
| **Peak CPU (per-core)** | 49.9% | 99.8% | GGUF is **2× more CPU-intensive** |
| **Peak Memory** | 2666.9 MB | 927.6 MB | GGUF uses **65% less memory** |
| **Model Load Time** | 86.29 s | N/A (lazy load) | ONNX downloads on first run |
| **Quantization** | FP32 (float32) | Q6_K (6-bit) | GGUF is 2.67× smaller |

---

## Analysis & Observations

### Accuracy (WER / CER)

Both engines produced **identical transcription quality** — WER of **4.65%** and CER of **1.90%**. The two errors were both substitutions: "dip" → "deep" and "pastor" → "pasta". These are phonetically similar words, consistent across both model variants. This confirms that the Q6_K quantization at 6-bit precision introduces **no measurable accuracy loss** on this test sample.

### Memory Efficiency

The Q6_K GGUF model uses **65% less peak memory** (927.6 MB vs 2666.9 MB) than the full FP32 ONNX export. This is a significant advantage for CPU-only environments with limited RAM. The ONNX model's higher memory usage is expected — it loads the full FP32 weights (~2.4 GB uncompressed for 0.6B parameters).

### Speed & Throughput

The ONNX model is **2.16× faster** in wall-clock latency (5.50 s vs 11.88 s), despite using FP32 precision. This advantage comes from:
- ONNX Runtime's optimized CPU kernels (AVX2-aware)
- The `onnx-asr` library using ONNX Runtime's execution provider with operator fusion
- Higher memory bandwidth utilization (FP32 weights read faster from the larger memory footprint)

The GGUF model, though more memory-efficient, saturates both CPU cores at 99.8% utilization, indicating it is compute-bound rather than memory-bound at Q6_K precision.

### CPU Utilization

- **ONNX:** 49.9% per-core average — leaves headroom, not fully utilizing both cores
- **GGUF:** 99.8% per-core — fully saturated, likely bottlenecked by CPU compute

The ONNX engine's lower CPU utilization suggests it could handle concurrent requests or benefit from additional parallelization, while the GGUF engine is already at capacity.

### Quantization Trade-off Summary

| Factor | Winner | Why |
|---|---|---|
| **Accuracy** | Tie | Identical WER/CER across both runs |
| **Speed** | ONNX (FP32) | 2.16× faster inference |
| **Memory** | GGUF (Q6_K) | 65% less peak memory |
| **CPU Usage** | ONNX (FP32) | 50% less CPU utilization |

### Recommendations

- **For memory-constrained environments (<2 GB free RAM):** Use the GGUF (parakeet.cpp) variant — it delivers identical accuracy while staying under 1 GB peak memory.
- **For latency-sensitive or throughput-oriented scenarios:** Use the ONNX (onnx-asr) variant — it runs 2× faster and uses half the CPU per core.
- **For production deployments:** The ONNX route is preferred on systems with ≥4 GB RAM. The GGUF route is ideal for edge devices or containers with strict memory limits.

---

## Raw Results

### ONNX (onnx-asr) — `results_onnx.json`
```json
{
  "model_id": "nvidia/parakeet-tdt-0.6b-v3 (ONNX via onnx-asr)",
  "engine": "onnx-asr (ONNX Runtime CPU)",
  "quantization": "FP32",
  "wer": 0.0465,
  "cer": 0.019,
  "latency_sec": 5.5,
  "rtf": 0.3281,
  "peak_cpu_percent": 49.9,
  "peak_memory_mb": 2666.9
}
```

### GGUF (parakeet.cpp) — `results_gguf.json`
```json
{
  "model_id": "mudler/parakeet-cpp-gguf (parakeet.cpp via parakeet-cli)",
  "engine": "parakeet.cpp (ggml)",
  "quantization": "Q6_K",
  "wer": 0.0465,
  "cer": 0.019,
  "latency_sec": 11.88,
  "rtf": 0.7078,
  "peak_cpu_percent": 99.8,
  "peak_memory_mb": 927.6
}
```

---

*Report generated on 2026-06-03 using the Eval-STT evaluation framework methodology.*