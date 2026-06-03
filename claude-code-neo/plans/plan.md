# STT Model Evaluation: Parakeet TDT 0.6B vs Parakeet GGUF

## Goal
Evaluate and compare two Parakeet-based speech-to-text models using the Eval-STT framework methodology:
1. **nvidia/parakeet-tdt-0.6b-v3** (via ONNX Runtime for CPU inference)
2. **mudler/parakeet-cpp-gguf** (via parakeet.cpp C++ inference engine)

## Research Summary
- **Eval-STT** framework provides WER/CER/latency/RTF/memory evaluation but only supports Whisper & Wav2Vec2 natively — both target models require custom integration
- **Parakeet TDT 0.6B V3 ONNX** (`istupakov/parakeet-tdt-0.6b-v3-onnx`) is available for CPU inference via `onnx-asr[cpu,hub]` library
- **Parakeet GGUF** (`mudler/parakeet-cpp-gguf`) requires building `parakeet.cpp` from source (cmake + C++17) and uses the `parakeet-cli` binary for inference
- **Hardware**: 2 CPU cores, no GPU, 7.7GB RAM — both models can run but will be slow; Q6_K quantization recommended for GGUF (~747MB)
- Test audio: Use gTTS or a small LibriSpeech sample with known reference text for accurate WER computation

## Approach
1. Clone Eval-STT repo and install its Python dependencies
2. Build custom evaluation scripts extending the Eval-STT methodology to support both Parakeet models
3. Use a standardized test audio sample with known ground truth text
4. Run both models on the same test data, measuring:
   - WER (Word Error Rate) and CER (Character Error Rate)
   - Latency (wall-clock inference time)
   - RTF (Real-Time Factor = latency / audio duration)
   - CPU and memory utilization
5. Generate a side-by-side comparison report with metrics table

## Subtasks
1. **Clone Eval-STT repo & install dependencies** — `git clone https://github.com/build-ai-applications/Eval-STT`, `pip install -r requirements.txt`, plus extras for both models
2. **Prepare test audio data** — Generate/download a short WAV file with known reference text (e.g., Harvard sentences or LibriSpeech sample)
3. **Build parakeet.cpp from source** — Clone mudler/parakeet.cpp, cmake build with CLI, download a GGUF model checkpoint
4. **Create Python evaluation wrapper for Model 1 (ONNX)** — Script that uses `onnx-asr` to load Parakeet TDT ONNX, transcribe audio, measure metrics
5. **Create Python evaluation wrapper for Model 2 (GGUF)** — Script that calls `parakeet-cli` binary, parses output, measures metrics
6. **Run full evaluation** — Execute both models on test audio, capture all metrics
7. **Generate comparison report** — Side-by-side table of WER, CER, latency, RTF, memory, formatted as markdown + JSON

## Deliverables
| File Path | Description |
|-----------|-------------|
| `/home/azureuser/stt_eval_ccneo/Eval-STT/` | Cloned Eval-STT repository |
| `/home/azureuser/stt_eval_ccneo/evaluate_onnx.py` | Evaluation script for ONNX Parakeet model |
| `/home/azureuser/stt_eval_ccneo/evaluate_gguf.py` | Evaluation script for GGUF Parakeet model via parakeet.cpp |
| `/home/azureuser/stt_eval_ccneo/parakeet.cpp/` | Built parakeet.cpp source with `parakeet-cli` binary |
| `/home/azureuser/stt_eval_ccneo/test_audio.wav` | Test audio file with known reference text |
| `/home/azureuser/stt_eval_ccneo/results.json` | Raw evaluation metrics for both models |
| `/home/azureuser/stt_eval_ccneo/COMPARISON_REPORT.md` | Final side-by-side comparison report |

## Evaluation Criteria
- Both models must complete transcription of test audio without errors
- Metrics captured: WER, CER, latency (seconds), RTF, peak CPU%, peak memory (MB)
- Report shows clear side-by-side comparison with observations

## Notes
- No GPU available — all inference on CPU only
- 2 cores, ~7.7GB RAM — use Q6_K quantized GGUF for Model 2 to fit memory
- Audio should be 16kHz mono WAV format for compatibility