#!/usr/bin/env python3
"""
Evaluate nvidia/parakeet-tdt-0.6b-v3 via ONNX Runtime CPU inference using onnx-asr.

Measures:
  - Latency (wall-clock inference time)
  - RTF (Real-Time Factor = latency / audio duration)
  - Peak CPU% and memory (MB)
  - WER and CER via jiwer

Saves results to: /home/azureuser/stt_eval_ccneo/results_onnx.json
"""

import os
import sys
import json
import time
import threading
import psutil
import jiwer
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_PATH = os.path.join(PROJECT_DIR, "test_audio.wav")
REFERENCE_PATH = os.path.join(PROJECT_DIR, "reference.txt")
RESULTS_PATH = os.path.join(PROJECT_DIR, "results_onnx.json")

# Read reference text
with open(REFERENCE_PATH, 'r') as f:
    REFERENCE_TEXT = f.read().strip()

def get_audio_duration(wav_path):
    """Read WAV header to get duration in seconds."""
    with open(wav_path, 'rb') as f:
        header = f.read(44)
        import struct
        channels = struct.unpack('<H', header[22:24])[0]
        sample_rate = struct.unpack('<I', header[24:28])[0]
        bits_per_sample = struct.unpack('<H', header[34:36])[0]
        file_size = os.path.getsize(wav_path)
        data_size = file_size - 44
        duration = data_size / (sample_rate * channels * (bits_per_sample // 8))
    return duration

class ResourceMonitor:
    """Monitors CPU and memory usage in a background thread."""
    def __init__(self, interval=0.5):
        self.interval = interval
        self.process = psutil.Process()
        self.cpu_percentages = []
        self.memory_mb = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread and self._thread.is_alive():
            self._stop.set()
            self._thread.join(timeout=5)

    def _monitor(self):
        while not self._stop.wait(self.interval):
            try:
                self.cpu_percentages.append(self.process.cpu_percent())
                mem_info = self.process.memory_info()
                self.memory_mb.append(mem_info.rss / (1024 * 1024))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

    def get_peak_cpu(self):
        return max(self.cpu_percentages) if self.cpu_percentages else 0.0

    def get_peak_memory_mb(self):
        return max(self.memory_mb) if self.memory_mb else 0.0

    def get_avg_cpu(self):
        return np.mean(self.cpu_percentages) if self.cpu_percentages else 0.0


def evaluate():
    print("=" * 60)
    print("EVALUATION: Parakeet TDT 0.6B V3 (ONNX via onnx-asr)")
    print("=" * 60)

    audio_duration = get_audio_duration(AUDIO_PATH)
    print(f"\nAudio file: {AUDIO_PATH}")
    print(f"Audio duration: {audio_duration:.2f} seconds")
    print(f"Reference text ({len(REFERENCE_TEXT.split())} words):")
    print(f"  \"{REFERENCE_TEXT[:80]}...\"")

    # Load model
    print("\n[1/4] Loading model via onnx-asr...")
    sys.stdout.flush()
    start_load = time.time()

    import onnx_asr
    model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3")
    load_time = time.time() - start_load
    print(f"  Model loaded in {load_time:.2f}s")

    # Transcribe with resource monitoring
    print("\n[2/4] Transcribing audio...")
    sys.stdout.flush()

    monitor = ResourceMonitor(interval=0.3)
    monitor.start()
    start_infer = time.time()

    transcription = model.recognize(AUDIO_PATH)

    end_infer = time.time()
    monitor.stop()

    latency = end_infer - start_infer
    rtf = latency / audio_duration if audio_duration > 0 else 0
    peak_cpu = monitor.get_peak_cpu()
    avg_cpu = monitor.get_avg_cpu()
    peak_mem = monitor.get_peak_memory_mb()

    # Normalize CPU% by core count
    cores = psutil.cpu_count()
    peak_cpu_normalized = min(peak_cpu / cores, 100.0) if cores > 0 else peak_cpu

    transcription = transcription.strip()
    print(f"  Transcription: \"{transcription}\"")
    print(f"  Inference time: {latency:.2f}s")
    print(f"  RTF: {rtf:.4f}")

    if peak_cpu > 0:
        print(f"  Peak CPU (per-core): {peak_cpu_normalized:.1f}% (raw: {peak_cpu:.1f}%)")
    if peak_mem > 0:
        print(f"  Peak memory: {peak_mem:.1f} MB")

    # Compute WER and CER
    print("\n[3/4] Computing WER/CER...")
    sys.stdout.flush()

    # Clean text for evaluation (lowercase, strip punctuation)
    def clean_text(text):
        return jiwer.RemovePunctuation()(text.lower().strip())

    ref_clean = clean_text(REFERENCE_TEXT)
    hyp_clean = clean_text(transcription)

    wer = jiwer.wer(ref_clean, hyp_clean)
    cer = jiwer.cer(ref_clean, hyp_clean)

    print(f"  Reference: \"{ref_clean}\"")
    print(f"  Hypothesis: \"{hyp_clean}\"")
    print(f"  WER: {wer:.4f} ({wer*100:.2f}%)")
    print(f"  CER: {cer:.4f} ({cer*100:.2f}%)")

    # Gather all results
    print("\n[4/4] Saving results...")
    results = {
        "model_id": "nvidia/parakeet-tdt-0.6b-v3 (ONNX via onnx-asr)",
        "engine": "onnx-asr (ONNX Runtime CPU)",
        "model_type": "parakeet-tdt-0.6b-v3",
        "quantization": "FP32 (default ONNX export)",
        "audio_file": AUDIO_PATH,
        "audio_duration_sec": round(audio_duration, 2),
        "reference_text": REFERENCE_TEXT,
        "transcription": transcription,
        "load_time_sec": round(load_time, 2),
        "latency_sec": round(latency, 2),
        "rtf": round(rtf, 4),
        "wer": round(wer, 4),
        "cer": round(cer, 4),
        "peak_cpu_percent": round(peak_cpu_normalized, 1),
        "peak_memory_mb": round(peak_mem, 1),
        "avg_cpu_percent": round(avg_cpu, 1),
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {RESULTS_PATH}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    return results


if __name__ == "__main__":
    evaluate()