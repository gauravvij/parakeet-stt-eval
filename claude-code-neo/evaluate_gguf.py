#!/usr/bin/env python3
"""
Evaluate mudler/parakeet-cpp-gguf via parakeet.cpp CLI binary.

Runs parakeet-cli as a subprocess, measures:
  - Latency (wall-clock inference time)
  - RTF (Real-Time Factor = latency / audio duration)
  - Peak CPU% and memory (MB) via psutil
  - WER and CER via jiwer

Saves results to: /home/azureuser/stt_eval_ccneo/results_gguf.json
"""

import os
import sys
import json
import time
import subprocess
import threading
import psutil
import jiwer
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_PATH = os.path.join(PROJECT_DIR, "test_audio.wav")
REFERENCE_PATH = os.path.join(PROJECT_DIR, "reference.txt")
RESULTS_PATH = os.path.join(PROJECT_DIR, "results_gguf.json")
PARAKEET_CLI = os.path.join(PROJECT_DIR, "parakeet.cpp", "build", "examples", "cli", "parakeet-cli")
GGUF_MODEL = os.path.join(PROJECT_DIR, "parakeet_gguf_q6k.gguf")

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

class ProcessResourceMonitor:
    """Monitors CPU and memory of a specific process in a background thread."""
    def __init__(self, pid, interval=0.3):
        self.interval = interval
        self.pid = pid
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
        try:
            process = psutil.Process(self.pid)
            while not self._stop.wait(self.interval):
                try:
                    self.cpu_percentages.append(process.cpu_percent())
                    mem_info = process.memory_info()
                    self.memory_mb.append(mem_info.rss / (1024 * 1024))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
        except psutil.NoSuchProcess:
            pass

    def get_peak_cpu(self):
        return max(self.cpu_percentages) if self.cpu_percentages else 0.0

    def get_peak_memory_mb(self):
        return max(self.memory_mb) if self.memory_mb else 0.0

    def get_avg_cpu(self):
        return np.mean(self.cpu_percentages) if self.cpu_percentages else 0.0


def evaluate():
    print("=" * 60)
    print("EVALUATION: Parakeet GGUF (parakeet.cpp via parakeet-cli)")
    print("=" * 60)

    audio_duration = get_audio_duration(AUDIO_PATH)
    print(f"\nAudio file: {AUDIO_PATH}")
    print(f"Audio duration: {audio_duration:.2f} seconds")
    print(f"Reference text ({len(REFERENCE_TEXT.split())} words):")
    print(f"  \"{REFERENCE_TEXT[:80]}...\"")

    # Verify parakeet-cli binary
    if not os.path.exists(PARAKEET_CLI):
        print(f"\nERROR: parakeet-cli not found at {PARAKEET_CLI}")
        sys.exit(1)
    if not os.path.exists(GGUF_MODEL):
        print(f"\nERROR: GGUF model not found at {GGUF_MODEL}")
        sys.exit(1)

    # Build command
    cmd = [
        PARAKEET_CLI,
        "transcribe",
        "--model", GGUF_MODEL,
        "--input", AUDIO_PATH,
        "--threads", "2",
    ]
    print(f"\n[1/4] Running command: {' '.join(cmd)}")
    sys.stdout.flush()

    # Run with resource monitoring
    print("\n[2/4] Transcribing audio...")
    sys.stdout.flush()

    start_infer = time.time()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    monitor = ProcessResourceMonitor(process.pid, interval=0.3)
    monitor.start()

    stdout, stderr = process.communicate()
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

    # Parse transcription from stdout
    stdout_clean = stdout.strip()
    stderr_clean = stderr.strip()

    print(f"  Inference completed in {latency:.2f}s")
    print(f"  RTF: {rtf:.4f}")
    if peak_cpu > 0:
        print(f"  Peak CPU (per-core): {peak_cpu_normalized:.1f}% (raw: {peak_cpu:.1f}%)")
    if peak_mem > 0:
        print(f"  Peak memory: {peak_mem:.1f} MB")

    if stderr_clean:
        print(f"  stderr: {stderr_clean[:300]}")

    # parakeet-cli outputs transcription directly to stdout
    transcription = stdout_clean

    # The parakeet-cli may output timing info or other metadata alongside transcription
    # Try to extract just the transcription by looking for non-timing lines
    lines = stdout_clean.split('\n')
    # Filter out lines that are clearly not transcription (timestamps, timing info)
    transcription_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip lines that contain metadata markers
        if line.startswith('{') and line.endswith('}'):
            continue
        if line.startswith('[') and line.endswith(']'):
            continue
        if 'ms' in line or 'time' in line.lower():
            continue
        if 'threads' in line.lower() or 'model' in line.lower():
            continue
        # Skip numerical-only lines or very short lines that look like debug output
        if line.replace('.', '').replace(',', '').strip().isdigit():
            continue
        transcription_lines.append(line)

    if transcription_lines:
        transcription = ' '.join(transcription_lines)
    else:
        # If we couldn't parse specific transcription lines, use full stdout minus stderr-like content
        transcription = stdout_clean

    transcription = transcription.strip()
    print(f"\n  Transcription: \"{transcription[:200]}\"")

    # Compute WER and CER
    print("\n[3/4] Computing WER/CER...")
    sys.stdout.flush()

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

    # Check if transcription is sensible
    if not transcription or len(transcription) < 3:
        print("\n  ⚠ WARNING: Transcription appears empty or too short!")
        print(f"  Full stdout: {stdout_clean[:500]}")
        if stderr:
            print(f"  Full stderr: {stderr_clean[:500]}")

    # Save results
    print("\n[4/4] Saving results...")
    results = {
        "model_id": "mudler/parakeet-cpp-gguf (parakeet.cpp via parakeet-cli)",
        "engine": "parakeet.cpp (ggml)",
        "model_type": "tdt-0.6b-v3",
        "quantization": "Q6_K",
        "audio_file": AUDIO_PATH,
        "audio_duration_sec": round(audio_duration, 2),
        "reference_text": REFERENCE_TEXT,
        "transcription": transcription,
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