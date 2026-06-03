#!/usr/bin/env python3
"""
Evaluation: nvidia/parakeet-tdt-0.6b-v3 vs mudler/parakeet-cpp-gguf (tdt-0.6b-v3-q4_k)
Extends the Eval-STT STTEvaluator framework.
"""

import subprocess
import sys
import os
import time
import gc
import json
import psutil
import tempfile
import soundfile as sf
import numpy as np

import torch
from jiwer import wer

PARAKEET_CLI = "/home/azureuser/parakeet_cpp/build/examples/cli/parakeet-cli"
GGUF_MODEL   = "/home/azureuser/stt_eval_cconly/models/tdt-0.6b-v3-q4_k.gguf"
RESULTS_OUT  = "/home/azureuser/stt_eval_cconly/parakeet_results.json"

REFERENCE_TEXT = (
    "the stale smell of old beer lingers "
    "it takes heat to bring out the odor "
    "a cold dip restores health and zest "
    "a salt pickle tastes fine with ham "
    "tacos al pastor are my favorite "
    "a zestful food is the hot cross bun"
)
TEST_WAV = "/tmp/test_audio.wav"


# ---------------------------------------------------------------------------
# Audio helper
# ---------------------------------------------------------------------------

def get_test_audio():
    print(f"Using test audio: {TEST_WAV}")
    data, sr = sf.read(TEST_WAV)
    duration = len(data) / sr
    print(f"Duration: {duration:.1f}s  sr={sr}")
    print(f"Reference: {REFERENCE_TEXT}")
    return TEST_WAV, REFERENCE_TEXT, duration


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class ParakeetEvaluator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.results = {}
        print(f"Device: {self.device}")

    # ---- memory helpers -------------------------------------------------------

    def _mem(self):
        cpu = psutil.Process().memory_info().rss / 1024 / 1024
        gpu = torch.cuda.memory_allocated() / 1024 / 1024 if self.device == "cuda" else 0.0
        return cpu, gpu

    # ---- loaders / transcribers -----------------------------------------------

    def _load_parakeet_nemo(self, _model_id):
        from transformers import pipeline as hf_pipeline
        pipe = hf_pipeline(
            "automatic-speech-recognition",
            model="nvidia/parakeet-tdt-0.6b-v3",
            device=self.device,
        )
        return pipe, None

    def _transcribe_parakeet_nemo(self, model, _proc, audio_path, _wf, _sr):
        result = model(audio_path)
        return result["text"].lower().strip()

    def _load_parakeet_gguf(self, _model_id):
        if not os.path.exists(PARAKEET_CLI):
            raise FileNotFoundError(f"parakeet-cli not found at {PARAKEET_CLI}")
        if not os.path.exists(GGUF_MODEL):
            raise FileNotFoundError(f"GGUF model not found at {GGUF_MODEL}")
        return (PARAKEET_CLI, GGUF_MODEL), None

    def _transcribe_parakeet_gguf(self, model, _proc, audio_path, _wf, _sr):
        cli, gguf = model
        threads = str(os.cpu_count() or 2)
        result = subprocess.run(
            [cli, "transcribe", "--model", gguf, "--input", audio_path,
             "--threads", threads],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())
        return result.stdout.strip().lower()

    # ---- main evaluation loop -------------------------------------------------

    models_cfg = {
        "parakeet-tdt-0.6b-v3-hf": (
            "nvidia/parakeet-tdt-0.6b-v3",
            "_load_parakeet_nemo",
            "_transcribe_parakeet_nemo",
        ),
        "parakeet-tdt-0.6b-v3-gguf-q4k": (
            GGUF_MODEL,
            "_load_parakeet_gguf",
            "_transcribe_parakeet_gguf",
        ),
    }

    def evaluate_model(self, name, audio_path, ref_text, audio_duration):
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")

        model_id, load_fn, xc_fn = self.models_cfg[name]
        load_func = getattr(self, load_fn)
        xc_func   = getattr(self, xc_fn)

        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        cpu0, gpu0 = self._mem()

        # --- load ---
        t0 = time.time()
        try:
            model, proc = load_func(model_id)
        except Exception as e:
            print(f"  Load failed: {e}")
            self.results[name] = {"status": "failed", "error": str(e)}
            return
        load_time = time.time() - t0
        cpu1, gpu1 = self._mem()
        print(f"  Model loaded in {load_time:.1f}s")

        # --- inference ---
        data, sr = sf.read(audio_path)
        waveform = None  # passed to transcribers but only CLI path needs wav file
        t0 = time.time()
        try:
            transcription = xc_func(model, proc, audio_path, waveform, sr)
        except Exception as e:
            print(f"  Inference failed: {e}")
            self.results[name] = {"status": "failed", "error": str(e)}
            return
        inf_time = time.time() - t0

        rtf   = inf_time / audio_duration
        speed = 1.0 / rtf if rtf > 0 else 0.0
        error = wer(ref_text, transcription)

        print(f"  Transcription : {transcription[:120]}")
        print(f"  WER           : {error:.3f}")
        print(f"  Inference     : {inf_time:.2f}s  RTF={rtf:.3f}  Speed={speed:.1f}x")
        print(f"  CPU mem delta : {cpu1 - cpu0:.0f} MB")

        self.results[name] = {
            "status":         "success",
            "wer":            round(error, 4),
            "rtf":            round(rtf, 4),
            "latency_s":      round(inf_time, 3),
            "cpu_memory_mb":  round(cpu1 - cpu0, 1),
            "gpu_memory_mb":  round(gpu1 - gpu0, 1),
            "speed":          round(speed, 2),
            "loading_time_s": round(load_time, 2),
            "transcription":  transcription,
        }

        # cleanup
        del model, proc
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def run(self, audio_path, ref_text, audio_duration):
        for name in self.models_cfg:
            self.evaluate_model(name, audio_path, ref_text, audio_duration)

    def print_summary(self):
        print(f"\n{'='*90}")
        print(f"{'Model':<40} {'WER':>6} {'RTF':>6} {'Latency':>9} {'CPU MB':>8} {'Speed':>7} Status")
        print(f"{'-'*90}")
        for name, r in self.results.items():
            if r["status"] == "success":
                print(f"{name:<40} {r['wer']:>6.3f} {r['rtf']:>6.3f} "
                      f"{r['latency_s']:>9.2f} {r['cpu_memory_mb']:>8.0f} "
                      f"{r['speed']:>7.1f} {r['status']}")
            else:
                print(f"{name:<40} {'N/A':>6} {'N/A':>6} {'N/A':>9} {'N/A':>8} {'N/A':>7} FAILED: {r['error']}")
        print(f"{'='*90}")

    def save_results(self, path=RESULTS_OUT):
        payload = []
        for name, r in self.results.items():
            entry = {"model": name}
            entry.update(r)
            payload.append(entry)
        with open(path, "w") as f:
            json.dump({"models": payload}, f, indent=2)
        print(f"\nResults saved to {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    audio_path, ref_text, duration = get_test_audio()
    evaluator = ParakeetEvaluator()
    evaluator.run(audio_path, ref_text, duration)
    evaluator.print_summary()
    evaluator.save_results()
