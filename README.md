# Parakeet STT Evaluation — Claude Code Solo vs Claude Code + Neo

A real-world benchmark comparing two AI engineering workflows on the same task:
evaluating [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
vs [`mudler/parakeet-cpp-gguf`](https://huggingface.co/mudler/parakeet-cpp-gguf)
using the [Eval-STT](https://github.com/build-ai-applications/Eval-STT) framework.

> **Read the full write-up:** [`claude-code-neo/blog_medium.md`](claude-code-neo/blog_medium.md)

---

## The Experiment

| | Claude Code Solo | Claude Code + Neo |
|---|---|---|
| **Total cost** | $1.96 | $0.74 |
| **WER (Model A)** | 20.9% | 4.65% |
| **RTF (Model A)** | 0.519 | 0.328 |
| **Human prompts** | Many iterative turns | 1 prompt + 1 reply |
| **Inference backend chosen** | HF Transformers bfloat16 | ONNX Runtime (researched) |

Same task. Same machine (CPU-only Azure VM, 2 vCPUs, 7.7 GB RAM). Same models. Different approach.

---

## Repository Structure

```
parakeet-stt-eval/
├── claude-code-solo/          # Run 1: Claude Code working interactively
│   ├── evaluate_parakeet.py   # Unified eval script (HF + GGUF)
│   ├── parakeet_evaluation_report.md
│   ├── parakeet_results.json
│   ├── results.json
│   └── requirements.txt
│
└── claude-code-neo/           # Run 2: Claude Code orchestrating Neo
    ├── evaluate_onnx.py       # ONNX Runtime evaluation (Model A)
    ├── evaluate_gguf.py       # parakeet.cpp evaluation (Model B)
    ├── generate_audio.py      # gTTS test audio generation
    ├── COMPARISON_REPORT.md   # Side-by-side metrics report
    ├── blog_medium.md         # Full write-up
    ├── reference.txt          # Harvard sentences reference text
    ├── test_audio.wav         # 16.78s, 16kHz mono test audio
    ├── results.json           # Combined results
    ├── results_onnx.json      # Per-model results (ONNX)
    ├── results_gguf.json      # Per-model results (GGUF)
    ├── plans/
    │   └── plan.md            # Neo's pre-execution research plan
    └── charts/                # Generated comparison visuals
        ├── hero.png
        ├── cost_comparison.png
        ├── metrics_4panel.png
        ├── rtf_comparison.png
        ├── workflow_diagram.png
        └── scorecard.png
```

---

## Reproducing the Evaluation

### Claude Code Solo approach

```bash
cd claude-code-solo
pip install -r requirements.txt
# Requires parakeet-cli binary and GGUF model — see evaluate_parakeet.py for paths
python evaluate_parakeet.py
```

### Claude Code + Neo approach (ONNX)

```bash
cd claude-code-neo
pip install onnx-asr[cpu,hub] jiwer psutil soundfile numpy gtts
python generate_audio.py          # generates test_audio.wav
python evaluate_onnx.py           # runs ONNX Runtime eval
```

### Claude Code + Neo approach (GGUF)

Requires building [parakeet.cpp](https://github.com/mudler/parakeet.cpp) from source and
downloading a Q6_K GGUF checkpoint from [mudler/parakeet-cpp-gguf](https://huggingface.co/mudler/parakeet-cpp-gguf).
See `evaluate_gguf.py` for the expected binary and model paths.

---

## Key Finding

The 37% RTF improvement (0.519 → 0.328) for the same model on the same hardware came from
choosing ONNX Runtime over PyTorch for CPU inference — a decision Neo made after researching
CPU inference benchmarks before writing any code. The interactive run used the most obvious
path (HF Transformers); the agent-orchestrated run used the best path.

That research discipline, applied upfront rather than discovered through iteration, is also
why the agent run cost 62% less.

---

## Tools Used

- [Claude Code](https://claude.ai/code) — AI coding assistant
- [Neo MCP](https://heyneo.so) — Local AI engineering agent
- [Eval-STT](https://github.com/build-ai-applications/Eval-STT) — Evaluation framework
- [onnx-asr](https://github.com/istupakov/onnx-asr) — ONNX Runtime inference for ASR
- [parakeet.cpp](https://github.com/mudler/parakeet.cpp) — C++ GGUF inference runtime
