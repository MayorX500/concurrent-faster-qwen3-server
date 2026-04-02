"""Benchmark qts (GGUF Q8) vs our candle server on Modal L4."""
import modal

app = modal.App("qts-q8-bench")

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu24.04", add_python="3.12")
    .apt_install("cmake", "pkg-config", "libssl-dev", "libasound2-dev", "git", "curl",
                 "libclang-dev", "clang")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "git clone https://github.com/yet-another-ai/qts.git /opt/qts",
        "cd /opt/qts && git submodule update --init --recursive",
        # Patch features to CUDA only (upstream defaults include metal+vulkan)
        "cd /opt/qts && sed -i 's/default = \\[\"metal\", \"vulkan\"\\]/default = [\"cuda\"]/' crates/qts_ggml_sys/Cargo.toml",
        "cd /opt/qts && sed -i 's/default = \\[\"metal\", \"vulkan\"\\]/default = []/' crates/qts_ggml/Cargo.toml",
        "cd /opt/qts && sed -i '/^native = /a cuda = [\"qts_ggml_sys/cuda\"]' crates/qts_ggml/Cargo.toml",
        "cd /opt/qts && sed -i 's/default = \\[\"metal\", \"vulkan\", \"coreml\", \"cuda\", \"nvrtx\", \"tensorrt\"\\]/default = [\"cuda\"]/' crates/qts/Cargo.toml crates/qts_cli/Cargo.toml",
        "cd /opt/qts && sed -i 's/cuda = \\[\"ort\\/cuda\"\\]/cuda = [\"ort\\/cuda\", \"qts_ggml\\/cuda\"]/' crates/qts/Cargo.toml",
        'bash -c "source /root/.cargo/env && cd /opt/qts && CUDA_PATH=/usr/local/cuda RUSTFLAGS=\'-L /usr/local/cuda/lib64 -L /usr/local/cuda/lib64/stubs -l cudart -l cublas -l cublasLt -l cuda\' cargo build --release -p qts_cli"',
        "pip install huggingface-hub[cli] hf-xet",
        "mkdir -p /opt/qts/models",
        'python3 -c "from huggingface_hub import hf_hub_download; '
        "[hf_hub_download('dsh0416/Qwen3-TTS-12Hz-0.6B-Base-QTS', f, local_dir='/opt/qts/models') "
        "for f in ['qwen3-tts-0.6b-f16.gguf','qwen3-tts-0.6b-q8_0.gguf','qwen3-tts-vocoder.onnx']]\"",
    )
)


@app.function(image=image, gpu="L4", timeout=600, memory=32768)
def bench():
    import subprocess, time, os, wave

    qts = "/opt/qts/target/release/qts_cli"
    models = "/opt/qts/models"
    text = "Buenos días, le habla el asistente virtual del centro de atención al cliente. ¿En qué puedo ayudarle?"

    results = {}

    for variant, gguf in [("f16", "qwen3-tts-0.6b-f16.gguf"), ("q8", "qwen3-tts-0.6b-q8_0.gguf")]:
        # Set model file
        model_dir_tmp = f"/tmp/models_{variant}"
        os.makedirs(model_dir_tmp, exist_ok=True)
        os.system(f"ln -sf {models}/{gguf} {model_dir_tmp}/qwen3-tts-0.6b-f16.gguf")
        os.system(f"ln -sf {models}/qwen3-tts-vocoder.onnx {model_dir_tmp}/qwen3-tts-vocoder.onnx")

        # Warmup
        subprocess.run([qts, "synthesize", "--model-dir", model_dir_tmp, "--text", "Hola", "--out", "/tmp/w.wav"],
                      capture_output=True, timeout=120)

        # 3 trials
        times = []
        for trial in range(3):
            out = f"/tmp/{variant}_{trial}.wav"
            t0 = time.time()
            r = subprocess.run([qts, "synthesize", "--model-dir", model_dir_tmp, "--text", text, "--out", out],
                              capture_output=True, text=True, timeout=120)
            elapsed = time.time() - t0
            if os.path.exists(out):
                with wave.open(out, 'r') as w:
                    dur = w.getnframes() / w.getframerate()
                times.append({"dur": round(dur, 2), "wall": round(elapsed, 2), "rtf": round(dur/elapsed, 2)})
                print(f"{variant} trial {trial}: {dur:.2f}s audio, {elapsed:.2f}s wall, {dur/elapsed:.2f}x RT")
            else:
                print(f"{variant} trial {trial}: FAILED - {r.stderr[:200]}")

        if times:
            avg_rtf = sum(t["rtf"] for t in times) / len(times)
            avg_wall = sum(t["wall"] for t in times) / len(times)
            results[variant] = {"avg_rtf": round(avg_rtf, 2), "avg_wall": round(avg_wall, 2), "trials": times}

    # GPU info
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total",
                       "--format=csv,noheader"], capture_output=True, text=True)
    results["gpu"] = r.stdout.strip()
    return results


@app.local_entrypoint()
def main():
    import json
    results = bench.remote()
    print("\n=== RESULTS ===")
    print(json.dumps(results, indent=2))
