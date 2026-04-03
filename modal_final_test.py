"""Final ICL test with ok-mujer-5s.wav — verify duration and no artifacts."""
import modal

app = modal.App("qwen3-tts-final-test")

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu24.04", add_python="3.12")
    .apt_install("cmake", "pkg-config", "libssl-dev", "libasound2-dev", "git", "curl",
                 "libclang-dev", "clang", "sox")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "pip install huggingface-hub[cli] hf-xet",
        "mkdir -p /src/models/0.6b-base",
        'python3 -c "from huggingface_hub import hf_hub_download; '
        "[hf_hub_download('Qwen/Qwen3-TTS-12Hz-0.6B-Base', f, local_dir='/src/models/0.6b-base') "
        "for f in ['model.safetensors','config.json','generation_config.json',"
        "'preprocessor_config.json','tokenizer_config.json','vocab.json','merges.txt',"
        "'speech_tokenizer/model.safetensors','speech_tokenizer/config.json']]\"",
        # Download both ref audios
        "curl -s -o /tmp/ok-mujer.wav https://okbot.apulab.info/static/debug-audio/ok-mujer.wav",
        "curl -s -o /tmp/ok-mujer-5s.wav https://okbot.apulab.info/static/debug-audio/ok-mujer-5s.wav || "
        "sox /tmp/ok-mujer.wav /tmp/ok-mujer-5s.wav trim 0 5",
    )
    .add_local_dir(".", remote_path="/src", ignore=[
        "target", "models", "__pycache__", ".git",
        "*.wav", "*.gguf", "*.onnx", "*.pyc",
    ])
)


@app.function(image=image, gpu="L4", timeout=1200, memory=32768)
def test():
    import subprocess, os, json, time, base64, urllib.request, wave
    os.chdir("/src")
    env = {**os.environ, "CUDA_COMPUTE_CAP": "89"}

    r = subprocess.run(
        ["bash", "-c", "source /root/.cargo/env && cargo build --release --features cuda,flash-attn"],
        capture_output=True, text=True, timeout=1200, env=env
    )
    if r.returncode != 0:
        print(f"BUILD FAILED: {r.stderr[-500:]}")
        return

    server = subprocess.Popen(
        ["/src/target/release/qwen3-tts-server"],
        env={**os.environ, "MODEL_DIR": "/src/models/0.6b-base", "PORT": "8090",
             "MAX_BATCH": "1", "RUST_LOG": "info"},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    time.sleep(45)

    # Try both ref files
    for ref_name in ["ok-mujer-5s.wav", "ok-mujer.wav"]:
        ref_path = f"/tmp/{ref_name}"
        if not os.path.exists(ref_path):
            print(f"SKIP {ref_name} — not found")
            continue

        w = wave.open(ref_path)
        ref_dur = w.getnframes() / w.getframerate()
        w.close()

        ref_wav = open(ref_path, "rb").read()
        ref_b64 = base64.b64encode(ref_wav).decode()
        ref_text = "Hola, buenos días, mi nombre es la asistente virtual"

        print(f"\n=== {ref_name} ({ref_dur:.1f}s) ===")

        # No clone baseline
        req = urllib.request.Request("http://localhost:8090/v1/audio/speech",
            data=json.dumps({"text": "Hola, buenas tardes", "language": "spanish"}).encode(),
            headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=60)
        data = resp.read()
        dur_no = (len(data) - 44) / (24000 * 2)
        print(f"  No clone: {dur_no:.2f}s")

        # ICL clone
        try:
            req = urllib.request.Request("http://localhost:8090/v1/audio/speech",
                data=json.dumps({
                    "text": "Hola, buenas tardes",
                    "language": "spanish",
                    "ref_audio": ref_b64,
                    "ref_text": ref_text,
                }).encode(),
                headers={"Content-Type": "application/json"})
            resp = urllib.request.urlopen(req, timeout=120)
            data_icl = resp.read()
            dur_icl = (len(data_icl) - 44) / (24000 * 2)

            # Verify: samples per frame
            samples = len(data_icl) - 44
            print(f"  ICL clone: {dur_icl:.2f}s ({samples} bytes, {samples//2} samples)")
            print(f"  Ratio: {dur_icl/dur_no:.2f}x vs no-clone")

            # Save for inspection
            with open(f"/tmp/test_{ref_name}", "wb") as f:
                f.write(data_icl)

        except Exception as e:
            print(f"  ICL clone: FAILED - {e}")
            try:
                print(f"  Body: {e.read().decode()[:200]}")
            except:
                pass

    # Get server logs
    server.terminate()
    _, stderr = server.communicate(timeout=5)
    logs = stderr.decode()
    for line in logs.split('\n'):
        if 'ICL' in line:
            print(f"  LOG: {line.strip()}")


@app.local_entrypoint()
def main():
    test.remote()
