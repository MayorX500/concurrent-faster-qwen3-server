"""Compile qwen3-tts-server with flash-attn on Modal H100.

Targets CUDA_COMPUTE_CAP=89 (Ada Lovelace / L4) so the binary
runs on our production GPU. Saves the binary to a Modal Volume
for download.
"""
import modal

app = modal.App("qwen3-tts-compile")

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu24.04", add_python="3.12")
    .apt_install("cmake", "pkg-config", "libssl-dev", "libasound2-dev",
                 "libclang-dev", "clang", "curl")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    )
)

vol = modal.Volume.from_name("tts-compiled", create_if_missing=True)
src_mount = modal.Mount.from_local_dir(
    ".", remote_path="/src",
    condition=lambda p: not any(x in p for x in ["target/", "models/", "__pycache__", ".git/"]),
)


@app.function(
    image=image, gpu="H100", timeout=2400, memory=65536,
    mounts=[src_mount], volumes={"/out": vol},
)
def compile():
    import subprocess, os, shutil

    os.chdir("/src")

    # Target L4 (Ada Lovelace sm_89)
    env = {**os.environ, "CUDA_COMPUTE_CAP": "89"}

    print("=== Building qwen3-tts-server with flash-attn (target: sm_89 for L4) ===")
    r = subprocess.run(
        ["bash", "-c",
         "source /root/.cargo/env && "
         "cargo build --release --features cuda,flash-attn"],
        env=env, capture_output=True, text=True, timeout=2000,
    )
    print(r.stdout[-2000:] if r.stdout else "")
    if r.returncode != 0:
        print(f"BUILD FAILED:\n{r.stderr[-3000:]}")
        return {"error": "build failed", "stderr": r.stderr[-3000:]}

    binary = "/src/target/release/qwen3-tts-server"
    if not os.path.exists(binary):
        print("Binary not found at expected path, checking...")
        for f in os.listdir("/src/target/release/"):
            if not f.startswith(".") and os.access(f"/src/target/release/{f}", os.X_OK):
                print(f"  executable: {f}")
        return {"error": "binary not found"}

    size = os.path.getsize(binary)
    print(f"Binary size: {size / 1024 / 1024:.1f} MB")

    # Copy to volume
    shutil.copy2(binary, "/out/qwen3-tts-server")
    vol.commit()
    print("Binary saved to volume tts-compiled:/qwen3-tts-server")

    return {"status": "ok", "size_mb": round(size / 1024 / 1024, 1)}


@app.function(image=image, volumes={"/out": vol})
def download():
    """Return the compiled binary bytes for local download."""
    with open("/out/qwen3-tts-server", "rb") as f:
        return f.read()


@app.local_entrypoint()
def main():
    import json, sys

    if len(sys.argv) > 1 and sys.argv[1] == "download":
        print("Downloading binary from volume...")
        data = download.remote()
        out = "qwen3-tts-server-flash"
        with open(out, "wb") as f:
            f.write(data)
        import os
        os.chmod(out, 0o755)
        print(f"Saved: {out} ({len(data) / 1024 / 1024:.1f} MB)")
    else:
        result = compile.remote()
        print("\n=== RESULT ===")
        print(json.dumps(result, indent=2))
        if result.get("status") == "ok":
            print("\nTo download: modal run modal_compile.py download")
