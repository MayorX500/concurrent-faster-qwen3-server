# Installation Guide

## Prerequisites

- NVIDIA GPU with CUDA 12.x and compute capability >= 8.9 (L4, L40S, A100, H100)
- Minimum 6GB VRAM (8GB+ recommended for 1.7B model)
- Linux x86_64

## Option 1: Download Pre-built Binary

```bash
# Download latest release
curl -L -o qwen3-tts-server \
  "https://github.com/alfonsodg/concurrent-faster-qwen3-server/releases/latest/download/qwen3-tts-server-v0.7.8-linux-x86_64"
chmod +x qwen3-tts-server
```

## Option 2: Build from Source

```bash
# Dependencies (Ubuntu/Debian)
sudo apt install cmake pkg-config libssl-dev libasound2-dev libclang-dev clang

# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone and build
git clone https://github.com/alfonsodg/concurrent-faster-qwen3-server.git
cd concurrent-faster-qwen3-server
CUDA_COMPUTE_CAP=89 cargo build --release --features cuda,flash-attn
# Binary at target/release/qwen3-tts-server
```

## Download Model

```bash
pip install huggingface-hub[cli] hf-xet

# 0.6B (lighter, 2.7GB VRAM)
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --local-dir models/0.6b-base \
  --include "model.safetensors" "config.json" "generation_config.json" \
  "preprocessor_config.json" "tokenizer_config.json" "vocab.json" "merges.txt" \
  "speech_tokenizer/model.safetensors" "speech_tokenizer/config.json"

# 1.7B (better quality, 5.2GB VRAM)
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --local-dir models/1.7b-base \
  --include "model.safetensors" "config.json" "generation_config.json" \
  "preprocessor_config.json" "tokenizer_config.json" "vocab.json" "merges.txt" \
  "speech_tokenizer/model.safetensors" "speech_tokenizer/config.json"
```

## Run

```bash
MODEL_DIR=models/1.7b-base PORT=8090 MAX_BATCH=12 ./qwen3-tts-server
```

## Verify

```bash
curl http://localhost:8090/health
# {"status":"ok","queue_depth":0,"max_batch":12}
```

## Systemd Service

```bash
sudo cp qwen3-tts-server.service /etc/systemd/system/
# Edit paths in service file
sudo systemctl daemon-reload
sudo systemctl enable --now qwen3-tts-server
```
