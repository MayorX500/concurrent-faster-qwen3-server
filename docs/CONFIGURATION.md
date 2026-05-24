# Configuration

All configuration is via environment variables. No config files.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `models/0.6b-base` | Path to Qwen3-TTS model directory |
| `PORT` | `8090` | HTTP listen port |
| `MAX_BATCH` | `8` | Maximum batch size for non-streaming |
| `MAX_WAIT_MS` | `200` | Max wait to fill batch before processing (ms) |
| `STREAM_MAX_BATCH` | `8` | Max concurrent streaming requests per batch |
| `STREAM_CHUNK_FRAMES` | `6` | Frames per vocoder decode (~500ms audio) |
| `STREAM_WAIT_MS` | `50` | Wait window to collect streaming batch (ms) |
| `STREAM_POLL_MS` | `5` | Poll interval for streaming batch collection |
| `MAX_REF_AUDIO_BYTES` | `10485760` | Max ref_audio size (10MB) |
| `RUST_LOG` | `info` | Log level (debug, info, warn, error) |
| `CUDA_VISIBLE_DEVICES` | — | GPU device index |

## Production Configuration (L40S, 1.7B)

```bash
MODEL_DIR=/home/ubuntu/qwen3-tts-server/models/1.7b-base
MAX_BATCH=12
MAX_WAIT_MS=500
PORT=8090
STREAM_MAX_BATCH=12
STREAM_CHUNK_FRAMES=6
RUST_LOG=info
CUDA_VISIBLE_DEVICES=0
```

## Development Configuration (L4, 0.6B)

```bash
MODEL_DIR=models/0.6b-base
MAX_BATCH=8
MAX_WAIT_MS=200
PORT=8090
RUST_LOG=info
```
