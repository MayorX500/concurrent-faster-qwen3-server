# qwen3-tts-server

High-performance Rust TTS server for Qwen3-TTS-12Hz-0.6B-Base. Batched inference with voice cloning, streaming, and flash-attention on NVIDIA GPUs.

## Performance (L4, 23GB)

| Batch | Throughput | Latency/req |
|-------|-----------|-------------|
| 1 | 2.12x RT | 2.4s |
| 8 | 11.49x RT | 0.4s |
| 16 | 16.59x RT | 0.3s |

Streaming TTFA: 450ms. VRAM: 2.7GB idle.

## API

### `POST /v1/audio/speech`

```json
{
  "text": "Buenos días, ¿en qué puedo ayudarle?",
  "language": "spanish",
  "stream": false,
  "temperature": 0.7,
  "ref_audio": "<base64 WAV for voice cloning>",
  "ref_text": "Reference transcript"
}
```

Returns `audio/wav`. With `"stream": true`, returns chunked WAV stream (TTFA ~450ms).

### `GET /health`

```json
{"status": "ok", "queue_depth": 0, "max_batch": 8}
```

### `GET /metrics`

Prometheus-compatible: `tts_requests_total`, `tts_avg_rtf`, `tts_queue_depth`, etc.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `models/0.6b-base` | Path to Qwen3-TTS model |
| `MAX_BATCH` | `8` | Maximum batch size |
| `MAX_WAIT_MS` | `200` | Max wait to fill batch |
| `PORT` | `8090` | HTTP listen port |

## Build

Requires Modal H100 for flash-attn cross-compilation targeting L4 (sm_89):

```bash
modal run modal_compile.py          # compile on H100
modal run modal_compile.py download # download binary
modal run modal_flash_batch.py      # benchmark on L4
```

## Architecture

- Axum HTTP server with batch engine (dedicated thread)
- `Arc<Qwen3TTS>` shared model weights across batch + streaming workers
- Batched transformer forward pass (N sequences per GPU call)
- Batched vocoder decoding (single ONNX pass)
- Adaptive `max_length` for call center text (~6 frames/word)
- OOM recovery with automatic batch splitting

See [DEVELOPMENT.md](DEVELOPMENT.md) for full technical details.
