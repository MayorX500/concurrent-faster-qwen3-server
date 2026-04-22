# Development Standards — qwen3-tts-server

## Language & Tooling

- Language: Rust (stable, 2021 edition)
- Build: `cargo` with features `cuda`, `flash-attn`
- Cross-compile: Modal H100 targeting sm_89 (L4/L40S)
- Linter: `clippy` (CI enforced)
- Formatter: `rustfmt`

## Project Structure

```
server/src/main.rs      — HTTP server (axum), streaming, sentence split
server/src/batch.rs     — Batch engine, adaptive max_length
vendor/qwen3-tts-rs/    — TTS library (transformer, vocoder, voice clone)
scripts/                — Python benchmarks
docs/                   — Project documentation
models/                 — Model weights (not tracked)
```

## Coding Patterns

- `Arc<Qwen3TTS>` shared across batch + streaming workers
- KV cache: `AnyKVCache` enum (Concat or PreAlloc)
- Streaming: `std::sync::mpsc` for vocoder→forward thread, `tokio::sync::mpsc` for HTTP
- Cross-fade: 48 samples (~2ms) at chunk boundaries
- Early stop: token repetition threshold (6 consecutive identical tokens)
- Sentence split: automatic for voice clone + text >20 words

## Commits

- Conventional Commits: `<type>(<scope>): <subject> (#issue)`
- Types: feat, fix, perf, docs, chore, refactor
- Every commit references an issue

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `models/0.6b-base` | Model path |
| `MAX_BATCH` | `8` | Batch size |
| `MAX_WAIT_MS` | `200` | Batch wait window |
| `PORT` | `8090` | HTTP port |
| `STREAM_MAX_BATCH` | `8` | Max concurrent streams |
| `STREAM_CHUNK_FRAMES` | `6` | Frames per vocoder decode |
| `RUST_LOG` | `info` | Log level |

## Testing

- Compile: `modal run modal_compile.py`
- Bench: `python3 scripts/bench_streaming.py`
- Audio validation: generate samples, listen, compare with previous version
- No automated tests required unless explicitly requested
