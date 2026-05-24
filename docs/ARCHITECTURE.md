# Architecture

## Overview

qwen3-tts-server is a high-performance Text-to-Speech server written in Rust.
It wraps the Qwen3-TTS models (0.6B/1.7B params) with an async HTTP layer,
batched GPU inference, and real-time streaming. Runs on NVIDIA GPUs with
flash-attention and CUDA 12.x.

## System Design

```
┌─────────────────────────────────────────────────────────┐
│                    Axum HTTP Server                       │
│  /v1/audio/speech  /v1/embeddings/preload  /health      │
└────────┬──────────────────────┬─────────────────────────┘
         │                      │
    ┌────▼────┐           ┌─────▼─────┐
    │  Batch  │           │ Streaming │
    │ Engine  │           │  Worker   │
    │(thread) │           │ (thread)  │
    └────┬────┘           └─────┬─────┘
         │                      │
    ┌────▼──────────────────────▼─────┐
    │       Arc<Qwen3TTS> (shared)     │
    │  Talker (transformer + KV cache) │
    │  Code Predictor (acoustic codes) │
    │  Decoder12Hz (vocoder)           │
    │  Speaker Encoder (voice clone)   │
    └──────────────────────────────────┘
                    │
              ┌─────▼─────┐
              │  CUDA GPU  │
              │ (L4/L40S)  │
              └────────────┘
```

## Components

### HTTP Layer (server/src/main.rs)
- Axum router with state sharing via `Arc<AppState>`
- Request validation, sentence splitting, sample rate conversion
- Streaming response with chunked WAV or auto sentence-split

### Batch Engine (server/src/batch.rs)
- Dedicated thread collecting requests into batches
- Configurable max batch size and wait window
- OOM recovery with automatic batch splitting
- Adaptive max_length based on word count

### Streaming Worker (server/src/main.rs)
- Batched streaming: multiple streams in one generation loop
- Batched vocoder decode: all streams decoded in single GPU pass
- Cross-fade at chunk boundaries (48 samples)
- Early stop on token repetition (threshold=6)

### TTS Library (vendor/qwen3-tts-rs/)
- Transformer with flash-attention and pre-allocated KV caches
- Code predictor for acoustic token generation
- 12Hz vocoder decoder (codec transformer + upsample stages)
- Speaker encoder (ECAPA-TDNN) for voice cloning
- Speech encoder for ICL voice cloning

## Data Flow

1. Request arrives → validate → check sentence split threshold
2. Batch path: queue → batch engine collects → batched forward pass → vocoder → WAV
3. Stream path: streaming worker → frame-by-frame generation → batched vocoder every 6 frames → chunked response
4. Voice clone: preload encodes speaker embedding once → reused via voice_id

## Memory Management

- Pre-allocated KV caches (PreAlloc on CUDA, Concat fallback)
- bucket32: all tensor sizes rounded to multiples of 32 to reduce CUDA allocator fragmentation
- Voice prompt cache with configurable capacity
