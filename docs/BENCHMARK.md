# Benchmark Results

**Date:** 2026-04-22
**Model:** Qwen3-TTS-12Hz-1.7B-Base
**GPU:** NVIDIA L40S (46GB)
**Config:** MAX_BATCH=12, STREAM_MAX_BATCH=12, STREAM_CHUNK_FRAMES=6

## Streaming Concurrent

| CCU | TTFA | Wall | Throughput | Real-time |
|-----|------|------|-----------|-----------|
| 1 | 334ms | 4.16s | 1.7x RT | yes |
| 2 | 355ms | 4.67s | 3.0x RT | yes |
| 4 | 376ms | 3.32s | 5.1x RT | yes |
| 8 | 420ms | 4.47s | 9.2x RT | yes |
| 12 | 408ms | 5.21s | 11.8x RT | yes |

## Voice Clone (streaming, denise ref)

| Phrase | Words | TTFA | Total | Audio | RTF |
|--------|-------|------|-------|-------|-----|
| Medium | 21 | 339ms | 5.16s | 8.31s | 0.62x |
| Long | 45 | 337ms | 11.40s | 18.39s | 0.62x |

## Batch Non-Streaming (L40S)

| CCU | Avg latency | Throughput |
|-----|-------------|------------|
| 1 | 4.14s | 2.0x RT |
| 4 | 4.15s | 7.1x RT |
| 8 | 4.65s | 13.1x RT |
| 12 | 4.34s | 20.3x RT |

## Batch Non-Streaming (L4, 0.6B)

| CCU | Throughput |
|-----|------------|
| 1 | 2.12x RT |
| 4 | 6.99x RT |
| 8 | 11.49x RT |
| 16 | 16.59x RT |

## Memory

| Metric | Value |
|--------|-------|
| VRAM idle (1.7B) | 5.2 GB |
| VRAM under load | ~8 GB (stabilized with bucket32) |
| VRAM idle (0.6B) | 2.7 GB |
| RSS | ~580 MB |
| Warmup time | ~12s (1.7B), ~6s (0.6B) |
