# Changelog

All notable changes to this project will be documented in this file.
Format based on [Conventional Commits](https://www.conventionalcommits.org/).

## v0.7.8 (2026-05-17)

- fix(memory): bucket tensor sizes to multiples of 32 — CUDA allocator stabilizes at ~8GB instead of growing to 22GB+
- fix(security): validate WAV bits_per_sample + atomic temp file creation
- chore: remove prohibited files from tracking, update .gitignore

## v0.7.7 (2026-04-22)

- perf(streaming): STREAM_CHUNK_FRAMES default 3→6 — halves vocoder calls
- 12 CCU streaming at 11.8x RT on L40S
- Deploy 1.7B model on production L40S

## v0.7.6 (2026-04-22)

- feat(streaming): auto sentence-split for voice clone — prevents truncation on long texts
- config: STREAM_MAX_BATCH=12 on production L40S
- docs: API usage guide with Python examples

## v0.7.5 (2026-04-22)

- perf(streaming): batched vocoder decode — all streams in one GPU pass
- fix(streaming): rep_threshold 3→6 to prevent premature cutoff
- 8 concurrent real-time streams on L40S

## v0.7.4 (2026-04-10)

- fix(streaming): stop generation on send failure (stream close delay)
- perf: fast clone stop with AtomicBool signal

## v0.7.3 (2026-04-08)

- perf: fast clone stop — AtomicBool signal, vocoder ctx=4

## v0.7.2 (2026-04-06)

- perf(streaming): early stop on token repetition — 28% faster long phrases

## v0.7.1 (2026-04-06)

- fix(streaming): replace full-context vocoder decode O(n²) with fixed-window O(1) context
- fix(streaming): cross-fade chunk boundaries (48 samples, ~2ms) to eliminate clicks
- perf(streaming): adaptive max_length for streaming (words*4)+30

## v0.7.0 (2026-04-05)

- feat(api): voice_id preload endpoint — zero encoder TTFA
- Bug: fake streaming (full-context vocoder decode)

## v0.5.2 (2026-04-03)

- feat(perf): warmup speaker/speech encoder + transformer at startup
- feat(perf): speaker embedding cache, TTFA header, streaming silence trim
