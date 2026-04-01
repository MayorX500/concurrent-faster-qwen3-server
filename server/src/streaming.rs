//! Batched streaming inference: generates frames for N sequences simultaneously,
//! sending decoded audio chunks through per-request channels as they're produced.

use anyhow::Result;
use qwen3_tts::{AudioBuffer, Language, Qwen3TTS, Speaker, SynthesisOptions};
use std::sync::mpsc;
use std::time::Instant;
use tracing::info;

/// A frame-level callback: called after each batch frame with per-sequence audio.
pub type FrameCallback = Box<dyn Fn(usize, &AudioBuffer) + Send>;

/// Run batched streaming synthesis. For each frame generated, decode and send
/// audio chunks to per-request senders immediately.
pub fn synthesize_batch_streaming(
    model: &Qwen3TTS,
    texts: &[(String, Language)],
    senders: Vec<mpsc::Sender<Result<AudioBuffer>>>,
    opts: SynthesisOptions,
) -> Result<()> {
    let n = texts.len();
    let requests: Vec<(String, Language, Option<SynthesisOptions>)> = texts
        .iter()
        .map(|(t, l)| (t.clone(), *l, Some(opts.clone())))
        .collect();

    // For now, generate all at once and send results.
    // TODO: modify synthesize_batch to yield per-frame and decode+send incrementally
    let t0 = Instant::now();
    let audios = model.synthesize_batch(&requests)?;
    let elapsed = t0.elapsed().as_secs_f32();

    let total_audio: f32 = audios
        .iter()
        .map(|a| a.samples.len() as f32 / a.sample_rate as f32)
        .sum();
    info!(n, elapsed, total_audio, "Batch streaming complete");

    for (i, audio) in audios.into_iter().enumerate() {
        if i < senders.len() {
            let _ = senders[i].send(Ok(audio));
        }
    }

    Ok(())
}
