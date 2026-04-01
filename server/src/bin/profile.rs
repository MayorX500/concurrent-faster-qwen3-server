use qwen3_tts::{Qwen3TTS, Language, SynthesisOptions};
use std::time::Instant;

fn main() {
    let device = qwen3_tts::auto_device().unwrap();
    let model = Qwen3TTS::from_pretrained("models/0.6b-base", device).unwrap();
    println!("Model loaded");

    let text = "Buenos días, le habla el asistente virtual. ¿En qué puedo ayudarle?";

    // Warmup with synthesize (not synthesize_with_voice)
    let _ = model.synthesize("Hola", None);
    println!("Warmup done");

    // Streaming TTFA
    println!("\n=== STREAMING TTFA ===");
    let t0 = Instant::now();
    let mut stream = model.synthesize_streaming(
        text, qwen3_tts::Speaker::Serena, Language::Spanish,
        SynthesisOptions::default(),
    ).unwrap();

    let mut first = true;
    let mut chunks = 0;
    let mut total_samples = 0;
    loop {
        match stream.next_chunk() {
            Ok(Some(audio)) => {
                if first {
                    println!("TTFA: {:.0}ms", t0.elapsed().as_millis());
                    first = false;
                }
                chunks += 1;
                total_samples += audio.samples.len();
            }
            Ok(None) => break,
            Err(e) => { println!("Error: {e}"); break; }
        }
    }
    let total = t0.elapsed().as_secs_f32();
    let dur = total_samples as f32 / 24000.0;
    println!("{chunks} chunks, {dur:.2}s audio, {total:.2}s wall, TTFA shown above");
}
