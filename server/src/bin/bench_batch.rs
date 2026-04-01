use qwen3_tts::{Qwen3TTS, Language, SynthesisOptions};
use std::time::Instant;

fn bench(model: &Qwen3TTS, n: usize) {
    let requests: Vec<(String, Language, Option<SynthesisOptions>)> = (0..n)
        .map(|i| (
            format!("Buenos días, le habla el asistente número {}. ¿En qué puedo ayudarle?", i+1),
            Language::Spanish,
            None,
        ))
        .collect();

    println!("\n=== BATCH x{n} ===");
    let t0 = Instant::now();
    match model.synthesize_batch(&requests) {
        Ok(audios) => {
            let elapsed = t0.elapsed().as_secs_f32();
            let total_audio: f32 = audios.iter()
                .map(|a| a.samples.len() as f32 / a.sample_rate as f32)
                .sum();
            let per_req = elapsed / n as f32;
            println!("OK: {:.1}s audio, {:.1}s wall, {:.2}x RT, {:.1}s/req",
                total_audio, elapsed, total_audio / elapsed, per_req);
        }
        Err(e) => println!("FAILED: {:#}", e),
    }
}

fn main() {
    let device = qwen3_tts::auto_device().unwrap();
    println!("Device: {:?}", device);
    let model = Qwen3TTS::from_pretrained("models/0.6b-base", device).unwrap();
    println!("Model loaded");

    let _ = model.synthesize_with_voice("Hola", qwen3_tts::Speaker::Serena, Language::Spanish, None);
    println!("Warmup done");

    for n in [1, 2, 4, 8] {
        bench(&model, n);
    }

    // Sequential baseline
    println!("\n=== SEQUENTIAL x4 ===");
    let t0 = Instant::now();
    let mut total_audio = 0.0f32;
    for i in 0..4 {
        let text = format!("Buenos días, le habla el asistente número {}. ¿En qué puedo ayudarle?", i+1);
        let a = model.synthesize_with_voice(&text, qwen3_tts::Speaker::Serena, Language::Spanish, None).unwrap();
        total_audio += a.samples.len() as f32 / a.sample_rate as f32;
    }
    let elapsed = t0.elapsed().as_secs_f32();
    println!("OK: {:.1}s audio, {:.1}s wall, {:.2}x RT", total_audio, elapsed, total_audio / elapsed);
}
