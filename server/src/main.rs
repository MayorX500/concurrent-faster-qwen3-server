use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use base64::Engine;
use hound::{SampleFormat, WavSpec, WavWriter};
use serde::{Deserialize, Serialize};
use std::{io::Cursor, sync::Arc, time::Instant};
use tokio::sync::{mpsc, oneshot, Semaphore};
use tracing::{info, warn, error};

// Request/response types for the worker channel
struct InferRequest {
    text: String,
    language: String,
    ref_audio: Option<Vec<u8>>,
    ref_text: Option<String>,
    temperature: f64,
    reply: oneshot::Sender<Result<InferResult>>,
}

struct InferResult {
    wav_bytes: Vec<u8>,
    duration: f32,
    gen_time: f32,
}

struct AppState {
    tx: mpsc::Sender<InferRequest>,
    semaphore: Arc<Semaphore>,
    max_workers: usize,
}

#[derive(Deserialize)]
struct SpeechRequest {
    text: String,
    #[serde(default = "default_language")]
    language: String,
    #[serde(default)]
    ref_audio: Option<String>,
    #[serde(default)]
    ref_text: Option<String>,
    #[serde(default)]
    temperature: Option<f64>,
}

fn default_language() -> String {
    "spanish".into()
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    active_workers: usize,
    max_workers: usize,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let active = state.max_workers - state.semaphore.available_permits();
    Json(HealthResponse {
        status: "ok",
        active_workers: active,
        max_workers: state.max_workers,
    })
}

async fn synthesize(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SpeechRequest>,
) -> Response {
    let permit = match state.semaphore.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorResponse { error: "All workers busy".into() }),
            ).into_response();
        }
    };

    let ref_audio = if let Some(b64) = &req.ref_audio {
        match base64::engine::general_purpose::STANDARD.decode(b64) {
            Ok(bytes) => Some(bytes),
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse { error: format!("Invalid base64: {e}") }),
                ).into_response();
            }
        }
    } else {
        None
    };

    let (reply_tx, reply_rx) = oneshot::channel();
    let infer_req = InferRequest {
        text: req.text,
        language: req.language,
        ref_audio,
        ref_text: req.ref_text,
        temperature: req.temperature.unwrap_or(0.7),
        reply: reply_tx,
    };

    if state.tx.send(infer_req).await.is_err() {
        drop(permit);
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse { error: "Worker pool shut down".into() }),
        ).into_response();
    }

    match reply_rx.await {
        Ok(Ok(result)) => {
            drop(permit);
            let rtf = result.duration / result.gen_time;
            info!(duration = result.duration, gen_time = result.gen_time, rtf, "Done");
            (
                StatusCode::OK,
                [
                    ("content-type", "audio/wav"),
                    ("x-rtf", &format!("{rtf:.2}")),
                    ("x-duration", &format!("{:.2}", result.duration)),
                ],
                result.wav_bytes,
            ).into_response()
        }
        Ok(Err(e)) => {
            drop(permit);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse { error: format!("{e:#}") }),
            ).into_response()
        }
        Err(_) => {
            drop(permit);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse { error: "Worker dropped".into() }),
            ).into_response()
        }
    }
}

/// Worker thread — owns the model, processes requests sequentially.
/// Multiple workers run in parallel, each with the same Arc<model> but
/// candle handles GPU serialization internally.
fn run_worker(rx: Arc<std::sync::Mutex<mpsc::Receiver<InferRequest>>>, model_dir: String, worker_id: usize) {
    use qwen3_tts::{AudioBuffer, Language, Qwen3TTS, Speaker, SynthesisOptions};

    let device = qwen3_tts::auto_device().expect("Failed to detect device");
    let model = Qwen3TTS::from_pretrained(&model_dir, device).expect("Failed to load model");
    info!(worker_id, "Worker ready");

    loop {
        // Grab next request from shared channel
        let req = {
            let mut guard = rx.lock().unwrap();
            match guard.blocking_recv() {
                Some(r) => r,
                None => break, // channel closed
            }
        };

        let t0 = Instant::now();
        let lang = match req.language.to_lowercase().as_str() {
            "spanish" | "es" => Language::Spanish,
            "english" | "en" => Language::English,
            "french" | "fr" => Language::French,
            _ => Language::Spanish,
        };

        let opts = SynthesisOptions {
            temperature: req.temperature,
            ..SynthesisOptions::default()
        };

        let result = if let Some(ref_bytes) = &req.ref_audio {
            let tmp = std::env::temp_dir().join(format!("ref_w{}_{}.wav", worker_id, std::process::id()));
            std::fs::write(&tmp, ref_bytes)
                .and_then(|_| Ok(()))
                .map_err(anyhow::Error::from)
                .and_then(|_| {
                    let ref_buf = AudioBuffer::load(&tmp)?;
                    std::fs::remove_file(&tmp).ok();
                    let prompt = model.create_voice_clone_prompt(&ref_buf, req.ref_text.as_deref())?;
                    model.synthesize_voice_clone(&req.text, &prompt, lang, Some(opts))
                })
        } else {
            model.synthesize_with_voice(&req.text, Speaker::Serena, lang, Some(opts))
        };

        let gen_time = t0.elapsed().as_secs_f32();

        let reply = match result {
            Ok(audio) => {
                let duration = audio.samples.len() as f32 / audio.sample_rate as f32;
                match audio_to_wav_bytes(&audio) {
                    Ok(wav_bytes) => Ok(InferResult { wav_bytes, duration, gen_time }),
                    Err(e) => Err(e),
                }
            }
            Err(e) => Err(e),
        };

        let _ = req.reply.send(reply);
    }
    info!(worker_id, "Worker exiting");
}

fn audio_to_wav_bytes(audio: &qwen3_tts::AudioBuffer) -> Result<Vec<u8>> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: audio.sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut buf = Cursor::new(Vec::new());
    let mut writer = WavWriter::new(&mut buf, spec)?;
    for &s in &audio.samples {
        writer.write_sample((s * 32767.0).clamp(-32768.0, 32767.0) as i16)?;
    }
    writer.finalize()?;
    Ok(buf.into_inner())
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let model_dir = std::env::var("MODEL_DIR").unwrap_or_else(|_| "models/0.6b-base".into());
    let workers: usize = std::env::var("WORKERS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let port: u16 = std::env::var("PORT").ok().and_then(|v| v.parse().ok()).unwrap_or(8090);

    info!(model_dir = %model_dir, workers, port, "Starting qwen3-tts-server");

    // Channel for dispatching requests to workers
    let (tx, rx) = mpsc::channel::<InferRequest>(workers * 2);
    let rx = Arc::new(std::sync::Mutex::new(rx));

    // Spawn worker threads — each loads the model independently
    // (candle tensors are not Send, so each worker owns its own model instance)
    // VRAM: ~767MB per worker. With 8 workers on L4 = ~6.1GB
    // TODO: Issue #3 — refactor to shared weights with Arc to reduce VRAM
    for i in 0..workers {
        let rx = rx.clone();
        let dir = model_dir.clone();
        std::thread::spawn(move || run_worker(rx, dir, i));
    }

    let state = Arc::new(AppState {
        tx,
        semaphore: Arc::new(Semaphore::new(workers)),
        max_workers: workers,
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/audio/speech", post(synthesize))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}")).await?;
    info!("Listening on 0.0.0.0:{port}");
    axum::serve(listener, app).await?;
    Ok(())
}
