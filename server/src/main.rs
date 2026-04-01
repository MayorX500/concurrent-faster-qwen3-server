mod batch;
mod streaming;

use anyhow::Result;
use axum::{
    body::Body,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use base64::Engine;
use batch::{BatchEngine, BatchEngineConfig, BatchRequest, VoiceCloneData};
use hound::{SampleFormat, WavSpec, WavWriter};
use qwen3_tts::{Language, SynthesisOptions};
use serde::{Deserialize, Serialize};
use std::{io::Cursor, sync::Arc};
use tokio::sync::{mpsc, oneshot, Semaphore};
use tracing::info;

struct AppState {
    tx: mpsc::Sender<BatchRequest>,
    stream_tx: mpsc::Sender<StreamingRequest>,
    semaphore: Arc<Semaphore>,
    max_batch: usize,
    model_dir: String,
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
    #[serde(default)]
    stream: Option<bool>,
}

fn default_language() -> String { "spanish".into() }

#[derive(Serialize)]
struct HealthResponse { status: &'static str, queue_depth: usize, max_batch: usize }

#[derive(Serialize)]
struct ErrorResponse { error: String }

fn parse_language(s: &str) -> Language {
    match s.to_lowercase().as_str() {
        "spanish" | "es" => Language::Spanish,
        "english" | "en" => Language::English,
        "french" | "fr" => Language::French,
        _ => Language::Spanish,
    }
}

fn audio_to_wav_bytes(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    let spec = WavSpec { channels: 1, sample_rate, bits_per_sample: 16, sample_format: SampleFormat::Int };
    let mut buf = Cursor::new(Vec::new());
    let mut writer = WavWriter::new(&mut buf, spec)?;
    for &s in samples { writer.write_sample((s * 32767.0).clamp(-32768.0, 32767.0) as i16)?; }
    writer.finalize()?;
    Ok(buf.into_inner())
}

fn samples_to_pcm16(samples: &[f32]) -> Vec<u8> {
    let mut pcm = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let v = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
        pcm.extend_from_slice(&v.to_le_bytes());
    }
    pcm
}

fn wav_header(sample_rate: u32, data_len: u32) -> Vec<u8> {
    let mut h = Vec::with_capacity(44);
    h.extend_from_slice(b"RIFF");
    h.extend_from_slice(&(36 + data_len).to_le_bytes());
    h.extend_from_slice(b"WAVE");
    h.extend_from_slice(b"fmt ");
    h.extend_from_slice(&16u32.to_le_bytes());
    h.extend_from_slice(&1u16.to_le_bytes()); // PCM
    h.extend_from_slice(&1u16.to_le_bytes()); // mono
    h.extend_from_slice(&sample_rate.to_le_bytes());
    h.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    h.extend_from_slice(&2u16.to_le_bytes()); // block align
    h.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    h.extend_from_slice(b"data");
    h.extend_from_slice(&data_len.to_le_bytes());
    h
}

async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let queue = state.max_batch - state.semaphore.available_permits();
    Json(HealthResponse { status: "ok", queue_depth: queue, max_batch: state.max_batch })
}

async fn synthesize(State(state): State<Arc<AppState>>, Json(req): Json<SpeechRequest>) -> Response {
    if req.stream.unwrap_or(false) {
        return synthesize_streaming(state, req).await;
    }

    let permit = match state.semaphore.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => return (StatusCode::SERVICE_UNAVAILABLE, Json(ErrorResponse { error: "Queue full".into() })).into_response(),
    };

    let voice_clone = decode_ref_audio(&req);

    let (reply_tx, reply_rx) = oneshot::channel();
    let batch_req = BatchRequest {
        text: req.text, language: parse_language(&req.language), voice_clone,
        options: SynthesisOptions { temperature: req.temperature.unwrap_or(0.7), ..SynthesisOptions::default() },
        reply: reply_tx,
    };

    if state.tx.send(batch_req).await.is_err() {
        drop(permit);
        return (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: "Engine down".into() })).into_response();
    }

    match reply_rx.await {
        Ok(Ok(result)) => {
            drop(permit);
            let duration = result.audio.samples.len() as f32 / result.audio.sample_rate as f32;
            let rtf = duration / result.gen_time_secs;
            info!(duration, gen_time = result.gen_time_secs, rtf, "Done");
            match audio_to_wav_bytes(&result.audio.samples, result.audio.sample_rate) {
                Ok(wav) => (StatusCode::OK, [("content-type", "audio/wav"), ("x-rtf", &format!("{rtf:.2}"))], wav).into_response(),
                Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: format!("{e:#}") })).into_response(),
            }
        }
        Ok(Err(e)) => { drop(permit); (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: format!("{e:#}") })).into_response() }
        Err(_) => { drop(permit); (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: "Dropped".into() })).into_response() }
    }
}

async fn synthesize_streaming(state: Arc<AppState>, req: SpeechRequest) -> Response {
    let model_dir = state.model_dir.clone();
    let language = parse_language(&req.language);
    let text = req.text.clone();
    let temp = req.temperature.unwrap_or(0.7);

    // Use a dedicated streaming worker thread with pre-loaded model
    let (tx, rx) = mpsc::channel::<Result<Vec<u8>, String>>(32);

    // Submit to streaming worker pool
    let stream_req = StreamingRequest { text, language, temperature: temp, tx };
    if let Err(_) = state.stream_tx.send(stream_req).await {
        return (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: "Stream engine down".into() })).into_response();
    }

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    let body = Body::from_stream(stream.map(|r| r.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))));

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "audio/wav")
        .header("transfer-encoding", "chunked")
        .header("x-streaming", "true")
        .body(body)
        .unwrap()
}

struct StreamingRequest {
    text: String,
    language: Language,
    temperature: f64,
    tx: mpsc::Sender<Result<Vec<u8>, String>>,
}

fn start_streaming_worker(model_dir: String) -> mpsc::Sender<StreamingRequest> {
    let (tx, rx) = mpsc::channel::<StreamingRequest>(16);
    let rx = std::sync::Arc::new(std::sync::Mutex::new(rx));

    std::thread::spawn(move || {
        let device = qwen3_tts::auto_device().expect("No device");
        let model = qwen3_tts::Qwen3TTS::from_pretrained(&model_dir, device).expect("Failed to load");
        info!("Streaming worker ready");

        loop {
            let req = {
                let mut guard = rx.lock().unwrap();
                match guard.blocking_recv() {
                    Some(r) => r,
                    None => break,
                }
            };

            let opts = SynthesisOptions { temperature: req.temperature, ..SynthesisOptions::default() };
            let mut session = match model.synthesize_streaming(
                &req.text, qwen3_tts::Speaker::Serena, req.language, opts,
            ) {
                Ok(s) => s,
                Err(e) => { let _ = req.tx.blocking_send(Err(format!("{e}"))); continue; }
            };

            // WAV header
            let header = wav_header(24000, 0xFFFFFFFF);
            if req.tx.blocking_send(Ok(header)).is_err() { continue; }

            loop {
                match session.next_chunk() {
                    Ok(Some(audio)) => {
                        let pcm = samples_to_pcm16(&audio.samples);
                        if req.tx.blocking_send(Ok(pcm)).is_err() { break; }
                    }
                    Ok(None) => break,
                    Err(_) => break,
                }
            }
        }
    });

    tx
}

fn decode_ref_audio(req: &SpeechRequest) -> Option<VoiceCloneData> {
    req.ref_audio.as_ref().and_then(|b64| {
        base64::engine::general_purpose::STANDARD.decode(b64).ok().map(|bytes| {
            let tmp = std::env::temp_dir().join(format!("ref_{}.wav",
                std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
            std::fs::write(&tmp, &bytes).ok();
            VoiceCloneData { ref_audio_path: tmp, ref_text: req.ref_text.clone() }
        })
    })
}

use tokio_stream::StreamExt;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let model_dir = std::env::var("MODEL_DIR").unwrap_or_else(|_| "models/0.6b-base".into());
    let max_batch: usize = std::env::var("MAX_BATCH").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let max_wait_ms: u64 = std::env::var("MAX_WAIT_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(200);
    let port: u16 = std::env::var("PORT").ok().and_then(|v| v.parse().ok()).unwrap_or(8090);

    info!(model_dir = %model_dir, max_batch, max_wait_ms, port, "Starting qwen3-tts-server");

    let tx = BatchEngine::start(BatchEngineConfig { max_batch_size: max_batch, max_wait_ms, model_dir: model_dir.clone() });
    let stream_tx = start_streaming_worker(model_dir.clone());
    let max_inflight = max_batch * 2;

    let state = Arc::new(AppState {
        tx, stream_tx, semaphore: Arc::new(Semaphore::new(max_inflight)), max_batch, model_dir,
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
