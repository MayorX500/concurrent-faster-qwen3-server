"""Benchmark OmniVoice on Modal L4 — compare with qwen3-tts-server."""
import modal

app = modal.App("omnivoice-bench")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.8.0",
        "torchaudio==2.8.0",
        "omnivoice",
    )
)


@app.function(image=image, gpu="L4", timeout=600, memory=32768)
def bench():
    import torch, time, torchaudio, json

    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}")

    from omnivoice import OmniVoice

    t0 = time.time()
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    # VRAM after load
    vram_idle = torch.cuda.max_memory_allocated() / 1024**3
    print(f"VRAM idle: {vram_idle:.2f}GB")

    results = {"gpu": gpu, "load_time": round(load_time, 1), "vram_idle_gb": round(vram_idle, 2)}

    # Warmup
    _ = model.generate(text="Hola", num_step=16)

    texts = {
        "es_short": "Buenos días, le habla el asistente virtual.",
        "es_medium": "Buenos días, le habla el asistente virtual del centro de atención al cliente. ¿En qué puedo ayudarle?",
        "es_long": "Su cita ha sido confirmada para el día martes a las tres de la tarde. Por favor recuerde traer su documento de identidad y llegar con quince minutos de anticipación. ¿Necesita algo más?",
        "en_medium": "Good morning, this is the virtual assistant from the customer service center. How can I help you today?",
    }

    for name, text in texts.items():
        for steps in [16, 32]:
            torch.cuda.reset_peak_memory_stats()
            times = []
            for trial in range(3):
                t0 = time.time()
                audio = model.generate(text=text, num_step=steps)
                elapsed = time.time() - t0
                dur = audio[0].shape[1] / 24000
                times.append({"dur": round(dur, 2), "wall": round(elapsed, 2), "rtf": round(dur / elapsed, 2)})

            vram = torch.cuda.max_memory_allocated() / 1024**3
            avg_rtf = sum(t["rtf"] for t in times) / len(times)
            avg_wall = sum(t["wall"] for t in times) / len(times)
            key = f"{name}_s{steps}"
            results[key] = {"avg_rtf": round(avg_rtf, 2), "avg_wall": round(avg_wall, 2),
                           "vram_gb": round(vram, 2), "trials": times}
            print(f"{key}: {avg_rtf:.2f}x RT, {avg_wall:.2f}s wall, {vram:.2f}GB VRAM")

    # Voice cloning test
    print("\n=== Voice Cloning ===")
    # Generate a ref audio first
    ref = model.generate(text="Esta es una prueba de referencia para clonación de voz.", num_step=32)
    torchaudio.save("/tmp/ref.wav", ref[0], 24000)

    t0 = time.time()
    cloned = model.generate(
        text="Buenos días, le habla el asistente. ¿En qué puedo ayudarle?",
        ref_audio="/tmp/ref.wav",
        num_step=32,
    )
    clone_time = time.time() - t0
    clone_dur = cloned[0].shape[1] / 24000
    results["voice_clone"] = {"dur": round(clone_dur, 2), "wall": round(clone_time, 2),
                              "rtf": round(clone_dur / clone_time, 2)}
    print(f"Voice clone: {clone_dur:.2f}s audio, {clone_time:.2f}s wall, {clone_dur/clone_time:.2f}x RT")

    return results


@app.local_entrypoint()
def main():
    import json
    r = bench.remote()
    print("\n=== RESULTS ===")
    print(json.dumps(r, indent=2))
