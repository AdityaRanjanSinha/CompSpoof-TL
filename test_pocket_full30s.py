import os
import soundfile as sf
from pocket_tts import TTSModel

out_dir = r"C:\Users\ratho\CompSpoof-TL\arijit_partial_spoof"
full_31s_wav = os.path.join(out_dir, "arijit_full_31s_ref.wav")

# 1. Load Pocket-TTS
print("Loading Kyutai Pocket-TTS model...")
model = TTSModel.load_model()

# 2. Extract speaker conditioning state from the ENTIRE 31-second audio!
print("Extracting speaker conditioning from the ENTIRE 31s audio...")
full_state = model.get_state_for_audio_prompt(full_31s_wav, truncate=False)

# 3. Generate speech
texts = [
    ("pocket_full30s_thankyou", "Thank you guys."),
    ("pocket_full30s_fakemyvoice", "fake my voice"),
    ("pocket_full30s_changemyvoice", "change my voice")
]

for name, text in texts:
    print(f"Generating zero-shot speech with 31s conditioning for: '{text}'...")
    audio_tensor = model.generate_audio(full_state, text)
    audio_np = audio_tensor.detach().cpu().numpy()
    if audio_np.ndim > 1:
        audio_np = audio_np.squeeze()
        
    out_path = os.path.join(out_dir, f"{name}.wav")
    sf.write(out_path, audio_np, model.sample_rate)
    print(f"Saved: {out_path} ({len(audio_np)/model.sample_rate:.2f}s, {model.sample_rate}Hz)")

print("=== Full 31s Pocket-TTS Generation Complete! ===")
