import os
import torch
import soundfile as sf
from pocket_tts import TTSModel

out_dir = r"C:\Users\ratho\CompSpoof-TL\arijit_partial_spoof"
ref_wav_path = os.path.join(out_dir, "arijit_ref_4s.wav")

# 2. Load Pocket-TTS model
print("Loading Kyutai Pocket-TTS model...")
model = TTSModel.load_model()
print("Model loaded successfully!")

# 3. Extract speaker conditioning state from Arijit's real audio
print("Extracting speaker conditioning from genuine Arijit audio...")
state = model.get_state_for_audio_prompt(ref_wav_path)

# 4. Generate speech in Arijit's voice
texts = [
    ("pocket_thank_you_guys", "Thank you guys."),
    ("pocket_fake_my_voice", "fake my voice"),
    ("pocket_change_my_voice", "change my voice")
]

for name, text in texts:
    print(f"Generating zero-shot speech for: '{text}'...")
    audio_tensor = model.generate_audio(state, text)
    audio_np = audio_tensor.detach().cpu().numpy()
    if audio_np.ndim > 1:
        audio_np = audio_np.squeeze()
        
    out_path = os.path.join(out_dir, f"{name}.wav")
    sf.write(out_path, audio_np, model.sample_rate)
    print(f"Saved: {out_path} ({len(audio_np)/model.sample_rate:.2f}s, {model.sample_rate}Hz)")

print("=== Pocket-TTS Zero-Shot Generation Complete! ===")
