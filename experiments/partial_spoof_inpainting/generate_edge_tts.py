import os
import asyncio
import edge_tts
import soundfile as sf
import librosa

out_dir = r"C:\Users\ratho\CompSpoof-TL\arijit_partial_spoof"

phrases = [
    ("edge_prabhat_thankyou.mp3", "Thank you guys.", "en-IN-PrabhatNeural"),
    ("edge_prabhat_fakemyvoice.mp3", "fake my voice", "en-IN-PrabhatNeural"),
    ("edge_prabhat_changemyvoice.mp3", "change my voice", "en-IN-PrabhatNeural"),
]

async def generate_all():
    for fname, text, voice in phrases:
        out_path = os.path.join(out_dir, fname)
        print(f"Generating Edge-TTS ({voice}) for '{text}'...")
        communicate = edge_tts.Communicate(text, voice, rate="-5%", pitch="+0Hz")
        await communicate.save(out_path)
        # Convert mp3 to 40kHz wav for RVC
        y, sr = librosa.load(out_path, sr=40000)
        wav_path = out_path.replace(".mp3", ".wav")
        sf.write(wav_path, y, 40000)
        print(f"Saved: {wav_path} ({len(y)/40000:.2f}s)")

asyncio.run(generate_all())
print("All Edge-TTS phrases generated!")
