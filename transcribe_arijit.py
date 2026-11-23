import os
import json
from faster_whisper import WhisperModel

model_path = r"C:\Users\ratho\kokoro_project\whisper_engine\tiny-model"
audio_file = r"C:\Users\ratho\Downloads\Arijit Singh About His Voice Changed Truth Arijit Singh.mp3"

print("Loading Whisper model...")
model = WhisperModel(model_path, device="cpu", compute_type="int8")
print("Transcribing...")
segments, info = model.transcribe(audio_file, word_timestamps=True)

res = []
for s in segments:
    words = [{'word': w.word, 'start': round(w.start, 2), 'end': round(w.end, 2)} for w in s.words]
    res.append({'start': round(s.start, 2), 'end': round(s.end, 2), 'text': s.text, 'words': words})

print(f"Language: {info.language} (Probability: {info.language_probability:.2f})")
for r in res:
    print(f"[{r['start']}s -> {r['end']}s]: {r['text']}")
    for w in r['words']:
        print(f"    {w['start']}s - {w['end']}s: {w['word']}")

with open(r"C:\Users\ratho\CompSpoof-TL\arijit_transcription.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
print("Transcription saved to arijit_transcription.json")
