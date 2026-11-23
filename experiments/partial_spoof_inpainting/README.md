# Partial Speech Spoofing & Neural Inpainting Experiments

This directory contains research scripts and pipelines for synthesizing realistic, partial speech deepfakes on authentic audio without unnatural tone or boundary glitches.

---

## 1. Problem Statement & Motivation
Standard speech deepfake evaluation often pours random AI-generated speech onto authentic recordings at arbitrary timestamps. This creates severe timbre mismatch, acoustic phase jumps, and room tone disconnection, making spoof detection trivially easy.

To benchmark robust detectors against advanced spoofing:
1. Splicing must occur at natural phonetic word/phrase boundaries.
2. The voice generator must condition **bidirectionally** on both preceding prefix and succeeding suffix contexts to maintain room reverb, boundary pitch, and vocal tract resonance.

---

## 2. Experiments & Script Index

### A. F5-TTS Bidirectional Neural Inpainting (State-of-the-Art)
* **`run_f5_speech_edit.py`**:
  Uses non-autoregressive Flow Matching Transformer (Diffusion Transformer / DiT backbone) directly on mel-spectrograms. Generates seamless word/phrase replacements (`"fake my voice"`, `"Thank you so much"`) conditioned symmetrically on both the left and right acoustic context with zero phase pop or room tone break.
* **`process_f5_clips.py`**:
  Resamples inpainted audio to 16 kHz and extracts standardized 4.0-second benchmark clips for CompSpoof-TL frame-level evaluation.

### B. Hybrid TTS + Voice Conversion
* **`run_arijit_rvc.py`**:
  Retrieval-based Voice Conversion (RVC v2) with FAISS feature retrieval using RMVPE, Harvest, and PM pitch extraction algorithms on CPU.
* **`generate_edge_tts.py`**:
  Generates Indian English neural speech (`en-IN-PrabhatNeural`) matching target phrase transcripts.
* **`convert_edge_and_full30s_rvc.py`** & **`run_hybrid_and_resynth.py`**:
  Converts Edge-TTS and Pocket-TTS outputs through RVC v2 and tests neural resynthesis with ambient room tone preservation.

### C. Zero-Shot Cloning & Transcription
* **`transcribe_arijit.py`** & **`arijit_transcription.json`**:
  Whisper-based word-level timestamp alignment on the source audio (`Arijit Singh About His Voice Changed Truth Arijit Singh.mp3`).
* **`test_pocket_tts.py`** & **`test_pocket_full30s.py`**:
  Zero-shot cloning evaluation using Kyutai Pocket-TTS conditioned on 4.5s vs full 31.15s reference audio.

---

## 3. Results & Interactive Visualization
* **Share Pack:** See [`share_packs/arijit_f5_inpainting/`](../../share_packs/arijit_f5_inpainting) for standalone offline HTML player and the 4 benchmark WAV files.
* **Web Dashboard:** See [`web_demo/`](../../web_demo) for the interactive Scenario B localization inspector.
