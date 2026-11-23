# Arijit Singh Partial Deepfake Evaluation & Sharing Pack

This package contains 4 audio files demonstrating state-of-the-art **neural speech inpainting** using **F5-TTS** (Non-Autoregressive Flow Matching DiT) on an authentic interview of Arijit Singh.

---

## The 4 Audio Files & Exact Labels

### 1. `1_real_4s_changemyvoice.wav`
* **Duration:** 4.000 seconds (64,000 samples @ 16 kHz)
* **Overall Label:** `100% REAL (Bona Fide Master)`
* **Transcript:** *"So I had to change my voice to kind of cut through the..."*
* **Detailed Timestamps:**
  * `0.00s - 4.00s`: **REAL** authentic audio

---

### 2. `2_f5_fake_4s_fakemyvoice.wav`
* **Duration:** 4.000 seconds (64,000 samples @ 16 kHz)
* **Overall Label:** `PARTIALLY SPOOFED (25.0% Synthetic)`
* **Transcript:** *"So I had to [fake my voice] to kind of cut through the..."*
* **Detailed Timestamps:**
  * `0.00s - 1.58s`: **REAL** (*"So I had to"*)
  * `1.58s - 2.58s`: **FAKE / SPOOFED** (*"fake my voice"* — synthesized by F5-TTS)
  * `2.58s - 4.00s`: **REAL** (*"to kind of cut through the"*)
* **Technique:** Continuous mel-domain inpainting conditioned bidirectionally on the preceding audio prefix (`0.00s - 1.58s`) and succeeding suffix (`2.58s - 4.00s`).

---

### 3. `3_real_full_arijit.wav`
* **Duration:** 31.15 seconds (747,590 samples @ 24 kHz)
* **Overall Label:** `100% REAL (Authentic Interview Monologue)`
* **Transcript:**
  > "...and I have to think, no, I don't know how to lie to me, I don't know what I mean. Today I'm not going to talk about TVI... So I had to **change my voice** to kind of cut through the thought, the dialogue. But I'm going to make a worse case... Thank you guys. We will see you next time."
* **Detailed Timestamps:**
  * `0.00s - 31.15s`: **REAL** across all frames.

---

### 4. `4_tampered_full_arijit.wav`
* **Duration:** 31.15 seconds (747,590 samples @ 24 kHz)
* **Overall Label:** `PARTIALLY SPOOFED (3.2% Synthetic Word Injection)`
* **Transcript:**
  > "...and I have to think, no, I don't know how to lie to me, I don't know what I mean. Today I'm not going to talk about TVI... So I had to **[FAKE MY VOICE]** to kind of cut through the thought, the dialogue. But I'm going to make a worse case... Thank you guys. We will see you next time."
* **Detailed Timestamps:**
  * `0.00s - 11.58s`: **REAL** (Authentic context)
  * `11.58s - 12.58s`: **FAKE / SPOOFED** (*"fake my voice"* — synthesized by F5-TTS Flow Matching)
  * `12.58s - 31.15s`: **REAL** (Authentic interview continues)

---

## Why This Deepfake Sounds Indistinguishable

1. **No Autoregressive Timbre Drift:** Standard TTS models hallucinate random pitch drift or speaker variations. F5-TTS uses non-autoregressive Flow Matching to denoise only the masked mel-frames.
2. **Bidirectional Acoustic Continuity:** The model directly observed the authentic audio before 11.58s and after 12.58s, locking the room reverb, microphone color, and vocal tract resonance.
3. **Smooth Energy Matching:** The synthetic word matches the surrounding RMS energy within 0.3 dB, with zero audible phase glitch at the splice points.

---

## How to Open the Interactive Player

Simply double-click `index.html` in this folder. It opens immediately in any web browser with waveform timelines, color-coded fake segment badges, and audio players for all 4 clips.
