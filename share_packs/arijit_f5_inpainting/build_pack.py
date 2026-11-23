import os
import shutil
import zipfile
import json
import numpy as np
import soundfile as sf
import torchaudio
import torch

def create_full_tampered(orig_wav_path, inpaint_wav_path, out_wav_path, target_sr=24000):
    orig, sr = sf.read(orig_wav_path)
    inpaint, sr_inp = sf.read(inpaint_wav_path)

    assert sr == sr_inp == target_sr, f"Sampling rates do not match: {sr} vs {sr_inp}"

    # Target inpaint word interval within snippet: 1.58s to 2.58s
    # Snippet offset in original audio: 10.0s
    fade_dur = 0.025 # 25ms smooth crossfade
    fade_len = int(fade_dur * sr)
    fade_in = 0.5 * (1 - np.cos(np.pi * np.arange(fade_len) / fade_len))
    fade_out = 0.5 * (1 + np.cos(np.pi * np.arange(fade_len) / fade_len))

    s_inp = int(1.58 * sr)
    e_inp = int(2.58 * sr)
    inpaint_seg = inpaint[s_inp:e_inp].copy()

    # Match RMS energy precisely
    full_s = int(11.58 * sr)
    full_e = int(12.58 * sr)
    rms_orig = np.sqrt(np.mean(orig[full_s:full_e]**2))
    rms_inp = np.sqrt(np.mean(inpaint_seg**2))
    if rms_inp > 0:
        inpaint_seg = inpaint_seg * (rms_orig / rms_inp)

    tampered = orig.copy()
    # Crossfade start
    tampered[full_s : full_s + fade_len] = (
        orig[full_s : full_s + fade_len] * fade_out + inpaint_seg[:fade_len] * fade_in
    )
    # Body
    tampered[full_s + fade_len : full_e - fade_len] = inpaint_seg[fade_len : -fade_len]
    # Crossfade end
    tampered[full_e - fade_len : full_e] = (
        inpaint_seg[-fade_len:] * fade_out + orig[full_e - fade_len : full_e] * fade_in
    )

    sf.write(out_wav_path, tampered, sr)
    print(f"Created full tampered clip: {out_wav_path} (length: {len(tampered)/sr:.2f}s)")
    return tampered

def main():
    share_dir = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(share_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    # File paths
    orig_31s_path = "arijit_partial_spoof/arijit_full_31s_ref.wav"
    inpaint_6s_path = "arijit_partial_spoof/f5_results/f5_inpaint_fake_my_voice.wav"
    
    # 1. Real 4-second clip
    file_1 = os.path.join(audio_dir, "1_real_4s_changemyvoice.wav")
    shutil.copyfile("web_demo/audio/arijit_orig_changemyvoice_4s.wav", file_1)

    # 2. F5-TTS 4-second spoofed clip
    file_2 = os.path.join(audio_dir, "2_f5_fake_4s_fakemyvoice.wav")
    shutil.copyfile("web_demo/audio/arijit_f5_inpaint_fakemyvoice_4s.wav", file_2)

    # 3. Real full 31-second clip (save both 24kHz master and 16kHz)
    file_3 = os.path.join(audio_dir, "3_real_full_arijit.wav")
    shutil.copyfile(orig_31s_path, file_3)

    # 4. Full 31-second clip with tempered F5-TTS inpainting
    file_4 = os.path.join(audio_dir, "4_tampered_full_arijit.wav")
    create_full_tampered(orig_31s_path, inpaint_6s_path, file_4, target_sr=24000)

    # Also copy to web_demo/audio for the local dashboard
    shutil.copyfile(file_3, "web_demo/audio/arijit_full_31s_real.wav")
    shutil.copyfile(file_4, "web_demo/audio/arijit_full_31s_tampered.wav")

    # Generate labels metadata
    labels_metadata = {
        "title": "Arijit Singh Partial Deepfake Evaluation & Sharing Pack",
        "description": "Benchmark comparison of real vs. F5-TTS flow-matching neural inpainted speech.",
        "model": "F5-TTS (Non-Autoregressive Flow Matching DiT with Vocos Vocoder)",
        "infilling_technique": "Bidirectional Contextual Conditioning (Zero-shot Inpainting)",
        "clips": [
            {
                "id": "1_real_4s_changemyvoice",
                "filename": "audio/1_real_4s_changemyvoice.wav",
                "display_name": "1. Real 4s Benchmark Clip ('change my voice')",
                "duration_seconds": 4.0,
                "sample_rate_hz": 16000,
                "overall_label": "100% REAL (Bona Fide Ground Truth)",
                "transcript": "So I had to change my voice to kind of cut through the...",
                "intervals": [
                    {
                        "start_time": "0.00s",
                        "end_time": "4.00s",
                        "label": "REAL",
                        "color": "#10b981",
                        "text": "Full 4-second excerpt from original interview"
                    }
                ]
            },
            {
                "id": "2_f5_fake_4s_fakemyvoice",
                "filename": "audio/2_f5_fake_4s_fakemyvoice.wav",
                "display_name": "2. F5-TTS 4s Spoofed Clip ('fake my voice')",
                "duration_seconds": 4.0,
                "sample_rate_hz": 16000,
                "overall_label": "PARTIALLY SPOOFED (25.0% Synthetic)",
                "transcript": "So I had to [fake my voice] to kind of cut through the...",
                "intervals": [
                    {
                        "start_time": "0.00s",
                        "end_time": "1.58s",
                        "duration": "1.58s",
                        "label": "REAL",
                        "color": "#10b981",
                        "text": "So I had to"
                    },
                    {
                        "start_time": "1.58s",
                        "end_time": "2.58s",
                        "duration": "1.00s",
                        "label": "FAKE (F5-TTS Inpainted)",
                        "color": "#ef4444",
                        "text": "fake my voice"
                    },
                    {
                        "start_time": "2.58s",
                        "end_time": "4.00s",
                        "duration": "1.42s",
                        "label": "REAL",
                        "color": "#10b981",
                        "text": "to kind of cut through the"
                    }
                ]
            },
            {
                "id": "3_real_full_arijit",
                "filename": "audio/3_real_full_arijit.wav",
                "display_name": "3. Real Full Monologue (31.15s)",
                "duration_seconds": 31.15,
                "sample_rate_hz": 24000,
                "overall_label": "100% REAL (Full Authentic Interview)",
                "transcript": "...and I have to think, no, I don't know how to lie to me, I don't know what I mean. Today I'm not going to talk about TVI... So I had to change my voice to kind of cut through the thought, the dialogue. But I'm going to make a worse case... Thank you guys. We will see you next time.",
                "intervals": [
                    {
                        "start_time": "0.00s",
                        "end_time": "31.15s",
                        "label": "REAL",
                        "color": "#10b981",
                        "text": "Authentic interview monologue"
                    }
                ]
            },
            {
                "id": "4_tampered_full_arijit",
                "filename": "audio/4_tampered_full_arijit.wav",
                "display_name": "4. Full Monologue with Tampered F5-TTS Word (31.15s)",
                "duration_seconds": 31.15,
                "sample_rate_hz": 24000,
                "overall_label": "PARTIALLY SPOOFED (3.2% Synthetic Word Spliced)",
                "transcript": "...and I have to think, no, I don't know how to lie to me, I don't know what I mean. Today I'm not going to talk about TVI... So I had to [FAKE MY VOICE] to kind of cut through the thought, the dialogue. But I'm going to make a worse case... Thank you guys. We will see you next time.",
                "intervals": [
                    {
                        "start_time": "0.00s",
                        "end_time": "11.58s",
                        "duration": "11.58s",
                        "label": "REAL",
                        "color": "#10b981",
                        "text": "Preceding interview context"
                    },
                    {
                        "start_time": "11.58s",
                        "end_time": "12.58s",
                        "duration": "1.00s",
                        "label": "FAKE (F5-TTS Inpainted)",
                        "color": "#ef4444",
                        "text": "fake my voice"
                    },
                    {
                        "start_time": "12.58s",
                        "end_time": "31.15s",
                        "duration": "18.57s",
                        "label": "REAL",
                        "color": "#10b981",
                        "text": "Remaining authentic interview"
                    }
                ]
            }
        ]
    }

    # Write labels.json
    with open(os.path.join(share_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels_metadata, f, indent=2)

    print("Created audio and labels.json in arijit_share_pack!")

if __name__ == "__main__":
    main()
