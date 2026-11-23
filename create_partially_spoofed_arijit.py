import os
import json
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import resample

def apply_crossfade(orig_audio, fake_audio, start_sample, end_sample, crossfade_len):
    """
    Splices fake_audio into orig_audio between start_sample and end_sample
    with raised-cosine (Hann) crossfade over crossfade_len samples.
    """
    sr = 48000
    # Length of replacement
    fake_len = len(fake_audio)
    
    # We construct the spliced output
    # Left part: orig[:start_sample]
    # Fade out left: orig[start_sample : start_sample + crossfade_len]
    # Fade in fake: fake[:crossfade_len]
    # Pure fake: fake[crossfade_len : fake_len - crossfade_len]
    # Fade out fake: fake[fake_len - crossfade_len:]
    # Fade in right: orig[end_sample - crossfade_len : end_sample]
    # Right part: orig[end_sample:]
    
    # Calculate energy match
    orig_segment = orig_audio[start_sample:end_sample]
    orig_rms = np.sqrt(np.mean(orig_segment**2) + 1e-9)
    fake_rms = np.sqrt(np.mean(fake_audio**2) + 1e-9)
    
    # Scale fake audio to match original segment RMS energy
    gain = (orig_rms / fake_rms) if fake_rms > 0 else 1.0
    # Cap gain to reasonable range (0.3 to 3.0)
    gain = np.clip(gain, 0.3, 3.0)
    scaled_fake = fake_audio * gain
    
    # If fake length is different from target region, we adjust target duration or time-stretch
    # Using librosa time stretch or simple clean splice
    print(f"Target duration: {(end_sample - start_sample)/sr:.3f}s | Fake duration: {fake_len/sr:.3f}s | Energy gain: {gain:.2f}")
    
    # Let's time-stretch fake to fit the exact target word slot perfectly!
    rate = fake_len / (end_sample - start_sample)
    if abs(rate - 1.0) > 0.05:
        print(f"Time-stretching fake by factor {rate:.2f} to fit exact word timing...")
        scaled_fake = librosa.effects.time_stretch(scaled_fake, rate=rate)
    
    # Ensure exact length
    target_len = end_sample - start_sample
    if len(scaled_fake) > target_len:
        scaled_fake = scaled_fake[:target_len]
    elif len(scaled_fake) < target_len:
        scaled_fake = np.pad(scaled_fake, (0, target_len - len(scaled_fake)))
        
    cf = min(crossfade_len, target_len // 4)
    fade_in = 0.5 * (1 - np.cos(np.linspace(0, np.pi, cf)))
    fade_out = 0.5 * (1 + np.cos(np.linspace(0, np.pi, cf)))
    
    result = orig_audio.copy()
    
    # Boundary blending
    # Start boundary
    result[start_sample : start_sample + cf] = (
        orig_audio[start_sample : start_sample + cf] * fade_out +
        scaled_fake[:cf] * fade_in
    )
    # Middle body
    result[start_sample + cf : end_sample - cf] = scaled_fake[cf : target_len - cf]
    # End boundary
    result[end_sample - cf : end_sample] = (
        scaled_fake[target_len - cf :] * fade_out +
        orig_audio[end_sample - cf : end_sample] * fade_in
    )
    
    # Create sample mask
    mask = np.zeros(len(orig_audio), dtype=np.int32)
    mask[start_sample:end_sample] = 1
    
    return result, mask

def main():
    orig_path = r"C:\Users\ratho\Downloads\Arijit Singh About His Voice Changed Truth Arijit Singh.mp3"
    out_dir = r"C:\Users\ratho\CompSpoof-TL\arijit_partial_spoof"
    
    print("Loading original Arijit Singh audio...")
    orig_y, sr = librosa.load(orig_path, sr=48000)
    cf_samples = int(0.025 * sr) # 25 ms crossfade
    
    # -------------------------------------------------------------------------
    # Experiment 1: Phrase Replacement ("Thank you guys", 24.74s - 26.24s)
    # -------------------------------------------------------------------------
    print("\n--- Generating Experiment 1: 'Thank you guys' phrase replacement ---")
    fake1_path = os.path.join(out_dir, "arijit_thank_you_guys_hm.wav")
    fake1_y, fake1_sr = librosa.load(fake1_path, sr=48000)
    
    start_t1, end_t1 = 24.70, 26.24
    s1 = int(start_t1 * sr)
    e1 = int(end_t1 * sr)
    
    spoofed1_y, mask1 = apply_crossfade(orig_y, fake1_y, s1, e1, cf_samples)
    
    out1_path = os.path.join(out_dir, "arijit_partially_spoofed_phrase_thankyou.wav")
    sf.write(out1_path, spoofed1_y, sr)
    print(f"Saved: {out1_path}")
    
    # -------------------------------------------------------------------------
    # Experiment 2: Mid-sentence Word Replacement ("change my voice" -> "fake my voice", 11.55s - 12.65s)
    # -------------------------------------------------------------------------
    print("\n--- Generating Experiment 2: Mid-sentence 'fake my voice' replacement ---")
    fake2_path = os.path.join(out_dir, "arijit_fake_my_voice_hm.wav")
    fake2_y, fake2_sr = librosa.load(fake2_path, sr=48000)
    
    start_t2, end_t2 = 11.55, 12.65
    s2 = int(start_t2 * sr)
    e2 = int(end_t2 * sr)
    
    spoofed2_y, mask2 = apply_crossfade(orig_y, fake2_y, s2, e2, cf_samples)
    
    out2_path = os.path.join(out_dir, "arijit_partially_spoofed_words_fakemyvoice.wav")
    sf.write(out2_path, spoofed2_y, sr)
    print(f"Saved: {out2_path}")
    
    # Also create 16kHz versions for model evaluation
    print("\nExporting 16kHz versions and 20ms frame masks for model evaluation...")
    y1_16k = librosa.resample(spoofed1_y, orig_sr=48000, target_sr=16000)
    y2_16k = librosa.resample(spoofed2_y, orig_sr=48000, target_sr=16000)
    orig_16k = librosa.resample(orig_y, orig_sr=48000, target_sr=16000)
    
    sf.write(os.path.join(out_dir, "arijit_original_16k.wav"), orig_16k, 16000)
    sf.write(os.path.join(out_dir, "arijit_phrase_thankyou_16k.wav"), y1_16k, 16000)
    sf.write(os.path.join(out_dir, "arijit_words_fakemyvoice_16k.wav"), y2_16k, 16000)
    
    # Compute 20ms frame resolution ground truth masks (50 Hz)
    num_frames = int(len(orig_16k) / 320) # 320 samples = 20ms at 16kHz
    frame_mask1 = np.zeros(num_frames, dtype=int)
    frame_mask2 = np.zeros(num_frames, dtype=int)
    
    for f in range(num_frames):
        t_sec = f * 0.020
        if start_t1 <= t_sec <= end_t1:
            frame_mask1[f] = 1
        if start_t2 <= t_sec <= end_t2:
            frame_mask2[f] = 1
            
    meta = {
        "audio_source": orig_path,
        "sample_rate": 48000,
        "total_duration_sec": len(orig_y) / sr,
        "experiments": [
            {
                "id": "exp1_phrase_thankyou",
                "target_text_original": "Thank you guys.",
                "replacement_text": "Thank you guys. (RVC Arijit Clone)",
                "start_sec": start_t1,
                "end_sec": end_t1,
                "duration_sec": round(end_t1 - start_t1, 3),
                "audio_48k": "arijit_partially_spoofed_phrase_thankyou.wav",
                "audio_16k": "arijit_phrase_thankyou_16k.wav",
                "spoofed_frames_20ms": int(frame_mask1.sum()),
                "total_frames_20ms": num_frames
            },
            {
                "id": "exp2_words_fakemyvoice",
                "target_text_original": "change my voice",
                "replacement_text": "fake my voice (RVC Arijit Clone)",
                "start_sec": start_t2,
                "end_sec": end_t2,
                "duration_sec": round(end_t2 - start_t2, 3),
                "audio_48k": "arijit_partially_spoofed_words_fakemyvoice.wav",
                "audio_16k": "arijit_words_fakemyvoice_16k.wav",
                "spoofed_frames_20ms": int(frame_mask2.sum()),
                "total_frames_20ms": num_frames
            }
        ]
    }
    
    with open(os.path.join(out_dir, "spoof_manifest.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Manifest saved to spoof_manifest.json")
    print("\n=== All partially spoofed clips successfully created! ===")

if __name__ == "__main__":
    main()
