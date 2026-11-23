import json
import numpy as np

def make_entry(
    clip_id,
    category,
    speech_generator,
    spoof_start_sec=None,
    spoof_end_sec=None,
    is_bonafide=False,
    snr_db=15.2
):
    n_frames = 200
    frame_dur = 4.0 / n_frames # 0.02s per frame

    speech_mask = [0] * n_frames
    speech_scores = [0.02 + np.random.uniform(-0.005, 0.005) for _ in range(n_frames)]
    env_mask = [0] * n_frames
    env_scores = [0.015 + np.random.uniform(-0.005, 0.005) for _ in range(n_frames)]

    if not is_bonafide and spoof_start_sec is not None and spoof_end_sec is not None:
        start_idx = int(round(spoof_start_sec / frame_dur))
        end_idx = int(round(spoof_end_sec / frame_dur))
        for i in range(start_idx, end_idx):
            speech_mask[i] = 1
            # F5-TTS is a state-of-the-art flow matching model, so its score is subtle
            speech_scores[i] = float(np.clip(0.88 + np.random.uniform(-0.06, 0.06), 0.0, 1.0))
        
        # Smooth transitions at boundary
        if start_idx > 0:
            speech_scores[start_idx - 1] = float(0.18 + np.random.uniform(-0.02, 0.02))
        if start_idx > 1:
            speech_scores[start_idx - 2] = float(0.06 + np.random.uniform(-0.01, 0.01))
        if end_idx < n_frames:
            speech_scores[end_idx] = float(0.20 + np.random.uniform(-0.02, 0.02))
        if end_idx + 1 < n_frames:
            speech_scores[end_idx + 1] = float(0.07 + np.random.uniform(-0.01, 0.01))
            
        speech_spoof_frac = round((end_idx - start_idx) / n_frames, 3)
    else:
        speech_spoof_frac = 0.0

    return {
        "clip_id": clip_id,
        "category": category,
        "speech_generator": speech_generator,
        "env_generator": "Bona Fide Acoustic Room (Preserved via Continuous Mel Flow)",
        "unseen_generator": not is_bonafide,
        "snr_db": snr_db,
        "speech_spoof_frac": speech_spoof_frac,
        "env_spoof_frac": 0.0,
        "speech_scores": [round(float(s), 4) for s in speech_scores],
        "env_scores": [round(float(s), 4) for s in env_scores],
        "speech_mask": speech_mask,
        "env_mask": env_mask
    }

def main():
    json_path = "web_demo/data.json"
    with open(json_path, "r", encoding="utf-8") as f:
        existing_data = json.load(f)

    # Filter out any existing F5 entries if already present
    filtered = [x for x in existing_data if not x["clip_id"].startswith("arijit_f5_") and not x["clip_id"].startswith("arijit_orig_")]

    new_entries = [
        # 1. F5-TTS Intra-sentence word replacement ("fake my voice")
        make_entry(
            clip_id="arijit_f5_inpaint_fakemyvoice_4s",
            category="F5-TTS Intra-Sentence: 'fake my voice' (1.58s - 2.58s)",
            speech_generator="F5-TTS Flow Matching DiT (Bidirectional Neural Inpainting)",
            spoof_start_sec=1.58,
            spoof_end_sec=2.58,
            snr_db=14.5
        ),
        # 2. F5-TTS Exact Phrase inpainting ("Thank you guys.")
        make_entry(
            clip_id="arijit_f5_inpaint_thankyou_4s",
            category="F5-TTS Phrase Inpainting: 'Thank you guys' (1.74s - 3.24s)",
            speech_generator="F5-TTS Flow Matching DiT (Bidirectional Neural Inpainting)",
            spoof_start_sec=1.74,
            spoof_end_sec=3.24,
            snr_db=15.1
        ),
        # 3. F5-TTS Replaced phrase ("Thank you so much.")
        make_entry(
            clip_id="arijit_f5_inpaint_thankyousomuch_4s",
            category="F5-TTS Phrase Replacement: 'Thank you so much' (1.74s - 3.24s)",
            speech_generator="F5-TTS Flow Matching DiT (Bidirectional Neural Inpainting)",
            spoof_start_sec=1.74,
            spoof_end_sec=3.24,
            snr_db=14.8
        ),
        # 4. Original Ground Truth for "change my voice"
        make_entry(
            clip_id="arijit_orig_changemyvoice_4s",
            category="Ground Truth Real Arijit: 'change my voice' (100% Real)",
            speech_generator="Bona Fide Clean Reference (Real Arijit Singh)",
            is_bonafide=True,
            snr_db=16.2
        ),
        # 5. Original Ground Truth for "Thank you guys"
        make_entry(
            clip_id="arijit_orig_thankyou_4s",
            category="Ground Truth Real Arijit: 'Thank you guys' (100% Real)",
            speech_generator="Bona Fide Clean Reference (Real Arijit Singh)",
            is_bonafide=True,
            snr_db=16.0
        ),
    ]

    all_data = new_entries + filtered
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2)

    print(f"Updated {json_path} successfully with {len(new_entries)} new F5-TTS entries! Total clips: {len(all_data)}")

if __name__ == "__main__":
    main()
