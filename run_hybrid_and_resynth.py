import os
import torch
import soundfile as sf
import librosa
import fairseq.data.dictionary
from rvc_python.infer import RVCInference

torch.serialization.add_safe_globals([fairseq.data.dictionary.Dictionary])

out_dir = r"C:\Users\ratho\CompSpoof-TL\arijit_partial_spoof"
model_path = r"C:\Users\ratho\Downloads\Arijit Singh (RVC) (Crepe v2) 450 Epoch\arijit.pth"
index_path = r"C:\Users\ratho\Downloads\Arijit Singh (RVC) (Crepe v2) 450 Epoch\added_IVF1040_Flat_nprobe_1_v2.index"
orig_mp3 = r"C:\Users\ratho\Downloads\Arijit Singh About His Voice Changed Truth Arijit Singh.mp3"

# 1. Extract the actual real speech snippets from the original mp3
y, sr = librosa.load(orig_mp3, sr=40000)
# Real snippet 1: "Thank you guys" (24.70 to 26.24)
real_ty = y[int(24.70 * sr) : int(26.24 * sr)]
real_ty_path = os.path.join(out_dir, "real_snippet_thankyou.wav")
sf.write(real_ty_path, real_ty, sr)

# Real snippet 2: "change my voice" (11.55 to 12.65)
real_cmv = y[int(11.55 * sr) : int(12.65 * sr)]
real_cmv_path = os.path.join(out_dir, "real_snippet_changemyvoice.wav")
sf.write(real_cmv_path, real_cmv, sr)

print("Extracted real snippets.")

# 2. Initialize RVC with index
rvc = RVCInference(device="cpu")
rvc.load_model(model_path, version="v2", index_path=index_path)
rvc.f0method = "pm"
rvc.index_rate = 0.85

conversions = [
    # Hybrid: Pocket-TTS Zero-Shot -> RVC
    (os.path.join(out_dir, "pocket_thank_you_guys.wav"), os.path.join(out_dir, "hybrid_pocket_rvc_thankyou.wav")),
    (os.path.join(out_dir, "pocket_fake_my_voice.wav"), os.path.join(out_dir, "hybrid_pocket_rvc_fakemyvoice.wav")),
    (os.path.join(out_dir, "pocket_change_my_voice.wav"), os.path.join(out_dir, "hybrid_pocket_rvc_changemyvoice.wav")),
    # Neural resynthesis of real snippet: Real -> RVC
    (real_ty_path, os.path.join(out_dir, "neural_resynth_rvc_thankyou.wav")),
    (real_cmv_path, os.path.join(out_dir, "neural_resynth_rvc_changemyvoice.wav"))
]

for src, dst in conversions:
    print(f"Converting {os.path.basename(src)} -> {os.path.basename(dst)}...")
    rvc.infer_file(src, dst)
    print(f"Saved: {dst}")

print("=== All RVC & Hybrid Conversions Done! ===")
