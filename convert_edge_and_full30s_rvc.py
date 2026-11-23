import os
import torch
import soundfile as sf
import fairseq.data.dictionary
from rvc_python.infer import RVCInference

torch.serialization.add_safe_globals([fairseq.data.dictionary.Dictionary])

out_dir = r"C:\Users\ratho\CompSpoof-TL\arijit_partial_spoof"
model_path = r"C:\Users\ratho\Downloads\Arijit Singh (RVC) (Crepe v2) 450 Epoch\arijit.pth"
index_path = r"C:\Users\ratho\Downloads\Arijit Singh (RVC) (Crepe v2) 450 Epoch\added_IVF1040_Flat_nprobe_1_v2.index"

rvc = RVCInference(device="cpu")
rvc.load_model(model_path, version="v2", index_path=index_path)
rvc.f0method = "rmvpe"
rvc.index_rate = 0.75
rvc.filter_radius = 3
rvc.protect = 0.40

tasks = [
    # Edge-TTS Indian Neural Voice -> Arijit RVC
    (os.path.join(out_dir, "edge_prabhat_thankyou.wav"), os.path.join(out_dir, "arijit_edge_prabhat_rvc_thankyou.wav")),
    (os.path.join(out_dir, "edge_prabhat_fakemyvoice.wav"), os.path.join(out_dir, "arijit_edge_prabhat_rvc_fakemyvoice.wav")),
    (os.path.join(out_dir, "edge_prabhat_changemyvoice.wav"), os.path.join(out_dir, "arijit_edge_prabhat_rvc_changemyvoice.wav")),
    # Full 31s Pocket-TTS -> Arijit RVC
    (os.path.join(out_dir, "pocket_full30s_thankyou.wav"), os.path.join(out_dir, "arijit_pocket_full30s_rvc_thankyou.wav")),
    (os.path.join(out_dir, "pocket_full30s_fakemyvoice.wav"), os.path.join(out_dir, "arijit_pocket_full30s_rvc_fakemyvoice.wav")),
    (os.path.join(out_dir, "pocket_full30s_changemyvoice.wav"), os.path.join(out_dir, "arijit_pocket_full30s_rvc_changemyvoice.wav"))
]

for src, dst in tasks:
    print(f"RVC Converting {os.path.basename(src)} -> {os.path.basename(dst)}...")
    rvc.infer_file(src, dst)
    print(f"Saved: {dst}")

print("=== All Edge-TTS and Full-31s RVC Conversions Done! ===")
