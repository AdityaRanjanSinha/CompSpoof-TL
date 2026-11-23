import os
import torch
import fairseq.data.dictionary
from rvc_python.infer import RVCInference

# Safe globals for torch load
torch.serialization.add_safe_globals([fairseq.data.dictionary.Dictionary])

model_path = r"C:\Users\ratho\Downloads\Arijit Singh (RVC) (Crepe v2) 450 Epoch\arijit.pth"
index_path = r"C:\Users\ratho\Downloads\Arijit Singh (RVC) (Crepe v2) 450 Epoch\added_IVF1040_Flat_nprobe_1_v2.index"
out_dir = r"C:\Users\ratho\CompSpoof-TL\arijit_partial_spoof"

print("Initializing RVC...")
rvc = RVCInference(device="cpu")
print("Loading Arijit Singh model...")
rvc.load_model(model_path, version="v2", index_path=index_path)
rvc.f0method = "pm"  # fast CPU pitch extraction
rvc.index_rate = 0.75

inputs = [
    ("fake_my_voice_hm.wav", "arijit_fake_my_voice_hm.wav"),
    ("fake_my_voice_adam.wav", "arijit_fake_my_voice_adam.wav"),
    ("change_my_voice_hm.wav", "arijit_change_my_voice_hm.wav"),
    ("thank_you_guys_hm.wav", "arijit_thank_you_guys_hm.wav")
]

for in_name, out_name in inputs:
    in_file = os.path.join(out_dir, in_name)
    out_file = os.path.join(out_dir, out_name)
    print(f"Converting {in_name} -> {out_name}...")
    rvc.infer_file(in_file, out_file)
    print(f"Done: {out_file}")

print("All RVC conversions completed!")
