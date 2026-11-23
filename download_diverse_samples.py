import os
import paramiko
import json

demo_dir = r"c:\Users\ratho\CompSpoof-TL\web_demo"
audio_dir = os.path.join(demo_dir, "audio")
os.makedirs(audio_dir, exist_ok=True)

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('pragya.iitd.ac.in', username='ee3231024', password='a3f1f4a1')
sftp = ssh.open_sftp()

# Download selected_samples.json
remote_json = "/flash/ee/btech/ee3231024/selected_samples.json"
local_json = os.path.join(demo_dir, "data.json")
print("Downloading data.json...")
sftp.get(remote_json, local_json)

with open(local_json, 'r') as f:
    samples = json.load(f)

for s in samples:
    cid = s['clip_id']
    remote_wav = f"/flash/ee/btech/ee3231024/CompSpoof-TL/eval/audio/{cid}.wav"
    local_wav = os.path.join(audio_dir, f"{cid}.wav")
    print(f"Downloading {cid}.wav...")
    sftp.get(remote_wav, local_wav)

sftp.close()
ssh.close()
print("All audio files downloaded successfully!")
