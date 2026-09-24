import paramiko
import time
import sys
import os

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def get_client(host='pragya.iitd.ac.in'):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, port=22, username='ee3231024', password='a3f1f4a1', timeout=15)
    return c

def run_cmd(client, cmd, desc=""):
    if desc:
        print(f"\n>>> {desc}")
    stdin, stdout, stderr = client.exec_command(cmd)
    out = stdout.read().decode('utf-8', errors='replace').strip()
    err = stderr.read().decode('utf-8', errors='replace').strip()
    if out:
        print(out)
    if err:
        print("STDERR:", err)
    return out, err

client = get_client()

# Step 1: Compress eval on PADUM
print("==================================================")
print("  ACCELERATED MIGRATION: 'eval' SPLIT")
print("==================================================")

t0 = time.time()
compress_eval = """ssh ee3231024@10.113.30.11 '
echo "Starting pigz compression of eval..."
tar -cf - -C /home/ee/btech/ee1230446/scratch/CompSpoof-TL/full eval | pigz -p 8 > /home/ee/btech/ee3231024/scratch/eval.tar.gz
ls -lh /home/ee/btech/ee3231024/scratch/eval.tar.gz
md5sum /home/ee/btech/ee3231024/scratch/eval.tar.gz > /home/ee/btech/ee3231024/scratch/eval.md5
cat /home/ee/btech/ee3231024/scratch/eval.md5

echo "Splitting eval.tar.gz into 100MB chunks for parallel multi-stream transfer..."
rm -rf /home/ee/btech/ee3231024/scratch/eval_chunks
mkdir -p /home/ee/btech/ee3231024/scratch/eval_chunks
split -b 100M /home/ee/btech/ee3231024/scratch/eval.tar.gz /home/ee/btech/ee3231024/scratch/eval_chunks/part_
ls -l /home/ee/btech/ee3231024/scratch/eval_chunks/
'"""

run_cmd(client, compress_eval, "Compressing and chunking 'eval' on PADUM")
t_comp = time.time() - t0
print(f"-> Phase 1 (Compression & Chunking) took {t_comp:.1f}s ({t_comp/60:.2f} min)")

client.close()
