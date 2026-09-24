import paramiko
import time
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

def get_client():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('pragya.iitd.ac.in', port=22, username='ee3231024', password='a3f1f4a1', timeout=30)
    c.get_transport().set_keepalive(15)
    return c

print("=======================================================")
print("  STEP 1: MONITORING PADUM 'train' CHUNKING")
print("=======================================================")

client = get_client()

# Polling loop until chunks and md5 are ready
while True:
    cmd_check = r"""ssh -o StrictHostKeyChecking=no ee3231024@10.113.30.11 '
    if [ -f /home/ee/btech/ee3231024/scratch/train.md5 ]; then
        CHUNKS=$(ls -1 /home/ee/btech/ee3231024/scratch/train_chunks 2>/dev/null | wc -l)
        IS_RUNNING=$(ps aux | grep -E "split -b 100M|pigz -p 8" | grep -v grep | wc -l)
        echo "MD5_EXISTS=1 CHUNKS=$CHUNKS RUNNING=$IS_RUNNING"
    else
        IS_RUNNING=$(ps aux | grep -E "pigz -p 8" | grep -v grep | wc -l)
        SIZE=$(ls -lh /home/ee/btech/ee3231024/scratch/train.tar.gz 2>/dev/null | awk "{print \$5}")
        echo "MD5_EXISTS=0 SIZE=$SIZE RUNNING=$IS_RUNNING"
    fi
    '"""
    stdin, stdout, stderr = client.exec_command(cmd_check)
    res = stdout.read().decode('utf-8', errors='replace').strip()
    print(f"[{time.strftime('%H:%M:%S')}] {res}")
    
    if "MD5_EXISTS=1" in res and "RUNNING=0" in res:
        print("\n>>> Pigz and split complete! Chunks and MD5 ready on PADUM.")
        break
    
    time.sleep(10)

print("\n=======================================================")
print("  STEP 2: 16-WAY PARALLEL SCP PULL TO PRAGYA /flash")
print("=======================================================")

t0 = time.time()

script = r"""
TARGET_DIR="/flash/ee/btech/ee3231024/CompSpoof-TL"
CHUNKS_DIR="$TARGET_DIR/train_chunks"
PADUM_CHUNKS="/home/ee/btech/ee3231024/scratch/train_chunks"

mkdir -p "$CHUNKS_DIR"

echo "1. Fetching chunk manifest from PADUM..."
ssh -o StrictHostKeyChecking=no ee3231024@10.113.30.11 "ls -1 $PADUM_CHUNKS" > /tmp/train_parts.txt
NUM_CHUNKS=$(wc -l < /tmp/train_parts.txt)
echo "Total chunks to transfer: $NUM_CHUNKS"

echo "2. Launching 16-way parallel SCP transfer..."
cat /tmp/train_parts.txt | xargs -n 1 -P 16 -I {} scp -o StrictHostKeyChecking=no ee3231024@10.113.30.11:$PADUM_CHUNKS/{} $CHUNKS_DIR/{}

echo "3. Reassembling chunks into $TARGET_DIR/train.tar.gz..."
cat $CHUNKS_DIR/part_* > $TARGET_DIR/train.tar.gz
ls -lh $TARGET_DIR/train.tar.gz

echo "4. Checking MD5 integrity against PADUM..."
LOCAL_MD5=$(md5sum $TARGET_DIR/train.tar.gz | awk '{print $1}')
REMOTE_MD5=$(ssh -o StrictHostKeyChecking=no ee3231024@10.113.30.11 "cat /home/ee/btech/ee3231024/scratch/train.md5" | awk '{print $1}')
echo "Pragya MD5: $LOCAL_MD5"
echo "PADUM  MD5: $REMOTE_MD5"

if [ "$LOCAL_MD5" = "$REMOTE_MD5" ]; then
    echo ">>> MD5 VERIFIED MATCH! Extracting to /flash..."
    cd "$TARGET_DIR"
    tar -xzf train.tar.gz
    rm -rf "$CHUNKS_DIR" /tmp/train_parts.txt
    echo "=== Extraction Complete! ==="
    echo "Train Audio clips:"
    find train/audio -name '*.wav' 2>/dev/null | wc -l
    echo "Train Masks:"
    find train/masks -name '*.npy' 2>/dev/null | wc -l
else
    echo ">>> ERROR: MD5 MISMATCH!"
    exit 1
fi
"""

stdin, stdout, stderr = client.exec_command(script)
while not stdout.channel.exit_status_ready():
    if stdout.channel.recv_ready():
        data = stdout.channel.recv(4096).decode('utf-8', errors='replace')
        sys.stdout.write(data)
        sys.stdout.flush()
    time.sleep(0.5)

# read any remaining
sys.stdout.write(stdout.read().decode('utf-8', errors='replace'))
err = stderr.read().decode('utf-8', errors='replace')
if err.strip():
    print("STDERR:", err.strip())

t_total = time.time() - t0
print(f"\n-> Total transfer & extraction time: {t_total:.1f}s ({t_total/60:.2f} min)")

print("\n=======================================================")
print("  STEP 3: FULL DATASET AUDIT ON PRAGYA /flash")
print("=======================================================")
audit_cmd = r"""
TARGET_DIR="/flash/ee/btech/ee3231024/CompSpoof-TL"
echo "--- Split counts ---"
echo "Train audio: $(find $TARGET_DIR/train/audio -name '*.wav' 2>/dev/null | wc -l)"
echo "Train masks: $(find $TARGET_DIR/train/masks -name '*.npy' 2>/dev/null | wc -l)"
echo "Val   audio: $(find $TARGET_DIR/val/audio -name '*.wav' 2>/dev/null | wc -l)"
echo "Val   masks: $(find $TARGET_DIR/val/masks -name '*.npy' 2>/dev/null | wc -l)"
echo "Eval  audio: $(find $TARGET_DIR/eval/audio -name '*.wav' 2>/dev/null | wc -l)"
echo "Eval  masks: $(find $TARGET_DIR/eval/masks -name '*.npy' 2>/dev/null | wc -l)"
echo "--- Total Disk Usage on /flash ---"
du -sh $TARGET_DIR/*
"""
stdin, stdout, stderr = client.exec_command(audit_cmd)
print(stdout.read().decode('utf-8', errors='replace'))

client.close()
