import paramiko
import time
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def parallel_fetch(split_name="eval", concurrency=8):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('pragya.iitd.ac.in', port=22, username='ee3231024', password='a3f1f4a1', timeout=15)

    print(f"\n=======================================================")
    print(f"  PARALLEL MULTI-STREAM TRANSFER FOR '{split_name}' (x{concurrency} streams)")
    print(f"=======================================================")

    t0 = time.time()
    
    script = f"""
    TARGET_DIR="/flash/ee/btech/ee3231024/CompSpoof-TL"
    CHUNKS_DIR="$TARGET_DIR/{split_name}_chunks"
    PADUM_CHUNKS="/home/ee/btech/ee3231024/scratch/{split_name}_chunks"
    
    mkdir -p "$CHUNKS_DIR"
    
    echo "1. Fetching chunk manifest from PADUM..."
    ssh -o StrictHostKeyChecking=no ee3231024@10.113.30.11 "ls -1 $PADUM_CHUNKS" > /tmp/{split_name}_parts.txt
    NUM_CHUNKS=$(wc -l < /tmp/{split_name}_parts.txt)
    echo "Total chunks to transfer: $NUM_CHUNKS"
    
    echo "2. Launching {concurrency}-way parallel SCP transfer..."
    cat /tmp/{split_name}_parts.txt | xargs -n 1 -P {concurrency} -I {{}} scp -o StrictHostKeyChecking=no ee3231024@10.113.30.11:$PADUM_CHUNKS/{{}} $CHUNKS_DIR/{{}}
    
    echo "3. Reassembling chunks..."
    cat $CHUNKS_DIR/part_* > $TARGET_DIR/{split_name}.tar.gz
    ls -lh $TARGET_DIR/{split_name}.tar.gz
    
    echo "4. Checking MD5 integrity..."
    LOCAL_MD5=$(md5sum $TARGET_DIR/{split_name}.tar.gz | awk '{{print $1}}')
    REMOTE_MD5=$(ssh -o StrictHostKeyChecking=no ee3231024@10.113.30.11 "cat /home/ee/btech/ee3231024/scratch/{split_name}.md5" | awk '{{print $1}}')
    echo "Pragya MD5: $LOCAL_MD5"
    echo "PADUM  MD5: $REMOTE_MD5"
    
    if [ "$LOCAL_MD5" = "$REMOTE_MD5" ]; then
        echo ">>> MD5 VERIFIED MATCH! Extracting to /flash..."
        cd "$TARGET_DIR"
        tar -xzf {split_name}.tar.gz
        rm -rf "$CHUNKS_DIR" /tmp/{split_name}_parts.txt
        echo "=== Extraction Complete! ==="
        echo "Audio clips:"
        find {split_name}/audio -name '*.wav' 2>/dev/null | wc -l
        echo "Masks:"
        find {split_name}/masks -name '*.npy' 2>/dev/null | wc -l
    else
        echo ">>> ERROR: MD5 MISMATCH!"
        exit 1
    fi
    """

    stdin, stdout, stderr = client.exec_command(script)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    t_total = time.time() - t0

    print(out.strip())
    if err.strip():
        print("STDERR:", err.strip())
    print(f"\n-> Total transfer & extraction time: {t_total:.1f}s ({t_total/60:.2f} min)")

    client.close()

if __name__ == '__main__':
    split = sys.argv[1] if len(sys.argv) > 1 else 'eval'
    conc = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    parallel_fetch(split, conc)
