import paramiko
import time
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def get_client(host='pragya.iitd.ac.in'):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, port=22, username='ee3231024', password='a3f1f4a1', timeout=15)
    return c

print("==================================================")
print("  ACCELERATED MIGRATION: 'train' SPLIT (77k CLIPS)")
print("==================================================")

client = get_client()

compress_train = """ssh ee3231024@10.113.30.11 '
echo "Starting pigz compression of train split..."
tar -cf - -C /home/ee/btech/ee1230446/scratch/CompSpoof-TL/full train | pigz -p 8 > /home/ee/btech/ee3231024/scratch/train.tar.gz
ls -lh /home/ee/btech/ee3231024/scratch/train.tar.gz
md5sum /home/ee/btech/ee3231024/scratch/train.tar.gz > /home/ee/btech/ee3231024/scratch/train.md5
cat /home/ee/btech/ee3231024/scratch/train.md5

echo "Splitting train.tar.gz into 100MB chunks for parallel multi-stream transfer..."
rm -rf /home/ee/btech/ee3231024/scratch/train_chunks
mkdir -p /home/ee/btech/ee3231024/scratch/train_chunks
split -b 100M /home/ee/btech/ee3231024/scratch/train.tar.gz /home/ee/btech/ee3231024/scratch/train_chunks/part_
echo "Total chunks created:"
ls -1 /home/ee/btech/ee3231024/scratch/train_chunks | wc -l
'"""

stdin, stdout, stderr = client.exec_command(compress_train)
out = stdout.read().decode('utf-8', errors='replace')
err = stderr.read().decode('utf-8', errors='replace')

print(out.strip())
if err.strip():
    print("STDERR:", err.strip())

client.close()
