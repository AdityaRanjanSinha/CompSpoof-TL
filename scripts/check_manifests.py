import paramiko
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('pragya.iitd.ac.in', port=22, username='ee3231024', password='a3f1f4a1', timeout=15)

cmd = r"""
echo "=== Listing /scratch/.../full on PADUM ==="
ssh -o StrictHostKeyChecking=no ee3231024@10.113.30.11 "ls -la /scratch/ee/btech/ee1230446/CompSpoof-TL/full/"

echo ""
echo "=== Listing /scratch/.../CompSpoof-TL on PADUM ==="
ssh -o StrictHostKeyChecking=no ee3231024@10.113.30.11 "ls -la /scratch/ee/btech/ee1230446/CompSpoof-TL/"

echo ""
echo "=== Checking sample files in train on Pragya ==="
ls -lh /flash/ee/btech/ee3231024/CompSpoof-TL/train/audio | head -n 5
ls -lh /flash/ee/btech/ee3231024/CompSpoof-TL/train/masks | head -n 5
"""

stdin, stdout, stderr = c.exec_command(cmd)
print(stdout.read().decode('utf-8', errors='replace'))
c.close()
