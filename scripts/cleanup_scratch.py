import paramiko
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('pragya.iitd.ac.in', port=22, username='ee3231024', password='a3f1f4a1', timeout=15)

cmd = r"""
echo "Cleaning PADUM chunk temp directories..."
ssh -o StrictHostKeyChecking=no ee3231024@10.113.30.11 "rm -rf /home/ee/btech/ee3231024/scratch/*_chunks; ls -lh /home/ee/btech/ee3231024/scratch/"
echo ""
echo "Cleaning Pragya chunk temp directories..."
rm -rf /flash/ee/btech/ee3231024/CompSpoof-TL/*_chunks /tmp/*_parts.txt
ls -lh /flash/ee/btech/ee3231024/CompSpoof-TL/
"""

stdin, stdout, stderr = c.exec_command(cmd)
print(stdout.read().decode('utf-8', errors='replace'))
c.close()
