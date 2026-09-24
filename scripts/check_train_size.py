import paramiko
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('pragya.iitd.ac.in', port=22, username='ee3231024', password='a3f1f4a1', timeout=15)

cmd = """ssh -o StrictHostKeyChecking=no ee3231024@10.113.30.11 '
du -sb /home/ee/btech/ee1230446/scratch/CompSpoof-TL/full/train/masks /home/ee/btech/ee1230446/scratch/CompSpoof-TL/full/train/audio
'"""

stdin, stdout, stderr = c.exec_command(cmd)
print(stdout.read().decode('utf-8', errors='replace'))
c.close()
