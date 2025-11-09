#!/usr/bin/env python
import sys, subprocess
port = int(sys.argv[1]) if len(sys.argv)>1 else 8000
try:
    out = subprocess.check_output(["/usr/sbin/lsof","-ti",f":{port}"]).decode().strip().splitlines()
except Exception:
    out = subprocess.check_output(["bash","-lc",f"ss -lptn 'sport = :{port}' | awk 'NR>1 {{print $7}}' | sed 's/.*pid=\\([0-9]*\\).*/\\1/'"]).decode().strip().splitlines()
for pid in out:
    if pid:
        subprocess.call(["kill","-9",pid])
