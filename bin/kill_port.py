#!/usr/bin/env python3
import sys
import subprocess
import re

def pids_for_port(port: int):
    pids = set()
    # Try lsof first
    try:
        out = subprocess.check_output(["lsof", "-ti", f":{port}"], stderr=subprocess.DEVNULL)
        for line in out.decode().split():
            if line.strip().isdigit():
                pids.add(line.strip())
    except Exception:
        pass
    # Fallback to ss
    try:
        out = subprocess.check_output(["bash", "-lc", f"ss -lptn 'sport = :{port}'"], stderr=subprocess.DEVNULL)
        for m in re.finditer(r"pid=(\d+)", out.decode()):
            pids.add(m.group(1))
    except Exception:
        pass
    return pids

def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    for pid in pids_for_port(port):
        try:
            subprocess.run(["kill", "-9", pid], check=False)
        except Exception:
            pass

if __name__ == "__main__":
    main()
