import sys
import subprocess
from pathlib import Path
from contextlib import contextmanager

@contextmanager
def get_log(log_path):
    if not log_path:
        yield subprocess.DEVNULL
        return
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, 'w', encoding='utf-8')
    try:
        yield f
    finally:
        f.close()

def error(msg, fatal=False):
    print(f"[✘] {msg}")
    if fatal:
        sys.exit(1)

def all_done():
    print(f"[✔] All done")
