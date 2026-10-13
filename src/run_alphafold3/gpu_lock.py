import fcntl
import functools
import os
import struct
import subprocess
import time
from contextlib import contextmanager
from tqdm import tqdm

# An exclusive lock on the first byte of the device node, held while waiting for the GPU lock
TURNSTILE = struct.pack("hhqqi", fcntl.F_WRLCK, os.SEEK_SET, 0, 1, 0)
DEF_MAX_WAIT = 12 * 3600
# Seconds between attempts to take a lock: a blocking lock could not give up at a deadline
POLL = 1

class GPUWaitTimeout(Exception):
    pass

@functools.cache
def device_node(gpu):
    """The device node of a GPU, numbered by its minor number, which need not be its index."""
    out = subprocess.check_output(["nvidia-smi", "-i", gpu, "-q"], text=True)
    for line in out.splitlines():
        key, _, val = line.partition(":")
        if key.strip() == "Minor Number" and val.strip().isdigit():
            return f"/dev/nvidia{val.strip()}"
    raise RuntimeError(f"Could not find the device node of GPU {gpu}")

def lock_holder(path):
    """
    The PID that took the GPU lock on the device node, if it is visible from here. The lock may
    outlive it, held by a child it started.
    """
    st = os.stat(path)
    dev = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"
    try:
        with open("/proc/locks") as f:
            for line in f:
                # e.g. "2: FLOCK  ADVISORY  WRITE 12345 00:05:1111 0 EOF"; waiters have "->" after the number
                fields = line.split()
                if fields[1] == "FLOCK" and fields[5] == dev and int(fields[4]) > 0:
                    return int(fields[4])
    except OSError:
        pass
    return None

def duration(seconds):
    for unit, size in (("h", 3600), ("min", 60)):
        if seconds >= size:
            return f"{seconds / size:.3g} {unit}"
    return f"{seconds:.3g} s"

def locked_by(path):
    holder = lock_holder(path)
    if holder is None:
        return ""
    if os.path.exists(f"/proc/{holder}"):
        return f" (locked by PID {holder})"
    return f" (locked by PID {holder}, which has exited, but a process it started still runs)"

@contextmanager
def gpu_lock(gpu, max_wait=DEF_MAX_WAIT):
    """
    Holds a GPU among the processes that take this lock, waiting while another one holds it, for
    up to max_wait seconds, after which GPUWaitTimeout is raised.

    The lock is a flock on the GPU's device node, which every process that can use the GPU sees,
    and the kernel releases it whenever the process dies. It can also be taken from the shell
    with `flock /dev/nvidiaN <command>`.

    Yields the descriptor that holds the lock. Passed to a child with pass_fds, it keeps the GPU
    locked until the child exits too, even if the caller dies first.

    A process that has to wait first queues on a turnstile, an exclusive lock on the first byte of
    the device node. A process that releases the GPU and wants it back has to pass the turnstile,
    which lets the waiting process in first.
    """
    path = device_node(gpu)
    lock_fd = os.open(path, os.O_RDONLY)
    try:
        # A write lock needs a descriptor open for writing
        turnstile_fd = os.open(path, os.O_RDWR)
        try:
            deadline = time.monotonic() + max_wait
            waiting = False
            for take in (lambda: fcntl.fcntl(turnstile_fd, fcntl.F_OFD_SETLK, TURNSTILE),
                         lambda: fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)):
                while True:
                    try:
                        take()
                        break
                    except BlockingIOError:
                        pass
                    if time.monotonic() >= deadline:
                        raise GPUWaitTimeout(f"GPU {gpu} is still busy{locked_by(path)} after {duration(max_wait)} of waiting")
                    if not waiting:
                        tqdm.write(f"[*] GPU {gpu} is busy{locked_by(path)}, waiting")
                        waiting = True
                    time.sleep(POLL)
        finally:
            os.close(turnstile_fd)
        yield lock_fd
    finally:
        # Closing rather than unlocking: a child still holding the descriptor keeps the lock
        os.close(lock_fd)
