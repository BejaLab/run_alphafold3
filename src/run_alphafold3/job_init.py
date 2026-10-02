import sqlite3
import argparse
import subprocess
import tempfile
import shutil
import json
import time
from pathlib import Path
from tqdm import tqdm

from run_alphafold3.utils import MEMORY_ENV, SEARCHES_SCHEMA, PREDICTIONS_SCHEMA, get_data_paths, get_cache_paths, detect_compute_gpus
from run_alphafold3.environment import probe_environment, write_environment, gpu_uuids
from run_alphafold3.model_cache import warm_cmd, cache_keys
from run_alphafold3.logger import error, get_log, all_done

DEF_GPUS = detect_compute_gpus()

def gpu_memory(gpu):
    out = subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True)
    return int(out) * 2**20

def host_memory_available():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 2**10

def compile_buckets(buckets, mode, gpu, model_path, cache_path, log, progress_bar, previous=None):
    """
    Compiles the model for the buckets in one container, smallest first, until one does not fit
    into memory. Adds the cache entries of those that fit to cache_path and returns
    {bucket: (cache entry, peak memory)} for them.
    """
    extra = []
    if mode == "unified":
        # Unified memory spills into host RAM
        extra.append(f"--budget={gpu_memory(gpu) + host_memory_available()}")
    if previous:
        extra.append(f"--previous={previous[0]}:{previous[1]}")

    fitted = {}
    gib = lambda size: f"{size / 2**30:.1f}"
    with tempfile.TemporaryDirectory() as tmp_dir:
        start = time.time()
        compile_time = 0
        cmd = warm_cmd(buckets, MEMORY_ENV[mode], gpu, model_path, tmp_dir, extra)
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log, text=True) as proc:
            for line in proc.stdout:
                result = json.loads(line)
                bucket = result["bucket"]
                compile_time += result.get("seconds", 0)
                timing = f" (compile {result['seconds']:.0f} s)" if "seconds" in result else ""
                if result["fits"]:
                    outcome = f"fits, needs {gib(result['peak'])} of {gib(result['budget'])} GiB"
                    fitted[bucket] = result["entry"], result["peak"]
                    progress_bar.update(1)
                elif "estimate" in result:
                    outcome = f"skipped, would need about {gib(result['estimate'])} of {gib(result['budget'])} GiB"
                elif "error" in result:
                    if "RESOURCE_EXHAUSTED" not in result["error"] and "Out of memory" not in result["error"]:
                        error(f"Compilation failed for bucket {bucket}: {result['error']}", fatal=True)
                    outcome = "out of memory while compiling"
                else:
                    outcome = f"does not fit, needs {gib(result['peak'])} of {gib(result['budget'])} GiB"
                tqdm.write(f"    {bucket:>5} tokens, {mode:<7}: {outcome}{timing}")
        if proc.returncode != 0 and len(fitted) < len(buckets):
            # Killed without a Python error, e.g. by the OOM killer when unified memory spills into host RAM
            tqdm.write(f"    {buckets[len(fitted)]:>5} tokens, {mode:<7}: killed (exit code {proc.returncode})")
        overhead = time.time() - start - compile_time
        tqdm.write(f"    {mode} memory container: {time.time() - start:.0f} s, of which {overhead:.0f} s outside compilation")

        # Only the entries of models that fit: a killed container may leave a partial file
        for entry, peak in fitted.values():
            shutil.copy2(Path(tmp_dir) / entry, cache_path / entry)
        autotune_dir = Path(tmp_dir) / "xla_gpu_per_fusion_autotune_cache_dir"
        if autotune_dir.is_dir():
            shutil.copytree(autotune_dir, cache_path / autotune_dir.name, dirs_exist_ok=True)
    return fitted

def launch(data_dir, gpus, log_file, overwrite = False):
    search_db_path, pred_db_path, model_path, public_path = get_data_paths(data_dir, create=True)
    cache_path, env_path = get_cache_paths(data_dir)
    print(f"[*] Initializing the database")
    if search_db_path.is_file() and overwrite:
        search_db_path.unlink()
    if pred_db_path.is_file() and overwrite:
        pred_db_path.unlink()
    with sqlite3.connect(search_db_path) as conn:
        for statement in SEARCHES_SCHEMA:
            conn.execute(statement)
    with sqlite3.connect(pred_db_path) as conn:
        for statement in PREDICTIONS_SCHEMA:
            conn.execute(statement)

    if not model_path.exists():
        error(f"No models at {model_path}", fatal=True)
    if not gpus:
        error("No GPUs given", fatal=True)

    print(f"[*] Probing the environment on GPU(s) {','.join(gpus)}")
    try:
        env = probe_environment(gpus, model_path)
    except RuntimeError as e:
        error(str(e), fatal=True)

    # The cache must hold exactly what environment.txt describes
    env_path.unlink(missing_ok=True)
    shutil.rmtree(cache_path, ignore_errors=True)
    cache_path.mkdir()

    # Smallest bucket first: normal memory until a bucket does not fit, then unified memory
    # from that bucket on, until a bucket does not fit either
    print(f"[*] Compiling the model on GPU {gpus[0]} ({env['gpu']})")
    buckets = [int(b) for b in env["buckets"].split(',')]
    compiled = {}
    with get_log(log_file) as log:
        with tqdm(total=len(buckets)) as progress_bar:
            previous = None
            for mode in ["normal", "unified"]:
                remaining = [b for b in buckets if b not in compiled]
                if not remaining:
                    break
                fitted = compile_buckets(remaining, mode, gpus[0], model_path, cache_path, log, progress_bar, previous)
                for bucket, (entry, peak) in fitted.items():
                    compiled[bucket] = mode, entry
                    previous = bucket, peak

        if not compiled:
            error("The model does not fit into memory even for the smallest bucket", fatal=True)

        # alphafold3_predict computes the cache keys of the GPUs it runs on, to link the compiled
        # model under them: the key of this GPU must match the entry just compiled
        start = time.time()
        smallest = min(compiled)
        if cache_keys([smallest], gpus[0], model_path, log)[smallest] != compiled[smallest][1]:
            error("The computed compilation cache key does not match the compiled model", fatal=True)
        print(f"[*] Cache key computation checked on GPU {gpus[0]} ({time.time() - start:.0f} s)")

    env["compiled_on"] = gpu_uuids([gpus[0]])[0]
    for bucket, (mode, entry) in compiled.items():
        env[f"bucket_{bucket}"] = f"{mode} {entry}"
    write_environment(env_path, env)

    unified = [b for b, (mode, entry) in compiled.items() if mode == "unified"]
    print(f"[*] Largest bucket: {max(compiled)} tokens" + (f"; unified memory from {unified[0]} tokens" if unified else ""))
    print(f"[*] Environment written to {env_path}")
    all_done()

def cli():
    def list_of_str_arg(arg):
        return [x.strip() for x in arg.split(',')]

    parser = argparse.ArgumentParser(description="AlphaFold3 wrapper: Initialize the databases and the compilation cache")
    parser.add_argument("-D", "--data-dir", required=True, help="Base directory for data")
    parser.add_argument("-g", "--gpus", type=list_of_str_arg, default=DEF_GPUS, help=f"GPUs, all of the same model; the first one is used for compilation [{','.join(DEF_GPUS)}]")
    parser.add_argument("-l", "--log", type=str, help="Raw log file")
    parser.add_argument("--overwrite", action = 'store_true', help="Over-write the database files if exist")
    args = parser.parse_args()
    launch(args.data_dir, args.gpus, args.log, overwrite = args.overwrite)

if __name__ == "__main__":
    cli()
