import argparse
import subprocess
import tempfile
import threading
import queue
import sqlite3
import json
from collections import deque, namedtuple
from concurrent.futures import ThreadPoolExecutor as TPE
from pathlib import Path
from tqdm import tqdm

from run_alphafold3.utils import NUM_SAMPLES, MEMORY_ENV, get_input_jsons, get_data_paths, get_cache_paths, fetch_pred, get_results_dir_path, get_results_files_paths, write_results, docker_cmd, model_flags, detect_compute_gpus
from run_alphafold3.environment import probe_environment, read_environment, compare_environment, parse_compiled_buckets, gpu_uuids
from run_alphafold3.model_cache import cache_keys

# XLA's autotuning results, kept in the compilation cache
AUTOTUNE_DIR = "xla_gpu_per_fusion_autotune_cache_dir"
from run_alphafold3.classes import JSONpath, AF3json
from run_alphafold3.logger import error, get_log, all_done

# --- Constants

DEF_GPUS = detect_compute_gpus()
DEF_BATCH_SIZE = 10

# Runs inside the AlphaFold3 container (CPU only). Prints the number of tokens of each input
# json in a directory, as counted by AlphaFold3's own tokeniser: tokenisation is followed by
# the choice of the bucket, where we stop before the expensive MSA and conformer featurisation.
TOKEN_SCRIPT = """
import json, pathlib, sys
sys.path.insert(0, "/app/alphafold")
from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.data import featurisation
from alphafold3.model.pipeline import pipeline

class TokenCount(Exception):
    pass

def stop_at_bucket(num_tokens, buckets):
    raise TokenCount(num_tokens)

pipeline.calculate_bucket_size = stop_at_bucket
ccds = {}
for path in sorted(pathlib.Path(sys.argv[1]).glob("*.json")):
    record = {"name": path.stem}
    try:
        fold_input = folding_input.Input.from_json(path.read_text())
        if fold_input.user_ccd not in ccds:
            ccds[fold_input.user_ccd] = chemical_components.Ccd(user_ccd=fold_input.user_ccd)
        featurisation.featurise_input(fold_input=fold_input, ccd=ccds[fold_input.user_ccd], buckets=(1,))
        record["error"] = "Tokenisation did not reach the choice of the bucket"
    except TokenCount as e:
        record["tokens"] = e.args[0]
    except Exception as e:
        record["error"] = " ".join(str(e).split())
    print(json.dumps(record), flush=True)
"""

# An input json and the seeds that still have to be predicted for it
Job = namedtuple("Job", ["json_path", "seeds"])

def read_results(path):
    try:
        return [p.read_text() for p in get_results_files_paths(path)]
    except StopIteration:
        return None

def count_tokens(jobs, log):
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, job in enumerate(jobs):
            AF3json(job.json_path, seeds=job.seeds).write(JSONpath(Path(tmp_dir) / f"{i}.json"), name=str(i))
        cmd = docker_cmd(["python", "-c", TOKEN_SCRIPT, "/input"], volumes={tmp_dir: "/input:ro"})
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=log, text=True)
    records = {}
    for line in proc.stdout.splitlines():
        record = json.loads(line)
        records[int(record["name"])] = record
    return [records.get(i, {"error": "Could not count tokens. Check log file."}) for i in range(len(jobs))]

def link_compiled(buckets, compiled, compiled_on, gpus, links_path, model_path, log):
    """
    The cache key of the compiled model depends on the physical GPU, but GPUs of the same model
    run the same compiled model. For each GPU, makes a directory with links to the compiled model
    of the buckets under the GPU's cache keys, to be mounted as the compilation cache, with the
    compilation cache itself mounted at /compiled. Returns {gpu: links directory}.
    """
    uuids = dict(zip(gpus, gpu_uuids(gpus)))
    with TPE(max_workers=len(gpus)) as executor:
        # No need to compute the keys of the GPU the model was compiled on
        futures = {gpu: executor.submit(cache_keys, buckets, gpu, model_path, log) for gpu in gpus if uuids[gpu] != compiled_on}
        link_paths = {}
        for gpu in gpus:
            keys = futures[gpu].result() if gpu in futures else {bucket: compiled[bucket][1] for bucket in buckets}
            link_path = links_path / gpu
            link_path.mkdir()
            (link_path / AUTOTUNE_DIR).symlink_to(f"/compiled/{AUTOTUNE_DIR}")
            for bucket, key in keys.items():
                (link_path / key).symlink_to(f"/compiled/{compiled[bucket][1]}")
            link_paths[gpu] = link_path
    return link_paths

def run_batch(jobs, mode, gpu, output_path, model_path, cache_path, link_path, log):
    """
    Runs AlphaFold3 on a batch of jobs in one container. Returns the predictions that were
    obtained as (json_path, json_hash, seed, sample, results_path) and the unfinished jobs.
    """
    results = []
    unfinished = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = Path(tmp_dir) / "input"
        af3_output_path = Path(tmp_dir) / "output"
        input_path.mkdir()
        af3_output_path.mkdir()
        batch = []
        for i, job in enumerate(jobs):
            af3 = AF3json(job.json_path, seeds=job.seeds)
            af3.write(JSONpath(input_path / f"job_{i}.json"), name=f"job_{i}")
            batch.append((f"job_{i}", job, af3))

        # The compilation cache is read-only: a model that alphafold3_init has not compiled is an error
        cmd = docker_cmd(
            ["python", "run_alphafold.py", "--input_dir=/input", "--output_dir=/output", "--model_dir=/models",
             "--norun_data_pipeline", "--jax_compilation_cache_dir=/cache", *model_flags()],
            gpus=[gpu],
            volumes={input_path: "/input:ro", af3_output_path: "/output", model_path: "/models:ro",
                     cache_path: "/compiled:ro", link_path: "/cache:ro"},
            env={**MEMORY_ENV[mode], "JAX_RAISE_PERSISTENT_CACHE_ERRORS": "true"},
        )
        subprocess.run(cmd, stdout=log, stderr=log)

        # A failing job stops the container, but the jobs before it are complete
        for name, job, af3 in batch:
            json_hash = af3.get_hash()
            missing = []
            for seed in job.seeds:
                samples = [read_results(get_results_dir_path(af3_output_path, name, seed, sample)) for sample in range(NUM_SAMPLES)]
                if all(samples):
                    for sample, (cif, summ, conf) in enumerate(samples):
                        to_path = get_results_dir_path(output_path, af3.name, seed, sample)
                        write_results(to_path, cif, summ, conf)
                        results.append((job.json_path, json_hash, seed, sample, to_path))
                else:
                    missing.append(seed)
            if missing:
                unfinished.append(Job(job.json_path, missing))
    return results, unfinished

def launch(input_val, output_dir, data_dir, log_file, seeds, gpus, batch_size):
    json_paths = get_input_jsons(input_val)
    if not json_paths:
        error("No input files supplied", fatal=True)

    search_db_path, pred_db_path, model_path, public_path = get_data_paths(data_dir)
    cache_path, env_path = get_cache_paths(data_dir)

    if not pred_db_path.exists():
        error(f"Database at {pred_db_path} does not exist", fatal=True)
    if not model_path.exists():
        error(f"No models at {model_path}", fatal=True)
    if not env_path.exists() or not cache_path.exists():
        error(f"No compilation cache in {data_dir}: run alphafold3_init first", fatal=True)

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    failed = predict(json_paths, output_path, pred_db_path, model_path, cache_path, env_path, log_file, seeds, gpus, batch_size)
    for json_path in failed:
        error(f"No predictions were obtained for {json_path}")
    if failed:
        error("Something went wrong", fatal=True)
    all_done()

def predict(json_paths, output_path, pred_db_path, model_path, cache_path, env_path, log_file, seeds, gpus, batch_size):
    """Predicts structures for json_paths (stem -> path). Returns the paths that failed."""
    recorded = read_environment(env_path)
    try:
        current = probe_environment(gpus, model_path)
    except RuntimeError as e:
        error(str(e), fatal=True)
    mismatches = compare_environment(recorded, current)
    for key, recorded_val, current_val in mismatches:
        error(f"{key}: {recorded_val} (recorded) vs. {current_val} (current)")
    if mismatches:
        error(f"The environment differs from the one recorded in {env_path}: re-run alphafold3_init", fatal=True)
    compiled = parse_compiled_buckets(recorded)
    buckets = [int(b) for b in recorded["buckets"].split(',')]

    # Restore cached predictions, collect the rest
    jobs = []
    success_paths = set()
    with sqlite3.connect(pred_db_path) as conn:
        for json_stem, json_path in json_paths.items():
            try:
                af3 = AF3json(json_path, seeds=seeds)
            except Exception as e:
                error(f"Could not parse {json_path}: {e}", fatal=True)
            json_hash = af3.get_hash()
            missing = set()
            for seed in af3.seeds:
                for sample in range(NUM_SAMPLES):
                    cif, summ, conf = fetch_pred(conn, json_hash, seed, sample)
                    if cif:
                        write_results(get_results_dir_path(output_path, af3.name, seed, sample), cif, summ, conf)
                    else:
                        missing.add(seed)
                        break
            if missing:
                jobs.append(Job(json_path, sorted(missing)))
            else:
                success_paths.add(str(json_path))

    with get_log(log_file) as log, sqlite3.connect(pred_db_path) as conn, tqdm(total=len(json_paths)) as progress_bar, tempfile.TemporaryDirectory() as links_dir:
        progress_bar.update(len(success_paths))

        # Each bucket runs under the memory mode it was compiled with
        pending = {"normal": [], "unified": []}
        needed = set()
        for job, record in zip(jobs, count_tokens(jobs, log) if jobs else []):
            if "error" in record:
                error(f"{job.json_path}: {record['error']}")
                continue
            bucket = next((b for b in buckets if b >= record["tokens"]), None)
            if bucket not in compiled:
                error(f"{job.json_path} has {record['tokens']} tokens, more than the largest compiled bucket ({max(compiled)})")
                continue
            pending[compiled[bucket][0]].append(job)
            needed.add(bucket)

        link_paths = link_compiled(sorted(needed), compiled, recorded["compiled_on"], gpus, Path(links_dir), model_path, log) if needed else {}

        # Normal memory batches run on all GPUs. Unified memory batches spill into host RAM: they
        # come after all normal ones and only one runs at a time.
        batch_queues = {mode: deque(mode_jobs[i:i + batch_size] for i in range(0, len(mode_jobs), batch_size)) for mode, mode_jobs in pending.items()}
        cond = threading.Condition()
        state = {"active": 0, "unified_running": False}
        events = queue.Queue()

        def next_batch():
            with cond:
                while True:
                    if batch_queues["normal"]:
                        mode = "normal"
                    elif batch_queues["unified"] and not state["unified_running"]:
                        mode = "unified"
                        state["unified_running"] = True
                    elif not batch_queues["normal"] and not batch_queues["unified"] and state["active"] == 0:
                        return None
                    else:
                        cond.wait()
                        continue
                    state["active"] += 1
                    return mode, batch_queues[mode].popleft()

        def worker(gpu):
            while (item := next_batch()) is not None:
                mode, batch = item
                try:
                    results, unfinished = run_batch(batch, mode, gpu, output_path, model_path, cache_path, link_paths[gpu], log)
                except Exception as e:
                    error(f"Got exception: {e}")
                    results, unfinished = [], batch
                events.put(results)
                with cond:
                    if len(batch) > 1:
                        # Retry one by one, to isolate the job that failed the batch
                        batch_queues[mode].extend([job] for job in unfinished)
                    else:
                        for job in unfinished:
                            events.put(job)
                    state["active"] -= 1
                    if mode == "unified":
                        state["unified_running"] = False
                    cond.notify_all()
            events.put(None)

        remaining = {str(job.json_path): set(job.seeds) for job in jobs}
        threads = [threading.Thread(target=worker, args=(gpu,), daemon=True) for gpu in gpus]
        for thread in threads:
            thread.start()
        workers_done = 0
        while workers_done < len(threads):
            event = events.get()
            if event is None:
                workers_done += 1
            elif isinstance(event, Job):
                error(f"Prediction failed for {event.json_path} (seeds {','.join(map(str, event.seeds))}). Check log file.")
            else:
                for json_path, json_hash, seed, sample, to_path in event:
                    cif_path, summ_path, conf_path = get_results_files_paths(to_path)
                    conn.execute("INSERT OR IGNORE INTO predictions VALUES (?, ?, ?, ?, ?, ?)", (json_hash, seed, sample, cif_path.read_text(), summ_path.read_text(), conf_path.read_text()))
                    seeds_left = remaining[str(json_path)]
                    seeds_left.discard(seed)
                    if not seeds_left and str(json_path) not in success_paths:
                        success_paths.add(str(json_path))
                        progress_bar.update(1)
                conn.commit()

    return [json_path for json_path in json_paths.values() if str(json_path) not in success_paths]

def cli():
    def set_of_int_arg(arg):
        return set(int(x) for x in arg.split(','))
    def list_of_str_arg(arg):
        return [x.strip() for x in arg.split(',')]

    parser = argparse.ArgumentParser(description="AlphaFold3 wrapper: Predict")
    parser.add_argument("-i", "--input", required=True, nargs='+', help="Path to input json file(s) or a directory containing them")
    parser.add_argument("-O", "--output", required=True, help="Output directory")
    parser.add_argument("-D", "--data-dir", required=True, help="Directory for database")
    parser.add_argument("-g", "--gpus", type=list_of_str_arg, default=DEF_GPUS, help=f"GPUs to use [{','.join(DEF_GPUS)}]")
    parser.add_argument("-b", "--batch-size", type=int, default=DEF_BATCH_SIZE, help=f"Number of jobs per AlphaFold3 run (default: {DEF_BATCH_SIZE})")
    parser.add_argument("-s", "--seeds", type=set_of_int_arg, help="Seeds (overrides modelSeeds in json)")
    parser.add_argument("-l", "--log", type=str, help="Raw log file")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    launch(
        args.input, args.output, args.data_dir, args.log,
        seeds=args.seeds, gpus=args.gpus, batch_size=args.batch_size
    )

if __name__ == "__main__":
    cli()
