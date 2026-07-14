import os
import argparse
import subprocess
import tempfile
import shutil
import queue
import sqlite3
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor as TPE
from pathlib import Path
from tqdm import tqdm

from run_alphafold3.utils import NUM_SAMPLES, get_input_jsons, get_data_paths, fetch_pred, get_results_dir_path, get_results_files_paths, create_dummy_databases, detect_compute_gpus
from run_alphafold3.classes import JSONpath, AF3json
from run_alphafold3.logger import error, get_log, all_done

# --- Constants

DEF_GPUS = detect_compute_gpus()
MAX_LEN = 5200

def run_worker(data, output_path, model_path, gpu, seeds, log):
    long_query = len(data) >= 3500
    xla_preallocate = "false" if long_query else "true"
    unified_mem = "true" if long_query else ""
    mem_fraction = 3.20 if long_query else 0.95

    json_hash = data.get_hash()
    output = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        json_path = JSONpath(tmp_path / "query.json")
        data.write(json_path)
        public_path = create_dummy_databases(tmp_path)
        
        docker_cmd = [
            "docker", "run",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--volume", f"{json_path}:/input.json",
            "--volume", f"{tmp_path}:/output",
            "--volume", f"{model_path}:/models",
            "--volume", f"{public_path}:/public_databases",
            "--env", f"CUDA_VISIBLE_DEVICES={gpu}",
            "--env", f"XLA_PYTHON_CLIENT_PREALLOCATE={xla_preallocate}",
            "--env", f"TF_FORCE_UNIFIED_MEMORY={unified_mem}",
            "--env", f"XLA_CLIENT_MEM_FRACTION={mem_fraction}",
            "--gpus", "all",
            "alphafold3", "sh", "-c",
            f"python run_alphafold.py --json_path=/input.json --output_dir=/output"
        ]
        
        try:
            subprocess.run(docker_cmd, stdout=log, stderr=log, check=True)
            for seed in seeds:
                for sample in range(NUM_SAMPLES):
                    from_path = get_results_dir_path(tmp_path, "query", seed, sample)
                    to_path   = get_results_dir_path(output_path, data.name, seed, sample)
                    cif_path, summ_path, conf_path = get_results_files_paths(from_path)
                    if cif_path.exists() and summ_path.exists() and conf_path.exists():
                        shutil.move(from_path, to_path)
                        output.append((data.path, json_hash, seed, sample, to_path))
        except subprocess.CalledProcessError:
            error("Error: Docker command failed. Check log file.")
            
    return output

def launch(input_val, output_dir, data_dir, log_file, seeds, gpus, max_len):
    json_paths = get_input_jsons(input_val)
    if not json_paths:
        error("No input files supplied", fatal=True)

    search_db_path, pred_db_path, model_path, public_path = get_data_paths(data_dir)

    if not pred_db_path.exists():
        error(f"Database at {pred_db_path} does not exist", fatal=True)
    if not model_path.exists():
        error(f"No models at {model_path}", fatal=True)

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    gpu_queue = queue.Queue()
    for gpu in gpus:
        gpu_queue.put(gpu)

    def process_files(conn):
        for json_stem, json_path in json_paths.items():
            try:
                af3 = AF3json(json_path, seeds=seeds)
            except Exception as e:
                error(f"Could not parse {json_path}: {e}", fatal=True)
            if len(af3) > max_len:
                error(f"{json_path} has a total length > {max_len}")
            json_hash = af3.get_hash()
            missing = set()
            for seed in af3.seeds:
                for sample in range(NUM_SAMPLES):
                    cif, summ, conf = fetch_pred(conn, json_hash, seed, sample)
                    if cif:
                        to_path = get_results_dir_path(output_path, af3.name, seed, sample, create=True)
                        cif_path = to_path / "query_model.cif"
                        summ_path = to_path / "query_summary_confidences.json"
                        conf_path = to_path / "query_confidences.json"
                        cif_path.write_text(cif)
                        summ_path.write_text(summ)
                        conf_path.write_text(conf)
                    else:
                        missing.add(seed)
                        break
            yield af3, missing

    def wrapper(data, missing_seeds, log):
        gpu = gpu_queue.get()
        try:
            return run_worker(data, output_path, model_path, gpu, missing_seeds, log)
        except Exception as e:
            error(f"Got exception: {e}")
        finally:
            gpu_queue.put(gpu)

    def check_futures(futures, conn, max_num=1):
        successes = set()
        assert max_num > 0
        while len(futures) >= max_num:
            futures_done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in futures_done:
                for json_path, json_hash, seed, sample, to_path in future.result():
                    cif_path, summ_path, conf_path = get_results_files_paths(to_path)
                    cif  = cif_path.read_text()
                    summ = summ_path.read_text()
                    conf = conf_path.read_text()
                    conn.execute("INSERT OR IGNORE INTO predictions VALUES (?, ?, ?, ?, ?, ?)", (json_hash, seed, sample, cif, summ, conf))
                    successes.add(str(json_path))
                conn.commit()
        return futures, successes

    with get_log(log_file) as log, TPE(max_workers=len(gpus)) as executor, sqlite3.connect(pred_db_path) as conn, tqdm(total=len(json_paths)) as progress_bar:
        futures = set()
        success_paths = set()
        for data, missing in process_files(conn):
            futures, successes = check_futures(futures, conn, max_num=len(gpus))
            if missing:
                futures.add(executor.submit(wrapper, data, missing, log))
            else:
                successes.add(str(data.path))
            success_paths |= successes
            progress_bar.update(len(successes))
        futures, successes = check_futures(futures, conn)
        success_paths |= successes
        progress_bar.update(len(successes))

    ok = True
    for json_stem, json_path in json_paths.items():
        if str(json_path) not in success_paths:
            error(f"No predictions were obtained for {json_path}")
            ok = False
    if not ok:
        error("Something went wrong", fatal=True)
    all_done()

def cli():
    def set_of_int_arg(arg):
        return set(int(x) for x in arg.split(','))
    def set_of_str_arg(arg):
        return set(x.strip() for x in arg.split(','))

    parser = argparse.ArgumentParser(description="AlphaFold3 wrapper: Predict")
    parser.add_argument("-i", "--input", required=True, nargs='+', help="Path to input json file(s) or a directory containing them")
    parser.add_argument("-O", "--output", required=True, help="Output directory")
    parser.add_argument("-D", "--data-dir", required=True, help="Directory for database")
    parser.add_argument("-g", "--gpus", type=set_of_str_arg, default=DEF_GPUS, help=f"GPUs to use [{','.join(DEF_GPUS)}]")
    parser.add_argument("--max-len", type=int, default=MAX_LEN, help="Maximum total number of amino acid residues (default: no)")
    parser.add_argument("-s", "--seeds", type=set_of_int_arg, help="Seeds (overrides modelSeeds in json)")
    parser.add_argument("-l", "--log", type=str, help="Raw log file")
    args = parser.parse_args()
    launch(
        args.input, args.output, args.data_dir, args.log,
        seeds=args.seeds, gpus=args.gpus, max_len=args.max_len
    )

if __name__ == "__main__":
    cli()
