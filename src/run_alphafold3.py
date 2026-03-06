import sqlite3
import hashlib, base64
import argparse
import shutil
import tempfile
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor as TPE
from pathlib import Path
from tqdm import tqdm
import os, sys
import queue
import subprocess
from contextlib import contextmanager
import json

# --- Constants ---
NUM_SAMPLES = 5

# --- Helper Functions ---

@contextmanager
def get_log(log_path):
    if not log_path:
        yield subprocess.DEVNULL
        return
    Path(log_path).parent.mkdir(parents = True, exist_ok = True)
    f = open(log_path, 'w', encoding = 'utf-8')
    try:
        yield f
    finally:
        f.close()

def get_data_paths(data_dir, create = False):
    data_path = Path(data_dir).resolve()
    if create and not data_path.exists():
        data_path.mkdir(parents = True, exist_ok = True)
    model_path = (data_path / "models").resolve()
    db_path = data_path / "predictions.sq3"
    public_path = (data_path / "public_databases").resolve()
    return model_path, public_path, db_path

def fetch_pred(conn, json_hash, seed, sample):
    found = conn.execute("SELECT cif, summary, confidences FROM predictions WHERE json_hash = ? AND seed = ? AND sample = ?", (json_hash, str(seed), str(sample))).fetchone()
    return (found[0], found[1], found[2]) if found else (None, None, None)

def get_results_dir_path(prefix_path, stem, seed, sample, create = False):
    path = prefix_path / stem / f"seed-{seed}_sample-{sample}"
    if create:
        path.mkdir(parents = True, exist_ok = True)
    return path

def get_results_files_paths(path):
    return path / "model.cif", path / "summary_confidences.json", path / "confidences.json"

def launch_run(input, output_dir, data_dir, log_file, seeds, gpus, max_len):

    if not isinstance(input, list):
        input = [ input ]
    if len(input) == 1:
        path = Path(input[0])
        if path.is_dir():
            input = path.glob('*.json')

    if not input:
        error("No input files supplied", fatal = True)

    model_path, public_path, db_path = get_data_paths(data_dir)
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents = True, exist_ok = True)
    json_paths = {}
    for path in input:
        json_path = JSONpath(path)
        if not json_path.is_json() or not json_path.is_file():
            error(f"{json_path} is not a json file", fatal = True)
        if json_path.stem in json_paths:
            error(f"Stem {json_path.stem} in {json_path} is found in a different input json file", fatal = True)
        json_paths[json_path.stem] = json_path

    gpu_queue = queue.Queue()
    for gpu in gpus:
        gpu_queue.put(gpu)

    conn_queue = queue.Queue()
    conn_queue.put(1)

    def process_files(conn):
        for json_stem, json_path in json_paths.items():
            try:
                data = AF3json(json_path, seeds)
            except Exception as e:
                error(f"Could not parse {json_path}: {e}", fatal = True)
            if data.get_len() > max_len:
                error(f"{json_path} has a total length of {data.len} > {max_data_len}")
            json_hash = data.get_hash()
            missing = set()
            for seed in data.seeds:
                for sample in range(NUM_SAMPLES):
                    cif, conf, summ = fetch_pred(conn, json_hash, seed, sample)
                    if cif:
                        to_path = get_results_dir_path(output_path, data.name, seed, sample, create = True)
                        cif_path, summ_path, conf_path = get_results_files_paths(to_path)
                        cif_path.write_text(cif)
                        summ_path.write_text(summ)
                        conf_path.write_text(conf)
                    else:
                        missing.add(seed)
                        break
            yield data, missing

    def wrapper(data, missing_seeds, log):
        gpu = gpu_queue.get()
        try:
            return run_gpu_worker(data, output_path, data_dir, gpu, missing_seeds, log)
        except Exception as e:
            error(f"Got exception: {e}")
        finally:
            gpu_queue.put(gpu)

    def check_futures(futures, conn, max_num = 1):
        successes = set()
        assert max_num > 0
        while len(futures) >= max_num:
            futures_done, futures = concurrent.futures.wait(futures, return_when = concurrent.futures.FIRST_COMPLETED)
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

    with get_log(log_file) as log, TPE(max_workers = len(gpus)) as executor, sqlite3.connect(db_path) as conn, tqdm(total = len(json_paths)) as progress_bar:
        futures = set()
        success_paths = set()
        for data, missing in process_files(conn):
            futures, successes = check_futures(futures, conn)
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
        error("Something went wrong", fatal = True)
    print(f"[✔] All done")

def error(msg, fatal = False):
    print(f"[✘] {msg}")
    if fatal:
        sys.exit(1)

# --- CLI Entries ---

class JSONpath(Path):
    def read_json(self):
        with self.open('r') as file:
            return json.load(file)
    def write_json(self, data):
        with self.open('w') as file:
            return json.dump(data, file, separators = (',', ':'))
    def is_json(self):
        return self.suffix.lower() == ".json"

class AF3json:
    def __init__(self, path: JSONpath, seeds: set[int]):
        self.path = path
        self.name = path.stem
        self.data = path.read_json()
        if 'sequences' not in self.data:
            raise ValueError("No 'sequences' found in json")
        self.orig_seeds = self.data.pop('modelSeeds')
        self.seeds = list(seeds) if seeds else self.orig_seeds
        if not self.seeds:
            raise ValueError("No seeds supplied")
        self.orig_name = self.data.pop('name')
        if 'dialect' not in self.data:
            self.data['dialect'] = "alphafold3"
        elif self.data['dialect'] != "alphafold3":
            raise ValueError("Only 'alphafold3' dialect is supported")
        if 'version' not in self.data or self.data['version'] == "1":
            self.data['version'] = 1
        elif self.data['version'] != 1:
            raise ValueError("Only version 1 is supported")
        self.len = self.hash = None

    def get_len(self):
        if not self.len:
            self.len = 0
            for seq in self.data['sequences']:
                if 'protein' in seq:
                    protein = seq['protein']
                    if 'id' not in protein:
                        protein['id'] = [ "A" ]
                    elif not isinstance(protein['id'], list):
                        protein['id'] = [ protein['id'] ]
                    self.len += len(protein['sequence']) * len(protein['id'])
        if not self.len:
            raise ValueError("No protein sequences found in json")
        return self.len

    def get_hash(self):
        if not self.hash:
            serialized = json.dumps(self.data, sort_keys = True, separators = (',', ':')).encode('utf-8')
            raw_hash = hashlib.sha256(serialized).digest()
            self.hash = base64.urlsafe_b64encode(raw_hash).decode().rstrip("=")
        return self.hash

    def write(self, path: JSONpath):
        data = self.data.copy()
        data['modelSeeds'] = self.seeds
        data['name'] = 'query'
        path.write_json(data)

def create_dummy_databases(path):
    public_path = path / "public_databases"
    public_path.mkdir()
    dbs = [
        "bfd-first_non_consensus_sequences.fasta",
        "mgy_clusters_2022_05.fa",
        "uniprot_all_2021_04.fa", 
        "uniref90_2022_05.fa",
        "nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta",
        "rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta",
        "rnacentral_active_seq_id_90_cov_80_linclust.fasta",
        "mmcif_files",
        "pdb_seqres_2022_09_28.fasta"
    ]
    for db in dbs:
        (public_path / db).touch()
    return public_path

def run_gpu_worker(data, output_path, data_dir, gpu, seeds, log):

    model_path, public_path, db_path = get_data_paths(data_dir)

    env = {}
    long_query = data.get_len() >= 3500
    xla_preallocate = "false" if long_query else "true"
    unified_mem = "true" if long_query else ""
    mem_fraction = 3.20 if long_query else 0.95

    json_hash = data.get_hash()
    output = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        af3_path = tmp_path / "alphafold3"
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
        # Execute and Log
        try:
            subprocess.run(docker_cmd, stdout = log, stderr = log, check = True)
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

def launch_init(data_dir):
    model_path, public_path, db_path = get_data_paths(data_dir, create = True)
    print(f"[*] Initializing database at {db_path}")
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS predictions (json_hash TEXT, seed INT, sample INT, cif TEXT, summary TEXT, confidences TEXT, PRIMARY KEY (json_hash, seed, sample))")
    print(f"[✔] Init complete")

def run_cli():
    def set_of_int_arg(arg):
        return set(int(x) for x in arg.split(','))

    parser = argparse.ArgumentParser(description = "AlphaFold3 wrapper: Run")
    parser.add_argument("-i", "--input", required = True, nargs = '+', help = "Path to input json file(s) or a directory containing them")
    parser.add_argument("-O", "--output", required = True, help = "Output directory")
    parser.add_argument("-D", "--data-dir", required = True, help = "Base directory for data")
    parser.add_argument("-g", "--gpus", type = set_of_int_arg, default = [0], help = "GPU indices to use")
    parser.add_argument("--max-len", type = int, default = 5200, help = "Maximum total number of amino acid resudues (default: no)")
    parser.add_argument("-s", "--seeds", type = set_of_int_arg, help = "Seeds (overrides modelSeeds in json)")
    parser.add_argument("-l", "--log", type = str, help = "Raw log file")
    args = parser.parse_args()
    launch_run(
        args.input, args.output, args.data_dir, args.log,
        seeds = args.seeds, gpus = args.gpus, max_len = args.max_len
    )

# stump
def template_cli():
    parser = argparse.ArgumentParser(description = "AlphaFold3 wrapper: Select")
    parser.add_argument("-I", "--input", required = True, help = "Directory containing 'run' outputs")
    parser.add_argument("-O", "--output", required = True, help = "Output directory for the best models")
    parser.add_argument("-l", "--soft-link", action = 'store_true', help = 'Soft link instead of hard copy')
    args = parser.parse_args()
    launch_select(args.input, args.output, args.soft_link)

def init_cli():
    parser = argparse.ArgumentParser(description = "AlphaFold3 wrapper: Initialize the database")
    parser.add_argument("-D", "--data-dir", required = True, help = "Base directory for data")
    args = parser.parse_args()
    launch_init(args.data_dir)
