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
import string
import itertools as it
import copy
from urllib.parse import quote

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
    db_path = data_path / "predictions.sq3"
    model_path = data_path / "models"
    public_path = data_path / "public_databases"
    return db_path, model_path, public_path

def clean_record_seq(record_seq):
    return record_seq.upper().strip().replace("*", "").replace("-", "")

def get_seq_hash(clean_seq):
    raw_hash = hashlib.sha256(clean_seq.encode('ascii')).digest()
    return base64.urlsafe_b64encode(raw_hash).decode().rstrip("=")

def fetch_search(conn, seq_hash):
    found = conn.execute("SELECT unpaired_msa, templates FROM searches WHERE seq_hash = ?", (seq_hash,)).fetchone()
    return (found[0], json.loads(found[1])) if found else (None, None)

def fetch_pred(conn, json_hash, seed, sample):
    found = conn.execute("SELECT cif, summary, confidences FROM predictions WHERE json_hash = ? AND seed = ? AND sample = ?", (json_hash, str(seed), str(sample))).fetchone()
    return (found[0], found[1], found[2]) if found else (None, None, None)

def get_results_dir_path(prefix_path, stem, seed, sample, create = False):
    path = prefix_path / stem / f"seed-{seed}_sample-{sample}"
    if create:
        path.mkdir(parents = True, exist_ok = True)
    return path

def get_results_files_paths(path):
    cif_path = next(path.glob("*model.cif"))
    sum_conf_path = next(path.glob("*summary_confidences.json"))
    conf_path = next(path.glob("*confidences.json"))
    return cif_path, sum_conf_path, conf_path

def get_input_files(input, exts):
    files = input if isinstance(input, list) else [ input ]
    paths = list(map(Path, files))
    if len(paths) == 1:
        path = paths[0]
        if path.is_dir():
            paths = [ p for ext in exts for p in path.glob(f"*.{ext}", case_sensitive = False) ]
    return paths

def get_input_jsons(input):
    json_paths = {}
    for path in get_input_files(input, exts = [ 'json' ]):
        json_path = JSONpath(path)
        if not json_path.is_json() or not json_path.is_file():
            error(f"{json_path} is not a json file", fatal = True)
        if json_path.stem in json_paths:
            error(f"Stem {json_path.stem} in {json_path} is found in a different input json file", fatal = True)
        json_paths[json_path.stem] = json_path
    return json_paths

def launch_json(input, output_dir):
    from Bio import SeqIO
    fasta_paths = get_input_files(input, [ 'fa', 'fasta', 'faa', 'fas' ])
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents = True, exist_ok = True)
    for fasta_path in fasta_paths:
        for rec in SeqIO.parse(fasta_path, 'fasta'):
            name = quote(rec.id, safe = "-_") + ".json"
            json_path = JSONpath(output_path / name)
            af3 = AF3json(data = { "sequences": [ { "protein": { "id": "A", "sequence": str(rec.seq) } } ] }, name = rec.id, seeds = [1])
            af3.write(json_path)

def launch_search(input, output_dir, data_dir, log_file, workers, threads):

    json_paths = get_input_jsons(input)
    if not json_paths:
        error("No input files supplied", fatal = True)

    db_path, model_path, public_path = get_data_paths(data_dir)

    if not db_path.exists():
        error(f"Database at {db_path} does not exist", fatal = True)
    if not public_path.exists():
        error(f"No public databases found", fatal = True)

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents = True, exist_ok = True)

    def process_files(conn):
        for json_stem, json_path in json_paths.items():
            try:
                af3 = AF3json(json_path)
            except Exception as e:
                error(f"Could not parse {json_path}: {e}", fatal = True)
            present = dict()
            missing = dict()
            for seq in af3.iter_seq():
                sequence = seq['sequence']
                seq_hash = get_seq_hash(sequence)
                unpaired_msa, templates = fetch_search(conn, seq_hash)
                if unpaired_msa is None:
                    missing[seq_hash] = sequence
                else:
                    present[seq_hash] = unpaired_msa, templates
            yield json_path, present, missing

    def wrapper(json_path, present, missing, log):
        try:
            added = {}
            for seq_hash, sequence in missing.items():
                added[seq_hash] = run_search_worker(seq_hash, sequence, public_path, threads, log)
            return json_path, present, added
        except Exception as e:
            error(f"Got exception: {e}")

    def check_futures(futures, conn, max_num = 1):
        successes = set()
        assert max_num > 0
        while len(futures) >= max_num:
            futures_done, futures = concurrent.futures.wait(futures, return_when = concurrent.futures.FIRST_COMPLETED)
            for future in futures_done:
                json_path, present, added = future.result()
                for seq_hash, (unpaired_msa, templates) in added.items():
                    conn.execute("INSERT OR IGNORE INTO searches VALUES (?, ?, ?)", (seq_hash, unpaired_msa, json.dumps(templates)))
                conn.commit()
                af3 = AF3json(json_path)
                for seq in af3.iter_seq():
                    sequence = seq['sequence']
                    seq_hash = get_seq_hash(sequence)
                    if seq_hash in present:
                        seq['unpairedMsa'], seq['templates'] = present[seq_hash]
                        seq['pairedMsa'] = ""
                    elif seq_hash in added:
                        seq['unpairedMsa'], seq['templates'] = added[seq_hash]
                        seq['pairedMsa'] = ""
                    else:
                        raise ValueError(f"Something went wrong: templates and MSAs not generated for: {sequence}")
                output_json_file = JSONpath(output_path / json_path.name)
                af3.write(output_json_file)
                successes.add(str(json_path))
        return futures, successes

    with get_log(log_file) as log, TPE(max_workers = workers) as executor, sqlite3.connect(db_path) as conn, tqdm(total = len(json_paths)) as progress_bar:
        futures = set()
        success_paths = set()
        for json_path, present, missing in process_files(conn):
            futures, successes = check_futures(futures, conn, max_num = workers)
            success_paths |= successes
            progress_bar.update(len(successes))
            futures.add(executor.submit(wrapper, json_path, present, missing, log))
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

def launch_predict(input, output_dir, data_dir, log_file, seeds, gpus, max_len):

    json_paths = get_input_jsons(input)
    if not json_paths:
        error("No input files supplied", fatal = True)

    db_path, model_path, public_path = get_data_paths(data_dir)

    if not db_path.exists():
        error(f"Database at {db_path} does not exist", fatal = True)
    if not model_path.exists():
        error(f"No models at {model_path}", fatal = True)

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents = True, exist_ok = True)

    gpu_queue = queue.Queue()
    for gpu in gpus:
        gpu_queue.put(gpu)

    def process_files(conn):
        for json_stem, json_path in json_paths.items():
            try:
                af3 = AF3json(json_path, seeds = seeds)
            except Exception as e:
                error(f"Could not parse {json_path}: {e}", fatal = True)
            if len(af3) > max_len:
                error(f"{json_path} has a total length of {data.len} > {max_data_len}")
            json_hash = af3.get_hash()
            missing = set()
            for seed in af3.seeds:
                for sample in range(NUM_SAMPLES):
                    cif, conf, summ = fetch_pred(conn, json_hash, seed, sample)
                    if cif:
                        to_path = get_results_dir_path(output_path, af3.name, seed, sample, create = True)
                        cif_path, summ_path, conf_path = get_results_files_paths(to_path)
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
            return run_predict_worker(data, output_path, model_path, gpu, missing_seeds, log)
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
            futures, successes = check_futures(futures, conn, max_num = len(gpus))
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
    def __init__(self, path: JSONpath = None, data: dict = None, name: str = None, oligomer = 1, seeds: set[int] = set()):
        if path:
            self.path = path
            self.name = path.stem
            self.data = path.read_json()
        elif data:
            self.path = None
            self.name = "query" if name is None else name
            self.data = data
        else:
            raise ValueError("Need json path or data")
        if 'sequences' not in self.data:
            raise ValueError("No 'sequences' found in json")
        self.orig_seeds = self.data.pop('modelSeeds', [])
        self.seeds = list(seeds) if seeds else self.orig_seeds
        self.orig_name = self.data.pop('name', None)
        if 'dialect' not in self.data:
            self.data['dialect'] = "alphafold3"
        elif self.data['dialect'] != "alphafold3":
            raise ValueError("Only 'alphafold3' dialect is supported")
        self.data['version'] = 4
        self.len = self.hash = None
        self.ids = AF3json.id_gen()
        self.len = 0
        for seq in self.data['sequences']:
            for item in seq:
                id = seq[item].get('id', [])
                seq[item]['id'] = list(it.islice(self.ids, oligomer * len(id)))
                if item == 'protein':
                    self.len += len(id) * len(seq[item]['sequence'])
                    if 'templates' not in seq[item]:
                        seq[item]['templates'] = []
                    if 'pairedMsa' not in seq[item]:
                        seq[item]['pairedMsa'] = ""
                    if 'unpairedMsa' not in seq[item]:
                        seq[item]['unpairedMsa'] = ""
        ccd_str = self.data.pop('userCCD', None)
        self.ccd = {}
        ccd_name = None
        if ccd_str:
            for line in ccd_str.split('\n'):
                if line.startswith('data_'):
                    ccd_name = line
                    self.ccd[ccd_name] = [ line ]
                elif ccd_name:
                    self.ccd[ccd_name].append(line)
                elif line:
                    raise ValueError("Found non-empty 'userCCD' but no data_* line")

    def __len__(self):
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
        if self.ccd:
            data['userCCD'] = ''
            for ccd_name, ccd_lines in self.ccd.items():
                data['userCCD'] += '\n'.join(ccd_lines) + '\n'
        path.write_json(data)

    def iterseq(self):
        if 'sequences' not in self.data:
            raise ValueError("No 'sequences' found in json")
        for seq in self.data['sequences']:
            new_seq = seq.copy()
            new_seq['id'] = "A"
            result = copy.deepcopy(self)
            result.data["sequences"] = [ new_seq ]
        sequence = self.data["sequences"][index]["sequence"]
        result = copy.deepcopy(self)
        results.data = { "sequences": [ { "protein": { "id": "A", "sequence": sequence } } ] }
        return result

    @staticmethod
    def id_gen():
        letters = string.ascii_uppercase
        for let in letters:
            yield let
        for size in it.count(2):
            for p in it.product(letters, repeat = size):
                yield "".join(p)

    def __add__(self, other):
        if not isinstance(other, AF3json):
            return NotImplemented
        for seq in other.data['sequences']:
            for item in seq:
                id = seq[item].get('id', [])
                seq[item]['id'] = list(it.islice(self.ids, len(id)))
            self.data['sequences'].append(seq)
        for ccd_name, ccd_lines in other.ccd.items():
            if ccd_name not in self.ccd:
                self.ccd[ccf_name] = ccd_lines
        return self

    def iter_seq(self, key = 'protein'):
        for seq in self.data['sequences']:
            if key in seq:
                yield seq[key]

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

def run_search_worker(seq_hash, sequence, public_path, threads, log):

    top_dir = Path(public_path.parts[0], public_path.parts[1])
    output = None, None
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        json_path = JSONpath(tmp_path / "query.json")
        query_json = AF3json(data = { "sequences": [ { "protein": { "id": "A", "sequence": sequence } } ] }, seeds = [1])
        query_json.write(json_path)
        docker_cmd = [
            "docker", "run",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--volume", f"{json_path}:/input.json",
            "--volume", f"{tmp_path}:/output",
            "--volume", f"{public_path}:/public_databases",
            "--volume", f"{top_dir}:{top_dir}",
            "alphafold3", "sh", "-c",
            f"python run_alphafold.py --json_path=/input.json --output_dir=/output --run_inference=false --jackhmmer_n_cpu {threads}"
        ]
        try:
            subprocess.run(docker_cmd, stdout = log, stderr = log, check = True)
            output_json = JSONpath(tmp_path / "query" / "query_data.json")
            if output_json.exists():
                af3 = AF3json(output_json)
                seq = next(af3.iter_seq())
                output = seq['unpairedMsa'], seq['templates']
        except subprocess.CalledProcessError:
            error("Error: Docker command failed. Check log file.", fatal = True)
    return output

def run_predict_worker(data, output_path, model_path, gpu, seeds, log):

    env = {}
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

def launch_complex(input_files, output_file):
    input_paths = []
    for oligomer, input_file in input_files:
        input_path = JSONpath(input_file)
        if not input_path.exists():
            error(f"{input_file} does not exist", fatal = True)
        if not input_path.is_file() or not input_path.is_json():
            error(f"{input_file} is not a json file", fatal = True)
        input_paths.append((input_path, oligomer))
    if not input_paths:
        error("No input files provided", fatal = True)

    input_path, oligomer = input_paths.pop()
    output = AF3json(input_path, oligomer = oligomer)
    for input_path, oligomer in input_paths:
        output += AF3json(input_path, oligomer = oligomer)

    output_path = JSONpath(output_file)
    output_path.parent.mkdir(parents = True, exist_ok = True)
    output.write(output_path)

def launch_init(data_dir):
    db_path, *rest = get_data_paths(data_dir, create = True)
    print(f"[*] Initializing database at {db_path}")
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS proteins (seq_hash TEXT PRIMARY KEY, seq TEXT UNIQUE NOT NULL)")
        conn.execute("CREATE TABLE IF NOT EXISTS searches (seq_hash TEXT, unpaired_msa TEXT, templates TEXT, PRIMARY KEY (seq_hash))")
        conn.execute("CREATE TABLE IF NOT EXISTS predictions (json_hash TEXT, seed INT, sample INT, cif TEXT, summary TEXT, confidences TEXT, PRIMARY KEY (json_hash, seed, sample))")
    print(f"[✔] Init complete")

def predict_cli():
    def set_of_int_arg(arg):
        return set(int(x) for x in arg.split(','))

    parser = argparse.ArgumentParser(description = "AlphaFold3 wrapper: Predict")
    parser.add_argument("-i", "--input", required = True, nargs = '+', help = "Path to input json file(s) or a directory containing them")
    parser.add_argument("-O", "--output", required = True, help = "Output directory")
    parser.add_argument("-D", "--data-dir", required = True, help = "Directory for database")
    parser.add_argument("-g", "--gpus", type = set_of_int_arg, default = [0], help = "GPU indices to use")
    parser.add_argument("--max-len", type = int, default = 5200, help = "Maximum total number of amino acid resudues (default: no)")
    parser.add_argument("-s", "--seeds", type = set_of_int_arg, help = "Seeds (overrides modelSeeds in json)")
    parser.add_argument("-l", "--log", type = str, help = "Raw log file")
    args = parser.parse_args()
    launch_predict(
        args.input, args.output, args.data_dir, args.log,
        seeds = args.seeds, gpus = args.gpus, max_len = args.max_len
    )

def search_cli():
    def set_of_int_arg(arg):
        return set(int(x) for x in arg.split(','))

    parser = argparse.ArgumentParser(description = "AlphaFold3 wrapper: Template search")
    parser.add_argument("-i", "--input", required = True, nargs = '+', help = "Path to input json file(s) or a directory containing them")
    parser.add_argument("-O", "--output", required = True, help = "Output directory")
    parser.add_argument("-D", "--data-dir", required = True, help = "Data directory")
    parser.add_argument("-w", "--workers", type = int, default = 1, help = "Number of workers")
    parser.add_argument("-t", "--threads", type = int, default = 1, help = "Number of threads per worker")
    parser.add_argument("-l", "--log", type = str, help = "Raw log file")
    args = parser.parse_args()
    launch_search(
        args.input, args.output, args.data_dir, args.log,
        workers = args.workers, threads = args.threads
    )

# stump
def select_cli():
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

def json_cli():
    parser = argparse.ArgumentParser(description = "AlphaFold3 wrapper: Convert fasta to json")
    parser.add_argument("-i", "--input", required = True, nargs = '+', help = "Path to input fasta file(s) or a directory containing them")
    parser.add_argument("-O", "--output", required = True, help = "Output directory")
    args = parser.parse_args()
    launch_json(
        args.input, args.output
    )

def complex_cli():
    class WeightedFileAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string = None):
            items = getattr(namespace, self.dest, None) or []
            # just the filename
            if len(values) == 1:
                weight = 1
                filepath = values[0]
            # weight then filename
            elif len(values) == 2:
                try:
                    weight = int(values[0])
                    filepath = values[1]
                except ValueError:
                    raise argparse.ArgumentError(self, f"Expected weight (number), got '{values[0]}'")
            else:
                raise argparse.ArgumentError(self, "Expected 1 or 2 arguments per -i flag")
            items.append((weight, filepath))
            setattr(namespace, self.dest, items)

    parser = argparse.ArgumentParser(description = "AlphaFold3 wrapper: Create complex")
    parser.add_argument("-i", "--input", required = True, nargs = '+', action = WeightedFileAction, metavar = ('N', 'FILE'), help = "Path to input json file(s) with optional number of chains (-i 3 input.json or -i input.json")
    parser.add_argument("-O", "--output", required = True, help = "Output json file")
    args = parser.parse_args()
    launch_complex(args.input, args.output)
