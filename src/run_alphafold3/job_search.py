import os
import argparse
import subprocess
import tempfile
import json
import sqlite3
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor as TPE
from pathlib import Path
from tqdm import tqdm

from run_alphafold3.utils import get_input_jsons, get_data_paths, get_seq_hash, fetch_search
from run_alphafold3.classes import JSONpath, AF3json
from run_alphafold3.logger import error, get_log, all_done

def run_worker(seq_hash, sequence, public_path, threads, log):
    top_dir = Path(public_path.parts[0], public_path.parts[1])
    output = None, None
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        json_path = JSONpath(tmp_path / "query.json")
        query_json = AF3json(data={"sequences": [{"protein": {"id": "A", "sequence": sequence}}]}, seeds=[1])
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
        print(docker_cmd)
        try:
            subprocess.run(docker_cmd, stdout=log, stderr=log, check=True)
            output_json = JSONpath(tmp_path / "query" / "query_data.json")
            if output_json.exists():
                af3 = AF3json(output_json)
                seq = next(af3.iter_seq())
                output = seq['unpairedMsa'], seq['templates']
        except subprocess.CalledProcessError:
            error("Docker command failed. Check log file.", fatal=True)
            
    return output

def launch(input_val, output_dir, data_dir, log_file, workers, threads):
    json_paths = get_input_jsons(input_val)
    if not json_paths:
        error("No input files supplied", fatal=True)

    search_db_path, pred_db_path, model_path, public_path = get_data_paths(data_dir)

    if not search_db_path.exists():
        error(f"Database at {search_db_path} does not exist", fatal=True)
    if not public_path.exists():
        error(f"No public databases found", fatal=True)

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    def process_files(conn):
        for json_stem, json_path in json_paths.items():
            try:
                af3 = AF3json(json_path)
            except Exception as e:
                error(f"Could not parse {json_path}: {e}", fatal=True)
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
                added[seq_hash] = run_worker(seq_hash, sequence, public_path, threads, log)
            return json_path, present, added
        except Exception as e:
            error(f"Got exception: {e}")

    def check_futures(futures, conn, max_num=1):
        successes = set()
        assert max_num > 0
        while len(futures) >= max_num:
            futures_done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
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

    with get_log(log_file) as log, TPE(max_workers=workers) as executor, sqlite3.connect(search_db_path) as conn, tqdm(total=len(json_paths)) as progress_bar:
        futures = set()
        success_paths = set()
        for json_path, present, missing in process_files(conn):
            futures, successes = check_futures(futures, conn, max_num=workers)
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
        error("Something went wrong", fatal=True)
    all_done()

def cli():
    parser = argparse.ArgumentParser(description="AlphaFold3 wrapper: Template search")
    parser.add_argument("-i", "--input", required=True, nargs='+', help="Path to input json file(s) or a directory containing them")
    parser.add_argument("-O", "--output", required=True, help="Output directory")
    parser.add_argument("-D", "--data-dir", required=True, help="Data directory")
    parser.add_argument("-w", "--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("-t", "--threads", type=int, default=1, help="Number of threads per worker")
    parser.add_argument("-l", "--log", type=str, help="Raw log file")
    args = parser.parse_args()
    launch(
        args.input, args.output, args.data_dir, args.log,
        workers=args.workers, threads=args.threads
    )

if __name__ == "__main__":
    cli()
