import sqlite3
import argparse

from run_alphafold3.utils import get_data_paths
from run_alphafold3.logger import all_done

def launch(data_dir, overwrite = False):
    search_db_path, pred_db_path, *rest = get_data_paths(data_dir, create=True)
    print(f"[*] Initializing the database")
    if search_db_path.is_file() and overwrite:
        search_db_path.unlink()
    if pred_db_path.is_file() and overwrite:
        pred_db_path.unlink()
    with sqlite3.connect(search_db_path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS proteins (seq_hash TEXT PRIMARY KEY, seq TEXT UNIQUE NOT NULL)")
        conn.execute("CREATE TABLE IF NOT EXISTS searches (seq_hash TEXT, unpaired_msa TEXT, templates TEXT, PRIMARY KEY (seq_hash))")
    with sqlite3.connect(pred_db_path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS predictions (json_hash TEXT, seed INT, sample INT, cif TEXT, summary TEXT, confidences TEXT, PRIMARY KEY (json_hash, seed, sample))")
    all_done()

def cli():
    parser = argparse.ArgumentParser(description="AlphaFold3 wrapper: Initialize the database")
    parser.add_argument("-D", "--data-dir", required=True, help="Base directory for data")
    parser.add_argument("--overwrite", action = 'store_true', help="Over-write the database files if exist")
    args = parser.parse_args()
    launch(args.data_dir, overwrite = args.overwrite)

if __name__ == "__main__":
    cli()
