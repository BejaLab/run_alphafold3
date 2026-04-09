import sqlite3
import hashlib
import base64
import csv
import json
import importlib.resources as pkg_resources
from pathlib import Path

from run_alphafold3.classes import JSONpath
from run_alphafold3.logger import error

# --- Constants ---
NUM_SAMPLES = 5
ALT_ALNS = 10

# --- Helper Functions ---
def parse_mod_resources(selected_mods):
    mods_dir = Path(pkg_resources.files('run_alphafold3.mods'))
    metadata_path = mods_dir.joinpath('mods.csv')
    
    metadata = []
    profile_paths = {}
    cif_paths = {}
    with open(metadata_path, 'r') as f:
        for line in f:
            profile, pos, res, mod, lig = line.rstrip().split(',')
            if not selected_mods or mod in selected_mods:
                cif_path = mods_dir / "cif" / (mod + '.cif')
                a3m_path = mods_dir / "a3m" / (profile + '.a3m')
                
                # Note: mod_path isn't defined in the original, assuming it means a3m_path
                assert a3m_path.is_file() 
                assert cif_path.is_file()
                
                metadata.append({ 'profile': profile, 'pos': int(pos), 'res': res, 'mod': mod })
                cif_paths[mod] = cif_path
                profile_paths[profile] = a3m_path

    if selected_mods:
        for mod in selected_mods:
            if mod not in cif_paths:
                error(f"Modification {mod} is not supported")

    return metadata, profile_paths, cif_paths

def load_metadata(txt_path: Path) -> list:
    metadata = []
    with open(txt_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata.append(row)
    return metadata

def get_data_paths(data_dir, create=False):
    data_path = Path(data_dir).resolve()
    if create and not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)
    pred_db_path = data_path / "predictions.sq3"
    search_db_path = data_path / "searches.sq3"
    model_path = data_path / "models"
    public_path = data_path / "public_databases"
    return pred_db_path, search_db_path, model_path, public_path

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

def get_results_dir_path(prefix_path, stem, seed, sample, create=False):
    path = prefix_path / stem / f"seed-{seed}_sample-{sample}"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path

def get_results_files_paths(path):
    cif_path = next(path.glob("*model.cif"))
    sum_conf_path = next(path.glob("*summary_confidences.json"))
    conf_path = next(path.glob("*confidences.json"))
    return cif_path, sum_conf_path, conf_path

def get_input_files(input_val, exts):
    files = input_val if isinstance(input_val, list) else [input_val]
    paths = list(map(Path, files))
    if len(paths) == 1:
        path = paths[0]
        if path.is_dir():
            paths = [p for ext in exts for p in path.glob(f"*.{ext}", case_sensitive=False)]
    return paths

def get_input_jsons(input_val):
    json_paths = {}
    for path in get_input_files(input_val, exts=['json']):
        json_path = JSONpath(path)
        if not json_path.is_file():
            error(f"{json_path} not found", fatal=True)
        if not json_path.is_json():
            error(f"{json_path} is not a json file", fatal=True)
        if json_path.stem in json_paths:
            error(f"Stem {json_path.stem} in {json_path} is found in a different input json file", fatal=True)
        json_paths[json_path.stem] = json_path
    return json_paths

def create_dummy_databases(path):
    public_path = path / "public_databases"
    public_path.mkdir(exist_ok=True)
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

def hh_mapping(hit, min_conf):
    mapping = {}
    q_seq = hit['query']['alignment']
    t_seq = hit['template']['alignment']
    confs = hit['confidence']
    
    assert len(q_seq) == len(t_seq)
    
    q_pos = hit['query']['coords'][0]
    t_pos = hit['template']['coords'][0]
    
    for q_char, t_char, conf in zip(q_seq, t_seq, confs):
        if q_char != '-' and t_char != '-' and int(conf) >= min_conf:
            mapping[t_pos] = q_pos
        if q_char != '-':
            q_pos += 1
        if t_char != '-':
            t_pos += 1
    return mapping

def add_ccd_data(af3_obj, mods_dir, modification: str):
    cif_path = mods_dir.joinpath(f"{modification}.cif")
    if not cif_path.is_file():
        raise FileNotFoundError(f"CCD CIF file missing for modification: {modification}")
        
    lines = cif_path.read_text().splitlines()
    data_line = next((line for line in lines if line.startswith('data_')), None)
    if not data_line:
        raise ValueError(f"No 'data_' block found in {cif_path}")
        
    if data_line not in af3_obj.ccd:
        af3_obj.ccd[data_line] = []
        ccd_name = None
        for line in lines:
            if line.startswith('data_'):
                ccd_name = line
            if ccd_name:
                af3_obj.ccd[ccd_name].append(line)
