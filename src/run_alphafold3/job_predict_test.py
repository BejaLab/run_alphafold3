import argparse
import json
import sqlite3
import tempfile
import time
from pathlib import Path
import gemmi

from run_alphafold3.utils import NUM_SAMPLES, PREDICTIONS_SCHEMA, get_data_paths, get_cache_paths, get_image_path, get_results_dir_path, get_results_files_paths, detect_compute_gpus
from run_alphafold3.classes import JSONpath
from run_alphafold3.job_predict import predict
from run_alphafold3.logger import error, all_done

DEF_GPUS = detect_compute_gpus()

# Single-sequence ubiquitin: small enough for the smallest bucket
TEST_NAME = "ubiquitin"
TEST_SEQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
TEST_SEED = 1

def test_gpu(gpu, tmp_path, json_path, model_path, cache_path, env_path, image_path, log_file):
    """Predicts the test protein on one GPU. Returns the coordinates and ranking score of each sample."""
    output_path = tmp_path / f"output_{gpu}"
    output_path.mkdir()
    # A fresh database, so that the prediction is not restored from predictions.sq3
    test_db_path = tmp_path / f"predictions_{gpu}.sq3"
    with sqlite3.connect(test_db_path) as conn:
        for statement in PREDICTIONS_SCHEMA:
            conn.execute(statement)

    print(f"[*] Predicting {TEST_NAME} ({len(TEST_SEQ)} residues) on GPU {gpu}")
    start = time.time()
    failed = predict({json_path.stem: json_path}, output_path, test_db_path, model_path, cache_path, env_path, image_path, log_file, None, [gpu], 1)
    elapsed = time.time() - start
    if failed:
        error(f"Prediction failed on GPU {gpu}. Check log file.", fatal=True)

    samples = []
    for sample in range(NUM_SAMPLES):
        results_path = get_results_dir_path(output_path, TEST_NAME, TEST_SEED, sample)
        try:
            cif_path, summ_path, conf_path = get_results_files_paths(results_path)
        except StopIteration:
            error(f"Missing output files in {results_path}", fatal=True)
        structure = gemmi.read_structure(str(cif_path))
        num_res = sum(len(chain) for chain in structure[0])
        if num_res != len(TEST_SEQ):
            error(f"Sample {sample} has {num_res} residues instead of {len(TEST_SEQ)}", fatal=True)
        coords = [atom.pos.tolist() for chain in structure[0] for res in chain for atom in res]
        json.loads(conf_path.read_text())
        samples.append((coords, json.loads(summ_path.read_text())["ranking_score"]))

    with sqlite3.connect(test_db_path) as conn:
        num_rows = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    if num_rows != NUM_SAMPLES:
        error(f"{num_rows} predictions stored in the database instead of {NUM_SAMPLES}", fatal=True)

    print(f"[*] {NUM_SAMPLES} samples in {elapsed:.0f} s, ranking scores: {', '.join(f'{score:.2f}' for coords, score in samples)}")
    return samples

def launch(data_dir, gpus, log_file):
    search_db_path, pred_db_path, model_path, public_path = get_data_paths(data_dir)
    cache_path, env_path = get_cache_paths(data_dir)
    image_path = get_image_path(data_dir)
    if not model_path.exists():
        error(f"No models at {model_path}", fatal=True)
    if not env_path.exists() or not cache_path.exists():
        error(f"No compilation cache in {data_dir}: run alphafold3_init first", fatal=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        json_path = JSONpath(tmp_path / f"{TEST_NAME}.json")
        json_path.write_json({
            "name": TEST_NAME, "modelSeeds": [TEST_SEED], "dialect": "alphafold3", "version": 4,
            "sequences": [{"protein": {
                "id": "A", "sequence": TEST_SEQ, "unpairedMsa": f">query\n{TEST_SEQ}\n", "pairedMsa": "", "templates": []
            }}],
        })
        # One GPU at a time: all GPUs must load the same compiled model and give identical results
        reference = None
        for gpu in gpus:
            samples = test_gpu(gpu, tmp_path, json_path, model_path, cache_path, env_path, image_path, log_file)
            if reference is None:
                reference = samples
            elif samples != reference:
                error(f"GPU {gpu} gives different results than GPU {gpus[0]}", fatal=True)

    if len(gpus) > 1:
        print(f"[*] Results are identical on GPUs {','.join(gpus)}")
    all_done()

def cli():
    def list_of_str_arg(arg):
        return [x.strip() for x in arg.split(',')]

    parser = argparse.ArgumentParser(description="AlphaFold3 wrapper: Test prediction on a small protein on each GPU, without touching the prediction database")
    parser.add_argument("-D", "--data-dir", required=True, help="Directory for database")
    parser.add_argument("-g", "--gpus", type=list_of_str_arg, default=DEF_GPUS, help=f"GPUs to test [{','.join(DEF_GPUS)}]")
    parser.add_argument("-l", "--log", type=str, help="Raw log file")
    args = parser.parse_args()
    launch(args.data_dir, args.gpus, args.log)

if __name__ == "__main__":
    cli()
