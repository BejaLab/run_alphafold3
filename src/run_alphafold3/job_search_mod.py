import argparse
import subprocess
import tempfile
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor as TPE
from pathlib import Path
from tqdm import tqdm

from .parsers.hhr import read_hhr
from run_alphafold3.utils import ALT_ALNS, get_input_jsons, get_seq_hash, parse_mod_resources, hh_mapping
from run_alphafold3.classes import JSONpath, AF3json
from run_alphafold3.logger import error, get_log, all_done

def run_worker(protein_dict, metadata, a3m_paths, log, min_prob, min_conf):
    output = {}
    with tempfile.TemporaryDirectory() as temp_dir:
        dir_path = Path(temp_dir)
        # Assuming protein_dict is passed directly
        query_seq = protein_dict.get('sequence', "")
        unpaired_msa = protein_dict.get('unpairedMsa', "")
        
        input_content = unpaired_msa if unpaired_msa else f">query\n{query_seq}\n"
        input_file = dir_path / "query.a3m"
        input_file.write_text(input_content)
        
        for a3m_path in a3m_paths.values():
            profile = a3m_path.stem
            out_file = dir_path / f"{profile}.hhr"
            cmd = [
                "hhalign",
                "-i", str(input_file),
                "-t", str(a3m_path),
                "-o", str(out_file),
                "-alt", str(ALT_ALNS)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"hhalign execution failed. Stderr:\n{result.stderr}")
            if not out_file.is_file():
                raise RuntimeError(f"hhalign did not generate the output")

            with open(out_file) as file:
                rules = [rule for rule in metadata if rule['profile'] == profile]
                for hit in read_hhr(file):
                    if hit['Probab'] >= min_prob:
                        mapping = hh_mapping(hit, min_conf=min_conf)
                        for rule in rules:
                            t_target_pos, expected_res, mod = rule['pos'], rule['res'], rule['mod']
                            if t_target_pos in mapping:
                                mapped_q_pos = mapping[t_target_pos]
                                seq_index = mapped_q_pos - 1 
                                assert -1 < seq_index < len(query_seq)
                                if query_seq[seq_index] == expected_res and mapped_q_pos not in output:
                                    output[mapped_q_pos] = {"ptmType": mod, "ptmPosition": mapped_q_pos}
    return list(output.values())

def launch(input_val, output_dir, log_file, min_prob, min_conf, selected_mods, workers):
    json_paths = get_input_jsons(input_val)
    if not json_paths:
        error("No input files supplied", fatal=True)

    metadata, a3m_paths, cif_paths = parse_mod_resources(selected_mods)

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    def process_files():
        for json_stem, json_path in json_paths.items():
            try:
                af3 = AF3json(json_path)
            except Exception as e:
                error(f"Could not parse {json_path}: {e}", fatal=True)
            seqs = list(af3.iter_seq())
            if not seqs:
                error(f"{json_path} contains no proteins", fatal=True)
            yield json_path, seqs

    def wrapper(json_path, seqs, log):
        try:
            mods = {}
            for protein in seqs:
                sequence = protein['sequence']
                seq_hash = get_seq_hash(sequence)
                mods[seq_hash] = run_worker(protein, metadata, a3m_paths, log, min_prob=min_prob, min_conf=min_conf)
            return json_path, mods
        except Exception as e:
            error(f"Got exception: {e}")

    def check_futures(futures, max_num=1):
        successes = set()
        assert max_num > 0
        while len(futures) >= max_num:
            futures_done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in futures_done:
                json_path, mods = future.result()
                af3 = AF3json(json_path)
                ptms = set()
                for seq in af3.iter_seq():
                    sequence = seq['sequence']
                    seq_hash = get_seq_hash(sequence)
                    if seq_hash in mods and mods[seq_hash]:
                        seq['modifications'] = mods[seq_hash]
                        ptms.add(mods[seq_hash][0]['ptmType']) # Adjusting index per data structure
                for ptm in ptms:
                    cif = cif_paths[ptm].read_text()
                    af3.add_ccd(cif)

                output_json_file = JSONpath(output_path / json_path.name)
                af3.write(output_json_file)
                successes.add(str(json_path))
        return futures, successes

    with get_log(log_file) as log, TPE(max_workers=workers) as executor, tqdm(total=len(json_paths)) as progress_bar:
        futures = set()
        success_paths = set()
        for json_path, seqs in process_files():
            futures, successes = check_futures(futures, max_num=workers)
            success_paths |= successes
            progress_bar.update(len(successes))
            futures.add(executor.submit(wrapper, json_path, seqs, log))
        futures, successes = check_futures(futures)
        success_paths |= successes
        progress_bar.update(len(successes))

    ok = True
    for json_stem, json_path in json_paths.items():
        if str(json_path) not in success_paths:
            error(f"No results were obtained for {json_path}")
            ok = False
    if not ok:
        error("Something went wrong", fatal=True)
    all_done()

def cli():
    parser = argparse.ArgumentParser(description="AlphaFold3 wrapper: Modification search using homology")
    parser.add_argument("-i", "--input", required=True, nargs='+', help="Path to input json file(s) containing alignments or a directory containing them")
    parser.add_argument("-O", "--output", required=True, help="Output directory")
    parser.add_argument("-p", "--prob", type=int, default=90, help="Minimum match probability (default: 90)")
    parser.add_argument("-c", "--conf", type=int, default=7, help="Minimum alignment position confidence (default: 7)")
    parser.add_argument("-m", "--mods", help="Only check for these modifications (default: all)")
    parser.add_argument("-w", "--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("-l", "--log", type=str, help="Raw log file")
    args = parser.parse_args()
    if args.mods:
        args.mods = set(mod.strip() for mod in args.mods.split(','))
    launch(args.input, args.output, args.log, min_prob=args.prob, min_conf=args.conf, workers=args.workers, selected_mods=args.mods)

if __name__ == "__main__":
    cli()
