import argparse
import json
import shutil
import os
from pathlib import Path

from run_alphafold3.logger import error

def parse_ranking_score(json_path: Path) -> float:
    if not json_path.exists():
        raise FileNotFoundError(f"Missing summary confidences file: {json_path}")
        
    data = json.loads(json_path.read_text())
    
    if 'ranking_score' not in data:
        raise KeyError(f"'ranking_score' not found in {json_path}")
        
    return float(data['ranking_score'])

def process_query_dir(query_dir: Path) -> dict:
    best_score = float('-inf')
    best_sample_dir = None
    
    seed_dirs = list(query_dir.glob("seed-*_sample-*"))
    if not seed_dirs:
        raise ValueError(f"No seed/sample directories found in {query_dir}")
        
    for sdir in seed_dirs:
        if not sdir.is_dir():
            continue
            
        summary_files = list(sdir.glob("*_summary_confidences.json"))
        
        if not summary_files:
            raise FileNotFoundError(f"No summary_confidences.json found in {sdir}")
        if len(summary_files) > 1:
            raise ValueError(f"Multiple summary files found in {sdir}. Cannot unambiguously determine the score.")
            
        score = parse_ranking_score(json_path = summary_files[0])
        
        if score > best_score:
            best_score = score
            best_sample_dir = sdir
            
    if best_sample_dir is None:
        raise ValueError(f"Could not determine best sample for {query_dir}")
             
    return { "best_dir": best_sample_dir, "score": best_score }

def launch(input_val, output_dir, soft_link = False, force = False):
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents = True, exist_ok = True)
    
    if isinstance(input_val, str):
        input_val = [ input_val ]
        
    for val in input_val:
        input_path = Path(val).resolve()
        if not input_path.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {input_path}")
            
        is_single_query = any(d.name.startswith("seed-") and "_sample-" in d.name for d in input_path.iterdir() if d.is_dir())
        
        queries = [ input_path ] if is_single_query else [ d for d in input_path.iterdir() if d.is_dir() ]
        
        if not queries:
            raise ValueError(f"No valid query directories found in {input_path}")

        for query_dir in queries:
            result = process_query_dir(query_dir = query_dir)
            best_dir = result["best_dir"]
            score = result["score"]
            
            query_name = query_dir.name if not is_single_query else input_path.name
            dest_dir = output_path / f"{query_name}_best"
            
            # Handle existing destinations
            if dest_dir.exists() or dest_dir.is_symlink():
                if force:
                    if dest_dir.is_symlink() or dest_dir.is_file():
                        dest_dir.unlink()
                    else:
                        shutil.rmtree(dest_dir)
                else:
                    error(f"Destination already exists: {dest_dir}, use --force to overwrite", fatal = True)
                
            if soft_link:
                rel_target = os.path.relpath(best_dir.resolve(), dest_dir.parent.resolve())
                dest_dir.symlink_to(rel_target, target_is_directory = True)
                action = "Symlinked"
            else:
                shutil.copytree(src = best_dir, dst = dest_dir)
                action = "Copied"
                
            print(f"Selected {best_dir.name} for [{query_name}] with score {score}. {action} to {dest_dir}")

def cli():
    parser = argparse.ArgumentParser(description = "AlphaFold3 job selection: Extract best structures based on ranking_score")
    parser.add_argument("-i", "--input", required = True, nargs = '+', help = "Input directory containing seed/sample folders or query subfolders")
    parser.add_argument("-O", "--output", required = True, help = "Output directory")
    parser.add_argument("-l", "--soft-link", action = 'store_true', help = "Soft link instead of hard copy")
    parser.add_argument("-f", "--force", action = 'store_true', help = "Overwrite the destination directory if it already exists")
    args = parser.parse_args()
    launch(input_val = args.input, output_dir = args.output, soft_link = args.soft_link, force = args.force)

if __name__ == "__main__":
    cli()
