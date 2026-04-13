import argparse
from pathlib import Path
from urllib.parse import quote
from Bio import SeqIO

from run_alphafold3.utils import get_input_files
from run_alphafold3.classes import JSONpath, AF3json

def launch(input_val, output_dir):
    fasta_paths = get_input_files(input_val, ['fa', 'fasta', 'faa', 'fas'])
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    
    for fasta_path in fasta_paths:
        for rec in SeqIO.parse(fasta_path, 'fasta'):
            name = quote(rec.id, safe="-_") + ".json"
            json_path = JSONpath(output_path / name)
            af3 = AF3json(data={"sequences": [{"protein": {"id": "A", "sequence": str(rec.seq)}}]}, name=rec.id, seeds=[1])
            af3.write(json_path)

def cli():
    parser = argparse.ArgumentParser(description="AlphaFold3 wrapper: Convert fasta to json")
    parser.add_argument("-i", "--input", required=True, nargs='+', help="Path to input fasta file(s) or a directory containing them")
    parser.add_argument("-O", "--output", required=True, help="Output directory")
    args = parser.parse_args()
    launch(args.input, args.output)

if __name__ == "__main__":
    cli()
