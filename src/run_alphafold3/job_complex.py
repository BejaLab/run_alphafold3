import argparse
from run_alphafold3.classes import JSONpath, AF3json
from run_alphafold3.logger import error

def launch(input_files, output_file):
    input_paths = []
    for oligomer, input_file in input_files:
        input_path = JSONpath(input_file)
        if not input_path.exists():
            error(f"{input_file} does not exist", fatal=True)
        if not input_path.is_file() or not input_path.is_json():
            error(f"{input_file} is not a json file", fatal=True)
        input_paths.append((input_path, oligomer))
        
    if not input_paths:
        error("No input files provided", fatal=True)

    input_path, oligomer = input_paths.pop()
    output = AF3json(input_path, oligomer=oligomer)
    for input_path, oligomer in input_paths:
        output += AF3json(input_path, oligomer=oligomer)

    output_path = JSONpath(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.write(output_path)

def cli():
    class WeightedFileAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            items = getattr(namespace, self.dest, None) or []
            if len(values) == 1:
                weight = 1
                filepath = values[0]
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

    parser = argparse.ArgumentParser(description="AlphaFold3 wrapper: Create complex")
    parser.add_argument("-i", "--input", required=True, nargs='+', action=WeightedFileAction, metavar=('N', 'FILE'), help="Path to input json file(s) with optional number of chains (-i 3 input.json or -i input.json")
    parser.add_argument("-O", "--output", required=True, help="Output json file")
    args = parser.parse_args()
    launch(args.input, args.output)

if __name__ == "__main__":
    cli()
