# run_alphafold3

A wrapper around [AlphaFold3](https://github.com/google-deepmind/alphafold3) that turns structure
prediction into a scriptable, resumable pipeline.

AlphaFold3 itself is run inside its official Docker image (`alphafold3`); this package takes care of
everything around it:

* **Input preparation** — build AlphaFold3 job JSONs from plain FASTA files, and combine them into
  homo- or hetero-oligomeric complexes.
* **Caching** — MSA/template searches and predictions are stored in SQLite databases keyed by
  sequence hash and job hash, so re-running a batch never repeats work that has already been done.
* **Batching and parallelism** — whole directories of jobs are processed with a configurable number
  of CPU workers (search) or across several GPUs (inference).
* **Post-translational modifications** — detect PTM sites (currently the retinal-binding lysine of
  microbial/animal rhodopsins) by profile–profile comparison against bundled HMM profiles, and inject
  the corresponding `modifications` entries and CCD blocks into the job JSON.
* **Result selection** — pick the best seed/sample per query by `ranking_score`.

The package installs seven command-line tools, all named `alphafold3_*`.

## Requirements

* Python ≥ 3.12
* [Docker](https://www.docker.com/) with an image tagged `alphafold3`
  (built as described in the AlphaFold3 repository)
* AlphaFold3 model weights and public sequence databases (see *Data directory* below)
* [`hhalign`](https://github.com/soedinglab/hh-suite) on `$PATH` — only for `alphafold3_search_mod`
* NVIDIA GPU(s) with drivers and the container toolkit — only for `alphafold3_predict`

## Installation

```bash
pip install git+https://github.com/BejaLab/run_alphafold3
```

Python dependencies (`gemmi`, `tqdm`, `biopython`) are installed automatically.

## Data directory

Most tools take `-D/--data-dir`, a single directory holding everything persistent:

```
<data-dir>/
├── searches.sq3        # cached MSAs and templates, keyed by sequence hash
├── predictions.sq3     # cached structures and confidences, keyed by job hash
├── models/             # AlphaFold3 model weights
└── public_databases/   # AlphaFold3 sequence databases
```

`models/` and `public_databases/` must be provided by the user; the two SQLite files are created by `alphafold3_init`.

## Typical pipeline

```bash
# 0. one-off: create the cache databases
alphafold3_init -D data

# 1. FASTA -> one AlphaFold3 job JSON per record
alphafold3_json -i sequences.fasta -O queries/json

# 2. MSA and template search (CPU, cached)
alphafold3_search -i queries/json -O queries/search -D data -w 4 -t 8 -l search.log

# 3. optional: annotate post-translational modifications
alphafold3_search_mod -i queries/search -O queries/search_mod -w 4

# 4. optional: assemble a complex out of several jobs
alphafold3_complex -i 3 queries/search_mod/protA.json -i queries/search_mod/protB.json \
                   -O queries/complex/AAAB.json

# 5. structure prediction (GPU, cached)
alphafold3_predict -i queries/search_mod -O queries/predict -D data -s 1,123,124 -l predict.log

# 6. keep the best-ranked sample per query
alphafold3_select -i queries/predict -O queries/select -l
```

Each stage reads JSONs and writes JSONs (of the same names) into a new directory, so stages can be
skipped or re-ordered. `-i` accepts either a list of files or a single directory to scan.

Predictions land in `<output>/<query>/seed-<seed>_sample-<sample>/`, five samples per seed.

## Tools

### `alphafold3_init` — initialize the database

Creates the data directory and the two SQLite caches. Run once before anything else.

```
usage: alphafold3_init [-h] -D DATA_DIR [--overwrite]

  -D, --data-dir DATA_DIR   Base directory for data
      --overwrite           Over-write the database files if exist
```

### `alphafold3_json` — convert FASTA to JSON

Writes one single-chain AlphaFold3 job JSON per FASTA record, named after the record ID
(URL-quoted) with `modelSeeds: [1]`.

```
usage: alphafold3_json [-h] -i INPUT [INPUT ...] -O OUTPUT

  -i, --input INPUT [INPUT ...]   Path to input fasta file(s) or a directory containing them
  -O, --output OUTPUT             Output directory
```

### `alphafold3_complex` — create a complex

Merges several job JSONs into one, re-lettering chain IDs. Each `-i` takes an optional copy number
in front of the file name, so `-i 3 protA.json` requests a trimer of that chain. MSAs, templates and
`userCCD` blocks already present in the inputs are carried over.

```
usage: alphafold3_complex [-h] -i N [FILE ...] -O OUTPUT

  -i, --input N [FILE ...]   Path to input json file(s) with optional number of chains
                             (-i 3 input.json or -i input.json)
  -O, --output OUTPUT        Output json file
```

### `alphafold3_search` — MSA and template search

Runs the AlphaFold3 data pipeline for every protein chain not yet in `searches.sq3`,
then writes copies of the input JSONs with `unpairedMsa` and `templates` filled in.
`pairedMsa` is blanked. `--workers` jobs run concurrently, each with `--threads` jackhmmer
CPUs.

```
usage: alphafold3_search [-h] -i INPUT [INPUT ...] -O OUTPUT -D DATA_DIR
                         [-w WORKERS] [-t THREADS] [-l LOG]

  -i, --input INPUT [INPUT ...]   Path to input json file(s) or a directory containing them
  -O, --output OUTPUT             Output directory
  -D, --data-dir DATA_DIR         Data directory
  -w, --workers WORKERS           Number of workers
  -t, --threads THREADS           Number of threads per worker
  -l, --log LOG                   Raw log file
```

### `alphafold3_search_mod` — modification search using homology

Aligns each query (its MSA, if present, otherwise the bare sequence) against the bundled profiles
with `hhalign`, maps annotated modification sites from the profile onto the query, and — when the
mapped residue is of the expected type — adds a `modifications` entry plus the matching CCD block to
the output JSON. Shipped profiles cover the retinal-binding lysine (`LYR`) of rhodopsins
(`PF01036`, `PF18761`, `SCOP d5dysa_`). Best run after `alphafold3_search`, so that a real MSA is
available.

```
usage: alphafold3_search_mod [-h] -i INPUT [INPUT ...] -O OUTPUT [-p PROB]
                             [-c CONF] [-m MODS] [-w WORKERS] [-l LOG]

  -i, --input INPUT [INPUT ...]   Path to input json file(s) containing alignments
                                  or a directory containing them
  -O, --output OUTPUT             Output directory
  -p, --prob PROB                 Minimum match probability (default: 90)
  -c, --conf CONF                 Minimum alignment position confidence (default: 7)
  -m, --mods MODS                 Only check for these modifications (default: all)
  -w, --workers WORKERS           Number of workers
  -l, --log LOG                   Raw log file
```

### `alphafold3_predict` — predict structures

Runs AlphaFold3 inference, one job per GPU at a time, and caches every seed/sample result
in `predictions.sq3`; cached results are written out without touching the GPU.
Memory settings are relaxed automatically for queries of ≥ 3500 residues.

```
usage: alphafold3_predict [-h] -i INPUT [INPUT ...] -O OUTPUT -D DATA_DIR
                          [-g GPUS] [--max-len MAX_LEN] [-s SEEDS] [-l LOG]

  -i, --input INPUT [INPUT ...]   Path to input json file(s) or a directory containing them
  -O, --output OUTPUT             Output directory
  -D, --data-dir DATA_DIR         Directory for database
  -g, --gpus GPUS                 GPUs to use (default: all detected, comma-separated)
      --max-len MAX_LEN           Maximum total number of amino acid residues (default: 5200)
  -s, --seeds SEEDS               Seeds (overrides modelSeeds in json)
  -l, --log LOG                   Raw log file
```

### `alphafold3_select` — extract the best structures

Scans `seed-*_sample-*` directories, reads `ranking_score` from each `*_summary_confidences.json`
and copies (or symlinks) the top-scoring one to `<output>/<query>_best`. The input may be a single
query directory or a directory of query directories.

```
usage: alphafold3_select [-h] -i INPUT [INPUT ...] -O OUTPUT [-l] [-f]

  -i, --input INPUT [INPUT ...]   Input directory containing seed/sample folders
                                  or query subfolders
  -O, --output OUTPUT             Output directory
  -l, --soft-link                 Soft link instead of hard copy
  -f, --force                     Overwrite the destination directory if it already exists
```

## License

GNU General Public License v3.
