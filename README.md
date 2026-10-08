# run_alphafold3

A wrapper around [AlphaFold3](https://github.com/google-deepmind/alphafold3) that turns structure
prediction into a scriptable, resumable pipeline.

AlphaFold3 itself is run with [Apptainer](https://apptainer.org/), from an image built out of its
official Docker image; this package takes care of everything around it:

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

The package installs eight command-line tools, all named `alphafold3_*`.

## Requirements

* Python ≥ 3.12
* [Apptainer](https://apptainer.org/) on `$PATH`. On Ubuntu ≥ 23.10, which restricts unprivileged
  user namespaces, install the system package (`ppa:apptainer/ppa`), which comes with the AppArmor
  profile it needs; an Apptainer from conda cannot run containers there
* The AlphaFold3 Docker image (built as described in the AlphaFold3 repository), or any other
  source `apptainer build` accepts — only for `alphafold3_init` to build the Apptainer image
* AlphaFold3 model weights and public sequence databases (see *Data directory* below)
* [`hhalign`](https://github.com/soedinglab/hh-suite) on `$PATH` — only for `alphafold3_search_mod`
* NVIDIA GPU(s) with drivers — for `alphafold3_init` and `alphafold3_predict`

## Installation

```bash
pip install git+https://github.com/BejaLab/run_alphafold3
```

Python dependencies (`gemmi`, `tqdm`, `biopython`) are installed automatically.

## Setup

1. Install Apptainer. On Ubuntu:

   ```bash
   sudo add-apt-repository -y ppa:apptainer/ppa
   sudo apt update
   sudo apt install -y apptainer
   ```

   Make sure no other `apptainer` (e.g. from conda) comes first on `$PATH`.

2. Build the AlphaFold3 Docker image from the AlphaFold3 repository. The labels are optional and end
   up in `environment.txt`:

   ```bash
   git clone --branch v3.0.4 https://github.com/google-deepmind/alphafold3
   cd alphafold3
   docker build -t alphafold3 -f docker/Dockerfile \
       --label alphafold3.commit=$(git rev-parse HEAD) --label alphafold3.version=v3.0.4 .
   ```

   Docker is only needed for `alphafold3_init` to build the Apptainer image; to build it elsewhere,
   save the image with `docker save alphafold3 -o alphafold3.tar` and pass
   `--image-source docker-archive://alphafold3.tar` to `alphafold3_init`.

3. Put the model weights into `<data-dir>/models/` and the sequence databases into
   `<data-dir>/public_databases/` (see *Data directory* below).

4. Build the image, create the databases and compile the model, then test the prediction:

   ```bash
   alphafold3_init -D <data-dir> -l init.log
   alphafold3_predict_test -D <data-dir>
   ```

To update AlphaFold3, rebuild the Docker image, delete `<data-dir>/alphafold3.sif` and run
`alphafold3_init` again. `alphafold3_predict` refuses to run until then.

## Data directory

Most tools take `-D/--data-dir`, a single directory holding everything persistent:

```
<data-dir>/
├── searches.sq3        # cached MSAs and templates, keyed by sequence hash
├── predictions.sq3     # cached structures and confidences, keyed by job hash
├── alphafold3.sif      # AlphaFold3 Apptainer image
├── jax_cache/          # compiled AlphaFold3 model, one entry per bucket size
├── environment.txt     # what the compiled model and the predictions depend on
├── models/             # AlphaFold3 model weights
└── public_databases/   # AlphaFold3 sequence databases
```

`models/` and `public_databases/` must be provided by the user; everything else is created by `alphafold3_init`.

`alphafold3_search` makes the top-level directory of `public_databases/` (e.g. `/data` for
`/data/af3/public_databases`) visible inside the container, so that links inside
`public_databases/` resolve; they must point within that same top-level directory.

Several data directories can share the image, the compiled model and the weights, e.g. to search
against different sequence databases while keeping the caches apart. Keep the real files in one
data directory and link to them from the others:

```bash
cd <other-data-dir>
ln -s <data-dir>/alphafold3.sif <data-dir>/jax_cache <data-dir>/environment.txt <data-dir>/models .
```

`searches.sq3`, `predictions.sq3` and `public_databases/` stay separate, as they depend on the
databases. Run `alphafold3_init` only in the data directory holding the real files: it re-creates
`jax_cache/` and `environment.txt`, which the links then follow.

## Typical pipeline

```bash
# 0. one-off (and after any change of AlphaFold3, GPU or driver): create the databases
#    and compile the model
alphafold3_init -D data -l init.log
alphafold3_predict_test -D data

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
alphafold3_predict -i queries/search_mod -O queries/predict -D data -s 1,123,124 -b 10 -l predict.log

# 6. keep the best-ranked sample per query
alphafold3_select -i queries/predict -O queries/select -l
```

Each stage reads JSONs and writes JSONs (of the same names) into a new directory, so stages can be
skipped or re-ordered. `-i` accepts either a list of files or a single directory to scan.

Predictions land in `<output>/<query>/seed-<seed>_sample-<sample>/`, five samples per seed.

## Tools

### `alphafold3_init` — initialize the database and compile the model

Creates the data directory and the two SQLite caches, builds the AlphaFold3 Apptainer image
`alphafold3.sif` unless it is already there (from `--image-source`, by default the local Docker
image `alphafold3:latest`; delete the image to rebuild it), then compiles the AlphaFold3 model for each
of AlphaFold3's bucket sizes (inputs are padded to the smallest bucket that fits them) and stores
the result in `jax_cache/`. Compilation dominates the run time of a small prediction and is done
here once, so that `alphafold3_predict` only loads the compiled model. Using the same compiled model
also makes predictions bit-for-bit reproducible.

Buckets are compiled on the first given GPU (any other given GPUs are only checked to be of the
same model), smallest first, with normal GPU memory
settings until a bucket does not fit and with unified memory (spilling into host RAM) from then on;
compilation stops at the first bucket that does not fit either. The largest bucket thus depends on
the available GPU and host memory. The outcome and timing of each attempt are printed, separating
the compilation itself from the overhead of the container run.

`environment.txt` records what the compiled model and the predictions depend on: the AlphaFold3
version and image, JAX, XLA flags, the GPU model and driver, the model weights and
settings and the bucket sizes, as well as the memory mode and the cache entry of each compiled
bucket and the GPU it was compiled on. Run `alphafold3_init` again whenever any of this changes;
`alphafold3_predict` refuses to run otherwise. The compiled model can be used on any GPU of the
same model (see `alphafold3_predict`).

```
usage: alphafold3_init [-h] -D DATA_DIR [-g GPUS] [-l LOG] [--image-source IMAGE_SOURCE]
                       [--overwrite]

  -D, --data-dir DATA_DIR       Base directory for data
  -g, --gpus GPUS               GPUs, all of the same model; the first one is used for
                                compilation (default: all detected, comma-separated)
  -l, --log LOG                 Raw log file
      --image-source IMAGE_SOURCE
                                Source to build the AlphaFold3 image from, if there is none in
                                the data directory, in any form accepted by `apptainer build`
                                (default: docker-daemon://alphafold3:latest)
      --overwrite               Over-write the database files if exist
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
CPUs. With `--read-only-cache`, `searches.sq3` is only read (see *Read-only caches* below).

```
usage: alphafold3_search [-h] -i INPUT [INPUT ...] -O OUTPUT -D DATA_DIR
                         [-w WORKERS] [-t THREADS] [-l LOG] [--read-only-cache]

  -i, --input INPUT [INPUT ...]   Path to input json file(s) or a directory containing them
  -O, --output OUTPUT             Output directory
  -D, --data-dir DATA_DIR         Data directory
  -w, --workers WORKERS           Number of workers
  -t, --threads THREADS           Number of threads per worker
  -l, --log LOG                   Raw log file
      --read-only-cache           Use cached searches, but do not add new ones to the cache
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

Runs AlphaFold3 inference and caches every seed/sample result in `predictions.sq3`; cached
results are written out without touching the GPU. Before anything else, the environment is checked
against `environment.txt` written by `alphafold3_init`.

Jobs are run in batches of `--batch-size` per AlphaFold3 run, so that the model is loaded once per
batch. Each job is assigned to a bucket by its number of tokens, counted by AlphaFold3 itself
(modified residues and ligands take one token per atom), and runs under the memory mode its bucket
was compiled with. Jobs larger than the largest compiled bucket are rejected. Batches in normal
memory mode run in parallel on all GPUs; batches in unified memory mode come after them and run one
at a time, since they share host RAM. If a batch fails, its unfinished jobs are retried one by one.

JAX keys compiled models by the physical GPU and the container runtime, but GPUs of the same model
run the same compiled model. For each GPU, the compiled model of the buckets needed is linked under
that GPU's cache keys in a temporary directory, which is used as the compilation cache; computing
the keys takes a short container run per GPU (in parallel). All GPUs thus run the same compiled
model and give identical results.

The compilation cache is mounted read-only, so a job never compiles the model. Every protein
chain needs `unpairedMsa`, `pairedMsa` and `templates` and every RNA chain `unpairedMsa`
(as written by `alphafold3_search`); set them to `""` and `[]` for single-sequence predictions.

```
usage: alphafold3_predict [-h] -i INPUT [INPUT ...] -O OUTPUT -D DATA_DIR
                          [-g GPUS] [-b BATCH_SIZE] [-s SEEDS] [-l LOG]
                          [--read-only-cache]

  -i, --input INPUT [INPUT ...]   Path to input json file(s) or a directory containing them
  -O, --output OUTPUT             Output directory
  -D, --data-dir DATA_DIR         Directory for database
  -g, --gpus GPUS                 GPUs to use (default: all detected, comma-separated)
  -b, --batch-size BATCH_SIZE     Number of jobs per AlphaFold3 run (default: 10)
  -s, --seeds SEEDS               Seeds (overrides modelSeeds in json)
  -l, --log LOG                   Raw log file
      --read-only-cache           Use cached predictions, but do not add new ones to the cache
```

#### Read-only caches

With `--read-only-cache`, `alphafold3_search` and `alphafold3_predict` look results up in
`searches.sq3` and `predictions.sq3` as usual but never write to them, and open them read-only, so
that they also work on a data directory the user cannot write, e.g. from a sandbox. Searches and
predictions that are not cached are run every time. Jobs writing to the caches at the same time
are waited for. If a writer crashed in the middle of a write, the database cannot be read read-only
until a run without `--read-only-cache` has rolled it back.

### `alphafold3_predict_test` — test prediction

Predicts a small protein (single-sequence ubiquitin) on each of the given GPUs in turn, through the
same code path as `alphafold3_predict` but with a temporary prediction database, and checks the
output: five samples with all residues, the confidences files and the database rows. Since the
compilation cache is read-only, this also checks that each GPU finds the compiled model, and the
results must be identical on all GPUs. Run it after `alphafold3_init`.

```
usage: alphafold3_predict_test [-h] -D DATA_DIR [-g GPUS] [-l LOG]

  -D, --data-dir DATA_DIR   Directory for database
  -g, --gpus GPUS           GPUs to test (default: all detected, comma-separated)
  -l, --log LOG             Raw log file
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
