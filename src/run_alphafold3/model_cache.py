import json
import subprocess
import tempfile

from run_alphafold3.utils import docker_cmd, model_flags
from run_alphafold3.logger import error

# Runs inside the AlphaFold3 container. For each bucket, smallest first, a poly-Ala chain of the
# bucket's length is featurised as run_alphafold.py would do it and the model forward pass is
# lowered. With --key_only, prints the compilation cache key under which run_alphafold.py would
# look up the compiled model on this GPU, without compiling. Otherwise compiles, but does not
# execute, the model, which writes its cache entry, and prints the memory it needs and the cache
# entry; stops at the first bucket that does not fit into memory. Peak memory grows with the
# square of the bucket size, so a bucket that cannot fit judging by the previous one is not tried.
WARM_SCRIPT = """
import argparse, json, pathlib, sys, time
sys.path.insert(0, "/app/alphafold")

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", required=True)
parser.add_argument("--jax_compilation_cache_dir", required=True)
parser.add_argument("--buckets", required=True)
parser.add_argument("--key_only", action="store_true")
parser.add_argument("--budget", type=int, help="Memory budget in bytes, if lower than the device limit")
parser.add_argument("--previous", help="bucket:peak of the previous bucket, for the estimate")
# Same names and meaning as in run_alphafold.py
parser.add_argument("--num_diffusion_samples", type=int, required=True)
parser.add_argument("--num_recycles", type=int, required=True)
parser.add_argument("--flash_attention_implementation", required=True)
args = parser.parse_args()

# tokamax reads absl flags lazily from sys.argv, so hand absl an empty command line
from absl import flags
sys.argv = sys.argv[:1]
flags.FLAGS(sys.argv)

import run_alphafold
from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.data import featurisation
from alphafold3.model.components import utils
import jax
from jax import numpy as jnp
# JAX computes the cache key before looking it up, and compiles only if it is not found
from jax._src import compiler

class CacheKey(Exception):
    pass

def stop_at_lookup(module_name, cache_key, *rest):
    raise CacheKey(cache_key)

def no_compile(*rest, **kwargs):
    raise RuntimeError("The compilation cache is not used")

def report(**result):
    print(json.dumps(result), flush=True)

cache_dir = pathlib.Path(args.jax_compilation_cache_dir)
jax.config.update("jax_compilation_cache_dir", str(cache_dir))
device = jax.local_devices(backend="gpu")[0]
runner = run_alphafold.ModelRunner(
    config=run_alphafold.make_model_config(
        flash_attention_implementation=args.flash_attention_implementation,
        num_diffusion_samples=args.num_diffusion_samples,
        num_recycles=args.num_recycles,
    ),
    device=device,
    model_dir=pathlib.Path(args.model_dir),
)
# ModelRunner._model is functools.partial(jax.jit(forward_fn.apply), params)
jitted = runner._model.func
params = runner._model.args[0]

budget = device.memory_stats()["bytes_limit"]
if args.budget:
    budget = min(budget, args.budget)
previous = tuple(map(int, args.previous.split(":"))) if args.previous else None

for bucket in (int(b) for b in args.buckets.split(",")):
    if not args.key_only and previous:
        estimate = previous[1] * (bucket / previous[0]) ** 2
        if estimate > budget:
            report(bucket=bucket, fits=False, estimate=estimate, budget=budget)
            break

    seq = "A" * bucket
    fold_input = folding_input.Input.from_json(json.dumps({
        "name": "warm", "modelSeeds": [1], "dialect": "alphafold3", "version": 2,
        "sequences": [{"protein": {
            "id": "A", "sequence": seq, "unpairedMsa": f">query\\n{seq}\\n", "pairedMsa": "", "templates": []
        }}],
    }))
    [example] = featurisation.featurise_input(
        fold_input=fold_input, buckets=(bucket,),
        ccd=chemical_components.Ccd(user_ccd=fold_input.user_ccd), verbose=False,
    )
    # Same preprocessing as ModelRunner.run_inference, so that the lowered module is identical
    example = jax.device_put(
        jax.tree_util.tree_map(jnp.asarray, utils.remove_invalidly_typed_feats(example)), device
    )
    lowered = jitted.lower(params, jax.random.PRNGKey(1), example)

    if args.key_only:
        # Only around the compilation of the model, not of the small helper functions
        cache_read, compile_and_load = compiler._cache_read, compiler.backend_compile_and_load
        compiler._cache_read, compiler.backend_compile_and_load = stop_at_lookup, no_compile
        try:
            lowered.compile()
        except CacheKey as e:
            report(bucket=bucket, key=e.args[0])
        finally:
            compiler._cache_read, compiler.backend_compile_and_load = cache_read, compile_and_load
        continue

    before = set(cache_dir.glob("jit_apply_fn-*-cache"))
    start = time.time()
    try:
        compiled = lowered.compile()
    except Exception as e:
        report(bucket=bucket, fits=False, seconds=time.time() - start, error=" ".join(str(e).split()))
        break
    seconds = time.time() - start
    stats = compiled.memory_analysis()
    peak = stats.argument_size_in_bytes + stats.output_size_in_bytes - stats.alias_size_in_bytes + stats.temp_size_in_bytes
    if peak > budget:
        report(bucket=bucket, fits=False, seconds=seconds, peak=peak, budget=budget)
        break
    [entry] = set(cache_dir.glob("jit_apply_fn-*-cache")) - before
    report(bucket=bucket, fits=True, seconds=seconds, peak=peak, budget=budget, entry=entry.name)
    previous = (bucket, peak)
    del compiled
"""

def warm_cmd(buckets, env, gpu, model_path, cache_path, extra=[]):
    return docker_cmd(
        ["python", "-c", WARM_SCRIPT, "--model_dir=/models", "--jax_compilation_cache_dir=/cache",
         f"--buckets={','.join(map(str, buckets))}", *extra, *model_flags()],
        gpus=[gpu],
        volumes={model_path: "/models:ro", cache_path: "/cache"},
        env=env,
    )

def cache_keys(buckets, gpu, model_path, log):
    """
    File names under which run_alphafold.py looks up the compiled model on this GPU, without
    compiling. They do not depend on the memory mode or on the cache contents, and need little
    GPU memory. XLA writes autotuning results of small helper functions into the cache, so the
    cache is a scratch one, mounted at the same path as the real one.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        cmd = warm_cmd(buckets, {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}, gpu, model_path, tmp_dir, ["--key_only"])
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=log, text=True)
    keys = {}
    for line in proc.stdout.splitlines():
        result = json.loads(line)
        keys[result["bucket"]] = result["key"] + "-cache"
    if proc.returncode != 0 or set(keys) != set(buckets):
        error(f"Could not compute compilation cache keys on GPU {gpu}. Check log file.", fatal=True)
    return keys
