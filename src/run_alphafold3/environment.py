import json
import base64
import hashlib
import subprocess

from run_alphafold3.utils import container_cmd, model_flags

# Runs inside the AlphaFold3 container
PROBE_SCRIPT = """
import json, os, sys, importlib.metadata
sys.path.insert(0, "/app/alphafold")
import run_alphafold
import jax, jaxlib
devices = jax.local_devices(backend="gpu")
print(json.dumps({
    "alphafold3": importlib.metadata.version("alphafold3"),
    "jax": jax.__version__,
    "jaxlib": jaxlib.__version__,
    "platform_version": devices[0].client.platform_version,
    "devices": [d.device_kind for d in devices],
    "xla_flags": os.environ.get("XLA_FLAGS", ""),
    "buckets": [int(b) for b in run_alphafold._BUCKETS.default],
}))
"""

def weights_digest(model_path):
    digest = hashlib.sha256()
    for path in sorted(p for p in model_path.iterdir() if p.is_file()):
        digest.update(path.name.encode())
        with open(path, 'rb') as f:
            digest.update(hashlib.file_digest(f, 'sha256').digest())
    return digest.hexdigest()

def image_identity(image_path):
    """The unique ID that Apptainer assigns to each image it builds, and the image's labels."""
    header = subprocess.check_output(["apptainer", "sif", "header", str(image_path)], text=True)
    sif_id = next(line.split(":", 1)[1].strip() for line in header.splitlines() if line.strip().startswith("ID:"))
    inspect = json.loads(subprocess.check_output(["apptainer", "inspect", "--json", "--labels", str(image_path)]))
    return sif_id, inspect["data"]["attributes"]["labels"] or {}

def probe_environment(gpus, model_path, image_path):
    """
    Collects everything that determines the compiled model, and hence the predictions:
    the AlphaFold3/JAX build, XLA flags, GPU model and driver, model weights, model flags
    and AlphaFold3's bucket sizes. Raises RuntimeError if the GPUs are not all the same model.
    """
    cmd = container_cmd(image_path, ["python", "-c", PROBE_SCRIPT], gpus=gpus, env={"XLA_PYTHON_CLIENT_PREALLOCATE": "false"})
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Could not probe the AlphaFold3 container:\n{proc.stderr}")
    info = json.loads(proc.stdout.strip().splitlines()[-1])

    gpu_names = set(info["devices"])
    if len(gpu_names) != 1:
        raise RuntimeError(f"GPUs {','.join(gpus)} are not all the same model: {', '.join(sorted(gpu_names))}")

    smi = subprocess.check_output(
        ["nvidia-smi", "-i", ",".join(gpus), "--query-gpu=driver_version", "--format=csv,noheader"], text=True
    )
    drivers = smi.split()

    sif_id, labels = image_identity(image_path)

    env = {
        "alphafold3_version": info["alphafold3"],
        "alphafold3_commit": labels.get("alphafold3.commit", "unknown"),
        "image_id": sif_id,
        "jax": info["jax"],
        "jaxlib": info["jaxlib"],
        "xla_backend": " ".join(info["platform_version"].split()),
        "xla_flags": info["xla_flags"],
        "gpu": gpu_names.pop(),
        "nvidia_driver": ",".join(sorted(set(drivers))),
        "model_weights_sha256": weights_digest(model_path),
        "model_flags": " ".join(model_flags()),
        "buckets": ",".join(map(str, info["buckets"])),
    }
    assert tuple(env) == PROBED_KEYS
    return env

# The keys of probe_environment(): what the predictions depend on. environment.txt adds the cache
# entries of the compiled buckets and the GPU they were compiled on, which do not change them.
PROBED_KEYS = (
    "alphafold3_version", "alphafold3_commit", "image_id", "jax", "jaxlib", "xla_backend", "xla_flags",
    "gpu", "nvidia_driver", "model_weights_sha256", "model_flags", "buckets",
)

def environment_record(env):
    """
    The environment that predictions are recorded with in the predictions cache, as (env_hash,
    JSON text), from probe_environment() or environment.txt. The hash is that of the text, which
    holds only PROBED_KEYS, so that it is the same for every GPU of a model and every alphafold3_init.
    """
    text = json.dumps({key: env[key] for key in PROBED_KEYS}, sort_keys=True)
    raw_hash = hashlib.sha256(text.encode()).digest()
    return base64.urlsafe_b64encode(raw_hash).decode().rstrip("="), text

def write_environment(env_path, env):
    with open(env_path, 'w') as f:
        f.write("# Written by alphafold3_init, checked by alphafold3_predict\n")
        for key, val in env.items():
            f.write(f"{key}: {val}\n")

def read_environment(env_path):
    env = {}
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, val = line.rstrip('\n').split(': ', 1)
                env[key] = val
    return env

def compare_environment(recorded, current):
    """Returns (key, recorded value, current value) for every probed key that differs."""
    return [(k, recorded.get(k), v) for k, v in current.items() if recorded.get(k) != v]

def gpu_uuids(gpus):
    smi = subprocess.check_output(["nvidia-smi", "-i", ",".join(gpus), "--query-gpu=uuid", "--format=csv,noheader"], text=True)
    return smi.split()

def parse_compiled_buckets(env):
    """'bucket_256: normal <cache entry>' lines -> {256: ('normal', '<cache entry>'), ...}"""
    compiled = {}
    for key, val in env.items():
        if key.startswith("bucket_"):
            mode, entry = val.split()
            compiled[int(key.removeprefix("bucket_"))] = mode, entry
    return dict(sorted(compiled.items()))
