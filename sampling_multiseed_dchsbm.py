"""Sample once per (seed_sample, eps), including every sampling order in one hypergraph.

Run this file before inference_shared_eps.py, both from the project directory.
SELECTED_HYE_SIZES_LIST sets default comparisons; SAMPLE_SIZES=None takes their
union. Fix SAMPLE_SIZES beforehand to compare additional subsets later. Reuse
the same results.pkl rather than resampling with the same seed.

Default experiment: DCHSBM, N=30000, K=4, cbar2=4, cbar3=12, Delta=0.8.
Each (seed_sample, zeta, eps) produces one hypergraph with orders 2 and 3.
For one seed_sample, labels and theta are fixed across eps; per-order activity
marginals stay fixed across zeta. A parameter configuration shares one parent
directory, with replicates in seed_sample... subdirectories. The manifest records
and strictly validates effective activity_seed, which is excluded from parent
directory names. Legacy directories containing its fingerprint are not migrated
automatically; inference must import this version's path helpers.
Retain parallel execution, locks, attempts, resume handling, and manifest publication.
allow_repeated=True is retained; --allow-repeated False uses Bernoulli simple
hypergraphs. --model hsbm accepts scalar --c; DCHSBM uses --c2/--c3 instead.
Inference must separately read the matching zeta path and theta.json.
This file only performs sampling.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


SELECTED_HYE_SIZES_LIST = [[2, 3]]
# Example comparisons: [[2], [2, 14], [2, 14, 15], [2, 14, 15, 16, 17]].
SAMPLE_SIZES = [2, 3]
EPS_LIST = [i / 1000 for i in range(240, 361, 5)]


def normalize_count_method(count_method):
    if not isinstance(count_method, str) or count_method.lower() not in ("exact", "approx"):
        raise ValueError("count_method must be 'exact' or 'approx'.")
    return count_method.lower()


def canonical_sampling_config(config):
    if config is None:
        return None

    config = dict(config)
    # Existing samples from the new backend without count_method used exact counts.
    config["count_method"] = normalize_count_method(config.get("count_method", "exact"))

    # block_size affects implementation/memory only and is excluded from model configuration comparisons.
    config.pop("block_size", None)

    return config

def selected_hye_sizes_list_setup_define():
    return [list(sizes) for sizes in SELECTED_HYE_SIZES_LIST]


def normalize_sizes(sizes):
    values = list(sizes)
    if not values or any(isinstance(d, bool) or int(d) != d or d < 2 for d in values):
        raise ValueError("Hyperedge sizes must be nonempty integers >= 2.")
    if len(set(values)) != len(values):
        raise ValueError("Duplicate hyperedge sizes.")
    return sorted(int(d) for d in values)


def normalize_order_sets(order_sets):
    result = [normalize_sizes(sizes) for sizes in order_sets]
    if not result or len({tuple(s) for s in result}) != len(result):
        raise ValueError("Order sets must be nonempty and distinct.")
    return result


def eps_name(eps):
    # Avoid :.2f so distinct eps values such as 0.101 and 0.104 do not share a directory.
    return "eps" + np.format_float_positional(float(eps), unique=True, trim="-")


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def node_assignments_setup(seed, N, K, mean=None, *, balanced=False):
    probs = np.ones(K) / K if mean is None else np.asarray(mean, dtype=float)
    probs = probs / probs.sum()
    if balanced:
        if N % K or not np.allclose(probs, np.ones(K) / K):
            raise ValueError("This DC experiment requires equal communities and N divisible by K.")
        labels = np.repeat(np.arange(K), N // K)
        np.random.default_rng(seed).shuffle(labels)
        return probs, labels
    labels = np.random.default_rng(seed).choice(K, size=N, p=probs)
    return probs, labels


def p_vals_set(c, eps, n, N, selected_hye_sizes, K, *, c_by_order=None):
    result = {}
    for d in selected_hye_sizes:
        # Inputs are mean affinity scales cbar_d: order-3 p uses 12 directly, not nu3=6 after division by 2.
        c_d = c if c_by_order is None else c_by_order[str(d)]
        p_in = c_d / (N * (eps + (1.0 - eps) * np.sum(n ** 2)))
        p_out = eps * p_in
        result[str(d)] = (np.full((K, K), p_out) +
                          (p_in - p_out) * np.eye(K)).tolist()
    return result


def validate_dc_parameters(N, K, sample_sizes, c2, c3, delta, zeta, activity_seed):
    if not set(sample_sizes).issubset({2, 3}):
        raise ValueError("This correlated-activity experiment only supports orders 2 and 3.")
    if N % K or (N // K) % 2:
        raise ValueError("DCHSBM requires N/K to be an even integer.")
    if any(not np.isfinite(v) or v <= 0 for v in (c2, c3)):
        raise ValueError("c2 and c3 must be finite and positive.")
    if not np.isfinite(delta) or not 0 <= delta < 1:
        raise ValueError("delta must be finite and in [0,1).")
    if not np.isfinite(zeta) or not -1 <= zeta <= 1:
        raise ValueError("zeta must be finite and in [-1,1].")
    hh = (N // K) * (1 + zeta) / 4
    if abs(hh - round(hh)) > 1e-9:
        raise ValueError(f"Community size {N//K} cannot realize zeta={zeta} exactly.")
    if (isinstance(activity_seed, (bool, np.bool_))
            or not isinstance(activity_seed, (int, np.integer)) or activity_seed < 0):
        raise ValueError("activity_seed must be a nonnegative integer.")


def correlated_activities(project_dir, labels, zeta, delta, seed):
    # Reuse the sampler's activity generator to retain its random assignment rule.
    root = str(Path(project_dir).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    sampling = importlib.import_module("src.model.sampling_dc")
    return sampling.make_correlated_activities(labels, zeta, delta, delta, seed=seed)


def get_root_save_dir(project_dir, seed_sample, N, K, c, sample_sizes, label_seed=100,
                      *, count_method="exact", model="hsbm", c2=4., c3=12., delta=.8,
                      zeta=0., activity_seed=1000, allow_repeated=None):
    sizes = "_".join(map(str, normalize_sizes(sample_sizes)))
    c_name = np.format_float_positional(float(c), unique=True, trim="-")
    count_method = normalize_count_method(count_method)
    if model == "dchsbm":
        validate_dc_parameters(N, K, sample_sizes, c2, c3, delta, zeta, activity_seed)
        # Group parent directories by configuration; distinguish replicates with seed_sample subdirectories.
        # activity_seed varies with seed_sample; including it in fingerprints would split one parameter group.
        # Keep the actual activity seed in sampling_config/manifest to prevent misuse or overwriting old samples.
        settings = dict(c2=float(c2), c3=float(c3), delta=float(delta), zeta=float(zeta),
                        label_seed=int(label_seed),
                        allow_repeated=True if allow_repeated is None else bool(allow_repeated))
        token = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
        directory = "shared_samples_dchsbm_new" + ("_approx" if count_method == "approx" else "")
        z_name = np.format_float_positional(float(zeta), unique=True, trim="-")
        return (Path(project_dir) / "data" / "synthetic_loop" / directory /
                f"N{N}_K{K}_z{z_name}_{token}_D{sizes}" / f"seed_sample{seed_sample}")
    if model != "hsbm":
        raise ValueError("model must be hsbm or dchsbm")
    directory = "shared_samples_hsbm_new" + ("_approx" if count_method == "approx" else "")
    if allow_repeated is False:
        directory += "_simple"
    # Exclude seed_mp, dropout, and selected_sizes so inference settings do not alter sampling paths.
    return (Path(project_dir) / "data" / "synthetic_loop" / directory /
            f"N{N}_K{K}_c{c_name}_labels{label_seed}_D{sizes}" /
            f"seed_sample{seed_sample}")


def sampling_config(N, K, c, eps, block_size, seed_sample, sample_sizes, label_seed,
                    *, count_method="exact", model="hsbm", c2=4., c3=12., delta=.8,
                    zeta=0., activity_seed=1000, allow_repeated=None):
    # Distinguish backend versions in cache configuration because their probability laws differ.
    config = dict(
        schema=2,
        model="hsbm",
        sampler="sampling_new_v1",
        count_method=normalize_count_method(count_method),
        N=int(N), K=int(K), c=float(c), eps=float(eps),
        block_size=None if block_size is None else int(block_size),
        seed_sample=int(seed_sample), label_seed=int(label_seed),
        sample_sizes=normalize_sizes(sample_sizes),
        # True: Poisson multiplicities at all orders; False: a Bernoulli simple hypergraph at all orders.
        # This setting controls both backend arguments and manifest records.
        allow_repeated=True if allow_repeated is None else bool(allow_repeated),
    )
    if model == "dchsbm":
        validate_dc_parameters(N, K, config["sample_sizes"], c2, c3, delta, zeta, activity_seed)
        config.pop("c")
        config.update(schema=3, model="dchsbm", label_mode="balanced",
                      c_by_order={str(d): float({2: c2, 3: c3}[d]) for d in config["sample_sizes"]},
                      delta=float(delta), zeta=float(zeta), activity_seed=int(activity_seed),
                      activity_rule="exact_joint_binary_within_each_community_v1")
    elif model != "hsbm":
        raise ValueError("model must be hsbm or dchsbm")
    return config


def resolve_script(project_dir, script):
    path = Path(script)
    path = path if path.is_absolute() else Path(project_dir) / path
    if not path.is_file():
        raise FileNotFoundError(f"Backend script not found: {path}")
    return path.resolve()


def run_logged(command, project_dir, log_file, threads=1):
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(RAYON_NUM_THREADS=str(threads), OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONIOENCODING="utf-8")
    with log_file.open("w", encoding="utf-8") as log:
        result = subprocess.run([str(s) for s in command], cwd=str(project_dir),
                                stdout=log, stderr=subprocess.STDOUT, env=env)
    if result.returncode:
        tail = "\n".join(log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-50:])
        raise RuntimeError(f"Subprocess failed ({result.returncode}). Log: {log_file}\n{tail}")


def parallel_map(function, items, n_jobs):
    if n_jobs < 1:
        raise ValueError("n_jobs must be positive.")
    # External processes do the heavy work; threads only schedule them, without joblib.
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        return list(executor.map(function, items))


def validate_sample(shared_dir, expected):
    shared_dir = Path(shared_dir)
    result = shared_dir / "sampling_results" / "results.pkl"
    manifest_file = shared_dir / "sampling_manifest.json"
    if not manifest_file.is_file() or not result.is_file():
        raise FileNotFoundError(f"Sampling missing/incomplete: {shared_dir}. Run sample_shared_eps.py first.")
    manifest = read_json(manifest_file)
    if (canonical_sampling_config(
            manifest.get("config")
        )
        != canonical_sampling_config(
            expected
        )
        or not manifest.get("complete")
        ):
        raise ValueError(
            f"Sampling configuration mismatch/incomplete: {shared_dir}"
        )
    if expected.get("model") == "dchsbm" and not {"theta.json", "activity_summary.json"}.issubset(manifest["sha256"]):
        raise ValueError(f"DCHSBM manifest does not protect its activity files: {shared_dir}")
    for name, digest in manifest["sha256"].items():
        if digest_file(shared_dir / name) != digest:
            raise ValueError(f"Sampling file changed: {shared_dir / name}")
    return result, manifest


@contextmanager
def sampling_lock(shared_dir):
    """Nonblocking OS lock released on process exit; retain the lock file to avoid unlink races."""
    shared_dir = Path(shared_dir)
    shared_dir.mkdir(parents=True, exist_ok=True)
    lock_path = shared_dir / ".sampling.lock"
    with lock_path.open("a+b") as stream:
        # Windows byte-range locks require a byte; all processes lock the same first byte.
        if os.fstat(stream.fileno()).st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(
                f"Cannot acquire sampling lock: {lock_path}. Another sampling process "
                "may be using this directory, or the filesystem may not support locking. "
                "Do not delete the lock file while jobs are running."
            ) from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def sample_test(project_dir, shared_dir, config, sampling_script="main_sampling_dc.py"):
    """Reuse complete samples; retry interrupted ones in a new attempt directory and preserve old files in previous/."""
    config = dict(config)
    config["count_method"] = normalize_count_method(config.get("count_method", "exact"))
    shared_dir = Path(shared_dir).resolve()
    with sampling_lock(shared_dir):
        return _sample_locked(project_dir, shared_dir, config, sampling_script)


def _sample_locked(project_dir, shared_dir, config, sampling_script):
    manifest_file = shared_dir / "sampling_manifest.json"

    if manifest_file.exists():
        try:
            old_manifest = read_json(manifest_file)
        except (json.JSONDecodeError, UnicodeDecodeError):
            old_manifest = None  # If a legacy run stopped while writing its completion marker, preserve its files and resample.

        if old_manifest is not None:
            if (
                not isinstance(old_manifest, dict)
                or canonical_sampling_config(
                    old_manifest.get("config")
                )
                != canonical_sampling_config(
                    config
                )
                ):
                    raise ValueError(
                        f"Sampling configuration mismatch: {shared_dir}"
                    )

            if old_manifest.get("complete"):
                # Report corrupt published results explicitly;
                # do not silently replace samples already used for inference.
                result, _ = validate_sample(
                    shared_dir,
                    config,
                )

                print(
                    f"[reuse sampling] {result}",
                    flush=True,
                )

                return result

    backend = resolve_script(
        project_dir,
        sampling_script,
    )

    # ============================================================
    # Include n_vals.txt.
    # ============================================================
    names = [
        "p_vals.json",
        "n_vals.txt",                 # <-- added
        "selected_hye_sizes.txt",
        "node_assignment_loop.txt",
        "sampling_results",
        "sampling_manifest.json",
    ]
    if config["model"] == "dchsbm":
        # Keep the completion marker last for atomic publication via names[:-1] below.
        names[-1:-1] = ["theta.json", "activity_summary.json"]

    had_previous = any(
        (shared_dir / name).exists()
        for name in names
    )

    attempts_dir = shared_dir / "attempts"
    attempts_dir.mkdir(exist_ok=True)

    attempt = Path(
        tempfile.mkdtemp(
            prefix=datetime.now().strftime(
                "attempt_%Y%m%d_%H%M%S_"
            ),
            dir=attempts_dir,
        )
    )

    write_json(
        attempt / "request.json",
        dict(
            config=config,
            backend=str(backend),
            backend_sha256=digest_file(backend),
        ),
    )

    if had_previous:
        print(
            "[resume sampling] "
            f"Unverified/incomplete files kept; "
            f"new attempt: {attempt}",
            flush=True,
        )

    # ============================================================
    # Generate community proportions n = probs and true labels.
    # ============================================================
    probs, labels = node_assignments_setup(
        config["label_seed"],
        config["N"],
        config["K"],
        balanced=config["model"] == "dchsbm",
    )

    # ============================================================
    # Generate p.
    # ============================================================
    p_vals = p_vals_set(
        config.get("c", 0.),
        config["eps"],
        probs,
        config["N"],
        config["sample_sizes"],
        config["K"],
        c_by_order=config.get("c_by_order"),
    )

    if config["model"] == "dchsbm":
        theta = correlated_activities(project_dir, labels, config["zeta"],
                                      config["delta"], config["activity_seed"])
        theta = {d: theta[d] for d in config["sample_sizes"]}
        write_json(attempt / "theta.json", {str(d): v.tolist() for d, v in theta.items()})
        stacked = np.array([theta[d] for d in config["sample_sizes"]])
        write_json(attempt / "activity_summary.json", {
            "orders": config["sample_sizes"], "zeta": config["zeta"], "delta": config["delta"],
            "Phi_empirical": (stacked @ stacked.T / config["N"]).tolist(),
            "activity_means_by_order": {str(d): float(v.mean()) for d, v in theta.items()},
            "activity_means_by_community": {
                str(a): {str(d): float(v[labels == a].mean()) for d, v in theta.items()}
                for a in range(config["K"])},
            "target_mean_degree_by_order": {str(d): config["c_by_order"][str(d)] / (d-1)
                                            for d in config["sample_sizes"]},
        })

    # ============================================================
    # Save p.
    # ============================================================
    write_json(
        attempt / "p_vals.json",
        p_vals,
    )

    # ============================================================
    # Save n = probs as well.
    #
    # For example, with K=4:
    # [
    #     0.25,
    #     0.25,
    #     0.25,
    #     0.25
    # ]
    # ============================================================
    np.savetxt(
        attempt / "n_vals.txt",
        probs,
        fmt="%.18g",
    )

    # Save the sampled hyperedge orders.
    np.savetxt(
        attempt / "selected_hye_sizes.txt",
        config["sample_sizes"],
        fmt="%d",
    )

    # Save true node labels.
    np.savetxt(
        attempt / "node_assignment_loop.txt",
        labels,
        fmt="%d",
    )

    # ============================================================
    # Sampling
    # ============================================================
    sampling_dir = attempt / "sampling_results"

    command = [
        sys.executable,
        backend,

        "--model",
        config["model"],

        "--p",
        attempt / "p_vals.json",

        "--selected_hye_sizes",
        attempt / "selected_hye_sizes.txt",

        "--node_assignments",
        attempt / "node_assignment_loop.txt",

        "--allow_repeated",
        str(config["allow_repeated"]),

        "--count_method",
        config["count_method"],

        # Subprocess arguments are text; the entry point's none_or_type(int) converts this to Python None.
        "--max_hyperedges",
        "None",

        "--max_compositions",
        "None",

        "--seed",
        str(config["seed_sample"]),

        "--save_dir",
        sampling_dir,
    ]

    if config["model"] == "dchsbm":
        command += ["--theta", attempt / "theta.json"]

    if config["block_size"] is not None:
        command += [
            "--block_size",
            str(config["block_size"]),
        ]

    print(
        f"[sampling] "
        f"seed={config['seed_sample']} "
        f"eps={config['eps']} "
        f"model={config['model']} zeta={config.get('zeta', 'NA')} "
        f"sizes={config['sample_sizes']}",
        flush=True,
    )

    run_logged(
        command,
        project_dir,
        sampling_dir / "run_log.txt",
    )

    result = sampling_dir / "results.pkl"

    if (
        not result.is_file()
        or result.stat().st_size == 0
    ):
        raise RuntimeError(
            "Sampling did not create "
            f"a nonempty file: {result}"
        )

    # ============================================================
    # Include the n_vals.txt hash.
    #
    # The manifest then pins p, n, labels, selected sizes,
    # and results.pkl together.
    # ============================================================
    hash_names = [
        "p_vals.json",
        "n_vals.txt",                 # <-- added
        "selected_hye_sizes.txt",
        "node_assignment_loop.txt",
        "sampling_results/results.pkl",
    ]
    if config["model"] == "dchsbm":
        hash_names += ["theta.json", "activity_summary.json"]

    manifest = dict(
        config=config,
        complete=True,
        backend=str(backend),
        backend_sha256=digest_file(backend),
        attempt=str(attempt),

        sha256={
            name: digest_file(attempt / name)
            for name in hash_names
        },
    )

    # ============================================================
    # Publish the completion marker atomically and last.
    # ============================================================
    ready_manifest = (
        attempt / "ready_manifest.json"
    )

    write_json(
        ready_manifest,
        manifest,
    )

    # ============================================================
    # Preserve any previous files.
    # ============================================================
    if had_previous:
        backup = attempt / "previous"
        backup.mkdir()

        for name in names:
            existing = shared_dir / name

            if existing.exists():
                existing.rename(
                    backup / name
                )

        print(
            f"[previous files preserved] {backup}",
            flush=True,
        )

    # ============================================================
    # Publish files.
    #
    # names[:-1] includes:
    #
    # p_vals.json
    # n_vals.txt
    # selected_hye_sizes.txt
    # node_assignment_loop.txt
    # sampling_results
    #
    # Thus n moves to shared_dir together with p.
    # ============================================================
    for name in names[:-1]:
        (attempt / name).rename(
            shared_dir / name
        )

    os.replace(
        ready_manifest,
        manifest_file,
    )

    return (
        shared_dir
        / "sampling_results"
        / "results.pkl"
    )


def loop_sampling_parallel_eps(N=30000, K=4, c=10, eps_list=None, n_jobs=25,block_size = None,
                               seed_sample=20, sample_sizes=None, label_seed=100,
                               project_dir=None, output_root=None,
                               sampling_script="main_sampling_dc.py", *, count_method="exact",
                               model="hsbm", c2=4., c3=12., delta=.8, zeta=0.,
                               activity_seed=1000, allow_repeated=None):
    count_method = normalize_count_method(count_method)
    project_dir = Path(project_dir or Path(__file__).resolve().parent).resolve()
    sample_sizes = normalize_sizes(sample_sizes if sample_sizes is not None else
                                   sorted({d for s in SELECTED_HYE_SIZES_LIST for d in s}))
    eps_list = list(EPS_LIST if eps_list is None else eps_list)
    validate_parameters(N, K, c, eps_list, sample_sizes)
    settings = dict(count_method=count_method, model=model, c2=c2, c3=c3, delta=delta,
                    zeta=zeta, activity_seed=activity_seed, allow_repeated=allow_repeated)
    if model == "dchsbm":
        validate_dc_parameters(N, K, sample_sizes, c2, c3, delta, zeta, activity_seed)
    root = (Path(output_root).resolve() if output_root is not None else
            get_root_save_dir(project_dir, seed_sample, N, K, c, sample_sizes, label_seed,
                              **settings))
    def task(eps):
        config = sampling_config(N, K, c, eps, block_size, seed_sample, sample_sizes, label_seed,
                                 **settings)
        return sample_test(project_dir, root / eps_name(eps) / "shared_sampling",
                           config, sampling_script)
    parallel_map(task, eps_list, n_jobs)
    return root


def validate_parameters(N, K, c, eps_list, sample_sizes):
    if N < 2 or K < 2 or K > N or not np.isfinite(c) or c <= 0:
        raise ValueError("Require N >= K >= 2 and finite c > 0.")
    if max(sample_sizes) > N:
        raise ValueError("Hyperedge size exceeds N.")
    if not eps_list or any(not np.isfinite(e) or e < 0 or e > 1 for e in eps_list):
        raise ValueError("eps values must be finite and in [0, 1].")
    if len({eps_name(e) for e in eps_list}) != len(eps_list):
        raise ValueError("Duplicate eps values.")


def common_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--N", type=int, default=30000)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--c", type=float, default=10, help="HSBM only: common affinity scale.")
    parser.add_argument("--model", choices=("hsbm", "dchsbm"), default="dchsbm")
    parser.add_argument("--c2", type=float, default=4., help="DCHSBM: cbar2=nu2.")
    parser.add_argument("--c3", type=float, default=12., help="DCHSBM: cbar3=2*nu3.")
    parser.add_argument("--delta", type=float, default=.8, help="Common Delta2=Delta3.")
    parser.add_argument("--zetas", "--zeta", nargs="+", type=float, default=[-1., 0., 1.])
    parser.add_argument("--activity-seed", type=int, default=1000,
                        help="Base seed; effective activity seed also depends on seed_sample.")
    parser.add_argument("--allow-repeated", "--allow_repeated", type=parse_bool, default=None,
                        help="True: Poisson copies (default); False: simple Bernoulli hypergraph.")
    parser.add_argument("--eps", nargs="+", type=float, default=EPS_LIST)
    parser.add_argument("--label-seed", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=123)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--count_method", "--count-method", type=str.lower,
                        choices=("exact", "approx"), default="exact",
                        help="Group-count sampling mode; approximate samples use a separate cache.")

    parser.add_argument("--seed-samples", nargs="+", type=int)
    parser.add_argument("--order-sets", help='JSON, e.g. "[[2], [2,3], [2,3,4]]"')
    parser.add_argument("--sample-sizes", nargs="+", type=int,
                        help="Fixed universe of sampled orders (must contain every order set).")
    return parser


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("Use True or False")


def resolve_options(args):
    orders = normalize_order_sets(json.loads(args.order_sets) if args.order_sets
                                  else selected_hye_sizes_list_setup_define())
    universe = args.sample_sizes if args.sample_sizes is not None else SAMPLE_SIZES
    universe = normalize_sizes(universe if universe is not None else sorted({d for s in orders for d in s}))
    if any(not set(s).issubset(universe) for s in orders):
        raise ValueError("Every selected order set must be a subset of --sample-sizes.")
    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive.")
    seeds = args.seed_samples if args.seed_samples is not None else [int(s) for s in
            np.random.default_rng(args.seed_base).integers(0, 2**31 - 1, size=args.num_samples)]
    if len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("Sampling seeds must be distinct nonnegative integers.")
    validate_parameters(args.N, args.K, args.c, args.eps, universe)
    if args.model == "dchsbm":
        if len(set(args.zetas)) != len(args.zetas):
            raise ValueError("Duplicate zeta values.")
        for zeta in args.zetas:
            validate_dc_parameters(args.N, args.K, universe, args.c2, args.c3,
                                   args.delta, zeta, args.activity_seed)
    return orders, universe, seeds


def main():
    parser = common_parser(__doc__)
    parser.add_argument("--n-jobs", "--n_jobs", type=int, default=1)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--sampling-script", default="main_sampling_dc.py")
    args = parser.parse_args()
    _, universe, seeds = resolve_options(args)
    print(f"model={args.model}; sample_sizes={universe}; seed_sample_list={seeds}")
    if args.model == "dchsbm":
        print(f"cbar2={args.c2}; cbar3={args.c3}; nu2={args.c2}; nu3={args.c3/2}; "
              f"delta={args.delta}; zetas={args.zetas}")
    for seed in seeds:
        # Exclude epsilon and zeta to compare one replicate with controlled label/activity correspondence.
        activity_seed = (int(np.random.SeedSequence([args.activity_seed, seed]).generate_state(1)[0])
                         if args.model == "dchsbm" else args.activity_seed)
        for zeta in (args.zetas if args.model == "dchsbm" else [0.]):
            root = loop_sampling_parallel_eps(
                args.N, args.K, args.c, args.eps, args.n_jobs, args.block_size,
                seed, universe, args.label_seed, args.project_dir,
                sampling_script=args.sampling_script, count_method=args.count_method,
                model=args.model, c2=args.c2, c3=args.c3, delta=args.delta, zeta=zeta,
                activity_seed=activity_seed, allow_repeated=args.allow_repeated)
            print(f"[completed sampling] {root}", flush=True)


if __name__ == "__main__":
    main()
