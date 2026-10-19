"""Inference only: reuse the shared results.pkl for each eps across selected_sizes.

Requires the latest sampling_multiseed_dchsbm.py in the same directory.
Only sampling configuration and common helpers are imported; sampling is not rerun.

Supported behavior:
1. Read p from shared_sampling/p_vals.json and retain selected_sizes for inference.
2. Read n from shared_sampling/n_vals.txt and copy it into the inference directory.
3. Use --selected-sizes for one order combination or --order-sets for several.
4. sample_sizes determines the shared results.pkl location; selected_sizes determines
   the orders used by BP/EM. These settings remain separate.
5. Retain em_iter=1, train_rounds=3, and the main_message_passing_loop.py entry.
   Both HSBM/DC node categories use dropout=0.8; true activities do not tune it.
6. --count_method exact/approx selects and validates existing sampling directories.
   It is not passed to BP and does not trigger sampling.
7. DC paths use the sampling batch's c2/c3/delta/zeta/activity_seed. These locate
   samples and record theory references; the backend receives no true theta/Phi.
8. DC initializes theta from observed degrees and then updates activities with EM.
   --em-iter defaults to 1; more outer iterations do not guarantee convergence.
9. --theta-constraint soft/hard controls the activity constraint; soft is the default.
10. --sample-indices 1 2 3 selects one-based entries from the original sampling seed
    list. Keep the original --num-samples and --seed-base; no samples are regenerated.
11. All selected sample x zeta x epsilon x order-combination tasks share one pool.
    --n-jobs limits total concurrency; results are saved separately per sample/zeta.
12. Summaries default to the common parameter directory beside seed_sample...
    directories. Each summary covers one sampling seed, explicitly named in
    final_results_eps_sweep_z{zeta}_seed_sample{seed_sample}.npz, for example
    final_results_eps_sweep_z0_seed_sample33158374.npz. NPZ seed_sample, seed_mp,
    and array formats are preserved. Python calls with output_root save there.
13. DCHSBM compute_epsilon_c uses activity cross-moments and layer densities for
    selected orders. theory_epsilon_c aligns with selected_hye_sizes_list and is
    for analysis only; it neither schedules dropout nor sets BP tolerance mp_thresh.

The default training entry is main_message_passing_loop.py.
"""

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


from sampling_multiseed_dchsbm import (
    EPS_LIST,
    SAMPLE_SIZES,
    common_parser,
    digest_file,
    eps_name,
    get_root_save_dir,
    normalize_count_method,
    normalize_order_sets,
    normalize_sizes,
    parallel_map,
    read_json,
    resolve_script,
    run_logged,
    sampling_config,
    selected_hye_sizes_list_setup_define,
    validate_parameters,
    validate_dc_parameters,
    validate_sample,
    write_json,
)


# ============================================================
# Theory
# ============================================================

def compute_epsilon_c_dchsbm(
    selected_hye_sizes,
    K=4,
    c_2=10,
    *,
    model="hsbm",
    c2=4.,
    c3=12.,
    delta=.8,
    zeta=0.,
):
    """Return the theoretical epsilon boundary, preserving legacy HSBM arguments/formula.

    DC calls must specify model="dchsbm", c2=..., c3=..., delta=..., zeta=....
    c_2 belongs only to legacy HSBM. DC c2/c3 are the sampler's layer c_bar values:
    mean incident hyperedge counts are nu_2=c2 and nu_3=c3/2, not identical degrees.

    DC assumes this order-2/3 binary-activity experiment: balanced communities,
    activities independent of communities, shared epsilon, and delta_2=delta_3=delta.
    First compute Lambda0=rho(Phi_selected @ diag(c2, c3/9)_selected) at unit signal,
    then solve Lambda0 * ((1-epsilon)/(1+(K-1)*epsilon))**2 = 1.
    If Lambda0<1, this instability boundary is absent in [0,1]; return np.nan.
    This local stability reference assumes known matched parameters and does not
    guarantee convergence when theta is learned.
    """
    if model not in ("hsbm", "dchsbm"):
        raise ValueError("model must be hsbm or dchsbm")
    if isinstance(K, (bool, np.bool_)) or not isinstance(K, (int, np.integer)) or K < 2:
        raise ValueError("K must be an integer >= 2")
    orders = normalize_sizes(selected_hye_sizes)

    if model == "dchsbm":
        if any(d not in (2, 3) for d in orders):
            raise ValueError("DCHSBM theory supports only orders 2 and 3")
        values = np.asarray([c2, c3, delta, zeta], dtype=float)
        if values.shape != (4,) or not np.isfinite(values).all():
            raise ValueError("c2, c3, delta and zeta must be finite scalars")
        c2, c3, delta, zeta = values
        if c2 < 0 or c3 < 0:
            raise ValueError("c2 and c3 must be nonnegative")
        if not 0 <= delta < 1 or not -1 <= zeta <= 1:
            raise ValueError("Require 0 <= delta < 1 and -1 <= zeta <= 1")

        phi = np.array([[1 + delta**2, 1 + zeta * delta**2],
                        [1 + zeta * delta**2, 1 + delta**2]])
        indices = [d - 2 for d in orders]
        # Order-3 gain is 2*nu_3/9=c3/9; keep only the orders used in this inference.
        gains = np.array([c2, c3 / 9.])[indices]
        root_gains = np.sqrt(gains)
        # Symmetrization preserves the eigenvalues of Phi @ diag(gains), including zero gains.
        matrix = root_gains[:, None] * phi[np.ix_(indices, indices)] * root_gains[None, :]
        lambda0 = float(np.linalg.eigvalsh(matrix)[-1])
        if lambda0 < 1:
            return float("nan")
        root = np.sqrt(lambda0)
        return float((root - 1) / (root + K - 1))

    sizes = np.asarray(
        orders,
        dtype=float,
    )

    C_beta = np.sum(
        2 / (sizes - 1)
    )

    P_d = 1 / (
        sizes * (sizes - 1)
    )

    P_d /= P_d.sum()

    F = np.sum(
        sizes * P_d
    )

    gamma = (
        np.exp(
            np.sum(
                P_d
                * np.log(
                    sizes
                    * (sizes - 1)
                    / 2
                )
            )
        )
        / np.sqrt(
            C_beta
            * (F - 1)
            / 2
        )
    )

    phi = (
        (K - 1)
        / K
        / np.sqrt(c_2)
        * gamma
    )

    return (
        1
        / (phi + 1 / K)
        - 1
    ) / (K - 1)


# ============================================================
# Dropout
# ============================================================

def get_dropout_for_case(
    selected_hye_sizes,
    eps,
    K,
    c,
):
    """Keep dropout fixed at 0.8 for both node categories; theory does not schedule it."""
    value = 0.8
    return value, value


# ============================================================
# Utilities
# ============================================================

def object_array(values):
    """Store variable-length hyperedge-order combinations in an object array."""

    result = np.empty(
        len(values),
        dtype=object,
    )

    result[:] = values

    return result


def default_sampling_universe():
    """Return the complete default hyperedge-size universe used by shared sampling.

    selected_sizes specifies the orders used in the current inference.
    sample_sizes specifies the orders originally sampled in the shared results.pkl
    and determines its directory name. Even when inference uses only [2, 3],
    sample_sizes must not be changed automatically to [2, 3].
    """

    if SAMPLE_SIZES is not None:

        return normalize_sizes(
            SAMPLE_SIZES
        )

    default_order_sets = (
        selected_hye_sizes_list_setup_define()
    )

    return normalize_sizes(
        sorted({
            d
            for sizes in default_order_sets
            for d in sizes
        })
    )


# ============================================================
# Main inference for one (eps, selected_sizes)
# ============================================================

def main_message_passing_test(
    project_dir,
    save_dir,
    shared_dir,
    selected_hye_sizes,
    N,
    K,
    seed_mp,
    dropout,
    dropout_vertex,
    dropout_vertex_FVS,
    dropout_vertex_rest,
    dropout_FVS,
    inner_n_jobs,
    backend,
    sample_manifest,
    *,
    model="dchsbm",
    em_iter=1,
    theta_constraint="soft",
):

    save_dir = Path(save_dir)
    shared_dir = Path(shared_dir)

    sizes = normalize_sizes(
        selected_hye_sizes
    )

    # --------------------------------------------------------
    # Shared sampling file
    # --------------------------------------------------------

    sampling_file = (
        shared_dir
        / "sampling_results"
        / "results.pkl"
    )

    # --------------------------------------------------------
    # Inference output
    # --------------------------------------------------------

    mp_dir = (
        save_dir
        / "mp_results"
    )

    inferred_file = (
        mp_dir
        / "inferred_params.npz"
    )

    manifest_file = (
        save_dir
        / "inference_manifest.json"
    )

    # --------------------------------------------------------
    # Inference configuration
    # --------------------------------------------------------

    config = dict(

        schema=2,
        model=model,
        fixed_p=True,
        fixed_n=True,
        theta_source="observed_graph" if model == "dchsbm" else "unit",
        learn_theta=(model == "dchsbm"),
        theta_constraint=theta_constraint if model == "dchsbm" else None,
        # Fingerprint core model files as well as the entry point so model edits invalidate inference reuse.
        model_sha256={
            name: digest_file(Path(project_dir) / "src/model" / name)
            for name in (
                "loop_hypergraph_block_model_rust.py",
                "loop_degree_corrected_hypergraph_block_model_rust.py",
            ) if (Path(project_dir) / "src/model" / name).is_file()
        },

        selected_hye_sizes=sizes,

        N=N,
        K=K,

        seed_mp=seed_mp,

        dropout=dropout,

        dropout_vertex=dropout_vertex,

        dropout_vertex_FVS=dropout_vertex_FVS,

        dropout_vertex_rest=dropout_vertex_rest,

        dropout_FVS=dropout_FVS,

        em_iter=em_iter,

        em_thresh=1e-5,

        mp_iter=1200,

        mp_thresh=1e-5,

        mp_patience=50,

        train_rounds=3,

        inner_n_jobs=inner_n_jobs,

        backend=str(
            backend
        ),

        backend_sha256=digest_file(
            backend
        ),

        sampling_file=str(
            sampling_file.resolve()
        ),

        # Include hashes of shared sampling files:
        # p, n, labels, results.pkl, and related files.
        sampling_sha256=sample_manifest[
            "sha256"
        ],
    )

    # ========================================================
    # Reuse completed inference
    # ========================================================

    if manifest_file.exists():

        old = read_json(
            manifest_file
        )

        if old.get("config") != config:

            raise ValueError(
                f"Inference configuration changed: "
                f"{save_dir}. "
                f"Use a new --run-tag."
            )

        if (
            old.get("complete")
            and inferred_file.exists()
            and digest_file(
                inferred_file
            )
            == old.get(
                "result_sha256"
            )
        ):

            print(
                f"[reuse inference] "
                f"{inferred_file}",
                flush=True,
            )

            return inferred_file

        raise RuntimeError(
            f"Incomplete/changed inference "
            f"result: {save_dir}. "
            f"Use a new --run-tag."
        )

    # ========================================================
    # Do not overwrite unknown directories
    # ========================================================

    if (
        save_dir.exists()
        and any(
            save_dir.iterdir()
        )
    ):

        raise FileExistsError(
            f"Unverified inference directory: "
            f"{save_dir}. "
            f"Use a new --run-tag."
        )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # p
    #
    # Read p for all sampled orders,
    # then write only p for the selected inference sizes.
    # ========================================================

    shared_p_file = (
        shared_dir
        / "p_vals.json"
    )

    if not shared_p_file.is_file():

        raise FileNotFoundError(
            f"Missing shared p file: "
            f"{shared_p_file}"
        )

    all_p = read_json(
        shared_p_file
    )

    missing_p_orders = [
        d
        for d in sizes
        if str(d) not in all_p
    ]

    if missing_p_orders:

        raise ValueError(
            f"p_vals.json does not contain "
            f"orders {missing_p_orders}. "
            f"Available orders: "
            f"{sorted(all_p.keys())}"
        )

    selected_p = {
        str(d): all_p[str(d)]
        for d in sizes
    }

    write_json(
        save_dir
        / "p_vals.json",
        selected_p,
    )

    # ========================================================
    # n
    #
    # Read n and p from the same shared sampling run.
    #
    # n has no hyperedge-order axis,
    # so selected_sizes does not slice it as it does p;
    # copy it in full.
    # ========================================================

    shared_n_file = (
        shared_dir
        / "n_vals.txt"
    )

    if not shared_n_file.is_file():

        raise FileNotFoundError(
            f"Missing shared n file: "
            f"{shared_n_file}"
        )

    n_vals = np.loadtxt(
        shared_n_file,
        dtype=float,
        ndmin=1,
    )

    # --------------------------------------------------------
    # Validate n
    # --------------------------------------------------------

    if n_vals.size != K:

        raise ValueError(
            f"Expected n to contain K={K} "
            f"values, but got "
            f"{n_vals.size}: "
            f"{shared_n_file}"
        )

    if not np.all(
        np.isfinite(n_vals)
    ):

        raise ValueError(
            f"n contains non-finite values: "
            f"{n_vals}"
        )

    if np.any(
        n_vals < 0
    ):

        raise ValueError(
            f"n contains negative values: "
            f"{n_vals}"
        )

    if not np.isclose(
        n_vals.sum(),
        1.0,
    ):

        raise ValueError(
            f"Community probabilities "
            f"must sum to 1. "
            f"Got sum={n_vals.sum()}, "
            f"n={n_vals}"
        )

    # --------------------------------------------------------
    # Save n into the case directory
    # --------------------------------------------------------

    np.savetxt(
        save_dir
        / "n_vals.txt",
        n_vals,
        fmt="%.18g",
    )

    # ========================================================
    # Selected hyperedge sizes
    # ========================================================

    np.savetxt(
        save_dir
        / "selected_hye_sizes.txt",
        sizes,
        fmt="%d",
    )

    np.savetxt(
        save_dir
        / "fix_hyper_edge.txt",
        sizes,
        fmt="%d",
    )

    # ========================================================
    # Write incomplete manifest before inference starts
    # ========================================================

    write_json(
        manifest_file,
        dict(
            config=config,
            complete=False,
        ),
    )

    # ========================================================
    # Message passing command
    # ========================================================

    command = [

        sys.executable,

        backend,

        # --------------------------
        # EM
        # --------------------------

        "--em_iter",
        em_iter,

        "--em_thresh",
        "1e-5",

        # --------------------------
        # MP
        # --------------------------

        "--mp_iter",
        "1200",

        "--mp_thresh",
        "1e-5",

        "--mp_patience",
        "50",

        "--train_rounds",
        "3",

        # --------------------------
        # Random seed
        # --------------------------

        "--seed",
        seed_mp,

        # --------------------------
        # Dropout
        # --------------------------

        "--dropout",
        dropout,

        "--dropout_vertex",
        dropout_vertex,

        "--dropout_vertex_FVS",
        dropout_vertex_FVS,

        "--dropout_vertex_rest",
        dropout_vertex_rest,

        "--dropout_FVS",
        dropout_FVS,

        # --------------------------
        # Ground truth labels
        # --------------------------

        "--node_assignments",
        shared_dir
        / "node_assignment_loop.txt",

        # --------------------------
        # Output
        # --------------------------

        "--save_dir",
        mp_dir,

        # --------------------------
        # Shared sampled hypergraph
        # --------------------------

        "--hye_file",
        sampling_file,

        # --------------------------
        # Current selected sizes
        # --------------------------

        "--selected_hye_sizes",
        save_dir
        / "selected_hye_sizes.txt",

        "--fix_hye_size",
        save_dir
        / "fix_hyper_edge.txt",

        # --------------------------
        # p
        # --------------------------

        "--p",
        save_dir
        / "p_vals.json",

        # --------------------------
        # n
        # --------------------------

        "--n",
        save_dir
        / "n_vals.txt",

        # --------------------------
        # Model size
        # --------------------------

        "--N",
        N,

        "--K",
        K,

        # --------------------------
        # Parallelism
        # --------------------------

        "--n_jobs",
        inner_n_jobs,
    ]

    # ========================================================
    # Run inference
    # ========================================================

    # --n fixes n; all selected orders enter --fix_hye_size, so p is also fixed.
    # Omit --theta and --fix_theta_sizes so true theta never enters inference.
    command += ["--model", model]
    if model == "dchsbm":
        command += ["--learn_theta", "--theta_constraint", theta_constraint]

    run_logged(
        command,
        project_dir,
        mp_dir
        / "run_log.txt",
        inner_n_jobs,
    )

    # ========================================================
    # Check inferred output
    # ========================================================

    if not inferred_file.is_file():

        raise RuntimeError(
            f"Inference did not create: "
            f"{inferred_file}"
        )

    # ========================================================
    # Validate inferred result
    # ========================================================

    with np.load(
        inferred_file,
        allow_pickle=True,
    ) as data:

        required = {

            "overlap",

            "best_free_energy",

            "n",

            "p",

            "em_total_time",

            "update_message_ratio",

            "update_vertex_ratio",
        }

        if model == "dchsbm":
            required.update({"theta", "selected_hye_sizes", "em_converged", "em_stop_reason"})

        missing = (
            required
            - set(
                data.files
            )
        )

        if missing:

            raise ValueError(
                f"Missing result fields: "
                f"{missing}"
            )

        # Preserve fixed parameters; align by order because backend output may reorder sizes.
        if not np.allclose(data["n"], n_vals, rtol=1e-12, atol=1e-14):
            raise ValueError("Backend changed the supplied fixed n")
        if "selected_hye_sizes" in data.files:
            output_orders = np.asarray(data["selected_hye_sizes"], dtype=int).tolist()
            if sorted(output_orders) != sizes:
                raise ValueError("Backend returned different hyperedge orders")
            expected_p = np.asarray([selected_p[str(d)] for d in output_orders])
            if data["p"].shape != expected_p.shape or not np.allclose(
                    data["p"], expected_p, rtol=1e-12, atol=1e-14):
                raise ValueError("Backend changed the supplied fixed p")

    # ========================================================
    # Complete inference manifest
    # ========================================================

    write_json(
        manifest_file,
        dict(

            config=config,

            complete=True,

            result_sha256=digest_file(
                inferred_file
            ),
        ),
    )

    return inferred_file


# ============================================================
# Sweep over eps and selected order sets
# ============================================================

def loop_inference_parallel_eps(
    N=30000,
    K=4,
    c=10,
    eps_list=None,
    n_jobs=50,
    seed_sample=20,
    seed_mp=123,
    dropout=0,
    dropout_vertex=0,
    dropout_FVS=0,
    selected_hye_sizes_list=None,
    sample_sizes=None,
    label_seed=100,
    project_dir=None,
    output_root=None,
    mp_script="main_message_passing_loop.py",
    run_tag="default",
    *,
    count_method="exact",
    model="dchsbm",
    c2=4., c3=12., delta=.8, zeta=0., activity_seed=1000,
    allow_repeated=None,
    em_iter=1,
    theta_constraint="soft",
    zetas=None,
):
    """Compatible single-sample interface; activity_seed is already resolved and is not derived again."""
    return loop_inference_parallel_samples(
        [seed_sample],
        N=N,
        K=K,
        c=c,
        eps_list=eps_list,
        n_jobs=n_jobs,
        seed_mp=seed_mp,
        dropout=dropout,
        dropout_vertex=dropout_vertex,
        dropout_FVS=dropout_FVS,
        selected_hye_sizes_list=selected_hye_sizes_list,
        sample_sizes=sample_sizes,
        label_seed=label_seed,
        project_dir=project_dir,
        output_root=output_root,
        mp_script=mp_script,
        run_tag=run_tag,
        count_method=count_method,
        model=model,
        c2=c2,
        c3=c3,
        delta=delta,
        zeta=zeta,
        activity_seed=activity_seed,
        allow_repeated=allow_repeated,
        em_iter=em_iter,
        theta_constraint=theta_constraint,
        zetas=zetas,
    )[int(seed_sample)]


def loop_inference_parallel_samples(
    seed_samples, *, n_jobs=50, activity_seeds=None, zetas=None, **options
):
    """Schedule sample x zeta x epsilon x order-combination tasks in one pool.

    activity_seeds may map {seed_sample: effective activity seed}; use its values
    directly. Without this mapping, use options.activity_seed for legacy single-sample
    compatibility. The CLI derives each seed once using the original sampling rule.

    Return {seed_sample: single-sample result}. With zetas=None each result is the
    original four-tuple; with explicit zetas, including one value, it is
    {zeta: original four-tuple}.
    """
    seeds = list(seed_samples)
    if (not seeds or any(isinstance(seed, (bool, np.bool_))
                         or not isinstance(seed, (int, np.integer)) or seed < 0
                         for seed in seeds)):
        raise ValueError("seed_samples must contain nonnegative integer seeds")
    seeds = [int(seed) for seed in seeds]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Sampling seeds must be distinct")
    if isinstance(n_jobs, (bool, np.bool_)) or not isinstance(n_jobs, (int, np.integer)) or n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer")
    if "seed_sample" in options:
        raise ValueError("Use seed_samples for the multi-sample interface")
    if len(seeds) > 1 and options.get("output_root") is not None:
        raise ValueError("Multiple samples cannot share one explicit output_root; "
                         "omit output_root to use separate seed_sample directories")

    if activity_seeds is not None:
        activity_seeds = dict(activity_seeds)
        for seed in seeds:
            if seed not in activity_seeds:
                raise ValueError(f"activity_seeds is missing sampling seed {seed}")
            value = activity_seeds[seed]
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer)) or value < 0):
                raise ValueError("activity_seeds must contain nonnegative integer values")

    multiple_zetas = zetas is not None
    zeta_values = ([float(options.get("zeta", 0.))] if zetas is None
                   else [float(value) for value in zetas])
    if not zeta_values or len(set(zeta_values)) != len(zeta_values):
        raise ValueError("zetas must be a nonempty list of distinct values")
    if any(not np.isfinite(value) or not -1 <= value <= 1 for value in zeta_values):
        raise ValueError("zetas must be finite and in [-1, 1]")
    if options.get("model", "dchsbm") != "dchsbm" and len(zeta_values) > 1:
        raise ValueError("Multiple zetas require model='dchsbm'")
    if options.get("output_root") is not None and len(zeta_values) > 1:
        raise ValueError("output_root is only supported for a single zeta; use the model-specific roots")

    # Materialize once before sharing across sample/zeta to avoid exhausting or reordering generators.
    requested_eps = options.get("eps_list")
    eps_list = list(EPS_LIST if requested_eps is None else requested_eps)
    requested_orders = options.get("selected_hye_sizes_list")
    order_sets = normalize_order_sets(
        selected_hye_sizes_list_setup_define() if requested_orders is None else requested_orders)
    if not eps_list or not order_sets:
        raise ValueError("At least one epsilon and one order set are required")
    options["eps_list"] = eps_list
    options["selected_hye_sizes_list"] = order_sets
    if options.get("sample_sizes") is not None:
        options["sample_sizes"] = normalize_sizes(options["sample_sizes"])

    # n_jobs limits concurrency across all samples/zeta; retain the CPU/Rayon budget allocation.
    total_cpus = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    total_tasks = len(seeds) * len(zeta_values) * len(eps_list) * len(order_sets)
    workers = min(int(n_jobs), total_cpus, total_tasks)
    inner_n_jobs = max(1, total_cpus // workers)
    print(f"[global pool] samples={len(seeds)}; zetas={zeta_values}; tasks={total_tasks}; "
          f"CPU={total_cpus}; workers={workers}; inner_n_jobs={inner_n_jobs}", flush=True)

    # Validate all existing samples first; each prepare closure binds its own directory and metadata.
    # Keep sample innermost so even small n_jobs covers distinct sampling seeds early.
    plan_keys = [(seed, value) for value in zeta_values for seed in seeds]
    plans = []
    for seed, value in plan_keys:
        sample_options = options | {"seed_sample": seed, "zeta": value}
        if activity_seeds is not None:
            sample_options["activity_seed"] = int(activity_seeds[seed])
        plans.append(_prepare_inference_sweep(
            **sample_options, n_jobs=int(n_jobs), total_cpus=total_cpus,
            workers=workers, inner_n_jobs=inner_n_jobs))

    jobs = [(group, plan[1][position])
            for position in range(max(len(plan[1]) for plan in plans))
            for group, plan in enumerate(plans) if position < len(plan[1])]

    def dispatch(job):
        group, item = job
        return group, plans[group][0](item)

    # Submit every task to this single pool, then save each group summary in sequence.
    completed = parallel_map(dispatch, jobs, workers)
    grouped_results = [[] for _ in plans]
    for group, row in completed:
        grouped_results[group].append(row)
    summaries = {seed: {} for seed in seeds}
    for (seed, value), plan, rows in zip(plan_keys, plans, grouped_results):
        summaries[seed][value] = plan[2](rows)
    return (summaries if multiple_zetas else
            {seed: summaries[seed][zeta_values[0]] for seed in seeds})


def _prepare_inference_sweep(
    N=30000,
    K=4,
    c=10,
    eps_list=None,
    n_jobs=50,
    seed_sample=20,
    seed_mp=123,
    dropout=0,
    dropout_vertex=0,
    dropout_FVS=0,
    selected_hye_sizes_list=None,
    sample_sizes=None,
    label_seed=100,
    project_dir=None,
    output_root=None,
    mp_script="main_message_passing_loop.py",
    run_tag="default",
    *,
    count_method="exact",
    model="dchsbm",
    c2=4., c3=12., delta=.8, zeta=0., activity_seed=1000,
    allow_repeated=None,
    em_iter=1,
    theta_constraint="soft",
    total_cpus=1, workers=1, inner_n_jobs=1,
):

    count_method = normalize_count_method(count_method)
    if model not in ("hsbm", "dchsbm"):
        raise ValueError("model must be hsbm or dchsbm")
    if not isinstance(em_iter, (int, np.integer)) or em_iter < 1:
        raise ValueError("em_iter must be a positive integer")
    if theta_constraint not in ("soft", "hard"):
        raise ValueError("theta_constraint must be soft or hard")
    # Use the exact sampling configuration to locate and validate the existing sample.
    sample_options = dict(model=model, c2=c2, c3=c3, delta=delta, zeta=zeta,
                          activity_seed=activity_seed, allow_repeated=allow_repeated)

    # ========================================================
    # Project directory
    # ========================================================

    project_dir = Path(
        project_dir
        or Path(__file__).resolve().parent
    ).resolve()

    # ========================================================
    # Selected sizes used by inference
    # ========================================================

    order_sets = normalize_order_sets(

        selected_hye_sizes_list

        if selected_hye_sizes_list is not None

        else selected_hye_sizes_list_setup_define()
    )

    # ========================================================
    # Sampling universe
    #
    # Do not change this automatically based on order_sets.
    #
    # Thus:
    #
    # shared sample:
    #   [2,3,4,5,6,14,15,16,17]
    #
    # inference:
    #   [2,3]
    #
    # the same results.pkl remains accessible.
    # ========================================================

    universe = normalize_sizes(

        sample_sizes

        if sample_sizes is not None

        else default_sampling_universe()
    )

    # ========================================================
    # Selected sizes must belong to the sampled universe.
    # ========================================================

    for sizes in order_sets:

        missing = (
            set(sizes)
            - set(universe)
        )

        if missing:

            raise ValueError(
                f"Selected orders {sizes} contain "
                f"orders not present in sampled "
                f"universe: {sorted(missing)}. "
                f"Sampling universe={universe}"
            )

    # ========================================================
    # eps
    # ========================================================

    eps_list = list(

        EPS_LIST

        if eps_list is None

        else eps_list
    )

    validate_parameters(
        N,
        K,
        c,
        eps_list,
        universe,
    )

    # ========================================================
    # General validation
    # ========================================================

    if n_jobs < 1:

        raise ValueError(
            "n_jobs must be positive."
        )

    if (
        not run_tag
        or any(
            ch not in
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789_-"
            for ch in run_tag
        )
    ):

        raise ValueError(
            "run-tag must use only letters, "
            "digits, '_' or '-'."
        )

    for value in (
        dropout,
        dropout_vertex,
        dropout_FVS,
    ):

        if not 0 <= value <= 1:

            raise ValueError(
                "Dropout values must "
                "be in [0, 1]."
            )

    # Save theory for the actual inference orders; true activities are not passed to BP.
    theory_epsilon_c = np.asarray([
        compute_epsilon_c_dchsbm(sizes, K, c, model=model,
                          c2=c2, c3=c3, delta=delta, zeta=zeta)
        for sizes in order_sets
    ])
    for sizes, epsilon_c in zip(order_sets, theory_epsilon_c):
        print(f"[theory model={model} zeta={zeta:g}] sizes={sizes} "
              f"epsilon_c={epsilon_c:.12g} (reference only)", flush=True)

    # ========================================================
    # MP backend
    # ========================================================

    backend = resolve_script(
        project_dir,
        mp_script,
    )

    # ========================================================
    # Root of shared samples
    # ========================================================

    root = (

        Path(
            output_root
        ).resolve()

        if output_root is not None

        else get_root_save_dir(
            project_dir,
            seed_sample,
            N,
            K,
            c,
            universe,
            label_seed,
            count_method=count_method,
            **sample_options,
        )
    )

    # ========================================================
    # CPU allocation
    # ========================================================

    # The single outer task pool allocates the CPU budget.

    # ========================================================
    # Run name
    # ========================================================

    run_name = (

        f"inference_"
        f"seed{seed_mp}_"
        f"dropout{dropout:g}_"
        f"vertex{dropout_vertex:g}_"
        f"FVS{dropout_FVS:g}_"
        f"{run_tag}"
    )

    # ========================================================
    # Validate all shared samples before inference
    #
    # Validate the large results.pkl only once per eps.
    # ========================================================

    manifests = {}

    for eps in eps_list:

        shared = (
            root
            / eps_name(eps)
            / "shared_sampling"
        )

        expected = sampling_config(
            N=N,
            K=K,
            c=c,
            eps=eps,
            # Validate sampling configuration without sampling; ignore block_size for this check.
            block_size=None,
            seed_sample=seed_sample,
            sample_sizes=universe,
            label_seed=label_seed,
            count_method=count_method,
            **sample_options,
        )

        _, manifests[eps] = (
            validate_sample(
                shared,
                expected,
            )
        )

    print(
        f"CPU={total_cpus}; "
        f"workers={workers}; "
        f"inner_n_jobs={inner_n_jobs}"
    )

    print(
        f"sample_sizes={universe}"
    )

    print(f"count_method={count_method}; shared_sample_root={root}")

    print(
        f"selected_hye_sizes_list="
        f"{order_sets}"
    )

    # ========================================================
    # One inference task
    # ========================================================

    def task(item):

        eps_idx, idx = item

        eps = eps_list[
            eps_idx
        ]

        sizes = order_sets[
            idx
        ]

        eps_dir = (
            root
            / eps_name(eps)
        )

        shared = (
            eps_dir
            / "shared_sampling"
        )

        case_dir = (

            eps_dir

            / run_name

            / (
                "sizes_"
                + "_".join(
                    map(
                        str,
                        sizes,
                    )
                )
            )
        )

        # ----------------------------------------------------
        # Fixed vertex dropout
        # ----------------------------------------------------

        # Unpack both return values; chained assignment would pass the tuple to each float parameter.
        vertex_fvs, vertex_rest = get_dropout_for_case(sizes, eps, K, c)

        print(
            f"[inference seed={seed_sample} zeta={zeta:g}] "
            f"eps={eps} "
            f"sizes={sizes} vertex_FVS={vertex_fvs:g} vertex_rest={vertex_rest:g}",
            flush=True,
        )

        inferred = (
            main_message_passing_test(

                project_dir,

                case_dir,

                shared,

                sizes,

                N,

                K,

                seed_mp,

                dropout,

                dropout_vertex,

                vertex_fvs,

                vertex_rest,

                dropout_FVS,

                inner_n_jobs,

                backend,

                manifests[eps],
                model=model,
                em_iter=em_iter,
                theta_constraint=theta_constraint,
            )
        )

        # ----------------------------------------------------
        # Read inference result
        # ----------------------------------------------------

        with np.load(
            inferred,
            allow_pickle=True,
        ) as data:

            row = {

                key: np.array(
                    data[key],
                    copy=True,
                )

                for key in (

                    "overlap",

                    "best_free_energy",

                    "n",

                    "p",

                    "em_total_time",

                    "update_message_ratio",

                    "update_vertex_ratio",
                )
            }

        # ----------------------------------------------------
        # Ground-truth empirical label ratio
        # ----------------------------------------------------

        labels = np.loadtxt(

            shared
            / "node_assignment_loop.txt",

            dtype=int,

            ndmin=1,
        )

        ratios = (
            np.bincount(
                labels,
                minlength=K,
            )
            / len(labels)
        )

        row.update(

            eps_idx=eps_idx,

            idx=idx,

            selected_hye_sizes=sizes,

            ratios=ratios,

            sampling_file=str(
                shared
                / "sampling_results"
                / "results.pkl"
            ),

            inferred_params_path=str(
                inferred
            ),
        )

        return row

    # ========================================================
    # Parallel inference
    # ========================================================

    task_list = [

        (e, i)

        for e in range(
            len(eps_list)
        )

        for i in range(
            len(order_sets)
        )
    ]

    def finalize(results):
        shape = (
            len(eps_list),
            len(order_sets),
        )

        arrays = {

            key: np.full(
                shape,
                np.nan,
            )

            for key in (

                "overlap",

                "best_free_energy",

                "em_total_time",

                "update_message_ratio",

                "update_vertex_ratio",
            )
        }

        arrays["best_n"] = (
            np.zeros(
                shape + (K,)
            )
        )

        for key in (

            "ratios_list",

            "best_p",

            "sampling_files",

            "inferred_params_paths",
        ):

            arrays[key] = np.empty(
                shape,
                dtype=object,
            )

        # ========================================================
        # Fill result arrays
        # ========================================================

        for row in results:

            pos = (
                row["eps_idx"],
                row["idx"],
            )

            for key in (

                "overlap",

                "best_free_energy",

                "em_total_time",

                "update_message_ratio",

                "update_vertex_ratio",
            ):

                arrays[key][pos] = float(
                    row[key]
                    .reshape(-1)[0]
                )

            # ----------------------------------------------------
            # Learned n
            # ----------------------------------------------------

            learned_n = (
                row["n"]
                .reshape(-1)
            )

            if learned_n.size != K:

                raise ValueError(
                    f"Inferred n has size "
                    f"{learned_n.size}, "
                    f"expected K={K}."
                )

            arrays[
                "best_n"
            ][pos] = learned_n

            # ----------------------------------------------------
            # Object fields
            # ----------------------------------------------------

            for key, source in (

                (
                    "ratios_list",
                    "ratios",
                ),

                (
                    "best_p",
                    "p",
                ),

                (
                    "sampling_files",
                    "sampling_file",
                ),

                (
                    "inferred_params_paths",
                    "inferred_params_path",
                ),
            ):

                arrays[key][pos] = (
                    row[source]
                )

        # ========================================================
        # Metadata
        # ========================================================

        metadata = dict(

            selected_hye_sizes_list=object_array(
                order_sets
            ),

            theory_epsilon_c=theory_epsilon_c,
            # Both explicit values override the backend dropout_vertex fallback; vertex0 alone can mislead.
            dropout_vertex_FVS=get_dropout_for_case(None, None, K, c)[0],
            dropout_vertex_rest=get_dropout_for_case(None, None, K, c)[1],

            sample_sizes=np.asarray(
                universe,
                dtype=int,
            ),

            N=N,

            K=K,

            c=c if model == "dchsbm" else np.nan,
            model=model,
            c_by_order={str(d): c2 if d == 2 else c3 for d in universe} if model == "dchsbm" else {},
            delta=delta if model == "dchsbm" else np.nan,
            zeta=zeta if model == "dchsbm" else np.nan,
            activity_seed=activity_seed,
            theta_source="observed_graph" if model == "dchsbm" else "unit",
            theta_constraint=theta_constraint if model == "dchsbm" else "none",
            em_iter=em_iter,

            seed_sample=seed_sample,

            seed_mp=seed_mp,

            label_seed=label_seed,

            count_method=count_method,

            dropout_list=dropout,

            dropout_vertex_list=dropout_vertex,

            dropout_FVS=dropout_FVS,
        )

        # ========================================================
        # Unique identifier for selected order sets
        # ========================================================

        order_tag = hashlib.sha256(

            json.dumps(
                order_sets
            ).encode()

        ).hexdigest()[:12]

        # ========================================================
        # Save each eps separately
        # ========================================================

        for e, eps in enumerate(
            eps_list
        ):

            result_dir = (
                root
                / eps_name(eps)
                / run_name
            )

            result_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            filename = (

                result_dir

                / (
                    f"final_results_eps_"
                    f"{order_tag}.npz"
                )
            )

            np.savez(

                filename,

                eps=eps,

                **metadata,

                **{
                    key: value[e]

                    for key, value
                    in arrays.items()
                },
            )

        # ========================================================
        # Summary
        # ========================================================

        # The default root is parameter-directory/seed_sample...; save summaries in the parameter directory.
        # Honor custom output_root so results stay within the caller-specified directory.
        summary_dir = (

            (root.parent if output_root is None else root)

            / "summary"

            / run_name
        )

        summary_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        sweep_tag = hashlib.sha256(

            json.dumps(
                [
                    float(e)
                    for e in eps_list
                ]
            ).encode()

        ).hexdigest()[:12]

        if model == "dchsbm":
            # Replace the summary filename hash with the actual zeta, before the sampling seed.
            zeta_tag = (
                "0" if float(zeta) == 0.0 else
                np.format_float_positional(float(zeta), unique=True, trim="-")
            )
            final_file = summary_dir / (
                f"final_results_eps_sweep_z{zeta_tag}_seed_sample{seed_sample}.npz"
            )
        else:
            # HSBM has no zeta parameter; keep its existing summary naming.
            final_file = summary_dir / (
                f"final_results_eps_sweep_seed_sample{seed_sample}_"
                f"{order_tag}_{sweep_tag}.npz"
            )

        np.savez(

            final_file,

            eps_list=np.asarray(
                eps_list,
                dtype=float,
            ),

            **metadata,

            **arrays,
        )

        print(
            f"[saved] {final_file}"
        )

        return (
            arrays["overlap"],
            arrays["best_free_energy"],
            arrays["ratios_list"],
            arrays["best_n"],
        )


    return task, task_list, finalize


# ============================================================
# Inference-specific option resolver
# ============================================================

def select_sample_seeds(seeds, indices=None):
    """Select existing sample seeds by one-based input-list positions, not directory order."""
    pool = list(seeds)
    if indices is None:
        return pool
    requested = list(indices)
    if not requested:
        raise ValueError("--sample-indices must contain at least one index")
    if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
           for i in requested):
        raise ValueError("--sample-indices must contain integer indices starting at 1")
    if len(set(requested)) != len(requested):
        raise ValueError("--sample-indices must be distinct")
    if any(i < 1 or i > len(pool) for i in requested):
        raise ValueError(
            f"--sample-indices must be between 1 and {len(pool)}; got {requested}. "
            "Keep --num-samples equal to the original sample count, or supply "
            "the original ordered list using --seed-samples."
        )
    return [pool[int(i) - 1] for i in requested]


def resolve_inference_options(
    args,
):
    """Handle sample_sizes and selected_sizes independently.

    sample_sizes is the complete order set originally used for shared sampling.
    It determines paths such as shared_samples/N..._D2_3_4_5_6_14_15_16_17/.
    selected_sizes is the order set used by current inference and may be any subset.

    Supported modes:
    1. --selected-sizes 2 3 15 16 17
    2. --order-sets '[[2,3],[2,3,4],[2,3,15,16,17]]'
    3. With neither option, use sampling_multiseed_dchsbm.py defaults.
    """

    # ========================================================
    # Sampling universe
    # ========================================================

    if args.sample_sizes is not None:

        universe = normalize_sizes(
            args.sample_sizes
        )

    else:

        universe = (
            default_sampling_universe()
        )

    # ========================================================
    # selected sizes
    # ========================================================

    if (
        args.selected_sizes is not None
        and args.order_sets is not None
    ):

        raise ValueError(
            "Use either --selected-sizes "
            "or --order-sets, not both."
        )

    # --------------------------------------------------------
    # One selected set
    # --------------------------------------------------------

    if args.selected_sizes is not None:

        orders = [

            normalize_sizes(
                args.selected_sizes
            )
        ]

    # --------------------------------------------------------
    # Multiple selected sets
    # --------------------------------------------------------

    elif args.order_sets is not None:

        try:

            raw_order_sets = json.loads(
                args.order_sets
            )

        except json.JSONDecodeError as error:

            raise ValueError(
                "Invalid JSON in --order-sets."
            ) from error

        orders = normalize_order_sets(
            raw_order_sets
        )

    # --------------------------------------------------------
    # Default selected sets
    # --------------------------------------------------------

    else:

        orders = normalize_order_sets(
            selected_hye_sizes_list_setup_define()
        )

    # ========================================================
    # Selected sets must be subsets of sampling universe
    # ========================================================

    for sizes in orders:

        missing = (
            set(sizes)
            - set(universe)
        )

        if missing:

            raise ValueError(
                f"Selected sizes {sizes} contain "
                f"orders that were not sampled: "
                f"{sorted(missing)}.\n"
                f"Sampling universe = {universe}\n"
                f"If the shared sample was created "
                f"with another universe, pass it "
                f"explicitly using --sample-sizes."
            )

    # ========================================================
    # Sampling seeds
    # ========================================================

    if args.num_samples < 1:

        raise ValueError(
            "--num-samples must be positive."
        )

    if args.seed_samples is not None:

        seeds = [
            int(seed)
            for seed
            in args.seed_samples
        ]

    else:

        rng = np.random.default_rng(
            args.seed_base
        )

        seeds = [

            int(seed)

            for seed in rng.integers(
                0,
                2**31 - 1,
                size=args.num_samples,
            )
        ]

    if (
        len(set(seeds))
        != len(seeds)
    ):

        raise ValueError(
            "Sampling seeds must be distinct."
        )

    if any(
        seed < 0
        for seed in seeds
    ):

        raise ValueError(
            "Sampling seeds must "
            "be nonnegative integers."
        )

    # ========================================================
    # Validate basic model parameters
    # ========================================================

    # Reconstruct the full original sampling seed list before selection; never regenerate or renumber the subset.
    # Without sample_indices, retain the original behavior of running every sample.
    seeds = select_sample_seeds(seeds, getattr(args, "sample_indices", None))

    validate_parameters(
        args.N,
        args.K,
        args.c,
        args.eps,
        universe,
    )

    if args.model == "dchsbm":
        if len(set(args.zetas)) != len(args.zetas):
            raise ValueError("zetas must be distinct")
        for zeta in args.zetas:
            validate_dc_parameters(args.N, args.K, universe, args.c2, args.c3,
                                   args.delta, zeta, args.activity_seed)
    if args.em_iter < 1:
        raise ValueError("--em-iter must be positive")

    return (
        orders,
        universe,
        seeds,
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = common_parser(
        __doc__
    )

    # ========================================================
    # Parallelism
    # ========================================================

    parser.add_argument(
        "--n-jobs", "--n_jobs",
        type=int,
        default=60,
        help="Maximum concurrent inference tasks across samples, zeta, epsilon and order sets.",
    )

    # ========================================================
    # MP random seed
    # ========================================================

    parser.add_argument(
        "--seed-mp",
        type=int,
        default=123,
    )

    # ========================================================
    # Dropout
    # ========================================================

    parser.add_argument(
        "--dropout",
        type=float,
        default=0,
    )

    parser.add_argument(
        "--dropout-vertex",
        type=float,
        default=0,
    )

    parser.add_argument(
        "--dropout-FVS",
        type=float,
        default=0,
    )

    # ========================================================
    # MP backend
    # ========================================================

    parser.add_argument(
        "--mp-script",
        default=(
            "main_message_passing_loop.py"
        ),
    )

    # ========================================================
    # Run tag
    # ========================================================

    parser.add_argument(
        "--run-tag",
        default="default",
        help=(
            "Use a new tag for a "
            "separate inference run."
        ),
    )

    # ========================================================
    # NEW:
    #
    # Custom selected hyperedge sizes
    #
    # Example:
    #
    # --selected-sizes 2 3 15 16 17
    #
    # aliases:
    #
    # --select-size
    # --select-sizes
    # ========================================================

    parser.add_argument(

        "--selected-sizes",
        "--select-size",
        "--select-sizes",

        dest="selected_sizes",

        nargs="+",

        type=int,

        default=None,

        help=(
            "Run inference only for one "
            "selected hyperedge-size set. "
            "Example: "
            "--selected-sizes 2 3 15 16 17"
        ),
    )

    parser.add_argument("--em-iter", "--em_iter", type=int, default=1,
                        help="EM outer iterations; fixed p,n, learn theta for DC (default: 1).")
    parser.add_argument("--theta-constraint", "--theta_constraint", choices=("soft", "hard"),
                        default="soft", help="Activity normalization for inferred theta.")

    parser.add_argument(
        "--sample-indices", "--sample_indices", nargs="+", type=int, default=None,
        help=("1-based positions in the original sampling seed list, e.g. 1 2 3. "
              "Keep the original --num-samples and --seed-base. "
              "With --seed-samples, positions refer to that explicit list. "
              "Omit to run all samples."),
    )

    args = parser.parse_args()

    # ========================================================
    # Resolve:
    #
    # orders
    #   = sizes used by current inference
    #
    # universe
    #   = all sizes originally sampled in shared sampling
    #
    # seeds
    #   = seed_sample
    # ========================================================

    (
        orders,
        universe,
        seeds,
    ) = resolve_inference_options(
        args
    )

    print(
        "=" * 70
    )

    print(
        f"sample_sizes "
        f"(shared sampling universe) "
        f"= {universe}"
    )

    print(
        f"selected_hye_sizes_list "
        f"(inference) "
        f"= {orders}"
    )

    print(
        f"seed_sample_list "
        f"= {seeds}"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # Run all selected sampling seeds in one shared pool
    # ========================================================

    sample_indices = (args.sample_indices if args.sample_indices is not None
                      else list(range(1, len(seeds) + 1)))
    print("sample index -> seed_sample: " + ", ".join(
        f"{index} -> {seed}" for index, seed in zip(sample_indices, seeds)
    ), flush=True)

    # Match sampling_multiseed_dchsbm.main exactly and derive each activity seed only once.
    # Reuse the effective seed across all eps/zeta of a sample to read the original hypergraph.
    activity_seeds = {
        seed: (int(np.random.SeedSequence([args.activity_seed, seed]).generate_state(1)[0])
               if args.model == "dchsbm" else args.activity_seed)
        for seed in seeds
    }
    loop_inference_parallel_samples(

        # --------------------------
        # Model
        # --------------------------

        N=args.N,

        K=args.K,

        c=args.c,

        eps_list=args.eps,

        # --------------------------
        # Parallelism
        # --------------------------

        n_jobs=args.n_jobs,

        # --------------------------
        # Sampling seed
        # --------------------------

        seed_samples=seeds,

        # --------------------------
        # MP seed
        # --------------------------

        seed_mp=args.seed_mp,

        # --------------------------
        # Dropout
        # --------------------------

        dropout=args.dropout,

        dropout_vertex=(
            args.dropout_vertex
        ),

        dropout_FVS=(
            args.dropout_FVS
        ),

        # --------------------------
        # Inference sizes
        # --------------------------

        selected_hye_sizes_list=(
            orders
        ),

        # --------------------------
        # Sampling universe
        # --------------------------

        sample_sizes=(
            universe
        ),

        # Match existing samples' counting method to avoid selecting another sampling directory.
        count_method=args.count_method,
        model=args.model,
        c2=args.c2, c3=args.c3, delta=args.delta,
        zetas=args.zetas if args.model == "dchsbm" else [0.],
        activity_seeds=activity_seeds, allow_repeated=args.allow_repeated,
        em_iter=args.em_iter, theta_constraint=args.theta_constraint,

        # --------------------------
        # Labels
        # --------------------------

        label_seed=(
            args.label_seed
        ),

        # --------------------------
        # Project
        # --------------------------

        project_dir=(
            args.project_dir
        ),

        # --------------------------
        # Backend
        # --------------------------

        mp_script=(
            args.mp_script
        ),

        # --------------------------
        # Output run tag
        # --------------------------

        run_tag=(
            args.run_tag
        ),
    )

if __name__ == "__main__":
    main()
