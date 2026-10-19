"""Inference only: reuse the shared results.pkl for each eps across selected_sizes.

Requires the latest sampling_multiseed_dc.py in the same directory. Import only
sampling configuration and common helpers; do not rerun sampling.

Supported behavior:
1. Read all p from shared_sampling/p_vals.json and retain selected_sizes.
2. Read n from shared_sampling/n_vals.txt and copy it into the inference directory.
3. --selected-sizes specifies one order combination; --order-sets specifies several.
4. sample_sizes locates shared results.pkl; selected_sizes chooses BP/EM orders.
5. Retain em_iter=1, train_rounds=3 by default (set with --train-rounds or
   --train_rounds), the dropout rule, and main_message_passing_loop.py.
6. --count_method exact/approx locates and validates existing samples only;
   it is not passed to BP and does not rerun sampling.
7. --sample-indices 1 2 3 selects the original list's first three hypergraphs.
   Keep --num-samples and --seed-base; --seed-samples may specify seed values.
   --seed-mp independently controls BP initialization regardless of selection.
8. Summary names include inference orders and the actual sampling seed, e.g.
   final_results_eps_sweep_2_3_4_5_16_seed_sample_33158374.npz. Multiple order
   groups are separated by double underscores. Different epsilon sweeps append
   a sweep identifier on collision to preserve existing summaries.
9. All selected sample x epsilon x order-combination tasks share one pool.
   --n-jobs limits total concurrency; --sample-indices filters the original list.

The imported sampling_multiseed_dc.py must be supplied separately; it is absent
from this source tree. The default training entry is main_message_passing_loop.py.
"""

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


from sampling_multiseed_dc import (
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
    validate_sample,
    write_json,
)


# ============================================================
# Theory
# ============================================================

def compute_epsilon_c(
    selected_hye_sizes,
    K=4,
    c_2=10,
):
    """Preserve the original theoretical threshold formula; this script reorganizes sampling/inference files."""

    sizes = np.asarray(
        selected_hye_sizes,
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
    """Preserve the existing dropout rule."""

    epsilon_c = compute_epsilon_c(
        selected_hye_sizes,
        K,
        c,
    )

    value = (
            0.8
    )

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

def _validate_train_rounds(value):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < 1):
        raise ValueError("train_rounds must be a positive integer.")
    return int(value)


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
    train_rounds=3,
):

    train_rounds = _validate_train_rounds(train_rounds)
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

        schema=1,

        selected_hye_sizes=sizes,

        N=N,
        K=K,

        seed_mp=seed_mp,

        dropout=dropout,

        dropout_vertex=dropout_vertex,

        dropout_vertex_FVS=dropout_vertex_FVS,

        dropout_vertex_rest=dropout_vertex_rest,

        dropout_FVS=dropout_FVS,

        em_iter=1,

        em_thresh=1e-5,

        mp_iter=1200,

        mp_thresh=1e-5,

        mp_patience=50,

        train_rounds=train_rounds,

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
        "1",

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
        str(train_rounds),

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
    train_rounds=3,
):

    """Compatible single-sample interface using the global scheduler; return the original four-tuple."""
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
        train_rounds=train_rounds,
    )[int(seed_sample)]


def loop_inference_parallel_samples(seed_samples, *, n_jobs=50, **options):
    """Run sample x epsilon x order-combination tasks in one pool and summarize per sample."""
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

    # Materialize once before sharing across samples to support generators and fix scan order.
    requested_eps = options.get("eps_list")
    eps_list = list(EPS_LIST if requested_eps is None else requested_eps)
    if not eps_list:
        raise ValueError("eps_list must be nonempty")
    requested_orders = options.get("selected_hye_sizes_list")
    order_sets = normalize_order_sets(
        selected_hye_sizes_list_setup_define() if requested_orders is None else requested_orders)
    options["eps_list"] = eps_list
    options["selected_hye_sizes_list"] = order_sets
    if options.get("sample_sizes") is not None:
        options["sample_sizes"] = normalize_sizes(options["sample_sizes"])

    # n_jobs limits processes across all samples; child processes retain the Rayon thread allocation.
    total_cpus = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    task_count = len(seeds) * len(eps_list) * len(order_sets)
    workers = min(int(n_jobs), total_cpus, task_count)
    inner_n_jobs = max(1, total_cpus // workers)
    print(f"[global pool] samples={len(seeds)}; tasks={task_count}; "
          f"CPU={total_cpus}; workers={workers}; inner_n_jobs={inner_n_jobs}", flush=True)

    # Validate all samples first; each preparation binds its own directory and metadata in a closure.
    plans = [_prepare_inference_sweep(
        seed_sample=seed, n_jobs=int(n_jobs), total_cpus=total_cpus,
        workers=workers, inner_n_jobs=inner_n_jobs, **options)
        for seed in seeds]
    # Interleave samples so one sample's entire eps sweep does not monopolize the queue.
    jobs = [(sample_index, plan[1][position])
            for position in range(max(len(plan[1]) for plan in plans))
            for sample_index, plan in enumerate(plans) if position < len(plan[1])]

    def dispatch(job):
        sample_index, item = job
        return sample_index, plans[sample_index][0](item)

    completed = parallel_map(dispatch, jobs, workers)
    grouped = [[] for _ in seeds]
    for sample_index, row in completed:
        grouped[sample_index].append(row)
    result = {seed: plan[2](rows) for seed, plan, rows in zip(seeds, plans, grouped)}
    write_combined_summary(
        dict(zip(seeds, grouped)),
        {seed: plan[2].summary_context for seed, plan in zip(seeds, plans)},
    )
    return result


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
    train_rounds=3,
    total_cpus,
    workers,
    inner_n_jobs,
):
    train_rounds = _validate_train_rounds(train_rounds)
    count_method = normalize_count_method(count_method)

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
        )
    )

    # ========================================================
    # CPU allocation
    # ========================================================




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
        # Adaptive vertex dropout
        # ----------------------------------------------------

        (
            vertex_fvs,
            vertex_rest,
        ) = get_dropout_for_case(
            sizes,
            eps,
            K,
            c,
        )

        print(
            f"[inference] "
            f"eps={eps} "
            f"sizes={sizes}",
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
                train_rounds=train_rounds,
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


    # ========================================================
    # Aggregate results
    # ========================================================

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

            sample_sizes=np.asarray(
                universe,
                dtype=int,
            ),

            N=N,

            K=K,

            c=c,

            seed_sample=seed_sample,

            seed_mp=seed_mp,

            train_rounds=train_rounds,

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

        summary_dir = (

            root

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

        # Use actual inference order combinations, preserving group boundaries, rather than the sampled universe.
        order_label = "__".join(
            "_".join(str(d) for d in sizes)
            for sizes in order_sets
        )

        final_file = (

            summary_dir

            / (
                f"final_results_"
                f"eps_sweep_"
                f"{order_label}_"
                f"seed_sample_{seed_sample}.npz"
            )
        )

        # Retain sweep_tag's ability to distinguish eps ranges when generating readable filenames.
        # Reuse names for identical sweeps; append a sweep identifier only when different sweeps collide.
        if final_file.exists():
            with np.load(final_file, allow_pickle=False) as previous:
                same_sweep = np.array_equal(previous["eps_list"], np.asarray(eps_list, dtype=float))
            if not same_sweep:
                final_file = final_file.with_name(f"{final_file.stem}_eps_{sweep_tag}.npz")

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

    finalize.summary_context = dict(
        root=root, run_name=run_name, seed_sample=seed_sample,
        eps_list=eps_list, order_sets=order_sets, N=N, K=K, c=c,
        universe=universe, label_seed=label_seed, seed_mp=seed_mp,
        count_method=count_method,
        train_rounds=train_rounds,
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
    3. With neither option, use sampling_multiseed_dc.py defaults.
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

    # Reconstruct the full sampling seed list, then select by original position without resampling.
    seeds = select_sample_seeds(seeds, getattr(args, "sample_indices", None))

    validate_parameters(
        args.N,
        args.K,
        args.c,
        args.eps,
        universe,
    )

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
        "--n-jobs",
        type=int,
        default=60,
        help="Total concurrent tasks across sampling seeds, epsilons and order sets.",
    )

    # ========================================================
    # MP random seed
    # ========================================================

    parser.add_argument(
        "--seed-mp",
        type=int,
        default=123,
    )

    parser.add_argument(
        "--train-rounds", "--train_rounds",
        dest="train_rounds", type=int, default=3,
        help="Independent training restarts per inference task (default: 3).",
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

    parser.add_argument(
        "--sample-indices", "--sample_indices", nargs="+", type=int, default=None,
        help=("1-based positions in the original sampling seed list, e.g. 1 2 3. "
              "Keep the original --num-samples and --seed-base. "
              "With --seed-samples, positions refer to that explicit list. "
              "Omit to run all samples."),
    )

    args = parser.parse_args()
    if args.train_rounds < 1:
        parser.error("--train-rounds must be a positive integer.")

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
    # Run all selected samples in one shared pool
    # ========================================================

    sample_indices = (args.sample_indices if args.sample_indices is not None
                      else list(range(1, len(seeds) + 1)))
    print("sample index -> seed_sample: " + ", ".join(
        f"{index} -> {seed}" for index, seed in zip(sample_indices, seeds)
    ), flush=True)

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

        train_rounds=args.train_rounds,

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


def write_combined_summary(seed_rows, contexts):
    """Add a dataset-level NPZ/CSV without changing inference or old summaries.

    Each p cell keeps the backend payload exactly as returned, including its
    original layer order and any object/dictionary representation.
    """
    import csv

    if not seed_rows or set(seed_rows) != set(contexts):
        raise ValueError("Summary requires matching, nonempty sample rows and contexts")
    datasets = {}
    for seed, context in contexts.items():
        root = Path(context["root"]).resolve()
        dataset = root.parent if root.name == f"seed_sample{seed}" else root
        datasets.setdefault(dataset, []).append(seed)
    scalar_keys = ("overlap", "best_free_energy", "em_total_time",
                   "update_message_ratio", "update_vertex_ratio")
    common_keys = ("run_name", "eps_list", "order_sets", "N", "K", "c", "universe",
                   "label_seed", "seed_mp", "count_method", "train_rounds")
    outputs = []
    for dataset, sample_seeds in datasets.items():
        seeds = sorted(sample_seeds)
        base = contexts[seeds[0]]
        if any(any(contexts[seed][key] != base[key] for key in common_keys)
               for seed in seeds[1:]):
            raise ValueError("Cannot combine samples with different inference settings")
        eps_list, order_sets, K = base["eps_list"], base["order_sets"], base["K"]
        shape = (len(seeds), len(eps_list), len(order_sets))
        arrays = {key: np.full(shape, np.nan) for key in scalar_keys}
        arrays["best_n"] = np.full(shape + (K,), np.nan)
        arrays["ratios_list"] = np.full(shape + (K,), np.nan)
        # Different order sets may return different numbers of p layers.
        for key in ("best_p", "sampling_files", "inferred_params_paths"):
            arrays[key] = np.empty(shape, dtype=object)
        records, seen = [], set()
        for sample_index, seed in enumerate(seeds):
            ordered = sorted(seed_rows[seed], key=lambda row: (row["eps_idx"], row["idx"]))
            for row in ordered:
                e, g = int(row["eps_idx"]), int(row["idx"])
                position = (sample_index, e, g)
                if not (0 <= e < shape[1] and 0 <= g < shape[2]) or position in seen:
                    raise ValueError("Duplicate or invalid cell in dataset summary")
                seen.add(position)
                record = dict(seed_sample=seed, seed_mp=base["seed_mp"], epsilon=eps_list[e],
                              train_rounds=base["train_rounds"],
                              selected_sizes="_".join(map(str, order_sets[g])),
                              N=base["N"], K=K, c=base["c"],
                              label_seed=base["label_seed"], count_method=base["count_method"])
                for key in scalar_keys:
                    # Match the original per-seed finalizer's scalar extraction.
                    value = float(np.asarray(row[key]).reshape(-1)[0])
                    arrays[key][position] = value
                    record[key] = value if np.isfinite(value) else ""
                for target, source in (("best_n", "n"), ("ratios_list", "ratios")):
                    vector = np.asarray(row[source]).reshape(-1)
                    if vector.size != K:
                        raise ValueError(f"Summary {source} must contain K={K} values")
                    arrays[target][position] = vector
                arrays["best_p"][position] = np.array(row["p"], copy=True)
                for target, source in (("sampling_files", "sampling_file"),
                                       ("inferred_params_paths", "inferred_params_path")):
                    arrays[target][position] = str(row[source])
                    record[source] = str(row[source])
                records.append(record)
        if len(seen) != int(np.prod(shape)):
            raise ValueError("Incomplete sample x epsilon x selected-order-set summary grid")
        for key in scalar_keys:
            arrays[key + "_mean"] = arrays[key].mean(axis=0)
            arrays[key + "_std"] = (arrays[key].std(axis=0, ddof=1) if len(seeds) > 1
                                     else np.full(shape[1:], np.nan))
        identity = dict(
            schema=1, run_name=base["run_name"], seed_samples=[int(seed) for seed in seeds],
            eps_list=[float(eps) for eps in eps_list], order_sets=order_sets,
            N=base["N"], K=K, c=float(base["c"]), sample_sizes=base["universe"],
            label_seed=base["label_seed"], seed_mp=base["seed_mp"],
            count_method=base["count_method"],
            train_rounds=base["train_rounds"],
        )
        tag = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
        metadata = dict(
            schema=1, N=base["N"], K=K, c=base["c"], label_seed=base["label_seed"],
            seed_mp=base["seed_mp"], count_method=base["count_method"], run_name=base["run_name"],
            train_rounds=base["train_rounds"],
            seed_samples=np.asarray(seeds), eps_list=np.asarray(eps_list),
            selected_hye_sizes_list=object_array(order_sets),
            sample_sizes=np.asarray(base["universe"]),
            axes="sample_seed, epsilon, selected_order_set", std_ddof=1,
            p_payload_policy="unchanged backend payload; internal layer ordering preserved",
            scan_identity_json=json.dumps(identity, sort_keys=True),
        )
        directory = dataset / "summary"
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"summary_{base['run_name']}_{tag}"
        npz_path, csv_path = directory / (stem + ".npz"), directory / (stem + ".csv")
        temporary = npz_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **metadata, **arrays)
        os.replace(temporary, npz_path)
        temporary = csv_path.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        os.replace(temporary, csv_path)
        print(f"[dataset summary] {npz_path}\n[dataset table] {csv_path}", flush=True)
        outputs.append((npz_path, csv_path))
    return outputs


if __name__ == "__main__":
    main()
