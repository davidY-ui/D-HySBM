"""Sample and infer fixed-incidence {2,d} experiments around the theoretical collapse boundary."""

import os
import json
import sys
import subprocess
import numpy as np

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False


# ============================================================
# Theory: fixed mean hyperedge incidence
# ============================================================

def fixed_budget_beta(d, x, c0, kbar_h):
    """Fix mean hyperedge incidence kbar_h = c0 * (beta_2 + beta_d / (d - 1)).

    Define x = beta_d / beta_2 and return {2: beta_2, d: beta_d}.
    """
    d = int(d)
    x = float(x)
    c0 = float(c0)
    kbar_h = float(kbar_h)

    if d <= 2:
        raise ValueError("d 必须满足 d >= 3")
    if x < 0:
        raise ValueError("x 必须非负")
    if c0 <= 0 or kbar_h <= 0:
        raise ValueError("c0 和 kbar_h 必须为正")

    beta_2 = (kbar_h / c0) / (1.0 + x / (d - 1))
    beta_d = x * beta_2

    return {2: beta_2, d: beta_d}


def compute_collapse_point(d, kbar_h):
    """Return the theoretical collapse point x_c from the manuscript.

        x_c = ((kbar_h - 1) d^2 (d-1)^2) / (d^2(d-1) - 4 kbar_h)

    Return 0.0 if kbar_h <= 1 (no detectable interval even in the pairwise limit),
    a finite value if finite x_c exists, or np.inf if collapse occurs only as
    x->inf or never occurs.
    """
    d = int(d)
    kbar_h = float(kbar_h)

    if kbar_h <= 1.0:
        return 0.0

    boundary = d**2 * (d - 1) / 4.0

    if kbar_h < boundary:
        denominator = d**2 * (d - 1) - 4.0 * kbar_h
        return (
            (kbar_h - 1.0)
            * d**2
            * (d - 1) ** 2
            / denominator
        )

    # kbar_h == boundary: only collapses as x -> inf
    # kbar_h > boundary: no collapse for finite x
    return np.inf


def compute_epsilon_c_fixed_budget(d, x, K=4, c0=10.0, kbar_h=3.0):
    """Compute the theoretical threshold from the order-resolved stability condition.

        lambda_inf,c = 1 / sqrt(c0 * sum_d beta_d chi_d^2)
        chi_d = 2 / [d(d-1)]

    Here D={2,d} and kbar_h is fixed.
    """
    beta = fixed_budget_beta(d=d, x=x, c0=c0, kbar_h=kbar_h)

    signal = 0.0
    for order, beta_order in beta.items():
        chi = 2.0 / (order * (order - 1))
        signal += beta_order * chi**2

    growth_max = c0 * signal  # At epsilon=0, lambda_inf=1.

    if growth_max <= 0.0:
        lambda_c = np.inf
        eps_c_raw = -np.inf
        eps_c_phys = 0.0
    else:
        lambda_c = 1.0 / np.sqrt(growth_max)
        eps_c_raw = (1.0 - lambda_c) / (1.0 + (K - 1) * lambda_c)
        eps_c_phys = max(0.0, float(eps_c_raw))

    return {
        "beta_2": float(beta[2]),
        "beta_d": float(beta[int(d)]),
        "signal": float(signal),
        "growth_max": float(growth_max),
        "lambda_c": float(lambda_c),
        "eps_c_raw": float(eps_c_raw),
        "eps_c_phys": float(eps_c_phys),
        "x_c": float(compute_collapse_point(d=d, kbar_h=kbar_h)),
    }


# ============================================================
# Model helpers
# ============================================================

def node_assignments_setup(seed, N, K, mean=None):
    """Generate node labels with a fixed label seed; only the hypergraph sampling seed varies."""
    if mean is None:
        probs = np.ones(K, dtype=float) / K
    else:
        probs = np.asarray(mean, dtype=float)
        probs = probs / probs.sum()

    rng = np.random.default_rng(seed)
    node_assignments = rng.choice(np.arange(K), size=N, p=probs)
    return probs, node_assignments


def p_vals_set_weighted(c0, eps, n, N, selected_hye_sizes, K, beta):
    """Build an affinity matrix for each selected order.

    Each order uses c^d = beta_d * c0, allowing distinct mean degree intensities.
    """
    n = np.asarray(n, dtype=float)
    denominator = eps + (1.0 - eps) * np.sum(n**2)

    p_vals = {}
    for size in selected_hye_sizes:
        size = int(size)
        if size not in beta:
            raise KeyError(f"beta 中缺少 order={size}")

        c_d = float(beta[size]) * float(c0)
        p_in = c_d / (N * denominator)
        p_out = eps * p_in

        p_vals[size] = (
            np.full((K, K), p_out, dtype=float)
            + (p_in - p_out) * np.eye(K)
        )

    return p_vals


def save_results(p_vals, selected_hye_sizes, node_assignments, save_dir, fix_hyper_edge):
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "p_vals.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v.tolist() for k, v in p_vals.items()}, f, indent=4)

    np.savetxt(
        os.path.join(save_dir, "selected_hye_sizes.txt"),
        np.asarray(selected_hye_sizes, dtype=int),
        fmt="%d",
    )
    np.savetxt(
        os.path.join(save_dir, "node_assignment_loop.txt"),
        np.asarray(node_assignments, dtype=int),
        fmt="%d",
    )
    np.savetxt(
        os.path.join(save_dir, "fix_hyper_edge.txt"),
        np.asarray(fix_hyper_edge, dtype=int),
        fmt="%d",
    )


def save_case_metadata(save_dir, d, x, K, c0, kbar_h, eps, beta, theory):
    info = {
        "d": int(d),
        "x": float(x),
        "K": int(K),
        "c0": float(c0),
        "kbar_h": float(kbar_h),
        "eps": float(eps),
        "selected_hye_sizes": [2, int(d)],
        "beta_2": float(beta[2]),
        "beta_d": float(beta[int(d)]),
        "c_2": float(beta[2] * c0),
        "c_d": float(beta[int(d)] * c0),
        "eps_c_raw": float(theory["eps_c_raw"]),
        "eps_c_phys": float(theory["eps_c_phys"]),
        "growth_max": float(theory["growth_max"]),
        "lambda_c": float(theory["lambda_c"]),
        "x_c": float(theory["x_c"]),
    }

    with open(os.path.join(save_dir, "fixed_budget_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=4)


# ============================================================
# Directory helpers
# ============================================================

def get_root_save_dir(project_dir, seed_sample, seed_mp, N, d, c0, kbar_h):
    return os.path.join(
        project_dir,
        "data",
        "synthetic_fixed_budget",
        (
            f"seed_sample{seed_sample}_seed_mp{seed_mp}_"
            f"N{N}_D2_{d}_c0{c0:g}_kbar{kbar_h:g}"
        ),
    )


def get_case_save_dir(root_save_dir, x, eps):
    """Sampling and inference must use exactly the same case path."""
    return os.path.join(
        root_save_dir,
        f"x{x:.6g}",
        f"eps{eps:.4f}",
    )


# ============================================================
# Sampling subprocess
# ============================================================

def sample_test(project_dir, save_dir, seed_sample=20, force=False):
    sampling_dir = os.path.join(save_dir, "sampling_results")
    os.makedirs(sampling_dir, exist_ok=True)

    result_file = os.path.join(sampling_dir, "results.pkl")

    if (not force) and os.path.isfile(result_file):
        print(f"[sampling exists] {result_file}")
        return result_file

    cmd = [
        sys.executable,
        os.path.join(project_dir, "main_sampling_loop.py"),
        "--p", os.path.join(save_dir, "p_vals.json"),
        "--selected_hye_sizes", os.path.join(save_dir, "selected_hye_sizes.txt"),
        "--node_assignments", os.path.join(save_dir, "node_assignment_loop.txt"),
        "--allow_repeated", "True",
        "--seed", str(seed_sample),
        "--save_dir", sampling_dir,
    ]

    log_file = os.path.join(sampling_dir, "run_log.txt")
    print(f"[sampling start] seed={seed_sample} dir={save_dir}")

    with open(log_file, "w", encoding="utf-8") as f:
        process = subprocess.Popen(
            cmd,
            cwd=project_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for i, line in enumerate(process.stdout, 1):
            f.write(line)
            if i <= 20 or i % 50 == 0:
                print(line, end="")
        process.wait()

    if process.returncode != 0:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            tail = "".join(f.readlines()[-50:])
        raise RuntimeError(
            "main_sampling_loop.py failed.\n"
            f"log: {log_file}\n\nLast 50 lines:\n{tail}"
        )

    if not os.path.isfile(result_file):
        raise RuntimeError(
            "Sampling process finished, but results.pkl was not created:\n"
            f"{result_file}\nlog: {log_file}"
        )

    print(f"[sampling done] {result_file}")
    return result_file


# ============================================================
# Message passing subprocess
# ============================================================

def main_message_passing_test(
    project_dir,
    save_dir,
    K,
    seed_mp,
    dropout,
    dropout_vertex,
    dropout_vertex_FVS,
    dropout_vertex_rest,
    dropout_FVS,
    inner_n_jobs=1,
):
    sampling_file = os.path.join(save_dir, "sampling_results", "results.pkl")
    if not os.path.isfile(sampling_file):
        raise FileNotFoundError(
            "Cannot start inference because sampling result does not exist:\n"
            f"{sampling_file}"
        )

    mp_save_dir = os.path.join(save_dir, "mp_results")
    os.makedirs(mp_save_dir, exist_ok=True)

    cmd = [
        sys.executable,
        os.path.join(project_dir, "main_message_passing_loop.py"),
        "--em_iter", "1",
        "--em_thresh", "1e-5",
        "--mp_iter", "2000",
        "--mp_thresh", "1e-5",
        "--mp_patience", "50",
        "--train_rounds", "3",
        "--seed", str(seed_mp),
        "--dropout", str(dropout),
        "--dropout_vertex", str(dropout_vertex),
        "--dropout_vertex_FVS", str(dropout_vertex_FVS),
        "--dropout_vertex_rest", str(dropout_vertex_rest),
        "--dropout_FVS", str(dropout_FVS),
        "--node_assignments", os.path.join(save_dir, "node_assignment_loop.txt"),
        "--save_dir", mp_save_dir,
        "--hye_file", sampling_file,
        "--selected_hye_sizes", os.path.join(save_dir, "selected_hye_sizes.txt"),
        "--fix_hye_size", os.path.join(save_dir, "fix_hyper_edge.txt"),
        "--p", os.path.join(save_dir, "p_vals.json"),
        "--K", str(K),
        "--n_jobs", str(inner_n_jobs),
    ]

    log_file = os.path.join(mp_save_dir, "run_log.txt")

    with open(log_file, "w", encoding="utf-8") as f:
        process = subprocess.Popen(
            cmd,
            cwd=project_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for i, line in enumerate(process.stdout, 1):
            f.write(line)
            if i <= 20 or i % 50 == 0:
                print(line, end="")
        process.wait()

    if process.returncode != 0:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            tail = "".join(f.readlines()[-50:])
        raise RuntimeError(
            "main_message_passing_loop.py failed.\n"
            f"log: {log_file}\n\nLast 50 lines:\n{tail}"
        )

    inferred_file = os.path.join(mp_save_dir, "inferred_params.npz")
    if not os.path.isfile(inferred_file):
        raise RuntimeError(
            "Inference finished but inferred_params.npz was not created:\n"
            f"{inferred_file}"
        )

    print(f"[inference done] {inferred_file}")
    return inferred_file


# ============================================================
# One sampling case
# ============================================================

def loop_sample_step(
    x_idx,
    eps_idx,
    d,
    x,
    eps,
    project_dir,
    root_save_dir,
    n_probs,
    node_assignments,
    N,
    K,
    c0,
    kbar_h,
    seed_sample,
):
    selected_hye_sizes = [2, int(d)]
    save_dir = get_case_save_dir(root_save_dir, x=x, eps=eps)
    os.makedirs(save_dir, exist_ok=True)

    beta = fixed_budget_beta(d=d, x=x, c0=c0, kbar_h=kbar_h)
    theory = compute_epsilon_c_fixed_budget(
        d=d,
        x=x,
        K=K,
        c0=c0,
        kbar_h=kbar_h,
    )

    p_vals = p_vals_set_weighted(
        c0=c0,
        eps=eps,
        n=n_probs,
        N=N,
        selected_hye_sizes=selected_hye_sizes,
        K=K,
        beta=beta,
    )

    # Preserve the original behavior: put all selected layers in fix_hyper_edge.
    fix_hyper_edge = selected_hye_sizes

    save_results(
        p_vals=p_vals,
        selected_hye_sizes=selected_hye_sizes,
        node_assignments=node_assignments,
        save_dir=save_dir,
        fix_hyper_edge=fix_hyper_edge,
    )

    save_case_metadata(
        save_dir=save_dir,
        d=d,
        x=x,
        K=K,
        c0=c0,
        kbar_h=kbar_h,
        eps=eps,
        beta=beta,
        theory=theory,
    )

    result_file = sample_test(
        project_dir=project_dir,
        save_dir=save_dir,
        seed_sample=seed_sample,
        force=False,
    )

    return x_idx, eps_idx, result_file


# ============================================================
# One inference case
# ============================================================

def loop_inference_step(
    x_idx,
    eps_idx,
    d,
    x,
    eps,
    project_dir,
    root_save_dir,
    K,
    inner_n_jobs,
    seed_mp,
    dropout,
    dropout_vertex,
    dropout_vertex_FVS,
    dropout_vertex_rest,
    dropout_FVS,
):
    save_dir = get_case_save_dir(root_save_dir, x=x, eps=eps)
    sampling_file = os.path.join(save_dir, "sampling_results", "results.pkl")

    if not os.path.isfile(sampling_file):
        raise FileNotFoundError(
            "Sampling file missing before inference:\n"
            f"{sampling_file}"
        )

    inferred_file = main_message_passing_test(
        project_dir=project_dir,
        save_dir=save_dir,
        K=K,
        seed_mp=seed_mp,
        dropout=dropout,
        dropout_vertex=dropout_vertex,
        dropout_vertex_FVS=dropout_vertex_FVS,
        dropout_vertex_rest=dropout_vertex_rest,
        dropout_FVS=dropout_FVS,
        inner_n_jobs=inner_n_jobs,
    )

    data = np.load(inferred_file, allow_pickle=True)

    labels = np.loadtxt(
        os.path.join(save_dir, "node_assignment_loop.txt"),
        dtype=int,
    ).ravel()
    _, counts = np.unique(labels, return_counts=True)
    ratios = counts / counts.sum()

    overlap_val = float(np.asarray(data["overlap"]).reshape(-1)[0])
    best_free_energy_val = float(np.asarray(data["best_free_energy"]).reshape(-1)[0])
    best_n_val = np.asarray(data["n"]).reshape(-1)
    best_p_val = np.asarray(data["p"])
    best_em_total_time = float(np.asarray(data["em_total_time"]).reshape(-1)[0])
    best_update_message_ratio = float(
        np.asarray(data["update_message_ratio"]).reshape(-1)[0]
    )
    best_update_vertex_ratio = float(
        np.asarray(data["update_vertex_ratio"]).reshape(-1)[0]
    )

    return (
        x_idx,
        eps_idx,
        overlap_val,
        best_free_energy_val,
        ratios,
        best_n_val,
        best_p_val,
        best_em_total_time,
        best_update_message_ratio,
        best_update_vertex_ratio,
    )


# ============================================================
# Sampling phase for ONE graph seed
# ============================================================

def loop_sampling_parallel_fixed_budget(
    N=10000,
    K=4,
    c0=10.0,
    kbar_h=3.0,
    d=3,
    x_list=None,
    eps_list=None,
    n_jobs=10,
    seed_sample=20,
    seed_mp=123,
):
    if not _HAS_JOBLIB:
        raise RuntimeError("joblib is required for this script")

    if x_list is None:
        x_list = [0.5, 1, 2, 4, 6, 8, 10, 11, 12, 13, 15, 20]
    if eps_list is None:
        eps_list = np.arange(0.0, 0.201, 0.01)

    # Fix planted labels; the five seed_sample values vary only the hypergraph realization.
    n_probs, node_assignments = node_assignments_setup(seed=100, N=N, K=K)

    project_dir = os.path.dirname(os.path.abspath(__file__))
    root_save_dir = get_root_save_dir(
        project_dir=project_dir,
        seed_sample=seed_sample,
        seed_mp=seed_mp,
        N=N,
        d=d,
        c0=c0,
        kbar_h=kbar_h,
    )
    os.makedirs(root_save_dir, exist_ok=True)

    tasks = [
        (x_idx, eps_idx, float(x), float(eps))
        for x_idx, x in enumerate(x_list)
        for eps_idx, eps in enumerate(eps_list)
    ]

    print("\n========================================")
    print("FIXED-BUDGET SAMPLING PHASE")
    print(f"D = {{2,{d}}}, c0={c0}, kbar_h={kbar_h}")
    print(f"seed_sample = {seed_sample}")
    print(f"number of cases = {len(tasks)}")
    print("========================================")

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(loop_sample_step)(
            x_idx=x_idx,
            eps_idx=eps_idx,
            d=d,
            x=x,
            eps=eps,
            project_dir=project_dir,
            root_save_dir=root_save_dir,
            n_probs=n_probs,
            node_assignments=node_assignments,
            N=N,
            K=K,
            c0=c0,
            kbar_h=kbar_h,
            seed_sample=seed_sample,
        )
        for x_idx, eps_idx, x, eps in tasks
    )

    missing_files = [r[2] for r in results if not os.path.isfile(r[2])]
    if missing_files:
        raise RuntimeError(
            "Sampling phase is incomplete. Missing files:\n"
            + "\n".join(missing_files)
        )

    print(f"Sampling completed: {len(results)} / {len(tasks)} cases OK.")
    return root_save_dir


# ============================================================
# Inference phase for ONE graph seed
# ============================================================

def loop_inference_parallel_fixed_budget(
    N=10000,
    K=4,
    c0=10.0,
    kbar_h=3.0,
    d=3,
    x_list=None,
    eps_list=None,
    n_jobs=50,
    seed_sample=20,
    seed_mp=123,
    dropout=0,
    dropout_vertex=0,
    dropout_vertex_FVS=0.8,
    dropout_vertex_rest=0.8,
    dropout_FVS=0,
):
    if not _HAS_JOBLIB:
        raise RuntimeError("joblib is required for this script")

    if x_list is None:
        x_list = [0.5, 1, 2, 4, 6, 8, 10, 11, 12, 13, 15, 20]
    if eps_list is None:
        eps_list = np.arange(0.0, 0.201, 0.01)

    x_list = np.asarray(x_list, dtype=float)
    eps_list = np.asarray(eps_list, dtype=float)

    project_dir = os.path.dirname(os.path.abspath(__file__))
    root_save_dir = get_root_save_dir(
        project_dir=project_dir,
        seed_sample=seed_sample,
        seed_mp=seed_mp,
        N=N,
        d=d,
        c0=c0,
        kbar_h=kbar_h,
    )

    total_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    inner_n_jobs = max(1, total_cpus // n_jobs)

    print(
        f"总 CPU 核心数: {total_cpus}, 外层并行: {n_jobs}, "
        f"每个 inference 子任务线程数: {inner_n_jobs}"
    )

    tasks = [
        (x_idx, eps_idx, float(x), float(eps))
        for x_idx, x in enumerate(x_list)
        for eps_idx, eps in enumerate(eps_list)
    ]

    print("\n========================================")
    print("FIXED-BUDGET INFERENCE PHASE")
    print(f"D = {{2,{d}}}, c0={c0}, kbar_h={kbar_h}")
    print(f"seed_sample = {seed_sample}, seed_mp = {seed_mp}")
    print(
        "fixed vertex rates: "
        f"FVS={dropout_vertex_FVS}, rest={dropout_vertex_rest}"
    )
    print(f"number of cases = {len(tasks)}")
    print("========================================")

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(loop_inference_step)(
            x_idx=x_idx,
            eps_idx=eps_idx,
            d=d,
            x=x,
            eps=eps,
            project_dir=project_dir,
            root_save_dir=root_save_dir,
            K=K,
            inner_n_jobs=inner_n_jobs,
            seed_mp=seed_mp,
            dropout=dropout,
            dropout_vertex=dropout_vertex,
            dropout_vertex_FVS=dropout_vertex_FVS,
            dropout_vertex_rest=dropout_vertex_rest,
            dropout_FVS=dropout_FVS,
        )
        for x_idx, eps_idx, x, eps in tasks
    )

    num_x = len(x_list)
    num_eps = len(eps_list)

    overlap = np.full((num_x, num_eps), np.nan, dtype=float)
    best_free_energy = np.full((num_x, num_eps), np.nan, dtype=float)
    em_total_time = np.full((num_x, num_eps), np.nan, dtype=float)
    update_message_ratio = np.full((num_x, num_eps), np.nan, dtype=float)
    update_vertex_ratio = np.full((num_x, num_eps), np.nan, dtype=float)
    best_n = np.full((num_x, num_eps, K), np.nan, dtype=float)
    ratios_list = np.empty((num_x, num_eps), dtype=object)
    ratios_list[...] = None
    best_p = np.empty((num_x, num_eps), dtype=object)
    best_p[...] = None

    for result in results:
        (
            x_idx,
            eps_idx,
            overlap_val,
            free_energy_val,
            ratios,
            best_n_val,
            best_p_val,
            best_em_total_time,
            best_update_message_ratio,
            best_update_vertex_ratio,
        ) = result

        overlap[x_idx, eps_idx] = overlap_val
        best_free_energy[x_idx, eps_idx] = free_energy_val
        ratios_list[x_idx, eps_idx] = ratios
        best_n[x_idx, eps_idx, :] = best_n_val
        best_p[x_idx, eps_idx] = best_p_val
        em_total_time[x_idx, eps_idx] = best_em_total_time
        update_message_ratio[x_idx, eps_idx] = best_update_message_ratio
        update_vertex_ratio[x_idx, eps_idx] = best_update_vertex_ratio

    beta_2_list = np.zeros(num_x, dtype=float)
    beta_d_list = np.zeros(num_x, dtype=float)
    eps_theory_raw = np.zeros(num_x, dtype=float)
    eps_theory_phys = np.zeros(num_x, dtype=float)
    growth_max = np.zeros(num_x, dtype=float)

    for i, x in enumerate(x_list):
        theory = compute_epsilon_c_fixed_budget(
            d=d,
            x=float(x),
            K=K,
            c0=c0,
            kbar_h=kbar_h,
        )
        beta_2_list[i] = theory["beta_2"]
        beta_d_list[i] = theory["beta_d"]
        eps_theory_raw[i] = theory["eps_c_raw"]
        eps_theory_phys[i] = theory["eps_c_phys"]
        growth_max[i] = theory["growth_max"]

    x_c = compute_collapse_point(d=d, kbar_h=kbar_h)

    final_file = os.path.join(
        root_save_dir,
        f"final_results_fixed_budget_D2_{d}_kbar{kbar_h:g}.npz",
    )

    np.savez(
        final_file,
        d=int(d),
        K=int(K),
        N=int(N),
        c0=float(c0),
        kbar_h=float(kbar_h),
        x_list=x_list,
        eps_list=eps_list,
        beta_2=beta_2_list,
        beta_d=beta_d_list,
        eps_theory_raw=eps_theory_raw,
        eps_theory_phys=eps_theory_phys,
        growth_max=growth_max,
        x_c=float(x_c),
        overlap=overlap,
        best_free_energy=best_free_energy,
        ratios_list=ratios_list,
        best_n=best_n,
        best_p=best_p,
        em_total_time=em_total_time,
        update_message_ratio=update_message_ratio,
        update_vertex_ratio=update_vertex_ratio,
        dropout=float(dropout),
        dropout_vertex=float(dropout_vertex),
        dropout_vertex_FVS=float(dropout_vertex_FVS),
        dropout_vertex_rest=float(dropout_vertex_rest),
        dropout_FVS=float(dropout_FVS),
        seed_sample=int(seed_sample),
        seed_mp=int(seed_mp),
    )

    print(f"\nFinal per-seed result saved:\n{final_file}")
    return final_file


# ============================================================
# Aggregate across graph seeds
# ============================================================

def aggregate_seed_results(
    project_dir,
    seed_sample_list,
    seed_mp,
    N,
    K,
    c0,
    kbar_h,
    d,
):
    files = []
    for seed_sample in seed_sample_list:
        root_save_dir = get_root_save_dir(
            project_dir=project_dir,
            seed_sample=seed_sample,
            seed_mp=seed_mp,
            N=N,
            d=d,
            c0=c0,
            kbar_h=kbar_h,
        )
        f = os.path.join(
            root_save_dir,
            f"final_results_fixed_budget_D2_{d}_kbar{kbar_h:g}.npz",
        )
        if not os.path.isfile(f):
            raise FileNotFoundError(f"缺少 per-seed 汇总文件: {f}")
        files.append(f)

    datasets = [np.load(f, allow_pickle=True) for f in files]

    x_list = np.asarray(datasets[0]["x_list"], dtype=float)
    eps_list = np.asarray(datasets[0]["eps_list"], dtype=float)

    overlap_stack = np.stack([np.asarray(ds["overlap"], dtype=float) for ds in datasets], axis=0)
    time_stack = np.stack([np.asarray(ds["em_total_time"], dtype=float) for ds in datasets], axis=0)
    free_energy_stack = np.stack(
        [np.asarray(ds["best_free_energy"], dtype=float) for ds in datasets],
        axis=0,
    )

    summary_dir = os.path.join(
        project_dir,
        "data",
        "synthetic_fixed_budget",
        "summary",
    )
    os.makedirs(summary_dir, exist_ok=True)

    summary_file = os.path.join(
        summary_dir,
        (
            f"final_results_fixed_budget_D2_{d}_N{N}_K{K}_"
            f"c0{c0:g}_kbar{kbar_h:g}_seeds{len(seed_sample_list)}.npz"
        ),
    )

    np.savez(
        summary_file,
        d=int(d),
        K=int(K),
        N=int(N),
        c0=float(c0),
        kbar_h=float(kbar_h),
        x_list=x_list,
        eps_list=eps_list,
        beta_2=np.asarray(datasets[0]["beta_2"], dtype=float),
        beta_d=np.asarray(datasets[0]["beta_d"], dtype=float),
        eps_theory_raw=np.asarray(datasets[0]["eps_theory_raw"], dtype=float),
        eps_theory_phys=np.asarray(datasets[0]["eps_theory_phys"], dtype=float),
        growth_max=np.asarray(datasets[0]["growth_max"], dtype=float),
        x_c=float(np.asarray(datasets[0]["x_c"]).reshape(-1)[0]),
        seed_sample_list=np.asarray(seed_sample_list, dtype=np.int64),
        seed_mp=int(seed_mp),
        overlap_all=overlap_stack,
        overlap_mean=np.mean(overlap_stack, axis=0),
        overlap_std=np.std(overlap_stack, axis=0),
        em_total_time_all=time_stack,
        em_total_time_mean=np.mean(time_stack, axis=0),
        em_total_time_std=np.std(time_stack, axis=0),
        best_free_energy_all=free_energy_stack,
        best_free_energy_mean=np.mean(free_energy_stack, axis=0),
        best_free_energy_std=np.std(free_energy_stack, axis=0),
    )

    for ds in datasets:
        ds.close()

    print(f"\nFive-seed summary saved:\n{summary_file}")
    return summary_file


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # --------------------------------------------------------
    # Paper example:
    #   D = {2,3}
    #   c0 = 10
    #   kbar_h = 3
    #   theoretical x_c = 12
    # --------------------------------------------------------
    N = 10000
    K = 4
    c0 = 10.0
    kbar_h = 12.0
    d = 3

    # Exclude x=0: beta_d=0 would still pass the empty d-layer to inference,
    # which can cause empty-layer/division-by-zero issues; use theory for the pairwise limit.
    x_list = np.array([1.0, 2.0, 4.0, 8.0, 12.0, 20.0])
    eps_list = np.round(np.arange(0.01, 0.401, 0.02), 2)

    # Keep node-update settings fixed across epsilon/x, independently of theory verification.
    dropout = 0
    dropout_vertex = 0
    dropout_vertex_FVS = 0.8
    dropout_vertex_rest = 0.8
    dropout_FVS = 0

    seed_mp = 123
    master_seed = 123

    rng = np.random.default_rng(master_seed)
    seed_sample_list = [
        int(x)
        for x in rng.integers(
            low=0,
            high=2**31 - 1,
            size=5,
        )
    ]

    print("seed_sample_list =", seed_sample_list)

    theory_at_xc = compute_epsilon_c_fixed_budget(
        d=d,
        x=12,
        K=K,
        c0=c0,
        kbar_h=kbar_h,
    )
    print("\nTheory check at x=12:")
    print(json.dumps(theory_at_xc, indent=4))

    # --------------------------------------------------------
    # STAGE 1: sampling for all five graph realizations
    # --------------------------------------------------------
    print("\n\n########################################")
    print("# STAGE 1: ALL FIXED-BUDGET SAMPLING")
    print("########################################")

    for seed_sample in seed_sample_list:
        loop_sampling_parallel_fixed_budget(
            N=N,
            K=K,
            c0=c0,
            kbar_h=kbar_h,
            d=d,
            x_list=x_list,
            eps_list=eps_list,
            n_jobs=10,
            seed_sample=seed_sample,
            seed_mp=seed_mp,
        )

    print("\n# ALL SAMPLING FINISHED")

    # --------------------------------------------------------
    # STAGE 2: inference
    # --------------------------------------------------------
    print("\n\n########################################")
    print("# STAGE 2: ALL FIXED-BUDGET INFERENCE")
    print("########################################")

    for seed_sample in seed_sample_list:
        loop_inference_parallel_fixed_budget(
            N=N,
            K=K,
            c0=c0,
            kbar_h=kbar_h,
            d=d,
            x_list=x_list,
            eps_list=eps_list,
            n_jobs=53,
            seed_sample=seed_sample,
            seed_mp=seed_mp,
            dropout=dropout,
            dropout_vertex=dropout_vertex,
            dropout_vertex_FVS=dropout_vertex_FVS,
            dropout_vertex_rest=dropout_vertex_rest,
            dropout_FVS=dropout_FVS,
        )

    # --------------------------------------------------------
    # STAGE 3: aggregate five graph seeds
    # --------------------------------------------------------
    project_dir = os.path.dirname(os.path.abspath(__file__))

    summary_file = aggregate_seed_results(
        project_dir=project_dir,
        seed_sample_list=seed_sample_list,
        seed_mp=seed_mp,
        N=N,
        K=K,
        c0=c0,
        kbar_h=kbar_h,
        d=d,
    )

    print("\n\n########################################")
    print("# ALL EXPERIMENTS FINISHED")
    print("########################################")
    print("Summary:", summary_file)
