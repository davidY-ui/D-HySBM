"""Run multiseed sampling, inference, and aggregation for order-dependent signal compensation."""

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


def fixed_budget_beta(d, x, c0, kbar_h):
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

    # Fix the mean incident-hyperedge budget while setting beta_d / beta_2 = x.
    beta_2 = (kbar_h / c0) / (1.0 + x / (d - 1.0))
    beta_d = x * beta_2

    return {2: beta_2, d: beta_d}


def lambda_signal(eps, K):
    eps = float(eps)
    K = int(K)
    return (1.0 - eps) / (1.0 + (K - 1.0) * eps)


def growth_signal_compensation(eps, d, x, r, K, kbar_h):
    eps = float(eps)
    d = int(d)
    x = float(x)
    r = float(r)
    K = int(K)
    kbar_h = float(kbar_h)

    if not (0.0 <= eps <= 1.0):
        raise ValueError("eps 必须位于 [0,1]")
    if r <= 0:
        raise ValueError("r 必须为正")

    chi_d = 2.0 / (d * (d - 1.0))
    lam2 = lambda_signal(eps, K)
    lamd = lambda_signal(eps ** r, K)

    return (
        kbar_h
        / (1.0 + x / (d - 1.0))
        * (lam2**2 + x * chi_d**2 * lamd**2)
    )


def compute_epsilon_c_signal(d, x, r, K=4, kbar_h=8.0, tol=1e-12):
    g0 = growth_signal_compensation(
        eps=0.0,
        d=d,
        x=x,
        r=r,
        K=K,
        kbar_h=kbar_h,
    )

    if g0 <= 1.0:
        return 0.0

    lo = 0.0
    hi = 1.0 - 1e-14

    for _ in range(200):
        mid = 0.5 * (lo + hi)

        gmid = growth_signal_compensation(
            eps=mid,
            d=d,
            x=x,
            r=r,
            K=K,
            kbar_h=kbar_h,
        )

        if gmid > 1.0:
            lo = mid
        else:
            hi = mid

        if hi - lo < tol:
            break

    return 0.5 * (lo + hi)


def compute_structural_collapse_point(d, kbar_h):
    d = int(d)
    kbar_h = float(kbar_h)

    if kbar_h <= 1.0:
        return 0.0

    boundary = d**2 * (d - 1.0) / 4.0

    if kbar_h < boundary:
        denominator = d**2 * (d - 1.0) - 4.0 * kbar_h

        return (
            (kbar_h - 1.0)
            * d**2
            * (d - 1.0) ** 2
            / denominator
        )

    return np.inf


def compute_r_star(d, K, kbar_h):
    d = int(d)
    K = int(K)
    kbar_h = float(kbar_h)

    if kbar_h <= 1.0:
        return {
            "exists": False,
            "r_star": np.inf,
            "eps0": 0.0,
            "lambda_d_star": np.inf,
            "z_star": np.nan,
            "boundary": d**2 * (d - 1.0) / 4.0,
        }

    boundary = d**2 * (d - 1.0) / 4.0
    sqrt_k = np.sqrt(kbar_h)

    eps0 = (
        (sqrt_k - 1.0)
        / (sqrt_k + K - 1.0)
    )

    lambda_d_star = (
        d
        * np.sqrt(d - 1.0)
        / (2.0 * sqrt_k)
    )

    if not (kbar_h > boundary) or lambda_d_star >= 1.0:
        return {
            "exists": False,
            "r_star": np.inf,
            "eps0": float(eps0),
            "lambda_d_star": float(lambda_d_star),
            "z_star": 0.0 if np.isclose(lambda_d_star, 1.0) else np.nan,
            "boundary": float(boundary),
        }

    z_star = (
        (1.0 - lambda_d_star)
        / (1.0 + (K - 1.0) * lambda_d_star)
    )

    r_star = (
        np.log(z_star)
        / np.log(eps0)
    )

    return {
        "exists": True,
        "r_star": float(r_star),
        "eps0": float(eps0),
        "lambda_d_star": float(lambda_d_star),
        "z_star": float(z_star),
        "boundary": float(boundary),
    }


def compute_signal_theory(d, x, r, K, c0, kbar_h):
    beta = fixed_budget_beta(
        d=d,
        x=x,
        c0=c0,
        kbar_h=kbar_h,
    )

    eps_c = compute_epsilon_c_signal(
        d=d,
        x=x,
        r=r,
        K=K,
        kbar_h=kbar_h,
    )

    growth_max = growth_signal_compensation(
        eps=0.0,
        d=d,
        x=x,
        r=r,
        K=K,
        kbar_h=kbar_h,
    )

    rstar_info = compute_r_star(
        d=d,
        K=K,
        kbar_h=kbar_h,
    )

    return {
        "beta_2": float(beta[2]),
        "beta_d": float(beta[int(d)]),
        "c_2": float(beta[2] * c0),
        "c_d": float(beta[int(d)] * c0),
        "eps_c": float(eps_c),
        "growth_max": float(growth_max),
        "x_c": float(
            compute_structural_collapse_point(
                d=d,
                kbar_h=kbar_h,
            )
        ),
        "r_star": float(rstar_info["r_star"]),
        "eps0": float(rstar_info["eps0"]),
        "lambda_d_star": float(rstar_info["lambda_d_star"]),
        "z_star": (
            float(rstar_info["z_star"])
            if np.isfinite(rstar_info["z_star"])
            else np.nan
        ),
        "compensation_exists": bool(rstar_info["exists"]),
        "compensation_boundary": float(rstar_info["boundary"]),
    }


def node_assignments_setup(seed, N, K, mean=None):
    if mean is None:
        probs = np.ones(K, dtype=float) / K
    else:
        probs = np.asarray(mean, dtype=float)
        probs = probs / probs.sum()

    rng = np.random.default_rng(seed)

    node_assignments = rng.choice(
        np.arange(K),
        size=N,
        p=probs,
    )

    return probs, node_assignments


def p_vals_set_weighted(
    c0,
    eps_by_order,
    n,
    N,
    selected_hye_sizes,
    K,
    beta,
):
    n = np.asarray(n, dtype=float)
    sum_n2 = float(np.sum(n**2))

    p_vals = {}

    for size in selected_hye_sizes:
        size = int(size)

        if size not in beta:
            raise KeyError(f"beta 中缺少 order={size}")

        if size not in eps_by_order:
            raise KeyError(f"eps_by_order 中缺少 order={size}")

        eps_d = float(eps_by_order[size])

        if not (0.0 <= eps_d <= 1.0):
            raise ValueError(
                f"order={size} 的 eps_d={eps_d} 不在 [0,1]"
            )

        denominator = (
            eps_d
            + (1.0 - eps_d) * sum_n2
        )

        c_d = (
            float(beta[size])
            * float(c0)
        )

        c_in = (
            c_d
            / denominator
        )

        c_out = (
            eps_d
            * c_in
        )

        p_in = c_in / N
        p_out = c_out / N

        p_vals[size] = (
            np.full(
                (K, K),
                p_out,
                dtype=float,
            )
            + (p_in - p_out) * np.eye(K)
        )

    return p_vals


def save_results(
    p_vals,
    n_probs,
    selected_hye_sizes,
    node_assignments,
    save_dir,
    fix_hyper_edge,
):
    os.makedirs(
        save_dir,
        exist_ok=True,
    )

    with open(
        os.path.join(
            save_dir,
            "p_vals.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                str(k): v.tolist()
                for k, v in p_vals.items()
            },
            f,
            indent=4,
        )

    np.savetxt(
        os.path.join(
            save_dir,
            "n_vals.txt",
        ),
        np.asarray(
            n_probs,
            dtype=float,
        ),
        fmt="%.18g",
    )

    np.savetxt(
        os.path.join(
            save_dir,
            "selected_hye_sizes.txt",
        ),
        np.asarray(
            selected_hye_sizes,
            dtype=int,
        ),
        fmt="%d",
    )

    np.savetxt(
        os.path.join(
            save_dir,
            "node_assignment_loop.txt",
        ),
        np.asarray(
            node_assignments,
            dtype=int,
        ),
        fmt="%d",
    )

    np.savetxt(
        os.path.join(
            save_dir,
            "fix_hyper_edge.txt",
        ),
        np.asarray(
            fix_hyper_edge,
            dtype=int,
        ),
        fmt="%d",
    )


def save_case_metadata(
    save_dir,
    d,
    x,
    r,
    K,
    c0,
    kbar_h,
    eps,
    beta,
    theory,
):
    eps_by_order = {
        2: float(eps),
        int(d): float(eps ** r),
    }

    info = {
        "d": int(d),
        "x": float(x),
        "r": float(r),
        "K": int(K),
        "c0": float(c0),
        "kbar_h": float(kbar_h),
        "eps_base": float(eps),
        "eps_2": float(eps_by_order[2]),
        "eps_d": float(eps_by_order[int(d)]),
        "selected_hye_sizes": [2, int(d)],
        "beta_2": float(beta[2]),
        "beta_d": float(beta[int(d)]),
        "c_2": float(beta[2] * c0),
        "c_d": float(beta[int(d)] * c0),
        "eps_c_theory": float(theory["eps_c"]),
        "growth_max": float(theory["growth_max"]),
        "x_c": float(theory["x_c"]),
        "r_star": float(theory["r_star"]),
        "eps0": float(theory["eps0"]),
        "compensation_boundary": float(
            theory["compensation_boundary"]
        ),
    }

    with open(
        os.path.join(
            save_dir,
            "signal_compensation_info.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            info,
            f,
            ensure_ascii=False,
            indent=4,
        )


def get_root_save_dir(
    project_dir,
    seed_sample,
    seed_mp,
    N,
    d,
    c0,
    kbar_h,
    r,
):
    return os.path.join(
        project_dir,
        "data",
        "synthetic_signal_compensation",
        (
            f"seed_sample{seed_sample}_seed_mp{seed_mp}_"
            f"N{N}_D2_{d}_c0{c0:g}_kbar{kbar_h:g}_r{r:.6f}"
        ),
    )


def get_case_save_dir(
    root_save_dir,
    x,
    eps,
):
    return os.path.join(
        root_save_dir,
        f"x{x:.6g}",
        f"eps{eps:.4f}",
    )


def sample_test(
    project_dir,
    save_dir,
    seed_sample=20,
    force=False,
):
    sampling_dir = os.path.join(
        save_dir,
        "sampling_results",
    )

    os.makedirs(
        sampling_dir,
        exist_ok=True,
    )

    result_file = os.path.join(
        sampling_dir,
        "results.pkl",
    )

    if (
        not force
        and os.path.isfile(result_file)
    ):
        print(
            f"[sampling exists] {result_file}"
        )
        return result_file

    cmd = [
        sys.executable,
        os.path.join(
            project_dir,
            "main_sampling_loop.py",
        ),

        "--p",
        os.path.join(
            save_dir,
            "p_vals.json",
        ),

        "--selected_hye_sizes",
        os.path.join(
            save_dir,
            "selected_hye_sizes.txt",
        ),

        "--node_assignments",
        os.path.join(
            save_dir,
            "node_assignment_loop.txt",
        ),

        "--allow_repeated",
        "True",

        "--seed",
        str(seed_sample),

        "--save_dir",
        sampling_dir,
    ]

    log_file = os.path.join(
        sampling_dir,
        "run_log.txt",
    )

    print(
        f"[sampling start] "
        f"seed={seed_sample} "
        f"dir={save_dir}"
    )

    with open(
        log_file,
        "w",
        encoding="utf-8",
    ) as f:

        process = subprocess.Popen(
            cmd,
            cwd=project_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for i, line in enumerate(
            process.stdout,
            1,
        ):
            f.write(line)

            if i <= 20 or i % 50 == 0:
                print(
                    line,
                    end="",
                )

        process.wait()

    if process.returncode != 0:

        with open(
            log_file,
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as f:

            tail = "".join(
                f.readlines()[-50:]
            )

        raise RuntimeError(
            "main_sampling_loop.py failed.\n"
            f"log: {log_file}\n\n"
            f"Last 50 lines:\n{tail}"
        )

    if not os.path.isfile(
        result_file
    ):
        raise RuntimeError(
            "Sampling process finished, "
            "but results.pkl was not created:\n"
            f"{result_file}\n"
            f"log: {log_file}"
        )

    print(
        f"[sampling done] {result_file}"
    )

    return result_file


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
    sampling_file = os.path.join(
        save_dir,
        "sampling_results",
        "results.pkl",
    )

    if not os.path.isfile(
        sampling_file
    ):
        raise FileNotFoundError(
            "Cannot start inference because "
            "sampling result does not exist:\n"
            f"{sampling_file}"
        )

    mp_save_dir = os.path.join(
        save_dir,
        "mp_results",
    )

    os.makedirs(
        mp_save_dir,
        exist_ok=True,
    )

    inferred_file = os.path.join(
        mp_save_dir,
        "inferred_params.npz",
    )

    print(
        f"[inference start] {save_dir}"
    )

    cmd = [
        sys.executable,

        os.path.join(
            project_dir,
            "main_message_passing_loop.py",
        ),

        "--em_iter",
        "1",

        "--em_thresh",
        "1e-5",

        "--mp_iter",
        "2000",

        "--mp_thresh",
        "1e-5",

        "--mp_patience",
        "50",

        "--train_rounds",
        "1",

        "--seed",
        str(seed_mp),

        "--dropout",
        str(dropout),

        "--dropout_vertex",
        "0.75",

        "--dropout_FVS",
        "0.75",

        "--node_assignments",
        os.path.join(
            save_dir,
            "node_assignment_loop.txt",
        ),

        "--save_dir",
        mp_save_dir,

        "--hye_file",
        sampling_file,

        "--selected_hye_sizes",
        os.path.join(
            save_dir,
            "selected_hye_sizes.txt",
        ),

        "--fix_hye_size",
        os.path.join(
            save_dir,
            "fix_hyper_edge.txt",
        ),

        "--p",
        os.path.join(
            save_dir,
            "p_vals.json",
        ),

        "--n",
        os.path.join(
            save_dir,
            "n_vals.txt",
        ),

        "--K",
        str(K),

        "--n_jobs",
        str(inner_n_jobs),
    ]

    log_file = os.path.join(
        mp_save_dir,
        "run_log.txt",
    )

    with open(
        log_file,
        "w",
        encoding="utf-8",
    ) as f:

        process = subprocess.Popen(
            cmd,
            cwd=project_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for i, line in enumerate(
            process.stdout,
            1,
        ):
            f.write(line)

            if i <= 20 or i % 50 == 0:
                print(
                    line,
                    end="",
                )

        process.wait()

    if process.returncode != 0:

        with open(
            log_file,
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as f:

            tail = "".join(
                f.readlines()[-50:]
            )

        raise RuntimeError(
            "main_message_passing_loop.py failed.\n"
            f"log: {log_file}\n\n"
            f"Last 50 lines:\n{tail}"
        )

    if not os.path.isfile(
        inferred_file
    ):
        raise RuntimeError(
            "Inference finished but "
            "inferred_params.npz was not created:\n"
            f"{inferred_file}"
        )

    print(
        f"[inference done] {inferred_file}"
    )

    return inferred_file


def loop_sample_step(
    x_idx,
    eps_idx,
    d,
    x,
    r,
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
    selected_hye_sizes = [
        2,
        int(d),
    ]

    save_dir = get_case_save_dir(
        root_save_dir,
        x=x,
        eps=eps,
    )

    os.makedirs(
        save_dir,
        exist_ok=True,
    )

    beta = fixed_budget_beta(
        d=d,
        x=x,
        c0=c0,
        kbar_h=kbar_h,
    )

    theory = compute_signal_theory(
        d=d,
        x=x,
        r=r,
        K=K,
        c0=c0,
        kbar_h=kbar_h,
    )

    eps_by_order = {
        2: float(eps),
        int(d): float(
            eps ** r
        ),
    }

    p_vals = p_vals_set_weighted(
        c0=c0,
        eps_by_order=eps_by_order,
        n=n_probs,
        N=N,
        selected_hye_sizes=selected_hye_sizes,
        K=K,
        beta=beta,
    )

    fix_hyper_edge = (
        selected_hye_sizes
    )

    save_results(
        p_vals=p_vals,
        n_probs=n_probs,
        selected_hye_sizes=selected_hye_sizes,
        node_assignments=node_assignments,
        save_dir=save_dir,
        fix_hyper_edge=fix_hyper_edge,
    )

    save_case_metadata(
        save_dir=save_dir,
        d=d,
        x=x,
        r=r,
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

    return (
        x_idx,
        eps_idx,
        result_file,
    )


def loop_inference_step(
    x_idx,
    eps_idx,
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
    save_dir = get_case_save_dir(
        root_save_dir,
        x=x,
        eps=eps,
    )

    sampling_file = os.path.join(
        save_dir,
        "sampling_results",
        "results.pkl",
    )

    if not os.path.isfile(
        sampling_file
    ):
        raise FileNotFoundError(
            "Sampling file missing before inference:\n"
            f"{sampling_file}"
        )

    inferred_file = (
        main_message_passing_test(
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
    )

    data = np.load(
        inferred_file,
        allow_pickle=True,
    )

    labels = np.loadtxt(
        os.path.join(
            save_dir,
            "node_assignment_loop.txt",
        ),
        dtype=int,
    ).ravel()

    _, counts = np.unique(
        labels,
        return_counts=True,
    )

    ratios = (
        counts
        / counts.sum()
    )

    overlap_val = float(
        np.asarray(
            data["overlap"]
        ).reshape(-1)[0]
    )

    best_free_energy_val = float(
        np.asarray(
            data["best_free_energy"]
        ).reshape(-1)[0]
    )

    best_n_val = np.asarray(
        data["n"]
    ).reshape(-1)

    best_p_val = np.asarray(
        data["p"]
    )

    best_em_total_time = float(
        np.asarray(
            data["em_total_time"]
        ).reshape(-1)[0]
    )

    best_update_message_ratio = float(
        np.asarray(
            data["update_message_ratio"]
        ).reshape(-1)[0]
    )

    best_update_vertex_ratio = float(
        np.asarray(
            data["update_vertex_ratio"]
        ).reshape(-1)[0]
    )

    data.close()

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


def loop_sampling_parallel_signal_compensation(
    N=10000,
    K=4,
    c0=10.0,
    kbar_h=8.0,
    d=3,
    r=1.0,
    x_list=None,
    eps_list=None,
    n_jobs=10,
    seed_sample=20,
    seed_mp=123,
):
    if not _HAS_JOBLIB:
        raise RuntimeError(
            "joblib is required for this script"
        )

    if x_list is None:
        x_list = [
            0.5,
            1,
            2,
            4,
            8,
            12,
            20,
        ]

    if eps_list is None:
        eps_list = np.arange(
            0.04,
            0.461,
            0.02,
        )

    n_probs, node_assignments = (
        node_assignments_setup(
            seed=100,
            N=N,
            K=K,
        )
    )

    project_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    root_save_dir = get_root_save_dir(
        project_dir=project_dir,
        seed_sample=seed_sample,
        seed_mp=seed_mp,
        N=N,
        d=d,
        c0=c0,
        kbar_h=kbar_h,
        r=r,
    )

    os.makedirs(
        root_save_dir,
        exist_ok=True,
    )

    tasks = [
        (
            x_idx,
            eps_idx,
            float(x),
            float(eps),
        )
        for x_idx, x in enumerate(x_list)
        for eps_idx, eps in enumerate(eps_list)
    ]

    print("\n========================================")
    print("SIGNAL-COMPENSATION SAMPLING PHASE")
    print(
        f"D = {{2,{d}}}, "
        f"c0={c0}, "
        f"kbar_h={kbar_h}, "
        f"r={r}"
    )
    print(
        f"seed_sample = {seed_sample}"
    )
    print(
        f"number of cases = {len(tasks)}"
    )
    print("========================================")

    results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        verbose=10,
    )(
        delayed(loop_sample_step)(
            x_idx=x_idx,
            eps_idx=eps_idx,
            d=d,
            x=x,
            r=r,
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

    missing_files = [
        r0[2]
        for r0 in results
        if not os.path.isfile(r0[2])
    ]

    if missing_files:
        raise RuntimeError(
            "Sampling phase is incomplete. Missing files:\n"
            + "\n".join(missing_files)
        )

    print(
        f"Sampling completed: "
        f"{len(results)} / {len(tasks)} cases OK."
    )

    return root_save_dir


def loop_inference_parallel_signal_compensation(
    N=10000,
    K=4,
    c0=10.0,
    kbar_h=8.0,
    d=3,
    r=1.0,
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
        raise RuntimeError(
            "joblib is required for this script"
        )

    if x_list is None:
        x_list = [
            0.5,
            1,
            2,
            4,
            8,
            12,
            20,
        ]

    if eps_list is None:
        eps_list = np.arange(
            0.04,
            0.461,
            0.02,
        )

    x_list = np.asarray(
        x_list,
        dtype=float,
    )

    eps_list = np.asarray(
        eps_list,
        dtype=float,
    )

    num_x = len(x_list)
    num_eps = len(eps_list)

    project_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    root_save_dir = get_root_save_dir(
        project_dir=project_dir,
        seed_sample=seed_sample,
        seed_mp=seed_mp,
        N=N,
        d=d,
        c0=c0,
        kbar_h=kbar_h,
        r=r,
    )

    os.makedirs(
        root_save_dir,
        exist_ok=True,
    )

    total_cpus = int(
        os.environ.get(
            "SLURM_CPUS_PER_TASK",
            os.cpu_count() or 1,
        )
    )

    inner_n_jobs = max(
        1,
        total_cpus // n_jobs,
    )

    print(
        f"总 CPU 核心数: {total_cpus}, "
        f"外层并行: {n_jobs}, "
        f"每个 inference 子任务线程数: {inner_n_jobs}"
    )

    tasks = [
        (
            x_idx,
            eps_idx,
            float(x),
            float(eps),
        )
        for x_idx, x in enumerate(x_list)
        for eps_idx, eps in enumerate(eps_list)
    ]

    print("\n========================================")
    print("SIGNAL-COMPENSATION INFERENCE PHASE")
    print(
        f"D = {{2,{d}}}, "
        f"c0={c0}, "
        f"kbar_h={kbar_h}, "
        f"r={r}"
    )
    print(
        f"seed_sample = {seed_sample}, "
        f"seed_mp = {seed_mp}"
    )
    print(
        "fixed vertex rates: "
        f"FVS={dropout_vertex_FVS}, "
        f"rest={dropout_vertex_rest}"
    )
    print(
        f"number of cases = {len(tasks)}"
    )
    print("========================================")

    results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        verbose=10,
    )(
        delayed(loop_inference_step)(
            x_idx=x_idx,
            eps_idx=eps_idx,
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

    overlap = np.full(
        (num_x, num_eps),
        np.nan,
        dtype=float,
    )

    best_free_energy = np.full(
        (num_x, num_eps),
        np.nan,
        dtype=float,
    )

    em_total_time = np.full(
        (num_x, num_eps),
        np.nan,
        dtype=float,
    )

    update_message_ratio = np.full(
        (num_x, num_eps),
        np.nan,
        dtype=float,
    )

    update_vertex_ratio = np.full(
        (num_x, num_eps),
        np.nan,
        dtype=float,
    )

    best_n = np.full(
        (num_x, num_eps, K),
        np.nan,
        dtype=float,
    )

    ratios_list = np.empty(
        (num_x, num_eps),
        dtype=object,
    )

    ratios_list[...] = None

    best_p = np.empty(
        (num_x, num_eps),
        dtype=object,
    )

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

        overlap[
            x_idx,
            eps_idx,
        ] = overlap_val

        best_free_energy[
            x_idx,
            eps_idx,
        ] = free_energy_val

        ratios_list[
            x_idx,
            eps_idx,
        ] = ratios

        best_n[
            x_idx,
            eps_idx,
            :,
        ] = best_n_val

        best_p[
            x_idx,
            eps_idx,
        ] = best_p_val

        em_total_time[
            x_idx,
            eps_idx,
        ] = best_em_total_time

        update_message_ratio[
            x_idx,
            eps_idx,
        ] = best_update_message_ratio

        update_vertex_ratio[
            x_idx,
            eps_idx,
        ] = best_update_vertex_ratio

    beta_2_list = np.zeros(
        num_x,
        dtype=float,
    )

    beta_d_list = np.zeros(
        num_x,
        dtype=float,
    )

    eps_theory = np.zeros(
        num_x,
        dtype=float,
    )

    growth_max = np.zeros(
        num_x,
        dtype=float,
    )

    for i, x in enumerate(x_list):
        theory = compute_signal_theory(
            d=d,
            x=float(x),
            r=r,
            K=K,
            c0=c0,
            kbar_h=kbar_h,
        )

        beta_2_list[i] = (
            theory["beta_2"]
        )

        beta_d_list[i] = (
            theory["beta_d"]
        )

        eps_theory[i] = (
            theory["eps_c"]
        )

        growth_max[i] = (
            theory["growth_max"]
        )

    rstar_info = compute_r_star(
        d=d,
        K=K,
        kbar_h=kbar_h,
    )

    x_c = compute_structural_collapse_point(
        d=d,
        kbar_h=kbar_h,
    )

    final_file = os.path.join(
        root_save_dir,
        (
            f"final_results_signal_compensation_D2_{d}_"
            f"kbar{kbar_h:g}_r{r:.6f}.npz"
        ),
    )

    np.savez(
        final_file,
        d=int(d),
        K=int(K),
        N=int(N),
        c0=float(c0),
        kbar_h=float(kbar_h),
        r=float(r),
        r_star=float(rstar_info["r_star"]),
        eps0=float(rstar_info["eps0"]),
        compensation_boundary=float(
            rstar_info["boundary"]
        ),
        x_list=x_list,
        eps_list=eps_list,
        beta_2=beta_2_list,
        beta_d=beta_d_list,
        eps_theory=eps_theory,
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
        dropout_vertex_FVS=float(
            dropout_vertex_FVS
        ),
        dropout_vertex_rest=float(
            dropout_vertex_rest
        ),
        dropout_FVS=float(dropout_FVS),
        seed_sample=int(seed_sample),
        seed_mp=int(seed_mp),
    )

    print(
        f"\nFinal per-seed result saved:\n"
        f"{final_file}"
    )

    return final_file


def aggregate_seed_results_one_r(
    project_dir,
    seed_sample_list,
    seed_mp,
    N,
    K,
    c0,
    kbar_h,
    d,
    r,
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
            r=r,
        )

        f = os.path.join(
            root_save_dir,
            (
                f"final_results_signal_compensation_D2_{d}_"
                f"kbar{kbar_h:g}_r{r:.6f}.npz"
            ),
        )

        if not os.path.isfile(f):
            raise FileNotFoundError(
                f"缺少 per-seed 汇总文件: {f}"
            )

        files.append(f)

    datasets = [
        np.load(
            f,
            allow_pickle=True,
        )
        for f in files
    ]

    x_list = np.asarray(
        datasets[0]["x_list"],
        dtype=float,
    )

    eps_list = np.asarray(
        datasets[0]["eps_list"],
        dtype=float,
    )

    overlap_stack = np.stack(
        [
            np.asarray(
                ds["overlap"],
                dtype=float,
            )
            for ds in datasets
        ],
        axis=0,
    )

    time_stack = np.stack(
        [
            np.asarray(
                ds["em_total_time"],
                dtype=float,
            )
            for ds in datasets
        ],
        axis=0,
    )

    free_energy_stack = np.stack(
        [
            np.asarray(
                ds["best_free_energy"],
                dtype=float,
            )
            for ds in datasets
        ],
        axis=0,
    )

    summary_dir = os.path.join(
        project_dir,
        "data",
        "synthetic_signal_compensation",
        "summary",
    )

    os.makedirs(
        summary_dir,
        exist_ok=True,
    )

    summary_file = os.path.join(
        summary_dir,
        (
            f"final_results_signal_compensation_D2_{d}_N{N}_K{K}_"
            f"c0{c0:g}_kbar{kbar_h:g}_r{r:.6f}_"
            f"seeds{len(seed_sample_list)}.npz"
        ),
    )

    np.savez(
        summary_file,
        d=int(d),
        K=int(K),
        N=int(N),
        c0=float(c0),
        kbar_h=float(kbar_h),
        r=float(r),
        r_star=float(
            np.asarray(
                datasets[0]["r_star"]
            ).reshape(-1)[0]
        ),
        eps0=float(
            np.asarray(
                datasets[0]["eps0"]
            ).reshape(-1)[0]
        ),
        compensation_boundary=float(
            np.asarray(
                datasets[0][
                    "compensation_boundary"
                ]
            ).reshape(-1)[0]
        ),
        x_list=x_list,
        eps_list=eps_list,
        beta_2=np.asarray(
            datasets[0]["beta_2"],
            dtype=float,
        ),
        beta_d=np.asarray(
            datasets[0]["beta_d"],
            dtype=float,
        ),
        eps_theory=np.asarray(
            datasets[0]["eps_theory"],
            dtype=float,
        ),
        growth_max=np.asarray(
            datasets[0]["growth_max"],
            dtype=float,
        ),
        x_c=float(
            np.asarray(
                datasets[0]["x_c"]
            ).reshape(-1)[0]
        ),
        seed_sample_list=np.asarray(
            seed_sample_list,
            dtype=np.int64,
        ),
        seed_mp=int(seed_mp),
        overlap_all=overlap_stack,
        overlap_mean=np.mean(
            overlap_stack,
            axis=0,
        ),
        overlap_std=np.std(
            overlap_stack,
            axis=0,
        ),
        em_total_time_all=time_stack,
        em_total_time_mean=np.mean(
            time_stack,
            axis=0,
        ),
        em_total_time_std=np.std(
            time_stack,
            axis=0,
        ),
        best_free_energy_all=free_energy_stack,
        best_free_energy_mean=np.mean(
            free_energy_stack,
            axis=0,
        ),
        best_free_energy_std=np.std(
            free_energy_stack,
            axis=0,
        ),
    )

    for ds in datasets:
        ds.close()

    print(
        f"\nFive-seed summary saved:\n"
        f"{summary_file}"
    )

    return summary_file


def aggregate_r_summaries(
    project_dir,
    r_list,
    N,
    K,
    c0,
    kbar_h,
    d,
    num_seeds,
):
    summary_dir = os.path.join(
        project_dir,
        "data",
        "synthetic_signal_compensation",
        "summary",
    )

    files = []

    for r in r_list:
        f = os.path.join(
            summary_dir,
            (
                f"final_results_signal_compensation_D2_{d}_N{N}_K{K}_"
                f"c0{c0:g}_kbar{kbar_h:g}_r{r:.6f}_"
                f"seeds{num_seeds}.npz"
            ),
        )

        if not os.path.isfile(f):
            raise FileNotFoundError(
                f"缺少 r-summary 文件: {f}"
            )

        files.append(f)

    datasets = [
        np.load(
            f,
            allow_pickle=True,
        )
        for f in files
    ]

    x_list = np.asarray(
        datasets[0]["x_list"],
        dtype=float,
    )

    eps_list = np.asarray(
        datasets[0]["eps_list"],
        dtype=float,
    )

    overlap_mean = np.stack(
        [
            np.asarray(
                ds["overlap_mean"],
                dtype=float,
            )
            for ds in datasets
        ],
        axis=0,
    )

    overlap_std = np.stack(
        [
            np.asarray(
                ds["overlap_std"],
                dtype=float,
            )
            for ds in datasets
        ],
        axis=0,
    )

    eps_theory = np.stack(
        [
            np.asarray(
                ds["eps_theory"],
                dtype=float,
            )
            for ds in datasets
        ],
        axis=0,
    )

    all_r_file = os.path.join(
        summary_dir,
        (
            f"final_results_signal_compensation_ALLR_D2_{d}_N{N}_K{K}_"
            f"c0{c0:g}_kbar{kbar_h:g}_seeds{num_seeds}.npz"
        ),
    )

    np.savez(
        all_r_file,
        d=int(d),
        K=int(K),
        N=int(N),
        c0=float(c0),
        kbar_h=float(kbar_h),
        r_list=np.asarray(
            r_list,
            dtype=float,
        ),
        r_star=float(
            np.asarray(
                datasets[0]["r_star"]
            ).reshape(-1)[0]
        ),
        eps0=float(
            np.asarray(
                datasets[0]["eps0"]
            ).reshape(-1)[0]
        ),
        x_list=x_list,
        eps_list=eps_list,
        eps_theory=eps_theory,
        overlap_mean=overlap_mean,
        overlap_std=overlap_std,
    )

    for ds in datasets:
        ds.close()

    print(
        f"\nAll-r summary saved:\n"
        f"{all_r_file}"
    )

    return all_r_file


def run_theory_sanity_checks(
    d,
    K,
    c0,
    kbar_h,
    r_star,
    x_list,
):
    x_list = np.asarray(
        x_list,
        dtype=float,
    )

    eps_r1 = np.asarray([
        compute_epsilon_c_signal(
            d=d,
            x=float(x),
            r=1.0,
            K=K,
            kbar_h=kbar_h,
        )
        for x in x_list
    ])

    eps_rstar = np.asarray([
        compute_epsilon_c_signal(
            d=d,
            x=float(x),
            r=r_star,
            K=K,
            kbar_h=kbar_h,
        )
        for x in x_list
    ])

    eps_r3 = np.asarray([
        compute_epsilon_c_signal(
            d=d,
            x=float(x),
            r=7.0,
            K=K,
            kbar_h=kbar_h,
        )
        for x in x_list
    ])

    print("\nTheory sanity check")
    print(
        "x_list       =",
        x_list,
    )
    print(
        "eps_c(r=1)   =",
        np.round(
            eps_r1,
            6,
        ),
    )
    print(
        "eps_c(r=r*)  =",
        np.round(
            eps_rstar,
            6,
        ),
    )
    print(
        "eps_c(r=3)   =",
        np.round(
            eps_r3,
            6,
        ),
    )

    if not np.all(
        np.diff(eps_r1) < 0
    ):
        raise RuntimeError(
            "Sanity check failed: "
            "r=1 理论阈值没有随 x 单调下降"
        )

    if (
        np.max(eps_rstar)
        - np.min(eps_rstar)
        > 1e-6
    ):
        raise RuntimeError(
            "Sanity check failed: "
            "r=r_star 理论阈值不是近似水平"
        )

    if not np.all(
        np.diff(eps_r3) > 0
    ):
        raise RuntimeError(
            "Sanity check failed: "
            "r=3 理论阈值没有随 x 单调上升"
        )

    print(
        "Theory sanity checks passed.\n"
    )


if __name__ == "__main__":

    N = 10000
    K = 4
    c0 = 10.0
    kbar_h = 32
    d = 5

    rstar_info = compute_r_star(
        d=d,
        K=K,
        kbar_h=kbar_h,
    )

    if not rstar_info["exists"]:
        raise RuntimeError(
            "当前参数不存在有限 r_star。"
            "请检查 kbar_h > d^2(d-1)/4 是否成立。"
        )

    r_star = rstar_info[
        "r_star"
    ]

    r_list = np.asarray(
        [
            1.0,
            r_star,
            7.0,
        ],
        dtype=float,
    )

    x_list = np.asarray(
        [
            1,
            4,
            8,
            16,
        ],
        dtype=float,
    )

    eps_list = np.arange(
        0.25,
        0.63,
        0.01,
    )

    dropout = 0
    dropout_vertex = 0

    dropout_vertex_FVS = 0
    dropout_vertex_rest = 0

    dropout_FVS = 0

    seed_mp = 123
    master_seed = 123

    rng = np.random.default_rng(
        master_seed
    )

    seed_sample_list = [
        int(x)
        for x in rng.integers(
            low=0,
            high=2**31 - 1,
            size=5,
        )
    ]

    print("========================================")
    print("SIGNAL COMPENSATION EXPERIMENT")
    print("========================================")

    print(
        f"D = {{2,{d}}}"
    )

    print(
        f"N = {N}, "
        f"K = {K}, "
        f"c0 = {c0}, "
        f"kbar_h = {kbar_h}"
    )

    print(
        f"boundary = "
        f"{rstar_info['boundary']:.6f}"
    )

    print(
        f"eps0 = "
        f"{rstar_info['eps0']:.12f}"
    )

    print(
        f"lambda_d_star = "
        f"{rstar_info['lambda_d_star']:.12f}"
    )

    print(
        f"z_star = "
        f"{rstar_info['z_star']:.12f}"
    )

    print(
        f"r_star = "
        f"{r_star:.12f}"
    )

    print(
        "r_list =",
        r_list,
    )

    print(
        "x_list =",
        x_list,
    )

    print(
        "eps_list =",
        eps_list,
    )

    print(
        "seed_sample_list =",
        seed_sample_list,
    )

    run_theory_sanity_checks(
        d=d,
        K=K,
        c0=c0,
        kbar_h=kbar_h,
        r_star=r_star,
        x_list=x_list,
    )

    print("\n\n########################################")
    print("# STAGE 1: ALL SIGNAL-COMPENSATION SAMPLING")
    print("########################################")

    for r in r_list:
        for seed_sample in seed_sample_list:

            loop_sampling_parallel_signal_compensation(
                N=N,
                K=K,
                c0=c0,
                kbar_h=kbar_h,
                d=d,
                r=float(r),
                x_list=x_list,
                eps_list=eps_list,
                n_jobs=10,
                seed_sample=seed_sample,
                seed_mp=seed_mp,
            )

    print(
        "\n# ALL SAMPLING FINISHED"
    )

    print("\n\n########################################")
    print("# STAGE 2: ALL SIGNAL-COMPENSATION INFERENCE")
    print("########################################")

    for r in r_list:
        for seed_sample in seed_sample_list[:1]:

            loop_inference_parallel_signal_compensation(
                N=N,
                K=K,
                c0=c0,
                kbar_h=kbar_h,
                d=d,
                r=float(r),
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

    print("\n\n########################################")
    print("# STAGE 3: AGGREGATE FIVE SEEDS FOR EACH r")
    print("########################################")

    project_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    for r in r_list:

        aggregate_seed_results_one_r(
            project_dir=project_dir,
            seed_sample_list=seed_sample_list,
            seed_mp=seed_mp,
            N=N,
            K=K,
            c0=c0,
            kbar_h=kbar_h,
            d=d,
            r=float(r),
        )

    print("\n\n########################################")
    print("# STAGE 4: AGGREGATE ALL r VALUES")
    print("########################################")

    all_r_summary = aggregate_r_summaries(
        project_dir=project_dir,
        r_list=r_list,
        N=N,
        K=K,
        c0=c0,
        kbar_h=kbar_h,
        d=d,
        num_seeds=len(seed_sample_list),
    )

    print("\n\n########################################")
    print("# ALL SIGNAL-COMPENSATION EXPERIMENTS FINISHED")
    print("########################################")

    print(
        "All-r summary:",
        all_r_summary,
    )