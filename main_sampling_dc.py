"""Unified HSBM/DCHSBM sampling entry in the style of main_sampling_loop.py.

Place this file at the project root and the latest sampling_dc.py in src/model/.
Retain the original input and results.pkl/args.pkl output conventions.
"""
import argparse
import json
import pickle as pkl
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

try:
    from sem.str_to_type import bool_type, none_or_type
except ImportError:
    def bool_type(x):
        if isinstance(x, bool):
            return x
        x = str(x).strip().lower()
        if x in {"true", "t", "1", "yes", "y"}:
            return True
        if x in {"false", "f", "0", "no", "n"}:
            return False
        raise argparse.ArgumentTypeError(f"Cannot interpret {x!r} as bool.")

    def none_or_type(t):
        def _parse(x):
            if x is None:
                return None
            if isinstance(x, str) and x.strip().lower() in {"none", "null", ""}:
                return None
            return t(x)
        return _parse

# Use the unified sampler; retain the sampling alias for the existing calls below.
# This sampler needs no joblib directory or TEMP/TMP changes.
from src.model import sampling_dc as sampling


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "-p", "--p", type=str, required=True,
        help="Path to JSON {order: symmetric affinity matrix}, with p=c/N.",
    )
    parser.add_argument(
        "-max_hye_size", "--max_hye_size", type=int, default=None,
        help="Retained for compatibility; actual orders come from selected_hye_sizes.",
    )
    parser.add_argument(
        "-block_size", "--block_size", type=none_or_type(int), default=None,
        help="Compatibility parameter; sampling_dc warns and ignores positive values.",
    )
    parser.add_argument(
        "-selected_hye_sizes", "--selected_hye_sizes", type=str, required=True,
        help="Path to a text file containing the selected integer hyperedge sizes.",
    )
    parser.add_argument(
        "-node_assignments", "--node_assignments", type=str, required=True,
        help="Path to integer community labels, one node per row.",
    )
    parser.add_argument(
        "-allow_repeated", "--allow_repeated", type=bool_type, default=True,
        help=(
            "True: Poisson copies at every order, including order two. "
            "False: simple Bernoulli hyperedges. Original runner default True is retained."
        ),
    )
    parser.add_argument(
        "--count_method", "--count-method", type=str.lower,
        choices=("exact", "approx"), default="exact",
        help="Group-count sampling: exact target law or approximate counts (default: exact).",
    )
    parser.add_argument(
        "-seed", "--seed", type=none_or_type(int), default=None,
        help="Random seed; None uses fresh randomness.",
    )
    parser.add_argument(
        "-save_dir", "--save_dir", type=none_or_type(Path), required=True,
        help="Output directory for results.pkl and args.pkl.",
    )

    # Choose the model explicitly; DCHSBM reads an N-dimensional activity vector per order from JSON.
    parser.add_argument(
        "-model", "--model", type=str.lower,
        choices=["hsbm", "dchsbm"], default="hsbm",
        help="Sampling model (default: hsbm).",
    )
    parser.add_argument(
        "-theta", "--theta", type=none_or_type(Path), default=None,
        help="DCHSBM activity JSON: {order: [theta_0, ..., theta_(N-1)]}.",
    )
    parser.add_argument(
        "-max_hyperedges", "--max_hyperedges", "--max-hyperedges",
        type=none_or_type(int), default=None,
        help="Output-copy cap; None disables this cap (default: None).",
    )
    parser.add_argument(
        "-max_compositions", "--max_compositions", "--max-compositions",
        type=none_or_type(int), default=None,
        help="Per-order type-composition cap; None disables it (default: None).",
    )

    args = parser.parse_args()
    if args.save_dir is None:
        parser.error("--save_dir must be a directory path, not None.")
    if args.model == "dchsbm" and args.theta is None:
        parser.error("--model dchsbm requires --theta theta.json.")
    if args.model == "hsbm" and args.theta is not None:
        parser.error("--model hsbm does not accept --theta; choose dchsbm to use activities.")
    for name in ("max_hyperedges", "max_compositions"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name} must be a positive integer or None.")

    with open(args.p, "r", encoding="utf-8") as f:
        p_ = json.load(f)
    p: dict[int, np.ndarray] = {int(k): np.asarray(v, dtype=float) for k, v in p_.items()}
    selected_hye_sizes = np.atleast_1d(
        np.loadtxt(args.selected_hye_sizes, dtype=int)
    ).tolist()
    node_assignments = np.atleast_1d(np.loadtxt(args.node_assignments, dtype=int))

    # Convert JSON order keys to integers; retain node order without normalizing activity arrays.
    theta = None
    if args.model == "dchsbm":
        with open(args.theta, "r", encoding="utf-8") as f:
            theta_ = json.load(f)
        theta = {int(k): np.asarray(v, dtype=float) for k, v in theta_.items()}

    # Preserve the first six positional arguments; pass additional settings by keyword.
    hyg = sampling.explicit_sampling_loop(
        p,
        selected_hye_sizes,
        node_assignments,
        args.allow_repeated,
        args.seed,
        args.block_size,
        model=args.model,
        theta=theta,
        count_method=args.count_method,
        max_hyperedges=args.max_hyperedges,
        max_compositions=args.max_compositions,
    )

    # Keep the output structure; args.pkl includes new settings, and repeated runs overwrite matching files.
    base_out = args.save_dir
    base_out.mkdir(parents=True, exist_ok=True)
    with open(base_out / "results.pkl", "wb") as results_f:
        pkl.dump(hyg, results_f, protocol=pkl.HIGHEST_PROTOCOL)
    with open(base_out / "args.pkl", "wb") as args_f:
        pkl.dump(vars(args), args_f, protocol=pkl.HIGHEST_PROTOCOL)
