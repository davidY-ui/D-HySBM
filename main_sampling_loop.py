"""Sample hypergraphs from per-order affinities and labels, then save the graph and CLI settings."""

import os ##
import argparse ##
import pickle as pkl
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import json
##from sem.str_to_type import bool_type, none_or_type
try:
    from sem.str_to_type import bool_type, none_or_type
except Exception:
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
            if isinstance(x, str) and x.lower() in {"none", "null", ""}:
                return None
            return t(x)
        return _parse

_ASCII_TMP = r"C:\tmp_joblib"
os.makedirs(_ASCII_TMP, exist_ok=True)
os.environ.setdefault("JOBLIB_TEMP_FOLDER", _ASCII_TMP)
os.environ.setdefault("TEMP", _ASCII_TMP)
os.environ.setdefault("TMP", _ASCII_TMP)
##
from src.model import sampling

if __name__ == "__main__":
    parser = ArgumentParser()
    # Input parameters.
    parser.add_argument(
        "-p",
        "--p",
        type=str,
        help=(
            "Symmetric matrix of community interaction probabilities."
            "The path is to be opened via json.load"
        ),
    )
    parser.add_argument(
        "-max_hye_size",
        "--max_hye_size",
        type=int,
        default=None,
        help="The maximum hyperedge size considered.",
    )
    parser.add_argument(
        "-block_size",
        "--block_size",
        type=int,
        default=None,
        help="To save Memory.",
    )
    parser.add_argument(
        "-selected_hye_sizes",
        "--selected_hye_sizes",
        type=str,
        default=None,
        help="Path to a file with selected hyperedge sizes.",
    )
    parser.add_argument(
        "-node_assignments",
        "--node_assignments",
        type=str,
        default=None,
        help=(
            "Path to a file with node assignments to communities."
            "The path is to be opened via numpy.loadtxt"
        ),
    )
    parser.add_argument(
        "-allow_repeated",
        "--allow_repeated",
        type=bool_type,
        default=True,
        help=(
            "Allow repetition of hyperedges in sampling. In sparsity regimes these are "
            "negligible compared to the size of the total hyperedge ensemble."
        ),
    )
    parser.add_argument(
        "-seed",
        "--seed",
        type=none_or_type(int),
        default=None,
        help="Seed for random sampling hyepergraphs.",
    )
    parser.add_argument(
        "-save_dir",
        "--save_dir",
        type=none_or_type(Path),
        help="Directory where results are saved.",
    )

    args = parser.parse_args()

    with open(args.p, "r", encoding="utf-8") as f:
        p_ = json.load(f)
    p: dict[int, np.ndarray] = {int(k): np.array(v) for k, v in p_.items()}
    # Keep one-order input files as lists, matching the sampler's multi-order interface.
    selected_hye_sizes = np.atleast_1d(
                np.loadtxt(args.selected_hye_sizes, dtype=int)).astype(int).tolist()
    #selected_hye_sizes = [int(x) for x in selected_hye_sizes]

    node_assignments = np.loadtxt(args.node_assignments, dtype=int)

    hyg = sampling.explicit_sampling_loop(
        p,
        selected_hye_sizes,
        node_assignments,
        args.allow_repeated,
        args.seed,
        args.block_size
    )
    
    # Serialization
    base_out = args.save_dir
    base_out.mkdir(parents=True, exist_ok=True)

    results_file = base_out / "results.pkl"
    with open(results_file, "wb") as results_f:
        pkl.dump(hyg, results_f, protocol=pkl.HIGHEST_PROTOCOL)

    args_file = base_out / "args.pkl"
    with open(args_file, "wb") as args_f:
        pkl.dump(vars(args), args_f, protocol=pkl.HIGHEST_PROTOCOL)
