"""Train HSBM/DCHSBM with BP/EM and select the restart with minimum finite free energy."""

import logging
import os
import random
import sys
import json
import pickle
import tempfile
from argparse import ArgumentParser, BooleanOptionalAction
from collections.abc import Mapping
from pathlib import Path

import numpy as np
PREPROCESSED_REALDATA_DIR = (
    Path(__file__).absolute().parent / "data" / "processed_input"
)

try:
    from sem.str_to_type import none_or_type
except Exception:
    def none_or_type(t):
        def _parse(x):
            if x is None:
                return None
            if isinstance(x, str) and x.lower() in {"none", "null"}:
                return None
            return t(x)
        return _parse

os.environ.setdefault("JOBLIB_TEMP_FOLDER", tempfile.gettempdir())

# Optional read-only overlay: run this copy against an existing HSBM checkout.
# When installed into that checkout, its ordinary src imports work unchanged.
_HSBM_ROOT = os.environ.get("HSBM_ROOT") or os.environ.get("ORIGINAL_HSBM_ROOT")
if _HSBM_ROOT:
    sys.path.insert(0, str(Path(_HSBM_ROOT).resolve()))
    import src.model
    _overlay = str(Path(__file__).resolve().parent / "src" / "model")
    if _overlay not in src.model.__path__:
        src.model.__path__.insert(0, _overlay)

from src.data.data_io import load_data
from src.model import dynamic_updates
from src.model.loop_hypergraph_block_model_rust import loop_HypergraphBlockModel
# from src.model.loop_hypergraph_block_model import loop_HypergraphBlockModel

import itertools as it
from scipy.optimize import linear_sum_assignment


def _load_hypergraph_input(real_dataset, hye_file, pickle_file, N):
    """Keep every observation in a pickled edge list, including repeated edges.

    Other input routes retain the original loader's behavior. Pickle inputs must
    come from a trusted source, as with the original command-line entry point.
    """
    if (bool(real_dataset) + bool(hye_file) + bool(pickle_file) != 1
            or not hye_file or Path(hye_file).suffix.lower() != ".pkl"):
        return load_data(real_dataset, hye_file, pickle_file, N)

    from scipy import sparse
    from src.data.representation.incidence_hypergraph import IncidenceHypergraph

    if N is not None and (isinstance(N, (bool, np.bool_))
                          or not isinstance(N, (int, np.integer)) or N <= 0):
        raise ValueError("N must be a positive integer or None")
    with Path(hye_file).open("rb") as stream:
        edges = pickle.load(stream)
    if (not isinstance(edges, (list, tuple, np.ndarray))
            or isinstance(edges, np.ndarray) and edges.ndim == 0
            or len(edges) == 0):
        raise ValueError("Pickled hyperedge input must be a nonempty list of edges")

    indices, indptr = [], [0]
    for edge_index, edge in enumerate(edges):
        if (not isinstance(edge, (list, tuple, np.ndarray))
                or isinstance(edge, np.ndarray) and edge.ndim != 1
                or len(edge) < 2):
            raise ValueError(f"Hyperedge {edge_index} must contain at least two node IDs")
        if any(isinstance(node, (bool, np.bool_))
               or not isinstance(node, (int, np.integer)) for node in edge):
            raise ValueError(f"Hyperedge {edge_index} must contain integer node IDs")
        nodes = [int(node) for node in edge]
        if len(set(nodes)) != len(nodes):
            raise ValueError(f"Hyperedge {edge_index} repeats a node within the edge")
        if any(node < 0 or N is not None and node >= N for node in nodes):
            raise ValueError(f"Hyperedge {edge_index} contains a node outside [0, N)")
        indices.extend(sorted(nodes))
        indptr.append(len(indices))
    node_count = max(indices) + 1 if N is None else int(N)
    incidence = sparse.csc_array(
        (np.ones(len(indices), dtype=np.uint8),
         np.asarray(indices, dtype=np.int64), np.asarray(indptr, dtype=np.int64)),
        shape=(node_count, len(edges)),
    )
    return IncidenceHypergraph(incidence)


# Get nodes incident to hyperedges/edges, shape = (N_active,).
def get_active_nodes_from_hyg(hyg):
    incidence = hyg.get_binary_incidence_matrix()

    # Convert to CSR for row-wise counts.
    incidence = incidence.tocsr()

    # Each row represents a node.
    # A nonzero row count means the node belongs to at least one hyperedge.
    active_nodes = np.flatnonzero(np.diff(incidence.indptr) > 0)

    return active_nodes

    

def get_real_overlap_nx(
    marginals,
    node_assignments,
    n=None,
    active_nodes=None,
):
    """Compute overlap from marginals and true node assignments.

    marginals: inference results, shape = (N, K).
    node_assignments: true labels, shape = (N,).
    active_nodes: if provided, evaluate only nodes incident to hyperedges.
    """

    marginals = np.asarray(marginals)
    node_assignments = np.asarray(
        node_assignments
    ).reshape(-1)

    # Filter only once.

    if active_nodes is not None:

        active_nodes = np.asarray(
            active_nodes,
            dtype=int
        ).reshape(-1)

        marginals_eval = marginals[
            active_nodes
        ]

        true_labels = node_assignments[
            active_nodes
        ]

    else:

        marginals_eval = marginals

        true_labels = node_assignments[
            :marginals.shape[0]
        ]

    N_eval = marginals_eval.shape[0]
    q = marginals_eval.shape[1]

    # predicted labels

    pred = np.argmax(
        marginals_eval,
        axis=1
    )

    # label permutation

    perms = it.permutations(
        range(q)
    )

    best_correct = 0

    for perm in perms:

        perm = np.asarray(
            perm
        )

        mapped_pred = perm[pred]

        correct = np.sum(
            mapped_pred == true_labels
        )

        if correct > best_correct:
            best_correct = correct


    accuracy = (
        best_correct / N_eval
    )

    # baseline

    _, counts = np.unique(
        true_labels,
        return_counts=True
    )

    max_n = (
        counts.max()
        / counts.sum()
    )

    overlap = (
        accuracy - max_n
    ) / (
        1.0 - max_n + 1e-12
    )


    return overlap




def _load_order_list(path, name):
    if path is None:
        return None
    values = np.atleast_1d(np.loadtxt(path, dtype=float))
    if (values.ndim != 1 or values.size == 0 or np.any(~np.isfinite(values))
            or np.any(values != np.floor(values)) or np.any(values < 2)):
        raise ValueError(f"{name} must contain a nonempty list of integer orders >= 2")
    orders = values.astype(int).tolist()
    if len(orders) != len(set(orders)):
        raise ValueError(f"{name} must not repeat orders")
    return orders


def _load_order_mapping(path, name, *, allow_null=False):
    with open(path, "r", encoding="utf-8") as file:
        raw = json.load(file)
    if raw is None and allow_null:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must be a JSON object keyed by integer order")
    values = {}
    for key, value in raw.items():
        try:
            order = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} contains a non-integer order: {key!r}") from exc
        if order < 2 or order in values:
            raise ValueError(f"{name} orders must be unique integers >= 2")
        values[order] = value
    return values


def _degree_save_fields(model):
    fields = {
        "theta": model.theta,
        "activity_moments": model.activity_moments(),
        "theta_by_order": dict(model.theta_by_order),
    }
    for name in (
        "layer_scale", "realized_layer_scale", "clipping_by_group", "theta_diff",
        "em_stop_reason", "em_converged", "em_training_iter", "training_iter",
        "converged", "mp_stop_reason", "mp_training_iter", "residual_history",
        "fix_theta_sizes", "em_history",
        "theta_constraint", "activity_constraint_beliefs", "activity_balance_residual",
        "last_activity_update",
    ):
        if hasattr(model, name):
            value = getattr(model, name)
            fields[name] = dict(value) if isinstance(value, Mapping) else value
    for name in ("log_node_to_hye", "log_hye_to_node"):
        messages = getattr(model, name)
        for component in ("data", "indices", "indptr", "shape"):
            fields[f"{name}_{component}"] = np.asarray(getattr(messages, component))
    return fields


def run_message_passing(argv=None):
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", type=str.lower, choices=("hsbm", "dchsbm"), default="hsbm",
                        help="Inference model; hsbm preserves the original loop implementation.")
    parser.add_argument("--affinity-update", "--affinity_update",
                        choices=("paper-w", "direct-p"), default="paper-w",
                        help="HSBM affinity learning: normalized shared W with layer scales (default), "
                             "or a directly learned p shared within each order group.")
    parser.add_argument("--theta", type=none_or_type(str), default=None,
                        help="DCHSBM JSON order -> node activity vector (mean one), or None.")
    parser.add_argument("--learn_theta", action=BooleanOptionalAction, default=None,
                        help="Learn activities; omitted means freeze supplied theta, otherwise learn.")
    parser.add_argument("--fix_theta_sizes", type=none_or_type(str), default=None,
                        help="DCHSBM text file of orders whose activity vectors are fixed.")
    parser.add_argument("--damping", type=float, default=0.0,
                        help="DCHSBM retained old-message fraction, in [0, 1).")
    parser.add_argument("--theta_constraint", choices=("soft", "hard"), default="hard",
                        help="DCHSBM activity mean-one constraint: frozen hard beliefs (default), "
                             "or argmax groups (a degree-based plug-in estimate).")
    parser.add_argument("--activity_regularization", type=float, default=0.0,
                        help="DCHSBM nonnegative activity regularization strength.")
    parser.add_argument("--require_bp_convergence", action=BooleanOptionalAction, default=False,
                        help="Require DCHSBM BP convergence before parameter updates (default: false).")
    parser.add_argument("--prior_update", choices=("hard", "soft"), default="hard",
                        help="DCHSBM community-prior update when n is not supplied.")
    # Data IO.
    parser.add_argument(
        "--real_dataset",
        type=none_or_type(str),
        default=None,
        help="Name of a real dataset to be loaded.",
    )
    parser.add_argument(
        "--hye_file",
        type=none_or_type(str),
        default=None,
        help=(
            "Path to a file containing a list of hyperedges representing a "
            "hypergraph."
        ),
    )
    parser.add_argument(
        "--node_assignments",
        type=none_or_type(str),
        default=None,
        help=(
            "Path to a file containing the true node assignments for the hypergraph."

        ),
    )
    parser.add_argument(
        "--pickle_file",
        type=none_or_type(str),
        default=None,
        help="Path to a file containing a pickle serialized hypergraph.",
    )
    # Data parameters.
    parser.add_argument(
        "--max_hye_size",
        type=none_or_type(int),
        default=None,
        help=(
            "The maximum hyperedge size considered. This value is used to exclude "
            "hyperedges in the configurations hypergraph, as well as a parameter of "
            "the probabilistic model to compute internal quantities."
        ),
    )

    parser.add_argument(
        "--fix_hye_size",
        type=none_or_type(str),
        default=None,
        help=(
            "Path to a text file containing the hyperedge orders whose affinity matrices are fixed."
        ),
    )
    parser.add_argument(
        "--selected_hye_sizes",
        type=none_or_type(str),
        default=None,
        help=(
            "Path to a text file listing hyperedge sizes to be selected. This value is used to exclude "
            "hyperedges in the configurations hypergraph, as well as a parameter of "
            "the probabilistic model to compute internal quantities."
        ),
    )

    parser.add_argument(
        "--shared_hye_sizes",
        type=none_or_type(str),
        default=None,
        help=(
            "JSON groups of hyperedge orders sharing the normalized affinity shape W."
        ),
    )

    parser.add_argument(
        "--omega",
        type=none_or_type(str),
        default=None,
        help=(
            "JSON order -> nonnegative squared layer weight; its square root is passed as omega."
        ),
    )

    parser.add_argument(
        "--beta",
        type=none_or_type(str),
        default=None,
        help=(
            "to enhance the d layer signal by beta_d "
        ),
    )

    parser.add_argument(
        "--save_dir", type=none_or_type(Path), help="Directory where results are saved."
    )
    # Model parameters.
    parser.add_argument(
        "--N",
        type=none_or_type(int),
        default=None,
        help=(
            "Number of nodes in the configurations hypergraph. Only needed (optionally)"
            " when specifying hye_file."
        ),
    )
    parser.add_argument(
        "--K",
        type=int,
        help="Number of communities in the model.",
    )
    parser.add_argument(
        "--n",
        type=none_or_type(str),
        default=None,
        help=(
            "Prior parameters for the communities of the stochastic block model. "
            "This is a path to a file to be opened via numpy.loadtxt. "
            "If not provided, the value of n is initialized at random. "
        ),
    )
    parser.add_argument(
        "--p",
        type=none_or_type(str),
        default=None,
        help=(
            "Symmetric matrix of community interaction probabilities. "
            "This is a path to a file to be opened via json.load "
            "If not provided, the value of p is initialized at random. "
        ),
    )
    # Model training.
    parser.add_argument(
        "--train_rounds",
        type=int,
        default=1,
        help=(
            "Train with different various random initializations and "
            "choose the model with the lowest free-energy selection score. "
            "Nonunit DCHSBM layer weights give a heuristic score."
        ),
    )
    parser.add_argument(
        "--em_iter",
        type=int,
        default=20,
        help="Max iterations of the EM procedure.",
    )
    parser.add_argument(
        "--em_thresh",
        type=float,
        default=1.0e-5,
        help=(
            "Threshold for the parameter change during EM. The difference is computed "
            "with respect to p and n, and additionally theta for DCHSBM."
        ),
    )
    parser.add_argument(
        "--mp_iter",
        type=int,
        default=2000,
        help="Max iterations of the message passing procedure.",
    )
    parser.add_argument(
        "--mp_thresh",
        type=float,
        default=1.0e-5,
        help=(
            "Threshold for the parameter change during message passing. "
            "HSBM checks log-marginal changes; DCHSBM checks the full candidate "
            "message, marginal and external-field residual."
        ),
    )
    parser.add_argument(
        "--mp_patience",
        type=int,
        default=50,
        help=(
            "Number of consecutive steps with the model's convergence measure "
            "below mp_thresh before message passing is stopped."
        ),
    )
    parser.add_argument(
        "--dirichlet_init_alpha",
        type=none_or_type(float),
        default=None,
        help="Dirichlet alpha utilized for the model initialization.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout in the message passing edge updates.",
    )
    parser.add_argument(
        "--dropout_vertex",
        type=float,
        default=0.5,
        help="Dropout in the message passing marginal updates for all.",
    )
    parser.add_argument(
        "--dropout_vertex_FVS",
        type=none_or_type(float),
        default=None,
    )

    parser.add_argument(
        "--dropout_vertex_rest",
        type=none_or_type(float),
        default=None,
    )
    parser.add_argument(
        "--dropout_FVS",
        type=float,
        default=0.5,
        help="Dropout in the message passing updates for FVS.",
    )
    parser.add_argument(
        "--warm_p_prior",
        action="store_true",
        help="Whether to warm the prior for the affinity matrix p.",
    )
    parser.add_argument(
        "--share_beta_d",
        action="store_true",
        help="if True, also pool the target mean c within each group."
        "Pooling uses C_prime_d weights and preserves group intensity before clipping."
        "It overrides unequal beta targets within that group.",
    )

    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help=(
            "Maximum number of parallel jobs. "
            "1 means no parallelization, -1 means all the available cores."
        ),
    )
    parser.add_argument(
        "--seed",
        type=none_or_type(int),
        default=None,
        help="Random seed.",
    )
    parser.add_argument(
        "--logging",
        type=str,
        default="INFO",
        help="Logging level.",
    )

    args = parser.parse_args(raw_args)
    if args.model != "hsbm" and args.affinity_update == "direct-p":
        parser.error("--affinity-update direct-p requires --model hsbm")
    degree_flags = {
        "--theta", "--learn_theta", "--no-learn_theta", "--fix_theta_sizes", "--damping",
        "--theta_constraint", "--activity_regularization", "--require_bp_convergence",
        "--no-require_bp_convergence", "--prior_update",
    }
    explicit_degree = sorted({arg.split("=", 1)[0] for arg in raw_args} & degree_flags)
    if args.model == "hsbm" and explicit_degree:
        parser.error(f"{', '.join(explicit_degree)} requires --model dchsbm")
    if not np.isfinite(args.damping) or not 0 <= args.damping < 1:
        parser.error("--damping must be finite and in [0, 1)")
    if not np.isfinite(args.activity_regularization) or args.activity_regularization < 0:
        parser.error("--activity_regularization must be finite and nonnegative")
    if args.K is None or args.K < 2:
        parser.error("--K must be at least 2")
    if min(args.train_rounds, args.em_iter, args.mp_iter, args.mp_patience) < 1:
        parser.error("--train_rounds, --em_iter, --mp_iter and --mp_patience must be positive")

    random.seed(args.seed)
    logging.getLogger().setLevel(args.logging.upper())

    overlap_compute = False

    hyg = _load_hypergraph_input(
        args.real_dataset,
        args.hye_file,
        args.pickle_file,
        args.N,
    )
    if args.max_hye_size is not None:
        hyg = hyg.max_hye_size_select(args.max_hye_size)


    try:
        selected_hye_sizes = _load_order_list(args.selected_hye_sizes, "--selected_hye_sizes")
        if selected_hye_sizes is not None:
            hyg = hyg.hye_sizes_select(selected_hye_sizes)
        selected = {int(d) for d in hyg.selected_hye_sizes}
        if not selected:
            raise ValueError("No hyperedges remain after selecting their sizes")

        omega = {d: 1.0 for d in selected}
        if args.omega is not None:
            raw_omega = _load_order_mapping(args.omega, "--omega")
            omega = {d: float(value) for d, value in raw_omega.items()}
            if any(not np.isfinite(v) or v < 0 for v in omega.values()):
                raise ValueError("--omega values must be finite and nonnegative")
            if not selected.issubset(omega):
                raise ValueError("--omega must provide every selected order")
            omega = {d: float(np.sqrt(v)) for d, v in omega.items()}

        beta_d = None
        if args.beta is not None:
            beta_d = {d: float(v) for d, v in _load_order_mapping(args.beta, "--beta").items()}
            if any(not np.isfinite(v) or v <= 0 for v in beta_d.values()):
                raise ValueError("--beta values must be finite and positive")

        fix_hye_size = _load_order_list(args.fix_hye_size, "--fix_hye_size")
        fix_theta_sizes = _load_order_list(args.fix_theta_sizes, "--fix_theta_sizes")
        for name, orders in (("--fix_hye_size", fix_hye_size), ("--fix_theta_sizes", fix_theta_sizes)):
            if orders is not None and not set(orders).issubset(selected):
                raise ValueError(f"{name} must be a subset of selected orders")

        shared_hye_sizes = None
        if args.shared_hye_sizes is not None:
            with open(args.shared_hye_sizes, "r", encoding="utf-8") as file:
                shared_hye_sizes = json.load(file)
            if (not isinstance(shared_hye_sizes, list)
                    or any(not isinstance(group, list) or not group for group in shared_hye_sizes)):
                raise ValueError("--shared_hye_sizes must be a JSON list of nonempty groups")
            flattened = [d for group in shared_hye_sizes for d in group]
            if any(type(d) is not int or d not in selected for d in flattened):
                raise ValueError("--shared_hye_sizes must use selected integer orders")
            if len(flattened) != len(set(flattened)):
                raise ValueError("--shared_hye_sizes groups must not overlap or repeat orders")

        theta = None
        if args.theta is not None:
            raw_theta = _load_order_mapping(args.theta, "--theta", allow_null=True)
            if raw_theta is not None:
                theta = {d: np.asarray(value, dtype=float) for d, value in raw_theta.items()}
                if set(theta) != selected:
                    raise ValueError("--theta keys must equal selected orders")
                if any(v.shape != (hyg.N,) or np.any(~np.isfinite(v)) or np.any(v < 0)
                       or not np.isclose(v.mean(), 1.0, rtol=1e-10, atol=1e-12)
                       for v in theta.values()):
                    raise ValueError(f"--theta needs finite nonnegative vectors of length {hyg.N}, each with mean one")

        n = None if args.n is None else np.atleast_1d(np.loadtxt(args.n, dtype=float))
        if n is not None and (n.shape != (args.K,) or np.any(~np.isfinite(n))
                              or np.any(n < 0) or not np.isclose(n.sum(), 1.0)):
            raise ValueError("--n must be a nonnegative normalized vector of length K")
        p_list = None
        if args.p is not None:
            p_list = {d: np.asarray(v, dtype=float) for d, v in _load_order_mapping(args.p, "--p").items()}
            if any(v.shape != (args.K, args.K) or np.any(~np.isfinite(v)) or np.any(v < 0)
                   or not np.allclose(v, v.T) for v in p_list.values()):
                raise ValueError("--p needs finite nonnegative symmetric K by K matrices")
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    if args.node_assignments is not None:
        overlap_compute = True
        node_assignments = np.atleast_1d(
                np.loadtxt(args.node_assignments, dtype=int))
    else:
        logging.warning(
        "Warning: true node assignments are not provided. "
        "Overlap evaluation will be skipped."
        )

    # Set maximum number of parallel jobs.
    dynamic_updates.N_JOBS = args.n_jobs

    best_model = None
    best_free_energy = float("inf")
    all_free_energy = []
    logging.info(f"Selected hyperedge sizes: {hyg.selected_hye_sizes}, fix_hye_size: {fix_hye_size}")
    model_class = loop_HypergraphBlockModel
    constructor_extra = dict(affinity_update=args.affinity_update)
    em_extra = {}
    if args.model == "dchsbm":
        from src.model.loop_degree_corrected_hypergraph_block_model_rust import (
            loop_DegreeCorrectedHypergraphBlockModel,
        )
        model_class = loop_DegreeCorrectedHypergraphBlockModel
        constructor_extra = dict(theta=theta, fix_theta_sizes=fix_theta_sizes,
                                 theta_constraint=args.theta_constraint)
        em_extra = dict(
            learn_theta=args.learn_theta, damping=args.damping,
            activity_regularization=args.activity_regularization,
            require_bp_convergence=args.require_bp_convergence, prior_update=args.prior_update,
        )
    # Compare independent initializations by free energy before computing optional label-based metrics.
    for i in range(args.train_rounds):
        model = model_class(
            n=n, p_list=p_list, N=hyg.N, K=args.K, fix_hye_size=fix_hye_size,
            selected_hye_sizes=hyg.selected_hye_sizes, shared_hye_sizes=shared_hye_sizes,
            omega=omega, beta_d=beta_d, share_beta_d=args.share_beta_d, **constructor_extra,
        )
        model.em_inference(
            hypergraph=hyg,
            em_iter=args.em_iter,
            em_thresh=args.em_thresh,
            mp_iter=args.mp_iter,
            mp_thresh=args.mp_thresh,
            mp_patience=args.mp_patience,
            seed=args.seed + i * 1024 if args.seed is not None else None,
            dirichlet_alpha=args.dirichlet_init_alpha,
            dropout=args.dropout,
            dropout_vertex= args.dropout_vertex,
            dropout_vertex_FVS = args.dropout_vertex_FVS,
            dropout_vertex_rest = args.dropout_vertex_rest,
            dropout_FVS=args.dropout_FVS,
            warm_p_prior=args.warm_p_prior,
            **em_extra,
        )
        free_energy = model.free_energy(hyg)
        logging.info(f"free_energy:\n{free_energy}")
        all_free_energy.append(free_energy)
        if np.isfinite(free_energy) and free_energy < best_free_energy:
            best_model = model
            best_free_energy = free_energy

    if best_model is None:
        raise RuntimeError("No training round produced a finite free energy")
    active_nodes=get_active_nodes_from_hyg(hyg)
    overlap = None
    overlap_active = None
    if overlap_compute:
        active_nodes=get_active_nodes_from_hyg(hyg)
        overlap_active = get_real_overlap_nx(best_model.log_marginals, node_assignments, best_model.n, active_nodes=active_nodes)
        overlap = get_real_overlap_nx(best_model.log_marginals, node_assignments, best_model.n)

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)

        # Command line arguments.
        with open(args.save_dir / "args.txt", "w", encoding="utf-8") as file:
            file.write("\n".join(raw_args))

        # Inference results.
        np.savez(
            args.save_dir / "inferred_params.npz",
            model=args.model,
            affinity_update=args.affinity_update,
            overlap_active=overlap_active,
            overlap=overlap,
            log_marginals=best_model.log_marginals,
            # log_hye_to_node=best_model.log_hye_to_node,
            # log_node_to_hye=best_model.log_node_to_hye,
            external_field=best_model.external_field,
            best_free_energy=best_free_energy,
            all_free_energy=all_free_energy,
            p=best_model.p,
            n=best_model.n,
            n_diff=best_model.n_diff,
            c_diff=best_model.c_diff,
            log_marginal_diff=(np.hstack(best_model.log_marginal_diff)
                               if best_model.log_marginal_diff else np.empty(0)),
            mp_iter_per_em_iter=[len(x) for x in best_model.log_marginal_diff],
            em_total_time = best_model.em_total_time,
            em_time_per_iter = best_model.em_time_per_iter,
            update_message_ratio = best_model.update_message_ratio,
            update_vertex_ratio = best_model.update_vertex_ratio,
            omega = dict(best_model.omega),
            shared_hye_sizes=np.asarray(
            best_model.shared_hye_sizes, dtype=object
            ),
            selected_hye_sizes=np.asarray(
                best_model.selected_hye_sizes, dtype=int
            ),
            share_beta_d=best_model.share_beta_d,
            effective_beta_d=(dict(best_model.effective_beta_d)
                              if best_model.effective_beta_d is not None else None),
            c0=best_model.c0,
            **(_degree_save_fields(best_model) if args.model == "dchsbm" else {}),
        )

    return best_model, best_free_energy



if __name__ == "__main__":
    run_message_passing()
