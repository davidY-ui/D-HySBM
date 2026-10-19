"""Loop DCHSBM: directly extends the original loop_HypergraphBlockModel.

All DC validation, BP, learning and free-energy calculations live in this file.
Only the original HSBM modules and its rust_hysbm extension are required.
Theta is per order with mean one; nonunit omega retains the loop heuristic weighting.
"""
# Reading order: constructor -> parallel_message_passing (fixed-parameter inference)
#                   -> em_inference (alternating parameter learning) -> free_energy (scoring).
# Numerical helpers occupy the first half of this file; the model class follows.
#
# Notation and array shapes:
#     N: nodes; K: communities; L: selected orders; M: node-hyperedge incidences.
#     d: actual hyperedge order; layer: array index (also i in enumerate(self.orders)).
#     n[a]: community prior proportion, shape (K,).
#     p[layer,a,b] = c[layer,a,b]/N: pairwise affinity, shape (L,K,K).
#     theta[layer,node]: relative node activity, shape (L,N), with mean 1 in each layer.
#     q[node,a]: marginal probability of community a, shape (N,K).
#     Both log-message directions have shape (K,M) and share the original loop CSC structure.
# theta controls node activity, beta_d target layer intensity, and omega message weights.
from collections.abc import Mapping
from itertools import combinations
from math import comb
from types import MappingProxyType, SimpleNamespace
import time

import numpy as np
from scipy import sparse, special

# Reuse the original HSBM parent, message container, and Rust wrappers directly.
from .loop_hypergraph_block_model_rust import (
    loop_HypergraphBlockModel, SharedCSCMessages, CLIP_MIN,
)
from . import dynamic_updates as updates
from .kappa import compute_C_prime_d
from .numerical import log_binomial_coefficient
from .fvs_function import FcoreHD_FVS, FVS_random_mask_proration


def normalized_log(values, axis):
    """Normalize and floor probabilities, then normalize again.

    Keep the original SharedCSCMessages container, but do not silently accept
    an impossible all-zero factor or lose normalization after clipping.
    """
    values = np.asarray(values, dtype=np.float64)
    if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        raise ValueError("Message contains NaN or +infinity")
    norm = special.logsumexp(values, axis=axis, keepdims=True)
    if np.any(~np.isfinite(norm)):
        raise ValueError("Impossible observation: all community weights are zero")
    # Normalize in log space to avoid product underflow; renormalize after clipping tiny values.
    # Messages normalize across K communities on axis=0; node marginals use axis=1.
    result = np.maximum(values - norm, CLIP_MIN)
    # Use stable logsumexp first. After clipping, values are in a safe range,
    # so exp/sum/log avoids another general logsumexp scan and scaling pass.
    return result - np.log(np.exp(result).sum(axis=axis, keepdims=True))


def as_incidence(hypergraph):
    """Accept the original IncidenceHypergraph or a binary scipy CSC array."""
    # Each CSC column is a hyperedge; its nonzero positions identify member nodes.
    # Copy and canonicalize the sparse structure to preserve the caller's incidence.
    source = (hypergraph.get_binary_incidence_matrix()
              if hasattr(hypergraph, "get_binary_incidence_matrix") else hypergraph)
    B = sparse.csc_array(source, copy=True)
    B.sum_duplicates()
    B.eliminate_zeros()
    B.sort_indices()
    if np.any(~np.isfinite(B.data)) or np.any(B.data != 1):
        raise ValueError("Expected a binary, unweighted incidence matrix")
    return B


def _resolve_backend(backend):
    """All accepted names select the SAME extension used by HSBM."""
    # auto / rust / hsbm-rust are aliases for the same original HSBM backend.
    if backend not in {"auto", "rust", "hsbm-rust"}:
        raise ValueError("backend must be auto, rust, or hsbm-rust; all use the original rust_hysbm")
    native = getattr(updates, "rust_hysbm", None)
    required = ("compute_psi_all_loop_p_data", "compute_psi_tilde_all_loop_p")
    if not getattr(updates, "HAS_RUST_HYSBM", False) or any(
            not callable(getattr(native, name, None)) for name in required):
        raise ImportError(
            "Install the original HSBM rust_hysbm extension with its per-order APIs. "
            "No separate DCHSBM Rust extension is required.")
    return "hsbm-rust"


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _rate(value, name):
    if not np.isscalar(value) or not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return float(value)


def posterior_statistics(model, B):
    """Snapshot node beliefs and posterior responsibilities of latent pairs.

    One observed hyperedge contributes total responsibility one.  Symmetric
    off-diagonal entries each contain half the unordered label-pair count.
    Prefix outer products avoid cancellation in nearly deterministic messages.
    """
    # Hold BP messages fixed and collect statistics for parameter updates.
    # Responsibilities give each community pair's relative share in the hyperedge pair sum.
    # Shares sum to 1 per hyperedge; do not count it as d(d-1)/2 independent ordinary edges.
    q = np.exp(model.log_marginals)
    if (q.shape != (model.N, model.K) or not np.isfinite(q).all()
            or not np.allclose(q.sum(axis=1), 1, rtol=1e-8, atol=1e-12)):
        raise ValueError("Marginals must have finite normalized rows of shape (N,K)")
    incoming = np.exp(model.log_node_to_hye.data)
    if (incoming.shape != (model.K, B.nnz) or not np.isfinite(incoming).all()
            or not np.allclose(incoming.sum(axis=0), 1, rtol=1e-8, atol=1e-12)):
        raise ValueError("Cavity messages must be finite and normalized")
    layers = {d: layer for layer, d in enumerate(model.orders)}
    responsibility = np.zeros_like(model.p, dtype=float)
    degrees = np.zeros((len(model.orders), model.N), dtype=float)
    edge_count = np.zeros(len(model.orders), dtype=np.int64)
    for edge, d in enumerate(np.diff(B.indptr)):
        layer = layers[int(d)]
        start, stop = B.indptr[edge:edge + 2]
        nodes = B.indices[start:stop]
        degrees[layer, nodes] += 1
        edge_count[layer] += 1
        prefix = np.zeros(model.K)
        pair_count = np.zeros((model.K, model.K))
        # Accumulate pairs with preceding nodes. Store half of each off-diagonal
        # unordered community pair in (a,b) and (b,a) to avoid double counting with symmetric affinity.
        for cavity in incoming[:, start:stop].T:
            pair = np.outer(prefix, cavity)
            pair_count += (pair + pair.T) / 2
            prefix += cavity
        pmax = model.p[layer].max()
        if pmax <= 0:
            raise ValueError("Impossible observation: zero affinity in observed layer")
        # Dividing by pmax improves numerical scaling; normalization cancels this common factor.
        weighted = (model.p[layer] / pmax) * pair_count
        normalizer = weighted.sum()
        if not np.isfinite(normalizer) or normalizer <= 0:
            raise ValueError("Impossible observation: zero latent-pair normalizer")
        responsibility[layer] += weighted / normalizer
    return {"q": q.copy(), "responsibilities": responsibility,
            "degrees": degrees, "edge_count": edge_count}


def sparse_exposure(p, theta, q, orders):
    """Return Lambda_d = chi_d/2 (T_d/N)^(d-2) S_d.T p_d S_d."""
    # Aggregate the many absent hyperedges into per-layer Lambda_d under the sparse approximation.
    # S_d[a] = sum_i theta_i^(d) q_i(a)，T_d = sum_i theta_i^(d)。
    # Mean-one theta gives T_d/N=1. Local n is the node count, not the community prior.
    n = q.shape[0]
    result = np.empty(len(orders), dtype=float)
    for layer, d in enumerate(orders):
        mass = theta[layer] @ q
        ratio = theta[layer].sum() / n
        result[layer] = ratio ** (d - 2) * (mass @ p[layer] @ mass) / (d * (d - 1))
    return result


def _count_normalization(counts, lower):
    """Exact count fit on one group with sum(theta)=group size."""
    size = len(counts)
    if not counts.any() or lower == 1:
        return np.ones(size)
    # Active-set method: zero counts may reach the lower bound; allocate the remainder by effective counts.
    counts = counts / counts.max()
    active = counts > 0
    result = np.full(size, lower)
    while True:
        mass = size - lower * np.count_nonzero(~active)
        values = mass * counts[active] / counts[active].sum()
        below = values < lower
        if not below.any():
            result[active] = values
            result[np.argmax(result)] += size - result.sum()
            return result
        indices = np.flatnonzero(active)
        active[indices[below]] = False


def community_activity_proposal(counts, q, *, mode="soft", lower=1e-10):
    """Fit sum(counts*log(theta)) with community means fixed at one.

    soft: q.T@theta=q.T@ones; hard: the same constraint for argmax groups.
    The numerical barrier below vanishes; it is NOT activity_regularization.
    Rank reduction and diagonal Newton curvature keep storage O(N*K).
    """
    counts, q = np.asarray(counts, float), np.asarray(q, float)
    if (counts.ndim != 1 or counts.size == 0 or not np.isfinite(counts).all()
            or np.any(counts < 0)):
        raise ValueError("counts must be a nonempty finite nonnegative vector")
    if (q.ndim != 2 or q.shape[0] != len(counts) or q.shape[1] == 0
            or not np.isfinite(q).all() or np.any(q < 0)
            or not np.allclose(q.sum(1), 1., rtol=0, atol=1e-12)):
        raise ValueError("q must contain normalized nonnegative node beliefs")
    if mode not in ("soft", "hard"):
        raise ValueError("theta_constraint must be 'soft' or 'hard'")
    if not np.isfinite(lower) or not 0 < lower <= 1:
        raise ValueError("min_activity must lie in (0,1]")
    if lower == 1 or not counts.any():
        return np.ones(len(counts))
    labels = q.argmax(1)
    if mode == "hard" or np.all((q == 0) | (q == 1)):
        result = np.ones(len(counts))
        for label in np.unique(labels):
            selected = labels == label
            result[selected] = _count_normalization(counts[selected], lower)
        return result

    # The thin SVD of q uses only N x K arrays; verify the original equality after dropping numerical null directions.
    basis, singular, _ = np.linalg.svd(q, full_matrices=False)
    rank = np.count_nonzero(singular > np.finfo(float).eps * max(q.shape) * singular[0])
    basis = basis[:, :rank]
    if rank == 1:
        result = _count_normalization(counts, lower)
    else:
        # Uniform count scaling preserves the optimum; mean scaling handles extreme counts.
        scaled = counts / counts.max()
        scaled /= scaled.mean()
        result = np.ones(len(counts))
        for barrier in 10. ** -np.arange(12):
            for iteration in range(100):
                gap = result - lower
                gradient = -scaled / result - barrier / gap
                curvature = scaled / result**2 + barrier / gap**2
                # Remove the equality-direction gradient, then project with weights to avoid an N x N Hessian.
                gradient -= basis @ (basis.T @ gradient)
                root_inverse = 1. / np.sqrt(curvature)
                weighted_basis, _ = np.linalg.qr(root_inverse[:, None] * basis, mode="reduced")
                vector = root_inverse * gradient
                direction = -root_inverse * (vector - weighted_basis @ (weighted_basis.T @ vector))
                direction -= basis @ (basis.T @ direction)
                decrement = -float(gradient @ direction)
                if decrement <= max(1e-13, .01 * len(counts) * barrier):
                    break
                negative = direction < 0
                step = min(1., .99 * float(np.min(-gap[negative] / direction[negative]))) if negative.any() else 1.
                objective = -float(special.xlogy(scaled, result).sum() + barrier * np.log(gap).sum())
                for _ in range(60):
                    trial = result + step * direction
                    if np.all(trial > lower):
                        candidate = -float(special.xlogy(scaled, trial).sum()
                                           + barrier * np.log(trial - lower).sum())
                        if candidate <= objective - .01 * step * decrement + 1e-14 * (1 + abs(objective)):
                            result = trial
                            break
                    step *= .5
                else:
                    raise RuntimeError("Community activity Newton line search failed")
            else:
                raise RuntimeError("Community activity Newton did not converge")
        # Correct rounding drift only; retain the original constraint and objective.
        result -= basis @ (basis.T @ (result - 1))
    residual = q.T @ (result - 1)
    if (not np.isfinite(result).all() or result.min() < lower - 1e-12
            or np.max(np.abs(residual), initial=0) > 1e-9):
        raise RuntimeError("Community activity constraints did not converge to 1e-9")
    return result


def shared_shape_proposal(responsibilities, exposure, n, initial=None):
    """Optimize a shared shape at fixed beliefs, activities, scales and prior.

    Maximize ``sum(R*log(W) - A*W)`` for a symmetric nonnegative W with
    ``n.T @ W @ n == 1``. Full symmetric matrices count off-diagonal terms
    twice in both the objective and the constraint; no extra factor is needed.
    For positive counts the solution is ``R / (A + lambda*outer(n,n))``.

    The scalar solve uses x_ab=n_a*n_b*W_ab. At a boundary optimum, zero-count
    coordinates having the smallest linear cost receive any remaining mass.
    No artificial pseudocount is added. All-zero R is deliberately a no-op on
    the normalized initial shape (or ones), not a claim of optimality.

    This helper applies no probability cap. Clipping its result, changing n
    or theta, or applying heuristic weighted BP is outside its M-step guarantee.
    """
    # Write grouped layers as p_d = s_d * W and learn their common shape W.
    # R contains weighted observed-hyperedge statistics; A is the weighted absent-hyperedge cost.
    # Optimize sum(R*log(W) - A*W) subject to n.T @ W @ n = 1.
    # lambda / multiplier below are constraint multipliers, not community signal parameters.
    prior = np.asarray(n, dtype=float)
    if (prior.ndim != 1 or prior.size == 0 or not np.isfinite(prior).all()
            or np.any(prior <= 0)
            or not np.isclose(prior.sum(), 1., rtol=1e-10, atol=1e-12)):
        raise ValueError("n must be a finite positive normalized vector")
    shape = (prior.size, prior.size)

    def symmetric_matrix(value, name):
        result = np.asarray(value, dtype=float)
        if (result.shape != shape or not np.isfinite(result).all()
                or np.any(result < 0)
                or not np.allclose(result, result.T, rtol=1e-12, atol=1e-14)):
            raise ValueError(f"{name} must be a finite nonnegative symmetric K by K matrix")
        return .5 * result + .5 * result.T

    count = symmetric_matrix(responsibilities, "responsibilities")
    linear = symmetric_matrix(exposure, "exposure")
    base = None if initial is None else symmetric_matrix(initial, "initial")
    constraint = np.outer(prior, prior)
    if np.any(constraint == 0):
        raise ValueError("n is too small to represent the shape constraint")
    if base is not None:
        mean = float(prior @ base @ prior)
        if not np.isfinite(mean) or mean <= 0:
            raise ValueError("initial must have a positive finite prior-weighted mean")
        base = base / mean
        if not np.isfinite(base).all():
            raise ValueError("Normalized initial shape is not finite")
    if not np.any(count > 0):
        return np.ones(shape) if base is None else base

    # Common objective rescaling leaves the optimizer unchanged, while avoiding
    # large raw counts/exposures in the multiplier solve.
    scale = max(float(count.max()), float(linear.max()))
    weight = count / scale
    positive = count > 0
    if np.any(weight[positive] == 0):
        raise ValueError("Responsibilities and exposure have an unrepresentable dynamic range")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        cost = (linear / scale) / constraint
    if not np.isfinite(cost).all():
        raise ValueError("Prior-weighted exposure is not finite")
    shifted_cost = cost - cost.min()
    positive_cost = shifted_cost[positive]
    positive_weight = weight[positive]
    mass = np.zeros(shape)
    at_boundary = (float(np.sum(positive_weight / positive_cost))
                   if np.all(positive_cost > 0) else np.inf)
    # Do not add arbitrary pseudocounts: the optimum may lie on a boundary with some W_ab=0.
    if at_boundary <= 1.:
        mass[positive] = positive_weight / positive_cost
        flat = (~positive) & (shifted_cost == 0)
        mass[flat] = (1. - at_boundary) * constraint[flat] / constraint[flat].sum()
    else:
        low = max(float(np.max(positive_weight - positive_cost)), np.nextafter(0., 1.))
        high = float(positive_weight.sum())
        # After shifting the smallest cost to zero, t=lambda+min(cost) >= 0.
        # sum(R/(cost+t)) decreases continuously and is <=1 at t=sum(R).
        for _ in range(200):
            # Each coordinate has mass <=1, hence t >= max(R-cost).
            # A geometric midpoint resolves very small positive counts without
            # thousands of absolute-precision bisections from t=0.
            middle = (np.sqrt(low) * np.sqrt(high) if high > 2 * low
                      else low + .5 * (high - low))
            if middle == low or middle == high:
                break
            value = float(np.sum(positive_weight / (positive_cost + middle)))
            if value > 1.:
                low = middle
            else:
                high = middle
        multiplier = low + .5 * (high - low)
        mass[positive] = positive_weight / (positive_cost + multiplier)
        # Only correct scalar-solve roundoff, not an unconstrained proposal.
        total = float(mass.sum())
        if not np.isfinite(total) or not np.isclose(total, 1., rtol=1e-10, atol=1e-12):
            raise ValueError("Shared-shape multiplier did not converge")
        mass /= total
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        result = mass / constraint
    if not np.isfinite(result).all():
        raise ValueError("Shared shape is not finite")
    return result


def estimate_theta_from_degrees(hypergraph, orders, pseudocount=0.5):
    """Return (k_i^(d)+alpha)/(mean(k^(d))+alpha) separately for each order.

    This is a regularized moment INITIALIZATION, not a Bernoulli MLE or EM
    update. It uses no community labels. It is most interpretable under the
    equal-expected-degree, activity-independent-of-community assumptions.
    Fit on the training graph only. An unobserved layer with alpha>0 yields
    theta=1 as a prior convention, not evidence of homogeneous activity.
    """
    # Initialize theta_i^(d) = (order-specific degree + pseudocount) / (mean degree + pseudocount).
    # Initialization uses no true community labels; learn_theta enables subsequent iterative learning.
    if not np.isfinite(pseudocount) or pseudocount < 0:
        raise ValueError("pseudocount must be finite and nonnegative")
    B = as_incidence(hypergraph)
    sizes = np.diff(B.indptr)
    result = {}
    for d in orders:
        if isinstance(d, bool) or not isinstance(d, (int, np.integer)) or d < 2:
            raise ValueError("Orders must be integers >=2")
        degree = np.asarray(B[:, sizes == d].sum(axis=1)).ravel()
        mean = degree.mean()
        if mean + pseudocount == 0:
            raise ValueError("An empty layer has no estimable activity scale; use a positive pseudocount")
        result[d] = (degree + pseudocount) / (mean + pseudocount)
    return result


# Inherit c, hard community-prior updates, and shared-group synchronization from the original loop class.
# Override the external field, parameter constraints, and learning objective for degree correction.
class loop_DegreeCorrectedHypergraphBlockModel(loop_HypergraphBlockModel):
    """Per-order degree correction of the original loop HSBM.

    n supplied: fixed by default. Nonfixed p_list entries are initial values.
    theta supplied: fixed by default unless learn_theta is enabled.
    fix_hye_size freezes p; fix_theta_sizes separately freezes theta.
    c and the prior-update rules are inherited from loop_HypergraphBlockModel.
    DC-specific validation initializes this class without the parent's ordinary
    HSBM probability constraints. All numerical DC methods are defined here.
    """
    message_container_class = SharedCSCMessages

    def __init__(self, n, p_list, N, K, fix_hye_size,
                 selected_hye_sizes=None, shared_hye_sizes=None,
                 omega=None, beta_d=None, share_beta_d=False, *,
                 theta=None, fix_theta_sizes=None, backend="auto", theta_constraint="soft"):
        # p_list keys are actual orders, e.g. {2: p2, 3: p3}.
        # Supplied n / theta are fixed by default; p_list for unfrozen layers supplies initial values only.
        # fix_hye_size freezes affinities; fix_theta_sizes independently freezes node activities.
        if p_list is not None and not isinstance(p_list, Mapping):
            raise ValueError("p_list must be a dictionary keyed by order, or None")
        if selected_hye_sizes is None:
            if not p_list:
                raise ValueError("Supply selected_hye_sizes when p_list is omitted")
            selected_hye_sizes = list(p_list)
        orders = tuple(_integer(d, "order", 2) for d in selected_hye_sizes)
        if not orders or len(set(orders)) != len(orders):
            raise ValueError("selected_hye_sizes must be nonempty and unique")
        if p_list is not None and set(p_list) - set(orders):
            raise ValueError("p_list contains unselected orders")
        self.fix_hyper_layer = list(fix_hye_size or [])
        self.fix_theta_sizes = list(fix_theta_sizes or [])
        for name, values in (("fix_hye_size", self.fix_hyper_layer),
                             ("fix_theta_sizes", self.fix_theta_sizes)):
            if len(set(values)) != len(values) or any(d not in orders for d in values):
                raise ValueError(f"{name} must contain distinct selected orders")
        if self.fix_hyper_layer and n is None:
            raise ValueError("Fixed layers with fixed scales require a fixed n")
        if any(p_list is None or d not in p_list for d in self.fix_hyper_layer):
            raise ValueError("Supply p_list for every fixed layer")
        # Groups share the shape W; p_d need not have identical values across layers.
        # Ungrouped layers form singleton groups; a group cannot mix frozen and learnable layers.
        self.shared_hye_sizes = []
        seen = set()
        for group in shared_hye_sizes or []:
            group = [group] if np.isscalar(group) else list(group)
            if (not group or any(isinstance(d, (bool, np.bool_))
                                  or not isinstance(d, (int, np.integer))
                                  or d not in orders for d in group)
                    or len(set(group)) != len(group) or seen.intersection(group)):
                raise ValueError("Shared groups must be nonempty, selected and nonoverlapping")
            self.shared_hye_sizes.append([int(d) for d in group])
            seen.update(group)
        self.shared_hye_sizes.extend([d] for d in orders if d not in seen)
        for group in self.shared_hye_sizes:
            fixed = [d in self.fix_hyper_layer for d in group]
            if any(fixed) and not all(fixed):
                raise ValueError("Fixed and learnable layers need separate groups")
        if omega is not None and (not isinstance(omega, Mapping) or set(omega) != set(orders)):
            raise ValueError("omega must have exactly the selected order keys")
        # w_d weights messages, the external field, and learning statistics; w_d=0 excludes a layer from inference.
        # The constructor uses w directly; only the CLI takes square roots of omega-file values.
        self.omega = {d: float(1 if omega is None else omega[d]) for d in orders}
        if any(not np.isfinite(v) or v < 0 for v in self.omega.values()):
            raise ValueError("Layer weights must be finite and nonnegative")
        if beta_d is not None and not isinstance(beta_d, Mapping):
            raise ValueError("beta_d must be a dictionary or None")
        # beta_d sets relative target mean affinity, with second-order beta_2=1 as reference.
        # It does not directly specify the within/between-community affinity ratio.
        self.beta_d = None if beta_d is None else {**beta_d, 2: 1.}
        if self.beta_d is not None and any(not np.isfinite(v) or v <= 0 for v in self.beta_d.values()):
            raise ValueError("beta_d must be finite and positive")
        if not isinstance(share_beta_d, (bool, np.bool_)):
            raise ValueError("share_beta_d must be boolean")
        self.share_beta_d = bool(share_beta_d)
        self._initial_p_orders = set(p_list or {})
        ps = (None if p_list is None else
              {d: p_list.get(d, np.zeros((K, K))) for d in orders})
        self._provided_n = n is not None
        self._provided_p = p_list is not None
        self._provided_theta = theta is not None
        self.N = _integer(N, "N", 2)
        self.K = _integer(K, "K", 1)
        self.orders = orders
        self.theta_constraint = theta_constraint
        self.activity_constraint_beliefs = None
        self.activity_balance_residual = np.full((len(orders), self.K), np.nan)
        self._activity_learned_layers = np.zeros(len(orders), dtype=bool)
        self.last_activity_update = None
        self.max_hye_size = max(orders)
        if self.max_hye_size > self.N:
            raise ValueError("Selected orders cannot exceed N")
        # Initialize DC state here: ordinary HSBM checks cannot enforce the theta-product constraint.
        self.n, self.p, self.theta = self._validated_parameter_arrays(n, ps, theta)
        self.p_by_order = MappingProxyType({d: self.p[i] for i, d in enumerate(orders)})
        self.theta_by_order = MappingProxyType({d: self.theta[i] for i, d in enumerate(orders)})
        self._validate_bernoulli_parameters()
        self.backend = _resolve_backend(backend)
        self.log_hye_to_node = None
        self.log_node_to_hye = None
        self.log_marginals = None
        self.external_field = None
        self.rng = np.random.default_rng()
        self.log_marginal_diff = []
        self.residual_history = []
        self.update_vertex_ratio = 1.0
        self.update_message_ratio = 1.0
        self.training_iter = 0
        self.n_diff, self.c_diff, self.theta_diff = [], [], []
        self.em_history = []
        self.em_time_per_iter = []
        self.em_total_time = 0.0
        self.em_training_iter = 0
        self.em_stop_reason = "not_started"
        self.em_converged = False
        self.converged = False
        self.mp_stop_reason = "not_started"
        self.mp_total_time = 0.0
        self._incidence = None
        self.damping = 0.0
        self._affinity_initialized = self._initial_p_orders == set(orders)
        self.selected_hye_sizes = list(orders)
        self.p_list = self.p_by_order
        self.layer_scale = None
        self.realized_layer_scale = None
        self.effective_beta_d = None
        self.c0 = None
        self.average_degree = None
        self.omega_per_message = None
        self.clipping_by_group = {}
        self._loop_initialized = False
        self._loop_dropout_FVS = 0.
        self.mp_training_iter = 0

    def _validated_parameter_arrays(self, n, p, theta):
        """Check DC parameters and return independent, read-only arrays."""
        # Arrange dictionary values by self.orders. Each theta layer needs overall mean 1;
        # equal within-community theta means are not required here.
        if isinstance(p, Mapping) and set(p) != set(self.orders):
            raise ValueError("p keys must equal the selected orders")
        L = len(self.orders)
        n = np.ones(self.K) / self.K if n is None else np.array(n, dtype=float, copy=True)
        if (n.shape != (self.K,) or np.any(~np.isfinite(n))
                or np.any(n <= 0) or not np.isclose(n.sum(), 1, rtol=1e-10, atol=1e-12)):
            raise ValueError("n must be a positive normalized vector of length K")
        if p is None:
            ps = np.zeros((L, self.K, self.K))
        elif isinstance(p, Mapping):
            ps = np.array([p[d] for d in self.orders], dtype=float)
        else:
            ps = np.array(p, dtype=float, copy=True)
            if ps.shape == (self.K, self.K):
                ps = np.repeat(ps[None, :, :], L, axis=0)
        if (ps.shape != (L, self.K, self.K) or np.any(~np.isfinite(ps)) or np.any(ps < 0)
                or not np.allclose(ps, ps.transpose(0, 2, 1), rtol=1e-12, atol=0)):
            raise ValueError("p must contain finite nonnegative symmetric K by K matrices")
        if theta is None:
            ts = np.ones((L, self.N))
        elif isinstance(theta, Mapping):
            if set(theta) != set(self.orders):
                raise ValueError("theta keys must equal active orders")
            ts = np.array([theta[d] for d in self.orders], dtype=float)
        else:
            ts = np.array(theta, dtype=float, copy=True)
            if ts.shape == (self.N,):
                ts = np.repeat(ts[None, :], L, axis=0)
        if (ts.shape != (L, self.N) or np.any(~np.isfinite(ts)) or np.any(ts < 0)
                or not np.allclose(ts.mean(axis=1), 1, rtol=1e-10, atol=1e-12)):
            raise ValueError("theta must be nonnegative, finite, and have mean one in each order")
        p = np.ascontiguousarray(ps)
        theta = np.ascontiguousarray(ts)
        # Block external array mutation to keep p_list views, probability checks, and the field synchronized.
        # Change parameters through set_parameters or the training entry point.
        for array in (n, p, theta):
            array.setflags(write=False)
        return n, p, theta


    @property
    def theta_constraint(self):
        return self._theta_constraint

    @theta_constraint.setter
    def theta_constraint(self, value):
        if not isinstance(value, str) or value not in ("soft", "hard"):
            raise ValueError("theta_constraint must be 'soft' or 'hard'")
        changed = getattr(self, "_theta_constraint", value) != value
        self._theta_constraint = value
        if changed and hasattr(self, "activity_balance_residual"):
            self.activity_constraint_beliefs = None
            self.activity_balance_residual[:] = np.nan
            self._activity_learned_layers[:] = False
            self.last_activity_update = None

    def _activity_membership(self, q):
        return q if self.theta_constraint == "soft" else np.eye(self.K)[q.argmax(1)]

    def _refresh_activity_balance(self):
        # New BP changes q; the final residual differs from the constraint residual at the update snapshot.
        self.activity_balance_residual[:] = np.nan
        if self.log_marginals is not None and self._activity_learned_layers.any():
            membership = self._activity_membership(np.exp(self.log_marginals))
            active = self._activity_learned_layers
            self.activity_balance_residual[active] = (self.theta[active] - 1) @ membership

    def _commit_parameters(self, n, p, theta):
        # Validate candidates before committing and rebuilding order views; refresh the field if messages exist.
        n, p, theta = self._validated_parameter_arrays(n, p, theta)
        self._validate_bernoulli_parameters(p=p, theta=theta)
        self.n, self.p, self.theta = n, p, theta
        self.p_by_order = MappingProxyType({d: self.p[i] for i, d in enumerate(self.orders)})
        self.theta_by_order = MappingProxyType({d: self.theta[i] for i, d in enumerate(self.orders)})
        self.p_list = self.p_by_order
        self._affinity_initialized = True
        self.realized_layer_scale = {d: float(self.n @ self.p_by_order[d] @ self.n) for d in self.orders}
        if self.log_marginals is not None:
            self.external_field = self.compute_external_field_list()
        self.converged = False

    def set_parameters(self, *, n=None, p=None, theta=None):
        """Atomically apply supplied parameters, preserving immutable order views.

        Changed parameters count as explicitly supplied; n/theta are fixed by default in
        subsequent EM calls. Rerun BP to obtain beliefs for the new values.
        """
        if p is None and not self._affinity_initialized:
            raise ValueError("Supply p or initialize it with em_inference first")
        self._commit_parameters(self.n if n is None else n,
                                self.p if p is None else p,
                                self.theta if theta is None else theta)
        self._provided_n |= n is not None
        self._provided_p |= p is not None
        self._provided_theta |= theta is not None
        if p is not None:
            self._initial_p_orders = set(self.orders)


    def _check_hypergraph_vs_model_params(self, hypergraph):
        if not self._affinity_initialized:
            raise ValueError("Initialize missing p_list layers through em_inference first")
        B = as_incidence(hypergraph)
        if B.shape[0] != self.N:
            raise ValueError("Hypergraph N differs from model N")
        for e, d in enumerate(np.diff(B.indptr)):
            if d not in self.orders:
                raise ValueError("Observed order has no parameters")
            if self.omega[int(d)] == 0:
                continue
            nodes = B.indices[B.indptr[e]:B.indptr[e + 1]]
            if np.any(self.theta_by_order[int(d)][nodes] == 0) or not np.any(self.p_by_order[int(d)]):
                raise ValueError("Active observed hyperedge has zero activity or all-zero affinity")
        return B

    def average_degree_from_G_for_hye_size(self, hypergraph, hye_size=2):
        # Use the original loop intensity definition c_emp=d(d-1)E_d/N.
        # This differs from the mean incidence count d*E_d/N for that order.
        B = as_incidence(hypergraph)
        return float(hye_size * (hye_size - 1) * np.count_nonzero(np.diff(B.indptr) == hye_size) / self.N)

    def initialize_layer_scales(self, hypergraph):
        """Original loop beta convention, including all selected candidate layers."""
        B = as_incidence(hypergraph)
        if B.shape[0] != self.N or np.any(~np.isin(np.diff(B.indptr), self.orders)):
            raise ValueError("Training graph differs from selected orders or N")
        if 2 not in self.orders:
            raise ValueError("Order 2 is required as the beta reference")
        c_emp = np.array([self.average_degree_from_G_for_hye_size(B, d) for d in self.orders])
        if np.any(c_emp <= 0):
            raise ValueError("Every selected order needs positive empirical c, as in the loop model")
        beta_emp = c_emp / c_emp[self.orders.index(2)]
        beta = beta_emp if self.beta_d is None else np.array([
            self.beta_d.get(d, beta_emp[i]) for i, d in enumerate(self.orders)])
        chi = np.array([compute_C_prime_d(d) for d in self.orders])
        # Adjust relative layer intensities by beta, preserving the baseline sum_d chi_d*c_d.
        # Actual DC exposure also depends on theta-weighted community sizes and differs from this baseline.
        raw = c_emp * beta / beta_emp
        targets = raw * np.dot(chi, c_emp) / np.dot(chi, raw)
        # With share_beta_d=True, pool target intensities within each group using chi weights.
        if self.share_beta_d:
            for group in self.shared_hye_sizes:
                idx = [self.orders.index(d) for d in group]
                targets[idx] = np.dot(chi[idx], targets[idx]) / chi[idx].sum()
        self.average_degree = c_emp
        self.c0 = float(targets[self.orders.index(2)])
        self.layer_scale = {d: float(targets[i] / self.N) for i, d in enumerate(self.orders)}
        self.effective_beta_d = {d: float(targets[i] / self.c0) for i, d in enumerate(self.orders)}
        return self.layer_scale.copy()

    def _group_indices(self, shared_idx):
        # shared_idx takes array indices: for orders=[2,4], [0,1] selects the entire group;
        # the actual orders [2,4] are not valid shared_idx values.
        idx = [_integer(i, "layer index", 0) for i in shared_idx]
        if len(set(idx)) != len(idx) or any(i >= len(self.orders) for i in idx):
            raise ValueError("shared_idx must contain distinct valid layer indices")
        if idx and set(self.orders[i] for i in idx) not in [set(g) for g in self.shared_hye_sizes]:
            raise ValueError("Update a complete shared group, not a subset")
        return idx

    def _scaled_shape(self, indices, shape, n=None, theta=None, *, clip=True):
        if self.layer_scale is None:
            raise RuntimeError("Initialize layer_scale in em_inference or initialize_layer_scales first")
        n = self.n if n is None else n
        theta = self.theta if theta is None else theta
        W = np.array(shape, dtype=float, copy=True)
        if (W.shape != (self.K, self.K) or not np.isfinite(W).all()
                or np.any(W < 0) or not np.allclose(W, W.T, rtol=1e-10, atol=1e-14)):
            raise ValueError("Shared shape must be finite, nonnegative and symmetric")
        mean = float(n @ W @ n)
        if mean <= 0:
            raise ValueError("Shared shape must have positive prior mean")
        # First enforce n.T@W@n=1, then set p_d=layer_scale[d]*W.
        W = (W + W.T) / (2 * mean)
        cap = np.inf
        for i in indices:
            d = self.orders[i]
            scale = self.layer_scale[d]
            if not np.isfinite(scale) or not 0 < scale < 1:
                raise ValueError("Target mean p must be in (0,1)")
            if not clip:
                continue
            # Also retain the original loop p-entry cap. The DC bound controls
            # the full product of the d largest activities, not only p itself.
            cap = min(cap, (1 - 1e-12) / scale)
            top = np.sort(theta[i])[-d:]
            if np.all(top > 0):
                log_cap = (np.log1p(-1e-12) + log_binomial_coefficient(self.N - 2, d - 2, allow_approx=False)
                           - np.log(scale) - np.log(top).sum())
                cap = min(cap, float(np.exp(min(log_cap, 700))))
        # A common group bound preserves layer ratios. Do not rescale after clipping;
        # actual n.T@p_d@n may fall below target; realized_layer_scale records the result.
        clipped = bool(np.any(W > cap))
        W = np.minimum(W, cap)
        return {i: self.layer_scale[self.orders[i]] * W for i in indices}, clipped

    def apply_beta_scaling(self, shared_idx, *, shape_matrix=None):
        indices = self._group_indices(shared_idx)
        if not indices:
            return
        shape = self.p[indices[0]] if shape_matrix is None else shape_matrix
        fixed = all(self.orders[i] in self.fix_hyper_layer for i in indices)
        candidates, clipped = self._scaled_shape(indices, shape, clip=not fixed)
        if fixed:
            if any(not np.allclose(self.p[i], value, rtol=1e-9, atol=1e-14)
                   for i, value in candidates.items()):
                raise ValueError("Fixed p is incompatible with layer scales/shared shape/DC probability bounds")
            return
        p = self.p.copy()
        for i, value in candidates.items():
            p[i] = value
        self._commit_parameters(self.n, p, self.theta)
        self.clipping_by_group[tuple(self.orders[i] for i in indices)] = clipped

    def compute_external_field_list(self, use_omega=True):
        # DC external field: h_i(a)=sum_d w_d*theta_i^(d)*chi_d*[p_d*S_d]_a.
        # S_d[a]=sum_j theta_j^(d)*q_j(a)，chi_d=2/[d(d-1)]。
        # Node-dependent activity requires an (N,K) field rather than a single K-vector.
        if self.log_marginals is None:
            raise ValueError("Initialize marginals first")
        q = np.exp(self.log_marginals)
        if q.shape != (self.N, self.K) or not np.isfinite(q).all() or not np.allclose(q.sum(1), 1):
            raise ValueError("Marginals must be normalized finite rows")
        result = np.zeros((self.N, self.K))
        for i, d in enumerate(self.orders):
            omega = self.omega[d] if use_omega else 1.
            if omega:
                t = self.theta[i]
                result += omega * compute_C_prime_d(d) * t[:, None] * (self.p[i] @ (t @ q))[None, :]
        if not np.isfinite(result).all():
            raise ValueError("External field overflow")
        return result

    def compute_external_field(self):
        return self.compute_external_field_list()

    def _factor_messages(self, B, mask=None):
        # For observed hyperedges, prod(theta)/kappa is identical for each root community a
        # and cancels during message normalization, so the original Rust pair sum still applies.
        # theta still affects inference through the external field, learning, and free energy.
        prepared = getattr(self, "_bp_factor_inputs", None)
        if (prepared is not None and B is prepared[0]
                and self.log_node_to_hye is prepared[1]):
            # Internal BP path only: structure was validated at entry and normalized_log
            # maintains messages. Compute all active positions to retain the full-residual convergence definition.
            _, messages, indptr, orders, active = prepared
            if mask is not None:
                mask = np.asarray(mask)
                if mask.shape != active.shape or mask.dtype != bool:
                    raise ValueError("mask must be a boolean incidence vector")
                active = active & mask  # Preserve the active-layer mask for the next iteration.
            positions, factor = updates.rust_hysbm.compute_psi_all_loop_p_data(
                indptr, np.ascontiguousarray(messages.data, dtype=np.float64),
                np.ascontiguousarray(self.p, dtype=np.float64), orders, active)
            positions = np.asarray(positions, dtype=np.int64)
            factor = np.asarray(factor, dtype=np.float64)
            if factor.shape != (self.K, positions.size):
                raise ValueError("Unexpected Rust output shape")
            return positions, factor
        # Independent external calls retain the original strict validation and wrapper.
        B, view = self._observed_inputs(B)
        active = np.repeat([self.omega[int(d)] > 0 for d in np.diff(B.indptr)], np.diff(B.indptr))
        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape != active.shape or mask.dtype != bool:
                raise ValueError("mask must be a boolean incidence vector")
            active &= mask
        return updates.compute_psi_dynamic_programming_loop_rust(view, self, active)

    def _init_message_passing(self, hypergraph, dirichlet_alpha=10.0):
        if not np.isfinite(dirichlet_alpha) or dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be positive and finite")
        B = self._check_hypergraph_vs_model_params(hypergraph)
        self._incidence = B
        # Each direction owns (K,M) values, sharing the same indices / indptr arrays.
        self.log_node_to_hye, self.log_hye_to_node = self.message_container_class.create_pair(B, self.K)
        M = B.nnz
        node_probs = self.rng.dirichlet([dirichlet_alpha] * self.K, size=M).T
        node_probs = np.maximum(node_probs, np.exp(CLIP_MIN))
        self.log_node_to_hye.data[:] = np.log(node_probs / node_probs.sum(axis=0, keepdims=True))
        marginals = np.maximum(self.rng.dirichlet([dirichlet_alpha] * self.K, size=self.N), np.exp(CLIP_MIN))
        self.log_marginals = np.log(marginals / marginals.sum(axis=1, keepdims=True))
        self.external_field = self.compute_external_field()
        self.log_hye_to_node.data[:] = normalized_log(
            self.log_marginals[B.indices].T - self.log_node_to_hye.data, axis=0)
        sizes = np.diff(self._incidence.indptr)
        self.omega_per_message = np.repeat([self.omega[int(d)] for d in sizes], sizes)
        self.log_hye_to_node.data[:, self.omega_per_message == 0] = -np.log(self.K)

    def parallel_message_passing(self, hypergraph, mp_iter=2000, mp_thresh=1e-5,
                                 dirichlet_alpha=None, dropout=.99, dropout_vertex=.6,
                                 dropout_vertex_FVS=None, dropout_vertex_rest=None,
                                 patience=50, seed=None, warm_start=True,
                                 dropout_FVS=0, fvs_node_set=None, *, damping=0.):
        # Hold n, p, theta fixed and iterate messages and node community probabilities.
        # dropout randomly retains old messages; damping mixes old probabilities into accepted updates.
        # Default damping=0 applies no additional probability mixture.
        from src.data.representation.incidence_hypergraph import IncidenceHypergraph
        self._loop_dropout_FVS = _rate(dropout_FVS, "dropout_FVS")
        B = self._check_hypergraph_vs_model_params(hypergraph)
        self._fvs_graph = (hypergraph if hasattr(hypergraph, "get_binary_incidence_matrix")
                           else IncidenceHypergraph(B) if B.shape[1] else None)
        fvs_rate = dropout_vertex if dropout_vertex_FVS is None else dropout_vertex_FVS
        rest_rate = dropout_vertex if dropout_vertex_rest is None else dropout_vertex_rest
        if fvs_node_set is None and (dropout_FVS or fvs_rate != rest_rate):
            fvs_seed = seed if seed is not None else int(self.rng.integers(0, 2**32))
            # FVS is the feedback vertex set used for scheduling. Reverse reinsertion avoids
            # the original csr_array.getnnz compatibility issue and may differ from degree reinsertion.
            # The original helper's degree reinsertion calls csr_array.getnnz,
            # which current SciPy arrays do not provide. Reuse its supported
            # reverse reinsertion without patching SciPy or the source module.
            fvs_node_set = ([] if self._fvs_graph is None else
                            FcoreHD_FVS(self._fvs_graph, random_tie=True, seed=fvs_seed,
                                        reinsertion_order="reverse").fvs_nodes)
        self.fvs_node_set = fvs_node_set
        sizes = np.diff(B.indptr)
        self.omega_per_message = np.repeat([self.omega[int(d)] for d in sizes], sizes)
        mp_iter = _integer(mp_iter, "mp_iter", 0)
        patience = _integer(patience, "patience", 1)
        if not np.isfinite(mp_thresh) or mp_thresh < 0:
            raise ValueError("mp_thresh must be nonnegative and finite")
        _rate(dropout, "dropout")
        _rate(dropout_vertex, "dropout_vertex")
        self.damping = _rate(damping, "damping")
        if self.damping == 1:
            raise ValueError("damping must be less than one")
        if dropout_vertex_FVS is not None:
            _rate(dropout_vertex_FVS, "dropout_vertex_FVS")
        if dropout_vertex_rest is not None:
            _rate(dropout_vertex_rest, "dropout_vertex_rest")
        B = self._check_hypergraph_vs_model_params(hypergraph)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        initialized = all(x is not None for x in (self.log_node_to_hye, self.log_hye_to_node,
                                                 self.log_marginals, self.external_field, self._incidence))
        # warm_start reuses messages only for the same graph and CSC ordering.
        # Changed incidences require reinitialization; matching node counts alone is insufficient.
        if warm_start and initialized:
            old = self._incidence
            if (old.shape != B.shape or not np.array_equal(old.indices, B.indices)
                    or not np.array_equal(old.indptr, B.indptr)):
                raise ValueError("Warm start graph changed; use warm_start=False")
        else:
            alpha = 10 * self.K if dirichlet_alpha is None else dirichlet_alpha
            self._init_message_passing(B, alpha)
        self.log_marginal_diff.append([])
        self.residual_history = []
        self.converged = False
        self.mp_stop_reason = "max_iter"
        self.training_iter = 0
        stable_steps = 0
        started = time.perf_counter()
        self._bp_factor_inputs = None
        try:
            if mp_iter:
                # Cache fixed structure only during this BP run; still perform two row_sum operations per iteration.
                # Initialization may copy B, so use the self._incidence actually referenced by messages.
                observed, _ = self._observed_inputs(self._incidence)
                self._bp_factor_inputs = (
                    observed, self.log_node_to_hye,
                    np.ascontiguousarray(observed.indptr, dtype=np.int64),
                    np.ascontiguousarray(self.selected_hye_sizes, dtype=np.int64),
                    np.ascontiguousarray(self.omega_per_message > 0, dtype=bool),
                )
            for step in range(mp_iter):
                old_q = self.log_marginals.copy()
                residual = self._parallel_message_passing_step(B, fvs_node_set, dropout, dropout_vertex,
                                                              dropout_vertex_FVS, dropout_vertex_rest, dropout_FVS)
                denom = self.N * self.update_vertex_ratio if self.update_vertex_ratio > 0 else self.N
                self.log_marginal_diff[-1].append(float(np.abs(self.log_marginals - old_q).sum() / denom))
                self.residual_history.append(residual)
                self.training_iter = step + 1
                # As in HSBM, test convergence using changes in the log marginals actually written back,
                # averaged over updated nodes without an additional division by K.
                # residual_history stores the maximum candidate residual for diagnostics, not stopping.
                # Set converged=True only after meeting the threshold for patience consecutive iterations.
                stable_steps = stable_steps + 1 if self.log_marginal_diff[-1][-1] <= mp_thresh else 0
                if stable_steps >= patience:
                    self.converged = True
                    self.mp_stop_reason = "converged"
                    break
        finally:
            # Public calls reenter strict validation; changed parameters, graphs, or weights invalidate old inputs.
            self._bp_factor_inputs = None
        self.mp_total_time = time.perf_counter() - started
        self.mp_training_iter = self.training_iter
        self._refresh_activity_balance()


    def _parallel_message_passing_step(self, hypergraph, fvs_node_set=None, dropout=.99,
                                      dropout_vertex=.6, dropout_vertex_FVS=None,
                                      dropout_vertex_rest=None, dropout_FVS=0):
        B = self._incidence
        M = B.nnz
        weights = self.omega_per_message
        old_h = self.external_field
        # 1. Node -> hyperedge: sum incoming messages, then subtract the target's own w_d*log message
        # to form a cavity message excluding its feedback. Products become sums in log space.
        total = self.log_hye_to_node.row_sum(position_weights=weights).T + np.log(self.n) - old_h
        node = normalized_log(total[B.indices].T - weights[None, :] * self.log_hye_to_node.data, axis=0)
        residual = float(np.max(np.abs(node - self.log_node_to_hye.data), initial=0))
        # Use the FVS incidence mask instead of ordinary message dropout when scheduling is enabled.
        if self._loop_dropout_FVS and M:
            _, _, node_mask = FVS_random_mask_proration(
                self._fvs_graph, self.log_node_to_hye.as_csc(0),
                dropout_FVS=self._loop_dropout_FVS,
                seed=int(self.rng.integers(0, 2**32)), fvs_node_set=fvs_node_set)
        else:
            node_mask = np.ones(M, bool) if dropout == 0 else self.rng.random(M) >= dropout
        damped = self._damped_log(self.log_node_to_hye.data, node)
        self.log_node_to_hye.data[:, node_mask] = damped[:, node_mask]
        # 2. Hyperedge -> node: Rust computes unnormalized factor messages; normalize over communities.
        # positions are global CSC incidence offsets; update only masked factors with nonzero weight.
        positions, factor = self._factor_messages(B)
        factor = normalized_log(factor, axis=0)
        residual = max(residual, float(np.max(np.abs(factor - self.log_hye_to_node.data[:, positions]), initial=0)))
        factor = self._damped_log(self.log_hye_to_node.data[:, positions], factor)
        factor_mask = (node_mask if self._loop_dropout_FVS else
                       np.ones(M, bool) if dropout == 0 else self.rng.random(M) >= dropout)
        selected = factor_mask[positions]
        self.log_hye_to_node.data[:, positions[selected]] = factor[:, selected]
        self.update_message_ratio = float((node_mask.sum() + selected.sum()) / (2 * M)) if M else 1.
        # 3. Node marginals include all incident hyperedges without subtracting a target hyperedge.
        proposed = normalized_log(self.log_hye_to_node.row_sum(position_weights=weights).T
                                  + np.log(self.n) - old_h, axis=1)
        residual = max(residual, float(np.max(np.abs(proposed - self.log_marginals), initial=0)))
        proposed = self._damped_log(self.log_marginals, proposed)
        vertex_mask = self._make_vertex_update_mask(dropout_vertex, dropout_vertex_FVS,
                                                     dropout_vertex_rest, fvs_node_set)
        self.log_marginals[vertex_mask] = proposed[vertex_mask]
        # 4. Recompute the next external field from updated marginals and include its change in the residual.
        self.external_field = self.compute_external_field_list()
        return max(residual, float(np.max(np.abs(self.external_field - old_h), initial=0)))

    def _learning_statistics(self, hypergraph):
        # For omega=0 layers, temporary nonzero affinities avoid division by zero before zero-weighting statistics.
        # These temporary values do not modify the layer affinities stored in the model.
        B = self._check_hypergraph_vs_model_params(hypergraph)
        self._observed_inputs(B)
        # Harmless dummy affinities prevent zero denominators on ignored layers.
        p = self.p.copy()
        for i, d in enumerate(self.orders):
            if self.omega[d] == 0:
                p[i] = 1.
        proxy = SimpleNamespace(N=self.N, K=self.K, orders=self.orders, p=p,
                                log_marginals=self.log_marginals, log_node_to_hye=self.log_node_to_hye)
        stats = posterior_statistics(proxy, B)
        omega = np.array([self.omega[d] for d in self.orders])
        # Returned responsibilities, degrees, and edge counts already include w; do not weight them again.
        stats["responsibilities"] *= omega[:, None, None]
        stats["degrees"] *= omega[:, None]
        stats["edge_count"] = stats["edge_count"] * omega
        return stats

    def updated_affinity_matrix_list(self, hypergraph, shared_idx, *, stats=None):
        # Update a whole shared group: learn common W, then recover p_d=s_d*W for each layer.
        # Reuse the E-step snapshot; each group uses responsibilities from its own unmodified layers.
        indices = self._group_indices(shared_idx)
        if not indices or all(self.orders[i] in self.fix_hyper_layer for i in indices):
            return
        if self.layer_scale is None:
            raise RuntimeError("Initialize layer_scale first")
        if stats is None:
            stats = self._learning_statistics(hypergraph)
        R = stats["responsibilities"][indices].sum(axis=0)
        if not R.any():
            # The likelihood is flat, but n may have changed its scale constraint.
            self.apply_beta_scaling(indices)
            return
        # R already includes observation weights w; add exposure weights omega and scales s_d explicitly to A.
        # S_d is the activity-weighted community size; ordinary HSBM N*n is not a substitute.
        A = np.zeros((self.K, self.K))
        for i in indices:
            d = self.orders[i]
            S = self.theta[i] @ stats["q"]
            A += self.omega[d] * self.layer_scale[d] * compute_C_prime_d(d) * np.outer(S, S) / 2
        shape = shared_shape_proposal(R, A, self.n, initial=self.p[indices[0]])
        self.apply_beta_scaling(indices, shape_matrix=shape)

    def updated_affinity_matrix(self, hypergraph, project=False, *, activity_regularization=0.):
        # Loop models update by group; reject the ordinary single-matrix update entry point.
        raise ValueError("Use updated_affinity_matrix_list(hypergraph, shared_idx) for loop shared shapes")

    def _weighted_surrogate(self, stats, n, p, theta, alpha=0.):
        # Parameter auxiliary objective with posterior statistics fixed: prior + pairwise affinity
        # + node activity - sparse exposure from absent hyperedges.
        # Q_before / Q_after in history refer to this objective, not the recovery overlap Q.
        value = special.xlogy(stats["q"], n).sum() + special.xlogy(stats["responsibilities"], p).sum()
        omega = np.array([self.omega[d] for d in self.orders])
        value += special.xlogy(stats["degrees"] + alpha * (omega[:, None] > 0), theta).sum()
        value -= np.dot(omega, sparse_exposure(p, theta, stats["q"], self.orders))
        return float(value)

    def updated_activities(self, hypergraph, *, regularization=0., min_activity=1e-10,
                           stats=None):
        """Community-normalized count fit with an optional current E-step snapshot.

        A supplied stats snapshot must match the current graph and BP beliefs;
        hard mode remains a plug-in estimator.
        """
        if not np.isfinite(regularization) or regularization < 0:
            raise ValueError("regularization must be finite and nonnegative")
        if not np.isfinite(min_activity) or not 0 < min_activity <= 1:
            raise ValueError("min_activity must lie in (0,1]")
        if regularization > 0 and any(
                d in self.fix_theta_sizes and self.omega[d] > 0 and np.any(self.theta[i] == 0)
                for i, d in enumerate(self.orders)):
            raise ValueError("Positive regularization requires positive fixed theta")
        if stats is None:
            stats = self._learning_statistics(hypergraph)
        q = stats["q"].copy()
        membership = self._activity_membership(q)
        active = np.array([d not in self.fix_theta_sizes and self.omega[d] > 0 for d in self.orders])
        proposed, baseline = self.theta.copy(), self.theta.copy()
        solver = {}
        for i in np.flatnonzero(active):
            # Degrees already include layer weights w; alpha optionally regularizes log(theta) separately.
            counts = stats["degrees"][i] + regularization
            proposed[i] = community_activity_proposal(counts, q, mode=self.theta_constraint, lower=min_activity)
            baseline[i] = 1.
            solver[int(self.orders[i])] = dict(
                method="community count normalization" if self.theta_constraint == "hard" else
                       "rank-reduced constrained log optimization",
                objective_at_ones=0., objective_at_proposal=float(special.xlogy(counts, proposed[i]).sum()))
        # Old theta may violate the new q constraints; backtrack from ones, without mixing infeasible old values.
        try:
            self._validate_bernoulli_parameters(theta=baseline)
        except ValueError as error:
            raise ValueError("Community activity ones baseline violates probability constraints") from error
        fraction = 0.
        trial = baseline
        for attempt in range(41):
            fraction = 2. ** -attempt
            candidate = baseline + fraction * (proposed - baseline)
            try:
                self._validate_bernoulli_parameters(theta=candidate)
            except ValueError:
                continue
            trial = candidate
            break
        else:
            fraction = 0.
        residual = np.full((len(self.orders), self.K), np.nan)
        residual[active] = (trial[active] - 1) @ membership
        if np.max(np.abs(residual[active]), initial=0) > 1e-9:
            raise RuntimeError("Accepted activities violate the community constraints")
        self.activity_constraint_beliefs = q
        self._activity_learned_layers = active
        self.last_activity_update = dict(
            mode=self.theta_constraint, constraint_residual=residual, accepted_fraction=fraction,
            objective_before=self._weighted_surrogate(stats, self.n, self.p, self.theta, regularization),
            objective_after=self._weighted_surrogate(stats, self.n, self.p, trial, regularization),
            solver=solver,
            interpretation=("fixed-belief community-constrained count fit" if self.theta_constraint == "soft" else
                            "hard-group degree plug-in; not an exact soft-belief M-step"))
        # Return candidates without committing theta; record current-parameter residuals against current beliefs.
        self._refresh_activity_balance()
        return trial.copy()

    def _initialize_loop_learning(self, B, learn_n, learn_p, learn_theta, warm_p_prior):
        if not learn_p and not self._affinity_initialized:
            raise ValueError("learn_p=False requires supplied or initialized affinities for every layer")
        fixed_p_and_n = not learn_n and (
            not learn_p or len(self.fix_hyper_layer) == len(self.orders))
        if fixed_p_and_n:
            # Do not recalibrate known fixed p and n from observed edge counts; no second-order reference is needed.
            # theta still undergoes degree initialization, probability checks, and subsequent learning.
            self.layer_scale = self.effective_beta_d = None
            self.c0 = self.average_degree = None
        else:
            self.initialize_layer_scales(B)
        n = self.n.copy()
        theta = self.theta.copy()
        p = self.p.copy()
        if not self._loop_initialized:
            if learn_n and not self._provided_n:
                n = self.rng.dirichlet(np.full(self.K, 100.))
            if learn_theta and not self._provided_theta:
                # If theta is absent and learnable, initialize each order from its node degrees.
                initial = estimate_theta_from_degrees(B, self.orders, pseudocount=.5)
                for i, d in enumerate(self.orders):
                    if d not in self.fix_theta_sizes and self.omega[d] > 0:
                        theta[i] = initial[d]
                # Degree-based activities are only an initializer. Keep supplied
                # fixed affinities feasible by shrinking this proposal if needed.
                fixed_p = np.zeros_like(p)
                for i, d in enumerate(self.orders):
                    if not learn_p or d in self.fix_hyper_layer:
                        fixed_p[i] = p[i]
                for attempt in range(41):
                    trial = self.theta + 2.**-attempt * (theta - self.theta)
                    try:
                        self._validate_bernoulli_parameters(p=fixed_p, theta=trial)
                    except ValueError:
                        continue
                    theta = trial
                    break
                else:
                    theta = self.theta.copy()
            for i, d in enumerate(self.orders):
                if d not in self._initial_p_orders:
                    if d in self.fix_hyper_layer:
                        raise ValueError("Missing fixed layer")
                    shape = np.ones((self.K, self.K)) / (10 * max(self.K - 1, 1))
                    noise = self.rng.uniform(0, .02, size=shape.shape)
                    shape += (noise + noise.T) / 2
                    np.fill_diagonal(shape, 1 + self.rng.random(self.K) / 50)
                    p[i] = shape / self.N
                elif learn_p and warm_p_prior and d not in self.fix_hyper_layer:
                    if self.K > 1 and p[i, 0, 0] > 0:
                        eps = p[i, 0, 1] / p[i, 0, 0]
                        p[i] = eps * np.ones((self.K, self.K)) + (1 - eps) * np.eye(self.K)
        clipping = {}
        for group in self.shared_hye_sizes:
            if fixed_p_and_n:
                clipping[tuple(group)] = False
                continue
            idx = [self.orders.index(d) for d in group]
            fixed = not learn_p or all(d in self.fix_hyper_layer for d in group)
            shape = p[idx[0]]
            if not fixed and not shape.any() and all(self.omega[d] == 0 for d in group):
                shape = np.ones((self.K, self.K))
            values, clipped = self._scaled_shape(idx, shape, n, theta, clip=not fixed)
            if fixed:
                if any(not np.allclose(p[i], v, rtol=1e-9, atol=1e-14) for i, v in values.items()):
                    raise ValueError("Fixed p is incompatible with layer scales or activity bounds")
            else:
                for i, v in values.items():
                    p[i] = v
            clipping[tuple(group)] = clipped
        self._commit_parameters(n, p, theta)
        self.clipping_by_group = clipping
        self._loop_initialized = True

    def em_inference(self, hypergraph, em_iter=20, em_thresh=1e-5,
                     mp_iter=2000, mp_thresh=1e-5, mp_patience=50, seed=None,
                     dirichlet_alpha=None, dropout=.99, dropout_vertex=.6,
                     dropout_vertex_FVS=None, dropout_vertex_rest=None,
                     dropout_FVS=0, warm_p_prior=False, *, learn_n=None,
                     learn_p=True, learn_theta=None, prior_update="hard",
                     activity_regularization=0., min_activity=1e-10,
                     damping=0., require_bp_convergence=False, fvs_node_set=None):
        """Loop-scale constrained approximate alternating learning.

        Shared normalization, hard n updates, clipping, and nonunit message
        weights prevent an exact likelihood-monotonicity claim. history records
        actual frozen auxiliary changes, including decreases, without hiding them.
        """
        # Outer loop: fixed-parameter BP -> posterior statistics -> enabled updates of n, theta, p.
        # This approximate alternating scheme with layer scales and sharing need not increase true marginal likelihood.
        em_iter = _integer(em_iter, "em_iter", 0)
        if not np.isfinite(em_thresh) or em_thresh < 0:
            raise ValueError("em_thresh must be finite and nonnegative")
        if not np.isfinite(activity_regularization) or activity_regularization < 0:
            raise ValueError("activity_regularization must be finite and nonnegative")
        if not np.isfinite(min_activity) or not 0 < min_activity < 1:
            raise ValueError("min_activity must lie in (0,1)")
        if prior_update not in {"hard", "soft"}:
            raise ValueError("prior_update must be hard or soft")
        for flag in (learn_n, learn_p, learn_theta, require_bp_convergence):
            if flag is not None and not isinstance(flag, (bool, np.bool_)):
                raise ValueError("Learning flags must be boolean")
        # None chooses defaults based on whether initial parameters were supplied; explicit True/False takes precedence.
        # learn_p and fix_hye_size freeze p; supplying p_list alone does not.
        learn_n = not self._provided_n if learn_n is None else bool(learn_n)
        learn_theta = not self._provided_theta if learn_theta is None else bool(learn_theta)
        if self.fix_hyper_layer and learn_n:
            raise ValueError("Fixed layers require fixed n")
        if not learn_p and learn_n:
            raise ValueError("Freeze n as well when freezing all loop-scale affinities")
        if activity_regularization > 0:
            for i, d in enumerate(self.orders):
                if self.omega[d] > 0 and (not learn_theta or d in self.fix_theta_sizes) and np.any(self.theta[i] == 0):
                    raise ValueError("Positive regularization requires positive fixed theta")
        B = as_incidence(hypergraph)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._initialize_loop_learning(B, learn_n, learn_p, learn_theta, warm_p_prior)
        self._check_hypergraph_vs_model_params(B)
        self.em_history = []
        self.n_diff, self.c_diff, self.theta_diff = [], [], []
        self.em_time_per_iter = []
        self.em_training_iter = 0
        self.em_stop_reason = "max_iter"
        self.em_converged = False
        start = time.perf_counter()
        options = dict(mp_iter=mp_iter, mp_thresh=mp_thresh, patience=mp_patience,
                       dirichlet_alpha=dirichlet_alpha, dropout=dropout, dropout_vertex=dropout_vertex,
                       dropout_vertex_FVS=dropout_vertex_FVS, dropout_vertex_rest=dropout_vertex_rest,
                       dropout_FVS=dropout_FVS, fvs_node_set=fvs_node_set, damping=damping, warm_start=True)
        # With fixed n and p, the activity update already recorded the same two absolute auxiliary objectives.
        reuse_activity_objectives = learn_theta and not learn_n and (
            not learn_p or len(self.fix_hyper_layer) == len(self.orders))
        final_needed = True
        for iteration in range(em_iter):
            step_start = time.perf_counter()
            # E step: run full BP at current parameters to update messages and node marginals.
            self.parallel_message_passing(B, **options)
            options["fvs_node_set"] = self.fvs_node_set
            final_needed = False
            # If require_bp_convergence=True, stop learning when BP fails to converge. Check these states for model selection;
            # finite free energy alone does not establish training convergence.
            if not self.converged and require_bp_convergence:
                self.em_stop_reason = "bp_not_converged"
                break
            if not learn_n and (not learn_p or len(self.fix_hyper_layer) == len(self.orders)) and not learn_theta:
                self.em_stop_reason = "fixed_parameters"
                break
            stats = self._learning_statistics(B)
            old = tuple(v.copy() for v in (self.n, self.p, self.theta))
            before = (None if reuse_activity_objectives else
                      self._weighted_surrogate(stats, *old, activity_regularization))
            bp_converged = self.converged
            # Update theta first; shared-shape learning then uses its actual weighted community masses.
            if learn_n:
                n = self.updated_community_prior() if prior_update == "hard" else self.updated_community_prior_soft()
                self._commit_parameters(n / n.sum(), self.p, self.theta)
            if learn_theta:
                theta = self.updated_activities(B, regularization=activity_regularization,
                                                min_activity=min_activity, stats=stats)
                self._commit_parameters(self.n, self.p, theta)
            if learn_p:
                for group in self.shared_hye_sizes:
                    if group[0] not in self.fix_hyper_layer:
                        # Shared groups do not overlap; committing another group's p preserves this group's responsibility snapshot.
                        self.updated_affinity_matrix_list(
                            B, [self.orders.index(d) for d in group], stats=stats)
            if reuse_activity_objectives:
                before = self.last_activity_update["objective_before"]
                after = self.last_activity_update["objective_after"]
            else:
                after = self._weighted_surrogate(stats, self.n, self.p, self.theta, activity_regularization)
            differences = [float(np.abs(a - b).sum() / max(np.abs(a).sum(), 1e-15))
                           for a, b in zip(old, (self.n, self.p, self.theta))]
            self.n_diff.append(differences[0]); self.c_diff.append(differences[1]); self.theta_diff.append(differences[2])
            self.em_training_iter = iteration + 1
            # Record actual objective changes and clipping; hard-prior updates or clipping may lower the objective.
            self.em_history.append(dict(iteration=iteration + 1, Q_before=before, Q_after=after,
                                        param_diff=sum(differences), bp_converged=bool(bp_converged),
                                        clipped=any(self.clipping_by_group.values())))
            self.em_time_per_iter.append(time.perf_counter() - step_start)
            final_needed = True
            if sum(differences) <= em_thresh:
                self.em_stop_reason = "em_thresh"
                break
        # Run BP after the last parameter change so saved messages match the final parameters.
        if final_needed:
            self.parallel_message_passing(B, **options)
        if not self.converged:
            self.em_stop_reason = "bp_not_converged"
        # converged describes the last BP run; em_converged describes outer training.
        # Reaching em_iter only stops iteration; it does not imply convergence.
        self.em_converged = bool(self.converged and self.em_stop_reason in {"em_thresh", "fixed_parameters"})
        self.training_iter = self.em_training_iter
        self.em_total_time = time.perf_counter() - start

    def single_hye_pi_list(self, assignments, hyperedge_size):
        # Compute only sum_pair p for the given community configuration; node IDs are absent, so theta is excluded.
        a = np.asarray(list(assignments))
        if len(a) != hyperedge_size or not np.issubdtype(a.dtype, np.integer) or np.any(a < 0) or np.any(a >= self.K):
            raise ValueError("Expected d community indices")
        p = self.p_by_order[hyperedge_size]
        return float(sum(p[a[i], a[j]] for i in range(len(a)) for j in range(i + 1, len(a))))

    def hye_pi_message_update(self, hypergraph, d, return_interactions=False):
        # For loop compatibility, d is a layer index here; self.orders[d] is the actual hyperedge order.
        B, _ = self._observed_inputs(hypergraph)
        d = _integer(d, "layer index", 0)
        if d >= len(self.orders):
            raise ValueError("Invalid layer index")
        edges = np.flatnonzero(np.diff(B.indptr) == self.orders[d])
        counts = np.zeros((len(edges), self.K, self.K))
        for row, e in enumerate(edges):
            prefix = np.zeros(self.K)
            for q in np.exp(self.log_node_to_hye.data[:, B.indptr[e]:B.indptr[e + 1]]).T:
                pair = np.outer(prefix, q)
                counts[row] += (pair + pair.T) / 2
                prefix += q
        pi = np.einsum("eab,ab->e", counts, self.p[d])
        return (pi, counts) if return_interactions else pi

    def hye_pi(self, hypergraph, d, return_interactions=False):
        # Use the parent layer-index convention; return hard-partition pair statistics, not full DC probabilities.
        B = as_incidence(hypergraph)
        d = _integer(d, "layer index", 0)
        if d >= len(self.orders):
            raise ValueError("Invalid layer index")
        assignments = self.community_assignments()
        counts = np.zeros((B.shape[1], self.K, self.K))
        for e in range(B.shape[1]):
            hist = np.bincount(assignments[B.indices[B.indptr[e]:B.indptr[e + 1]]], minlength=self.K)
            counts[e] = (np.outer(hist, hist) - np.diag(hist)) / 2
        pi = np.einsum("eab,ab->e", counts, self.p[d])
        return (pi, counts) if return_interactions else pi

    def c_diff_relative(self, c_old, c_new):
        return float(np.abs(c_new - c_old).sum() / max(np.abs(c_old).sum(), 1e-15))

    def free_energy(self, hypergraph, *, density=False, return_diagnostics=False):
        """DC sparse Bethe entropy functional; nonunit omega gives a heuristic score."""
        # Compute the sparse Bethe functional: exposure, factors, node-entropy correction, and community prior.
        # density=True returns F/N; diagnostics also report marginal consistency and BP status.
        B = self._check_hypergraph_vs_model_params(hypergraph)
        self.compute_external_field_list()
        q = np.exp(self.log_marginals)
        positions, raw_factor = self._factor_messages(B)
        sizes = np.diff(B.indptr)
        active_edges = np.array([self.omega[int(d)] > 0 for d in sizes])
        # Ignored factors may have log partition -inf; never multiply them by 0.
        log_z = self.compute_psi_tilde(B)
        if np.any(~np.isfinite(log_z[active_edges])):
            raise ValueError("Active observed factor has zero probability")
        incidence_edge = np.repeat(np.arange(B.shape[1]), sizes)
        log_cavity = self.log_node_to_hye.data[:, positions]
        # b_ei(a) is the single-node marginal from a hyperedge factor. With all active omega=1,
        # it matches q_i(a) at a BP fixed point; with nonunit weights, the difference is diagnostic only.
        factor_marginal = np.exp(log_cavity + raw_factor - log_z[incidence_edge[positions]][None, :])
        omega = np.array([self.omega[int(sizes[e])] for e in incidence_edge[positions]])
        terms = np.zeros_like(factor_marginal)
        np.multiply(factor_marginal, log_cavity, out=terms, where=factor_marginal > 0)
        factor_sum = float(np.sum(terms * omega[None, :]))
        for e in np.flatnonzero(active_edges):
            d = int(sizes[e]); nodes = B.indices[B.indptr[e]:B.indptr[e + 1]]
            # Retain prod(theta)/kappa in free energy even though message normalization cancels it.
            log_constant = (np.log(self.theta_by_order[d][nodes]).sum()
                            - log_binomial_coefficient(self.N - 2, d - 2, allow_approx=False) - np.log(comb(d, 2)))
            factor_sum -= self.omega[d] * (log_z[e] + log_constant)
        weighted_degree = np.bincount(B.indices[positions], weights=omega, minlength=self.N)
        weights = np.array([self.omega[d] for d in self.orders])
        exposure = sparse_exposure(self.p, self.theta, q, self.orders)
        total = float(np.dot(weights, exposure) + factor_sum
                      - np.dot(weighted_degree - 1, special.xlogy(q, q).sum(1))
                      - special.xlogy(q, self.n).sum())
        consistency = float(np.max(np.abs(factor_marginal.T - q[B.indices[positions]]), initial=0))
        # For omega in {0,1}, this is the sparse Bethe functional on active layers. Other weights use heuristic
        # message weighting; the score is neither strict power-likelihood Bethe free energy nor exact model evidence.
        exact_weights = all(omega in (0, 1) for omega in self.omega.values())
        if return_diagnostics:
            return dict(F=total / self.N if density else total, total_F=total,
                        local_consistency=consistency, bp_converged=bool(self.converged),
                        exposure_by_order=weights * exposure,
                        interpretation=("sparse Bethe functional on active layers" if exact_weights else
                                        "heuristic weighted score; not an exact powered-likelihood Bethe free energy"))
        return total / self.N if density else total

    def community_assignments(self):
        if self.log_marginals is None:
            raise ValueError("Run message passing first")
        # super() uses the original loop partition method with this DC object's current marginals.
        return super().community_assignments()


    def updated_community_prior_soft(self):
        if self.log_marginals is None:
            raise ValueError("Run message passing first")
        return super().updated_community_prior_soft()


    def _validate_bernoulli_parameters(self, p=None, theta=None):
        """Certify all candidate Bernoulli probabilities, without clipping them.

        Usually a cheap upper bound suffices. If it exceeds one, maximize the
        pair sum over community-count compositions; a loose bound alone is not
        a reason to reject a valid model. This does not certify sparsity.
        """
        # Check the full mu_e = prod(theta_i)*sum_pair(p)/kappa_d <= 1.
        # First bound it using the largest d activities and affinity entry. If that bound is loose, enumerate
        # community-count configurations for the maximum pair sum. This ensures valid probabilities, not sparsity.
        ps = self.p if p is None else np.asarray(p)
        ts = self.theta if theta is None else np.asarray(theta)
        for ell, d in enumerate(self.orders):
            layer = ps[ell]
            pmax = layer.max()
            top = np.sort(ts[ell])[-d:]
            if pmax == 0 or np.any(top == 0):
                continue
            log_choose = log_binomial_coefficient(self.N - 2, d - 2, allow_approx=False)
            log_bound = np.log(top).sum() + np.log(pmax) - log_choose
            if log_bound <= 1e-12:
                continue
            count = comb(d + self.K - 1, self.K - 1)
            if count > 200000:
                raise ValueError("Cannot certify Bernoulli probability <=1: sufficient bound fails and too many community compositions")
            # Stars and bars enumerate all K community counts summing to d.
            scaled = layer / pmax
            best = 0.0
            for bars in combinations(range(d + self.K - 1), self.K - 1):
                cuts = (-1,) + bars + (d + self.K - 1,)
                counts = np.diff(cuts) - 1
                pair = 0.5 * (counts @ scaled @ counts - np.dot(counts, scaled.diagonal()))
                best = max(best, pair / comb(d, 2))
            if best > 0 and log_bound + np.log(best) > 1e-12:
                raise ValueError(f"Order {d}: some Bernoulli hyperedge probability exceeds one")

    def _observed_inputs(self, hypergraph):
        """Adapt CSC inputs to the original wrappers; no new factor algorithm.

        A lightweight view also supports an empty incidence matrix, which the
        original IncidenceHypergraph constructor does not accept.
        """
        # Check exact alignment of message and incidence CSC positions before calling the original Rust wrapper.
        B = hypergraph if hypergraph is self._incidence else as_incidence(hypergraph)
        msg = self.log_node_to_hye
        if msg is None:
            raise ValueError("Initialize messages first")
        if (B.shape != (self.N, msg.E) or msg.data.shape != (self.K, B.nnz)
                or not np.array_equal(B.indices, msg.indices)
                or not np.array_equal(B.indptr, msg.indptr)):
            raise ValueError("Messages and incidence have different CSC structures")
        if np.any(~np.isin(np.diff(B.indptr), self.selected_hye_sizes)):
            raise ValueError("Observed hyperedge order has no parameters")
        logs = msg.data
        if np.any(np.isnan(logs)) or np.any(np.isposinf(logs)):
            raise ValueError("Invalid incoming log messages")
        z = special.logsumexp(logs, axis=0)
        if np.any(~np.isfinite(z)) or not np.allclose(z, 0, rtol=0, atol=1e-8):
            raise ValueError("Incoming message columns must be normalized log probabilities")
        return B, SimpleNamespace(get_binary_incidence_matrix=lambda: B)

    def compute_psi_tilde(self, hypergraph):
        """Original HSBM log E_q[sum_pair p], without activity/kappa constants.

        This local pair statistic is not a DC evidence or free-energy value.
        """
        # Return log E_cavity[sum_pair p] without theta/kappa; free energy restores that factor separately.
        _, graph_view = self._observed_inputs(hypergraph)
        return updates.compute_psi_tilde_dynamic_programming_loop_rust(graph_view, self)

    def _make_vertex_update_mask(self, dropout_vertex=0, dropout_vertex_FVS=None,
                                 dropout_vertex_rest=None, fvs_node_set=None):
        fvs_rate = dropout_vertex if dropout_vertex_FVS is None else dropout_vertex_FVS
        rest_rate = dropout_vertex if dropout_vertex_rest is None else dropout_vertex_rest
        rates = np.full(self.N, _rate(rest_rate, "dropout_vertex_rest"))
        _rate(fvs_rate, "dropout_vertex_FVS")
        if fvs_node_set is None and fvs_rate != rest_rate:
            raise ValueError("Supply fvs_node_set when using different vertex rates")
        if fvs_node_set is not None:
            nodes = np.asarray(list(fvs_node_set))
            if nodes.size:
                if (not np.issubdtype(nodes.dtype, np.integer) or np.any(nodes < 0)
                        or np.any(nodes >= self.N)):
                    raise ValueError("Invalid FVS node indices")
                rates[nodes] = fvs_rate
        mask = np.ones(self.N, dtype=bool) if np.all(rates == 0) else self.rng.random(self.N) >= rates
        self.update_vertex_ratio = float(mask.mean())
        return mask

    def _damped_log(self, old, proposed):
        # Mix new and old messages in probability space, then compute logs stably with logaddexp;
        # do not linearly average the two log values.
        if self.damping == 0:
            return proposed
        # Convex combination of probabilities, NOT interpolation of logits.
        return np.logaddexp(np.log(self.damping) + old,
                            np.log1p(-self.damping) + proposed)

    def _nodes(self, nodes):
        nodes = np.asarray(list(nodes))
        if (nodes.ndim != 1 or not np.issubdtype(nodes.dtype, np.integer)
                or len(nodes) not in self.orders or len(set(nodes)) != len(nodes)
                or np.any(nodes < 0) or np.any(nodes >= self.N)):
            raise ValueError("Use distinct valid node indices in an active order")
        return nodes

    def hyperedge_probability(self, nodes, assignments):
        """Exact conditional Bernoulli probability for one candidate hyperedge.

        assignments contains the community indices OF THE LISTED NODES, in
        the same order (length d), rather than a whole-network assignment.
        """
        # Full conditional probability under the generative model. nodes gives node IDs and assignments
        # their community labels in the same order, so include the product of those nodes' theta values.
        nodes = self._nodes(nodes)
        a = np.asarray(list(assignments))
        d = len(nodes)
        if (a.shape != (d,) or not np.issubdtype(a.dtype, np.integer)
                or np.any(a < 0) or np.any(a >= self.K)):
            raise ValueError("assignments must be d community indices")
        p = self.p_by_order[d]
        pair_sum = sum(p[a[i], a[j]] for i in range(d) for j in range(i + 1, d))
        return self._probability_from_pair_sum(nodes, pair_sum)

    def _probability_from_pair_sum(self, nodes, pair_sum):
        d = len(nodes)
        theta = self.theta_by_order[d][nodes]
        if pair_sum == 0 or np.any(theta == 0):
            return 0.0
        log_choose = log_binomial_coefficient(self.N - 2, d - 2, allow_approx=False)
        log_mu = np.log(theta).sum() + np.log(pair_sum) - np.log(float(comb(d, 2))) - log_choose
        if log_mu > 1e-12:
            raise ValueError("Bernoulli probability exceeds one; reduce affinities/activities")
        return float(np.exp(min(log_mu, 0.0)))

    def predict_hyperedge_probability(self, nodes):
        """Factorized marginal prediction; not a held-out conditional likelihood.

        For honest test prediction, fit only to training hyperedges first.
        """
        # Approximate unknown community assignments by the product of node marginals, not the full joint posterior.
        if self.log_marginals is None:
            raise ValueError("Run message passing first")
        nodes = self._nodes(nodes)
        q = np.exp(self.log_marginals[nodes])
        p = self.p_by_order[len(nodes)]
        pair_sum = sum(q[i] @ p @ q[j] for i in range(len(nodes)) for j in range(i + 1, len(nodes)))
        return self._probability_from_pair_sum(nodes, pair_sum)

    def activity_moments(self):
        """Empirical Phi_ds=mean_i theta_i^(d) theta_i^(s)."""
        # Return an (L,L) matrix: activity second moments on the diagonal and mean cross-order node products elsewhere.
        # These are raw second moments, not centered covariances or correlation coefficients.
        return self.theta @ self.theta.T / self.N


loop_DCHSBM = loop_DegreeCorrectedHypergraphBlockModel
DCHSBM = loop_DegreeCorrectedHypergraphBlockModel
