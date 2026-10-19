"""One sampler for HSBM and DCHSBM, selected by the ``model`` argument.

Reuse the HSBM count-vector, node-drawing, and pair-affinity helpers. No Rust
extension or sampling_dc.py is needed. The input affinity is p[d] = c[d] / N.

HSBM:   model="hsbm", theta=None (effective activities are all one).
DCHSBM: model="dchsbm", theta={d: array(N,)} (order-specific activities).

Both orders 2 and above use type-stratified sampling, without an N by N
binary probability matrix. Few-valued activities are required for efficiency.
"""
from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, localcontext
from itertools import combinations, product
from math import ceil, comb, exp, expm1, fsum, isfinite, log, log1p, prod
import warnings

import numpy as np

from src.model.numerical import hyperedge_pi
from src.model.sampling import _community_count_combinations, _sample_hye_from_count


def _labels(node_assignments):
    labels = np.asarray(node_assignments)
    if (labels.ndim != 1 or not labels.size or labels.dtype.kind not in "iu"
            or np.any(labels < 0)):
        raise ValueError("node_assignments must be a nonempty 1D array of nonnegative integers")
    return labels


def _real_scalar(value, name, lower, upper):
    if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
            or not np.isrealobj(value)):
        raise ValueError(f"{name} must be a real scalar in [{lower}, {upper}]")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not isfinite(result) or not lower <= result <= upper:
        raise ValueError(f"{name} must be a finite real scalar in [{lower}, {upper}]")
    return result


def _positive_integer(value, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value <= 0):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def make_correlated_activities(node_assignments, zeta, delta2=0.5, delta3=0.5, seed=None):
    """Make the exact two-order activity experiment, separately in each group.

    theta[2] = 1 + delta2*X, theta[3] = 1 + delta3*Y, where X,Y in {-1,+1}
    and the empirical fraction for a sign pair (x,y) is (1+zeta*x*y)/4.
    Each community must have even size n_a, and n_a*(1+zeta)/4 must be an
    integer. Infeasible proportions raise ValueError; no silent rounding.

    Returns {2: array(N,), 3: array(N,)}. With the same seed and labels,
    theta[2] is identical for different feasible zeta. Each order's activity
    histogram is fixed, including within every community. 0 <= delta < 1.
    """
    labels = _labels(node_assignments)
    zeta = _real_scalar(zeta, "zeta", -1., 1.)
    delta2 = _real_scalar(delta2, "delta2", 0., 1.)
    delta3 = _real_scalar(delta3, "delta3", 0., 1.)
    if delta2 == 1 or delta3 == 1:
        raise ValueError("delta2 and delta3 must be strictly smaller than 1")
    rng = np.random.default_rng(seed)
    theta2 = np.empty(labels.size, dtype=float)
    theta3 = np.empty(labels.size, dtype=float)
    for group in np.unique(labels):
        nodes = np.flatnonzero(labels == group)
        size = len(nodes)
        count_hh = size * (1 + zeta) / 4
        if size % 2 or abs(count_hh - round(count_hh)) > 1e-9:
            raise ValueError(
                f"Community {group} has size {size}: exact proportions require "
                "even size and size*(1+zeta)/4 integer; increase N or choose feasible zeta"
            )
        count_hh = int(round(count_hh))
        nodes = rng.permutation(nodes)
        half = size // 2
        # The first half have high second-order activity; vary only their overlap with high third-order activity.
        # Changing zeta with the same seed preserves the theta2 node assignment.
        theta2[nodes[:half]] = 1 + delta2
        theta2[nodes[half:]] = 1 - delta2
        theta3[nodes] = 1 - delta3
        theta3[nodes[:count_hh]] = 1 + delta3
        theta3[nodes[half:half + half - count_hh]] = 1 + delta3
    return {2: theta2, 3: theta3}


def _validate_inputs(p, selected_hye_sizes, node_assignments, theta):
    labels = _labels(node_assignments)
    try:
        orders = list(selected_hye_sizes)
    except TypeError as exc:
        raise ValueError("selected_hye_sizes must be an iterable of integer orders") from exc
    if (not orders or any(isinstance(d, (bool, np.bool_))
                         or not isinstance(d, (int, np.integer))
                         or not 2 <= d <= labels.size for d in orders)
            or len(set(orders)) != len(orders)):
        raise ValueError("selected_hye_sizes must contain distinct integers between 2 and N")
    orders = [int(d) for d in orders]
    for mapping, name in ((p, "p"), (theta, "theta")):
        if (not isinstance(mapping, Mapping)
                or any(isinstance(d, (bool, np.bool_))
                       or not isinstance(d, (int, np.integer)) for d in mapping)
                or set(mapping) != set(orders)):
            raise ValueError(f"{name} must map exactly the selected integer orders to arrays")
    affinities, activities = {}, {}
    K = None
    for d in orders:
        if not np.isrealobj(p[d]) or not np.isrealobj(theta[d]):
            raise ValueError("p and theta must be real arrays")
        matrix = np.asarray(p[d], dtype=float)
        activity = np.asarray(theta[d], dtype=float)
        if (matrix.ndim != 2 or not matrix.shape[0] or matrix.shape[0] != matrix.shape[1]
                or not np.isfinite(matrix).all() or np.any(matrix < 0)
                or not np.array_equal(matrix, matrix.T)):
            raise ValueError(f"p[{d}] must be a finite, nonnegative, symmetric square matrix")
        if K is None:
            K = matrix.shape[0]
        if matrix.shape != (K, K) or int(labels.max()) >= K:
            raise ValueError("All affinity matrices must have the same K and cover all labels")
        if (activity.shape != (labels.size,) or not np.isfinite(activity).all()
                or np.any(activity < 0)):
            raise ValueError(f"theta[{d}] must be a finite, nonnegative vector of length N")
        affinities[d], activities[d] = matrix, activity
    return labels, orders, affinities, activities


def _activity_types(labels, activity):
    # theta varies within communities, so community-only stratification is insufficient.
    # Group nodes by (community, theta); candidate edges with the same type composition have equal probability.
    groups = {}
    for node, (label, value) in enumerate(zip(labels, activity)):
        groups.setdefault((int(label), float(value)), []).append(node)
    keys = sorted(groups)
    nodes = {t: np.asarray(groups[key], dtype=np.int64) for t, key in enumerate(keys)}
    type_labels = np.array([key[0] for key in keys], dtype=np.int64)
    type_theta = np.array([key[1] for key in keys], dtype=float)
    return nodes, [len(nodes[t]) for t in nodes], type_labels, type_theta


def _all_candidates(nodes, count):
    choices = [combinations(nodes[t], m) for t, m in enumerate(count) if m]
    for selection in product(*choices):
        yield tuple(sorted(int(node) for subset in selection for node in subset))


def _uniform_unique_subset(nodes, count, population, size, rng):
    # Given the count, sample a uniform subset without replacement rather than deduplicating replacement draws.
    complement = size > population // 2
    draw_size = population - size if complement else size
    selected = set()
    while len(selected) < draw_size:
        selected.add(_sample_hye_from_count(nodes, count, rng))
    if complement:
        # Near a complete graph, sample missing edges to avoid repeated rejections of existing edges.
        return (edge for edge in _all_candidates(nodes, count) if edge not in selected)
    return iter(sorted(selected))


def _geometric_binomial_count(population, log_probability, rng, max_count=None):
    """Draw Binomial(population, p) via geometric gaps, with Python-int trials.

    This is the Bernoulli waiting-time construction, not a Poisson count
    approximation. Expected work is O(population*p + 1); use it for sparse
    strata whose candidate count exceeds NumPy's int64 Binomial interface.
    log_probability retains probabilities too small for a binary64 float.
    max_count is the remaining output allowance and can be zero internally.
    """
    if population == 0 or log_probability == -float("inf"):
        return 0
    if not isfinite(log_probability) or log_probability > 0:
        raise ValueError("Bernoulli log probability must be <= 0")

    if log_probability >= log(np.finfo(float).tiny):
        probability = exp(log_probability)
        if probability == 1:
            if max_count is not None and population > max_count:
                raise ValueError("Sample exceeds max_hyperedges; increase the limit, set it to None, or reduce density")
            return population
        hazard_num, hazard_den = (-log1p(-probability)).as_integer_ratio()
    else:
        # A positive probability may underflow in exp(log_probability) while M*p remains substantial.
        # Only this extreme branch uses standard-library high precision to retain significant digits in 1-p.
        with localcontext() as context:
            context.prec = max(50, ceil(-log_probability / log(10)) + 30)
            probability = Decimal.from_float(log_probability).exp()
            hazard = -(Decimal(1) - probability).ln()
            hazard_num, hazard_den = hazard.as_integer_ratio()

    remaining = population
    size = 0
    while remaining:
        # For E~Exp(1), G=floor(E/[-log(1-p)])+1 is the geometric gap between successes.
        # Compute floor using integer ratios to avoid float overflow or rounding errors for huge gaps.
        exp_num, exp_den = float(rng.exponential()).as_integer_ratio()
        gap = (exp_num * hazard_den) // (exp_den * hazard_num) + 1
        if gap > remaining:
            break
        remaining -= gap
        size += 1
        if max_count is not None and size > max_count:
            raise ValueError("Sample exceeds max_hyperedges; increase the limit, set it to None, or reduce density")
    return size


def _draw_hyperedge_count(population, log_mu, rng, allow_repeated,
                          count_method, max_count=None):
    """Choose a stratum count law separately from how its edges are selected.

    False targets Binomial(M, mu), True targets Poisson(M*mu). Approximate
    counts use rounded/clipped Normal draws or a Poisson approximation;
    they do not imply exact independent edge occurrences/multiplicities.
    """
    def exact_simple_count():
        if population > np.iinfo(np.int64).max:
            return _geometric_binomial_count(
                population, min(log_mu, 0.), rng, max_count=max_count,
            )
        return int(rng.binomial(population, exp(min(log_mu, 0.))))

    if not allow_repeated and count_method == "exact":
        size = exact_simple_count()
    else:
        log_mean = log(population) + log_mu
        try:
            mean = exp(log_mean)
        except OverflowError as exc:
            raise ValueError("Hyperedge count mean overflows floating-point arithmetic") from exc
        if not isfinite(mean):
            raise ValueError("Hyperedge count mean must be finite")

        if allow_repeated:
            # True targets Poisson counts: variance equals the mean, without the Bernoulli factor (1-mu).
            if count_method == "approx" and mean > 10:
                size = max(0, int(round(rng.normal(mean, exp(.5 * log_mean)))))
            else:
                size = int(rng.poisson(mean))
        else:
            # Use the original sampler's mean/variance thresholds; expm1 computes 1-mu stably.
            log_variance = (log_mean + log(-expm1(log_mu))
                            if log_mu < 0 else -float("inf"))
            if population > 20 and log_variance > log(10):
                size = max(0, int(round(rng.normal(mean, exp(.5 * log_variance)))))
            elif ((population > 20 and log_mean < -log(10))
                  or (population > 100 and log_mean < log(10))):
                size = int(rng.poisson(mean))
            else:
                # When approximation criteria fail, handle huge candidate counts without an out-of-range Binomial call.
                size = exact_simple_count()
            # The approximation is unbounded; restrict counts to [0,M] so selection without replacement terminates.
            size = min(population, size)

    if max_count is not None and size > max_count:
        raise ValueError("Sample exceeds max_hyperedges; increase the limit, set it to None, or reduce density")
    return size


def explicit_sampling_loop(
    p, selected_hye_sizes, node_assignments, allow_repeated=False, seed=None,
    block_size=None, *, model="hsbm", theta=None,
    max_hyperedges=1_000_000, max_compositions=100_000, count_method="exact",
):
    r"""Sample HSBM or DCHSBM through the same type-stratified implementation.

    Parameters
    ----------
    p : dict[int, ndarray(K,K)]
        Per-order symmetric, nonnegative affinities, in the convention p=c/N.
    selected_hye_sizes : iterable[int]
        Distinct orders between 2 and N. p and theta must have these exact keys.
    node_assignments : array(N,)
        Nonnegative integer community labels, indexing p's rows/columns.
    allow_repeated : bool, default False
        False targets Bernoulli simple hyperedges; True targets Poisson copies.
        This choice applies to ALL selected orders, including order two.
    count_method : {'exact', 'approx'}, default 'exact'
        Applies to BOTH models and BOTH multiplicity modes. Exact retains the
        target count law, up to floating-point/RNG precision. Approx borrows
        the original sampler's Binomial-to-Normal/Poisson thresholds in False
        mode; in True mode it uses Normal(mean, variance=mean) when mean>10,
        otherwise Poisson. Rounded Normal counts are clipped below at zero;
        approximate False counts are also bounded above by candidate count.
        Approx preserves simple/repeated output structure, but introduces
        distributional error and does not promise exact edge independence.
    seed : int or numpy.random.Generator, optional
        Passed to numpy.random.default_rng. Fresh identical seeds reproduce results.
    block_size : positive int or None
        Retained in the original argument position for call compatibility.
        A positive value issues a warning and is ignored: this algorithm never
        scans an N by N binary matrix and does not need matrix blocks.
    model : {'hsbm', 'dchsbm'}, default 'hsbm'
        Explicit model selector, case insensitive. It is not inferred from theta.
        HSBM requires theta=None; DCHSBM requires theta to be supplied.
    theta : dict[int, ndarray(N,)] or None
        Order-specific, finite nonnegative activities. HSBM uses effective ones.
        DCHSBM accepts zero activities and never rescales supplied values.
    max_hyperedges, max_compositions : positive int or None
        Output-copy cap and per-order type-composition upper-bound cap.
        None independently disables the corresponding cap. Defaults remain
        1_000_000 and 100_000; zero is not an unlimited sentinel.

    Returns
    -------
    list[tuple[int, ...]]
        Hyperedges with distinct nodes inside each tuple. Repeated tuples are
        preserved in Poisson mode. Ordered by size and lexicographically.

    Notes
    -----
    The first six positional slots match the old explicit_sampling_loop.
    The default allow_repeated is now False (the old default was True).
    The old implementation permitted repetition only for higher orders and
    approximated some counts. New True means a Poisson model at every order.
    Use False with count_method='exact' to preserve the independent-Bernoulli law;
    it does not reproduce the old sampler's random-number sequence.

    With the same seed, HSBM and DCHSBM with theta[d]=ones(N) give exactly the
    same output, in either multiplicity mode.

    For e of size d, define
      mu_e = prod(theta[d][i] for i in e) * sum_{i<j in e} p[d][z_i,z_j]
             / (comb(N-2,d-2) * comb(d,2)).

    With count_method='exact':
    allow_repeated=False: each distinct e is independently Bernoulli(mu_e).
                          Requires 0 <= mu_e <= 1. Default for simple graphs.
    allow_repeated=True:  each distinct e has independent Poisson(mu_e) copies.
                          mu_e is a rate and may exceed one.

    Different orders are conditionally independent given z,theta,p. HSBM is
    exactly the theta=1 special case. Input p and theta are never rescaled.

    Default exact class sampling is up to floating point / RNG precision.
    Approximate counts are opt-in. Intended for few-valued theta, especially the two-level
    controlled experiment. Many distinct theta values can be expensive:
    max_compositions caps the per-order count-vector upper bound, and
    max_hyperedges caps the number of returned copies (no truncation).
    Each resource cap can be disabled with None. In exact mode, candidate counts can exceed
    int64 in both modes: Poisson uses log-space total rates; Bernoulli uses
    geometric waiting times and Python integers for these large strata.
    Bernoulli strata within int64 retain NumPy's Binomial RNG. Waiting-time
    counting costs O(L+1) draws for L selected edges and avoids enumerating
    absent candidates. It does not replace Binomial counts by Poisson counts.

    Poisson mode is an explicit multihypergraph model, not the original HSBM
    implementation's Binomial/approximate count followed by replacement.
    """
    # Require explicit model selection so HSBM cannot silently ignore a supplied theta,
    # and DCHSBM cannot silently become HSBM when theta is omitted.
    if not isinstance(model, str) or model.lower() not in ("hsbm", "dchsbm"):
        raise ValueError("model must be 'hsbm' or 'dchsbm'")
    model = model.lower()
    if not isinstance(count_method, str) or count_method.lower() not in ("exact", "approx"):
        raise ValueError("count_method must be 'exact' or 'approx'")
    count_method = count_method.lower()
    labels = _labels(node_assignments)
    try:
        selected_hye_sizes = list(selected_hye_sizes)
    except TypeError as exc:
        raise ValueError("selected_hye_sizes must be an iterable of integer orders") from exc
    if model == "hsbm":
        if theta is not None:
            raise ValueError("HSBM requires theta=None; use model='dchsbm' for supplied activities")
        # Share one all-ones vector across orders, read-only by convention, rather than allocating N entries per order.
        # Downstream functions only read it; HSBM and explicit theta=1 use the same sampling path.
        unit_activity = np.ones(labels.size, dtype=float)
        try:
            theta = {d: unit_activity for d in selected_hye_sizes}
        except TypeError as exc:
            raise ValueError("selected_hye_sizes must contain integer orders") from exc
    elif theta is None:
        raise ValueError("DCHSBM requires theta={d: activity_vector} for every selected order")
    if block_size is not None:
        _positive_integer(block_size, "block_size")
        warnings.warn(
            "block_size is ignored by sampling_new: type-stratified sampling does not "
            "allocate an N by N matrix", UserWarning, stacklevel=2,
        )
    if not isinstance(allow_repeated, (bool, np.bool_)):
        raise ValueError("allow_repeated must be Boolean")
    # None disables only the corresponding resource cap; parameter validation still applies.
    if max_hyperedges is not None:
        max_hyperedges = _positive_integer(max_hyperedges, "max_hyperedges")
    if max_compositions is not None:
        max_compositions = _positive_integer(max_compositions, "max_compositions")
    labels, orders, affinities, activities = _validate_inputs(
        p, selected_hye_sizes, node_assignments, theta,
    )
    rng = np.random.default_rng(seed)
    sampled_hypergraph = []
    N = labels.size
    for d in sorted(orders):
        nodes, type_sizes, type_labels, type_theta = _activity_types(labels, activities[d])
        num_types = len(type_sizes)
        if (max_compositions is not None
                and comb(num_types + d - 1, d) > max_compositions):
            raise ValueError(
                f"Order {d} has too many (community,theta) compositions; "
                "use few-valued activities, increase max_compositions, or set it to None"
            )
        # Reuse HSBM affinity and composition enumeration, extending communities to (community, theta) types.
        augmented_p = affinities[d][np.ix_(type_labels, type_labels)]
        kappa = comb(N - 2, d - 2) * comb(d, 2)
        layer_edges = []
        for count in _community_count_combinations(d, type_sizes):
            occupied = [(t, m) for t, m in enumerate(count) if m]
            if any(type_theta[t] == 0 for t, m in occupied):
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                pi = float(hyperedge_pi(count, augmented_p))
            if not isfinite(pi):
                raise ValueError(f"Order {d} pair affinity overflows floating-point arithmetic")
            if pi == 0:
                continue
            # comb returns Python arbitrary-precision integers; neither branch casts huge candidate counts to int64.
            population = prod(comb(type_sizes[t], m) for t, m in occupied)
            log_mu = log(pi) + fsum(m * log(type_theta[t]) for t, m in occupied) - log(kappa)
            if not allow_repeated and log_mu > 1e-12:
                raise ValueError(f"Order {d} contains Bernoulli probability > 1; check p=c/N and theta")
            remaining_cap = (None if max_hyperedges is None else
                             max_hyperedges - len(sampled_hypergraph) - len(layer_edges))
            # Count sampling and repetition are controlled separately; True/False and HSBM/DC share this entry point.
            size = _draw_hyperedge_count(
                population, log_mu, rng, allow_repeated, count_method,
                max_count=remaining_cap,
            )
            if not size:
                continue
            if allow_repeated:
                layer_edges.extend(_sample_hye_from_count(nodes, count, rng) for _ in range(size))
            else:
                layer_edges.extend(_uniform_unique_subset(nodes, count, population, size, rng))
        sampled_hypergraph.extend(sorted(layer_edges))
    return sampled_hypergraph
