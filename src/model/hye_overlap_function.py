"""Measure hyperedge overlap within and between selected orders."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from math import comb
from typing import Protocol

import numpy as np
from scipy import sparse


class _HasIncidence(Protocol):
    def get_binary_incidence_matrix(self):
        ...


def _as_incidence_matrix(
    incidence_or_hypergraph: sparse.spmatrix | sparse.sparray | _HasIncidence,
) -> sparse.csc_array:
    if hasattr(incidence_or_hypergraph, "get_binary_incidence_matrix"):
        incidence_or_hypergraph = incidence_or_hypergraph.get_binary_incidence_matrix()

    incidence = sparse.csc_array(incidence_or_hypergraph)
    if not incidence.has_sorted_indices:
        incidence.sort_indices()

    return incidence


def incidence_to_hyperedges(
    incidence: sparse.spmatrix | sparse.sparray | _HasIncidence,
    deduplicate: bool = True,
) -> list[tuple[int, ...]]:
    """
    Convert an incidence matrix with shape (N, E) to hyperedge tuples.

    Rows are nodes and columns are hyperedges. Hyperedges with size < 2 are ignored.
    """
    incidence = _as_incidence_matrix(incidence)

    hyperedges = [
        tuple(map(int, incidence.indices[incidence.indptr[e] : incidence.indptr[e + 1]]))
        for e in range(incidence.shape[1])
    ]
    hyperedges = [hye for hye in hyperedges if len(hye) >= 2]

    if deduplicate:
        return sorted(set(hyperedges), key=lambda hye: (len(hye), hye))

    return hyperedges


def group_by_paper_order(
    hyperedges: list[tuple[int, ...]],
) -> dict[int, list[tuple[int, ...]]]:
    """
    Group hyperedges by paper order m = hyperedge size - 1.

    size=2 -> order 1, size=3 -> order 2, size=4 -> order 3.
    """
    by_order: dict[int, list[tuple[int, ...]]] = defaultdict(list)

    for hye in hyperedges:
        by_order[len(hye) - 1].append(hye)

    return dict(by_order)


def _q_min_required(k: int, m: int) -> int:
    """
    Minimum q such that C(q, m) >= k.

    For one node participating in k order-m hyperedges, q is the minimum number of
    distinct neighbors needed to form k different m-combinations.
    """
    if k <= 0:
        return 0
    if m < 1:
        raise ValueError(f"m must be at least 1, got {m}.")

    low = m
    high = m
    while comb(high, m) < k:
        high *= 2

    while low < high:
        mid = (low + high) // 2
        if comb(mid, m) >= k:
            high = mid
        else:
            low = mid + 1

    return low


def intra_order_overlap(
    hyperedges_m: list[tuple[int, ...]],
    m: int,
) -> float:
    """
    Compute diagonal term T^(m), the same-order hyperedge overlap.

    m is the paper order. For pairwise edges, T^(1) is defined as 0.
    """
    if m == 1:
        return 0.0
    if m < 1:
        raise ValueError(f"m must be at least 1, got {m}.")
    if not hyperedges_m:
        return np.nan

    incidence_count: dict[int, int] = defaultdict(int)
    unique_neighbors: dict[int, set[int]] = defaultdict(set)

    for hye in hyperedges_m:
        for v in hye:
            incidence_count[v] += 1
            unique_neighbors[v].update(u for u in hye if u != v)

    weighted_sum = 0.0
    weight_total = 0

    for v, k in incidence_count.items():
        q_i = len(unique_neighbors[v])
        q_max = m * k
        q_min = _q_min_required(k, m)

        if q_max == q_min:
            local_overlap = 0.0
        else:
            local_overlap = (q_max - q_i) / (q_max - q_min)

        weighted_sum += k * local_overlap
        weight_total += k

    return weighted_sum / weight_total if weight_total else np.nan


def inter_order_overlap(
    by_order: dict[int, list[tuple[int, ...]]],
    m: int,
    n: int,
) -> float:
    """
    Compute off-diagonal term I^(m,n).

    The lower-order layer is compared with all unique lower-order faces generated
    inside higher-order hyperedges.
    """
    if m == n:
        raise ValueError("inter_order_overlap requires m != n.")
    if m < 1 or n < 1:
        raise ValueError(f"orders must be at least 1, got m={m}, n={n}.")

    low = min(m, n)
    high = max(m, n)
    face_size = low + 1

    low_edges = set(by_order.get(low, ()))
    high_edges = by_order.get(high, ())

    if not low_edges or not high_edges:
        return np.nan

    possible_low_faces: set[tuple[int, ...]] = set()
    for hye in high_edges:
        if len(hye) >= face_size:
            possible_low_faces.update(combinations(hye, face_size))

    if not possible_low_faces:
        return np.nan

    return len(possible_low_faces & low_edges) / len(possible_low_faces)


def overlap_matrix_from_incidence(
    incidence: sparse.spmatrix | sparse.sparray | _HasIncidence,
    max_order: int | None = None,
    symmetric: bool = True,
    deduplicate: bool = True,
) -> tuple[np.ndarray, list[int]]:
    """
    Compute the paper overlap matrix O.

    Returns O and the corresponding order labels. order=1 means size-2 edges,
    order=2 means size-3 hyperedges, and so on.
    """
    hyperedges = incidence_to_hyperedges(incidence, deduplicate=deduplicate)
    by_order = group_by_paper_order(hyperedges)

    if max_order is None:
        max_order = max(by_order, default=1)
    if max_order < 1:
        raise ValueError(f"max_order must be at least 1, got {max_order}.")

    orders = list(range(1, max_order + 1))
    O = np.full((max_order, max_order), np.nan, dtype=float)

    for m in orders:
        for n in orders:
            i = m - 1
            j = n - 1

            if m == n:
                O[i, j] = intra_order_overlap(by_order.get(m, []), m=m)
            elif symmetric or m < n:
                O[i, j] = inter_order_overlap(by_order, m, n)

    O[0, 0] = 0.0

    return O, orders
