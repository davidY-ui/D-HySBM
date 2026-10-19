"""Build coarse hypergraphs by percolation or triangle contraction."""

import numpy as np
import random
import itertools
from scipy import sparse
from collections import defaultdict
from src.data.representation.incidence_hypergraph import IncidenceHypergraph


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    #Merge components and obtain the new root.
    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def k_clique_communities_incidence(
        H: IncidenceHypergraph, 
        k: int, selected_hye_sizes: list[int] | None = None,
        return_hyperedges=False):
    """
    Run k-clique / k-hyperedge percolation on an incidence hypergraph.

    Parameters
    ----------
    H : IncidenceHypergraph
        Hypergraph providing an N x E binary incidence matrix.
    k : int
        Percolation threshold. Two clusters are adjacent if they share
        at least k-1 nodes.
    selected_hye_sizes : list[int] or None
        None selects size-2 hyperedges. A list such as [3, 4, 5] selects
        those sizes. The subsequent size >= k filter applies in either case.
    return_hyperedges : bool
        False yields community node sets only. True also yields the original
        hyperedge IDs belonging to each community.

    Yields
    ------
    frozenset or tuple[frozenset, list[int]]
        Community nodes, optionally paired with their original hyperedge IDs.
    """
    if k < 2:
        raise ValueError(f"k={k}, k must be greater than 1.")

    # Convert to CSC for column-wise hyperedge access.
    H_incidence: sparse.csc_array = H.get_binary_incidence_matrix()
    H_incidence = H_incidence.astype(bool)
    H_incidence.eliminate_zeros()
    assert H_incidence.has_sorted_indices, (
        "The incidence matrix doesn't have a canonical sorted format. "
        "To fix this, call the sort_indices() method of scipy CSC matrices.",
    )

    N, E = H_incidence.shape

    # Size of each hyperedge
    hye_sizes = np.diff(H_incidence.indptr)

    if selected_hye_sizes is None:
        valid_hyperedges = np.where(hye_sizes == 2)[0]
    else:
        selected_hye_sizes = np.asarray(selected_hye_sizes)
        valid_hyperedges = np.where(np.isin(hye_sizes, selected_hye_sizes))[0]

    if len(valid_hyperedges) == 0:
        return

    # hyperedges[cid] contains the nodes of participating hyperedge cid.
    # original_hye_ids[cid] is its original column in the incidence matrix.
    hyperedges = []
    original_hye_ids = []

    for e in valid_hyperedges:
        start = H_incidence.indptr[e]
        end = H_incidence.indptr[e + 1]
        nodes = frozenset(H_incidence.indices[start:end])

        if len(nodes) >= k:
            hyperedges.append(nodes)
            original_hye_ids.append(e)

    M = len(hyperedges)

    if M == 0:
        return

    # membership_dict[node] lists the hyperedge IDs containing that node.
    membership_dict = defaultdict(list)

    for cid, hye in enumerate(hyperedges):
        for node in hye:
            membership_dict[node].append(cid)

    # Union-find connects hyperedges whose overlap is >= k-1.
    uf = UnionFind(M)

    for cid, hye in enumerate(hyperedges):
        overlap_count = defaultdict(int)

        # Find other hyperedges containing each node of the current hyperedge.
        for node in hye:
            for adj_cid in membership_dict[node]:
                if adj_cid > cid:
                    overlap_count[adj_cid] += 1

        # Two hyperedges are adjacent when they share >= k-1 nodes.
        for adj_cid, count in overlap_count.items():
            if count >= k - 1:
                uf.union(cid, adj_cid)

    # Collect connected components of hyperedges.
    components = defaultdict(list)

    for cid in range(M):
        root = uf.find(cid)
        components[root].append(cid)

    # The node union of each hyperedge component forms a community.
    for comp in components.values():
        community_nodes = frozenset.union(*(hyperedges[cid] for cid in comp))

        if return_hyperedges:
            community_hyperedges = [original_hye_ids[cid] for cid in comp]
            yield community_nodes, community_hyperedges
        else:
            yield community_nodes

def turn_to_factor_graph(
        H: IncidenceHypergraph,
        k: int,
        selected_hye_sizes: list[int] | None = None,
        return_hyperedges: bool = False
) -> IncidenceHypergraph | tuple[IncidenceHypergraph, list[list[int]]]:
    """
    Convert a hypergraph to a factor graph / coarse hypergraph.

    Treat each community from k-clique percolation as a new hyperedge.
    Return IncidenceHypergraph(HYG), where HYG is a list[tuple[int, ...]],
    for example [(0, 1, 2, 5), (3, 4, 6), ...].
    """

    communities = list(
        k_clique_communities_incidence(
            H,
            k=k,
            selected_hye_sizes=selected_hye_sizes,
            return_hyperedges=True
        )
    )

    if len(communities) == 0:
        new_HYG: list[tuple[int, ...]] = []
        new_H = IncidenceHypergraph(new_HYG)

        if return_hyperedges:
            return new_H, []

        return new_H

    new_HYG: list[tuple[int, ...]] = []
    factor_to_original_hyperedges: list[list[int]] = []

    for community_nodes, original_hye_ids in communities:
        # community_nodes is a frozenset.
        # Convert it to tuple[int, ...].
        new_hyperedge = tuple(sorted(community_nodes))

        new_HYG.append(new_hyperedge)
        factor_to_original_hyperedges.append(original_hye_ids)

    new_H = IncidenceHypergraph(new_HYG)

    if return_hyperedges:
        return new_H, factor_to_original_hyperedges

    return new_H



def _hyperedge_key(nodes) -> tuple[int, ...]:
    return tuple(sorted(map(int, nodes)))


def build_hyperedge_lookup(
    incidence: sparse.spmatrix,
) -> dict[tuple[int, ...], list[int]]:
    incidence = incidence.tocsc()
    lookup = defaultdict(list)

    for e in range(incidence.shape[1]):
        nodes = incidence.indices[incidence.indptr[e] : incidence.indptr[e + 1]]
        lookup[_hyperedge_key(nodes)].append(e)

    return lookup

##Find the ID of the specified hyperedge.
def find_hyperedge_id(
    incidence: sparse.spmatrix,
    target_nodes,
    lookup: dict[tuple[int, ...], list[int]] | None = None,
) -> list[int]:
    if lookup is None:
        lookup = build_hyperedge_lookup(incidence)

    return list(lookup.get(_hyperedge_key(target_nodes), []))


def construct_G3_from_size2_hypergraph(
    hypergraph: IncidenceHypergraph,
    seed=None,
) -> IncidenceHypergraph:
    """
    Greedily form triangular hyperedges from size-2 hyperedges.
    For each triangle (u, v, w), remove its three size-2 hyperedges and add
    one size-3 hyperedge unless that size-3 hyperedge already exists.
    """
    rng = random.Random(seed)

    incidence = sparse.csc_array(hypergraph.get_binary_incidence_matrix())
    if not incidence.has_sorted_indices:
        incidence.sort_indices()

    hyperedges = [_hyperedge_key(hye) for hye in hypergraph]
    existing_hyperedges = set(hyperedges)

    hye_sizes = np.diff(incidence.indptr)
    size2_ids = np.flatnonzero(hye_sizes == 2)

    edge_to_hye_ids = defaultdict(list)
    edges = set()

    for e in size2_ids:
        nodes = incidence.indices[incidence.indptr[e] : incidence.indptr[e + 1]]
        edge = _hyperedge_key(nodes)
        edges.add(edge)
        edge_to_hye_ids[edge].append(int(e))

    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    new_triangles = []
    new_triangle_set = set()
    deleted_hye_ids = set()

    while True:
        found_triangle = None

        for u in list(adj):
            for v in list(adj[u]):
                if u >= v:
                    continue

                common = adj[u] & adj[v]
                if common:
                    w = rng.choice(sorted(common))
                    found_triangle = tuple(sorted((u, v, w)))
                    break

            if found_triangle is not None:
                break

        if found_triangle is None:
            break

        if (
            found_triangle not in existing_hyperedges
            and found_triangle not in new_triangle_set
        ):
            new_triangles.append(found_triangle)
            new_triangle_set.add(found_triangle)

        for a, b in itertools.combinations(found_triangle, 2):
            edge = tuple(sorted((a, b)))

            adj[a].discard(b)
            adj[b].discard(a)

            deleted_hye_ids.update(edge_to_hye_ids.get(edge, []))

    keep_mask = np.ones(incidence.shape[1], dtype=bool)
    if deleted_hye_ids:
        keep_mask[list(deleted_hye_ids)] = False

    hyperedges_new = [
        hyperedges[i]
        for i in range(len(hyperedges))
        if keep_mask[i]
    ]
    hyperedges_new.extend(new_triangles)

    return IncidenceHypergraph.load_from_hye_list(hyperedges_new, N=hypergraph.N)
