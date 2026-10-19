"""Construct feedback vertex sets and masks for factor-graph update scheduling."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy import sparse

from src.data.representation.incidence_hypergraph import IncidenceHypergraph
from src.model.factor_G_simplize import construct_G3_from_size2_hypergraph


class FcoreHDResult(NamedTuple):
    fvs_nodes: list[int]
    remaining_nodes: np.ndarray
    remaining_hyes: np.ndarray
    remaining_hyes_list: list[np.ndarray]
    fvs_average_degree: float


def _as_sorted_csc(incidence: sparse.spmatrix | sparse.sparray) -> sparse.csc_array:
    incidence_csc = sparse.csc_array(incidence, dtype=np.int32)

    if not incidence_csc.has_sorted_indices:
        incidence_csc.sort_indices()

    return incidence_csc


def _factor_graph_2core(
    incidence_csc: sparse.csc_array,
    incidence_csr: sparse.csr_array,
    active_nodes: np.ndarray,
    active_hyes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the 2-core of the active incidence bipartite graph.

    Nodes and hyperedges with current degree less than 2 are peeled until no more
    such vertices remain. Returned degrees are the degrees after peeling.

    Parameters
    ----------
    incidence_csc:
        Binary node-hyperedge incidence matrix in CSC format, with shape
        ``(n_nodes, n_hyperedges)``. It is used for efficient traversal from each
        hyperedge to its incident nodes.
    incidence_csr:
        The same binary incidence matrix in CSR format. It is used for efficient
        traversal from each node to its incident hyperedges.
    active_nodes:
        Boolean mask of nodes that are still available before 2-core peeling.
        Inactive nodes are ignored and their returned degree is set to 0.
    active_hyes:
        Boolean mask of hyperedges that are still available before 2-core peeling.
        Inactive hyperedges are ignored and their returned degree is set to 0.

    Returns
    -------
    in_core_nodes:
        Boolean mask of active nodes that remain in the factor-graph 2-core after
        recursively removing nodes and hyperedges with degree less than 2.
    in_core_hyes:
        Boolean mask of active hyperedges that remain in the factor-graph 2-core
        after the same peeling process.
    node_deg:
        Degree of each node after 2-core peeling, measured as the number of
        remaining incident hyperedges in ``in_core_hyes``. Nodes outside the final
        2-core have degree less than 2 or 0; inactive nodes are set to 0.
    hye_deg:
        Degree of each hyperedge after 2-core peeling, measured as the number of
        remaining incident nodes in ``in_core_nodes``. Hyperedges outside the final
        2-core have degree less than 2 or 0; inactive hyperedges are set to 0.
    """
    node_deg = np.asarray(
        incidence_csr @ active_hyes.astype(np.int32),
        dtype=int,
    ).ravel()
    hye_deg = np.asarray(
        incidence_csc.T @ active_nodes.astype(np.int32),
        dtype=int,
    ).ravel()

    node_deg[~active_nodes] = 0
    hye_deg[~active_hyes] = 0

    in_core_nodes = active_nodes.copy()
    in_core_hyes = active_hyes.copy()

    q_nodes = list(np.flatnonzero(in_core_nodes & (node_deg < 2)))
    q_hyes = list(np.flatnonzero(in_core_hyes & (hye_deg < 2)))

    head_n = 0
    head_e = 0

    while head_n < len(q_nodes) or head_e < len(q_hyes):
        while head_n < len(q_nodes):
            v = int(q_nodes[head_n])
            head_n += 1

            if not in_core_nodes[v]:
                continue

            in_core_nodes[v] = False

            start, end = incidence_csr.indptr[v], incidence_csr.indptr[v + 1]
            for e in incidence_csr.indices[start:end]:
                if in_core_hyes[e]:
                    hye_deg[e] -= 1
                    if hye_deg[e] < 2:
                        q_hyes.append(int(e))

        while head_e < len(q_hyes):
            e = int(q_hyes[head_e])
            head_e += 1

            if not in_core_hyes[e]:
                continue

            in_core_hyes[e] = False

            start, end = incidence_csc.indptr[e], incidence_csc.indptr[e + 1]
            for v in incidence_csc.indices[start:end]:
                if in_core_nodes[v]:
                    node_deg[v] -= 1
                    if node_deg[v] < 2:
                        q_nodes.append(int(v))

    return in_core_nodes, in_core_hyes, node_deg, hye_deg


def FcoreHD_FVS(
    hypergraph: IncidenceHypergraph,
    random_tie: bool = False,
    simplify_g3: bool = True,
    seed: int | None = None,
    reinsertion=True,
    reinsertion_order="degree",
) -> FcoreHDResult:
    """
    Use FCoreHD to find an approximate feedback vertex set.

    The returned ``fvs_average_degree`` is the mean degree of the selected FVS
    nodes measured immediately before each selected node is removed.
    """
    rng = np.random.default_rng(seed)

    hypergraph_g3 = (
        construct_G3_from_size2_hypergraph(hypergraph, seed=seed)
        if simplify_g3
        else hypergraph
    )
    incidence_csc = _as_sorted_csc(hypergraph_g3.get_binary_incidence_matrix())
    incidence_csr = sparse.csr_array(incidence_csc)

    n_nodes, n_hyes = incidence_csc.shape
    active_nodes = np.ones(n_nodes, dtype=bool)
    active_hyes = np.ones(n_hyes, dtype=bool)

    fvs_nodes: list[int] = []
    fvs_degrees_before_deletion: list[int] = []

    while True:
        node_deg_before_deletion = np.asarray(
            incidence_csr @ active_hyes.astype(np.int32),
            dtype=int,
        ).ravel()
        node_deg_before_deletion[~active_nodes] = 0

        in_core_nodes, _, node_deg, _ = _factor_graph_2core(
            incidence_csc=incidence_csc,
            incidence_csr=incidence_csr,
            active_nodes=active_nodes,
            active_hyes=active_hyes,
        )

        if not np.any(in_core_nodes):
            break

        max_deg = int(np.max(node_deg[in_core_nodes]))
        candidates = np.flatnonzero(in_core_nodes & (node_deg == max_deg))

        if random_tie:
            v = int(rng.choice(candidates))
        else:
            v = int(candidates[0])

        fvs_nodes.append(v)
        fvs_degrees_before_deletion.append(int(node_deg_before_deletion[v]))
        active_nodes[v] = False

    # remaining_nodes = np.flatnonzero(active_nodes)
    # hye_deg = np.asarray(
    #     incidence_csc.T @ active_nodes.astype(np.int32),
    #     dtype=int,
    # ).ravel()
    # remaining_hyes_idx = np.flatnonzero(active_hyes & (hye_deg >= 2))
    # hyperedges = list(hypergraph_g3)
    # remaining_hyes = [hyperedges[e] for e in remaining_hyes_idx]
    # fvs_average_degree = (
    #     float(np.mean(fvs_degrees_before_deletion))
    #     if fvs_degrees_before_deletion
    #     else 0.0
    # )
        # ====== Reinsertion postprocessing starts here ======
    if reinsertion and fvs_nodes:
        degree_map = {
            v: d for v, d in zip(fvs_nodes, fvs_degrees_before_deletion)
        }

        fvs_nodes = reinsertion_prune_fvs_from_incidence(
            incidence_csc=incidence_csc,
            incidence_csr=incidence_csr,
            fvs_nodes=fvs_nodes,
            order=reinsertion_order,
            seed=seed,
        )

        active_nodes = np.ones(n_nodes, dtype=bool)
        active_nodes[np.asarray(fvs_nodes, dtype=int)] = False

        fvs_degrees_before_deletion = [
            degree_map[v] for v in fvs_nodes if v in degree_map
        ]

    remaining_nodes = np.flatnonzero(active_nodes)
    hye_deg = np.asarray(
        incidence_csc.T @ active_nodes.astype(np.int32),
        dtype=int,
    ).ravel()
    remaining_hyes_idx = np.flatnonzero(active_hyes & (hye_deg >= 2))
    hyperedges = list(hypergraph_g3)
    remaining_hyes = [hyperedges[e] for e in remaining_hyes_idx]
    fvs_average_degree = (
        float(np.mean(fvs_degrees_before_deletion))
        if fvs_degrees_before_deletion
        else 0.0
    )

    return FcoreHDResult(
        fvs_nodes,
        remaining_nodes,
        remaining_hyes_idx,
        remaining_hyes,
        fvs_average_degree,
    )


def FVS_random(
    hypergraph: IncidenceHypergraph,
    dropout_FVS: float = 0.5,
    seed: int | None = None,
    min_keep: int = 2,
) -> IncidenceHypergraph:
    """
    Return only the randomized FVS sub-hypergraph.
    """
    sub_hypergraph, _, _ = FVS_random_mask_proration(
        hypergraph=hypergraph,
        dropout_FVS=dropout_FVS,
        seed=seed,
        min_keep=min_keep,
    )
    return sub_hypergraph


def FVS_random_mask(
    hypergraph: IncidenceHypergraph,
    message_matrix,
    dropout_FVS: float = 0.5,
    seed: int | None = None,
) -> tuple[np.ndarray, list[int]]:
    """
    Build a message update mask.

    Non-FVS node messages are updated. FVS node messages are updated with
    probability ``dropout_FVS``.
    """
    if not 0 <= dropout_FVS <= 1:
        raise ValueError("dropout_FVS must be in [0, 1].")

    rng = np.random.default_rng(seed)
    fvs_node_set = FcoreHD_FVS(
        hypergraph,
        random_tie=True,
        seed=seed,
    ).fvs_nodes
    fvs_node_set = np.asarray(fvs_node_set, dtype=int)

    mat = message_matrix.tocsc()
    is_fvs_node = np.zeros(mat.shape[0], dtype=bool)
    is_fvs_node[fvs_node_set] = True
    fvs_data_mask = is_fvs_node[mat.indices]

    update_mask = np.ones(mat.nnz, dtype=bool)
    update_mask[fvs_data_mask] = rng.random(np.sum(fvs_data_mask)) > dropout_FVS

    return update_mask, list(map(int, fvs_node_set))





def FVS_random_mask_proration(
    hypergraph: IncidenceHypergraph,
    message_matrix=None,
    dropout_FVS: float = 0.5,
    seed: int | None = None,
    min_keep: int = 2,
    fvs_node_set: list[int] | np.ndarray | None = None,
) -> tuple[IncidenceHypergraph, np.ndarray, np.ndarray]:
    """
    Construct a randomized sub-hypergraph around FVS nodes.

    Rules:
        1. Non-FVS node incidences are kept.
        2. FVS node incidences are kept with probability ``dropout_FVS``.
        3. For each FVS node, at least ``min_keep`` incident hyperedges are kept.
        4. If an FVS node degree is <= ``min_keep``, all its incidences are kept.
        5. Hyperedges whose size becomes < 2 after masking are removed.

    ``message_matrix`` is accepted for backward compatibility but is not used to
    construct the sub-hypergraph. The returned ``update_mask`` is aligned with
    the CSC incidence matrix data order.
    """
    if not 0 <= dropout_FVS <= 1:
        raise ValueError("dropout_FVS must be in [0, 1].")
    if min_keep < 0:
        raise ValueError("min_keep must be non-negative.")

    rng = np.random.default_rng(seed)

    if fvs_node_set is None:
        fvs_node_set = FcoreHD_FVS(
            hypergraph,
            random_tie=True,
            seed=seed,
        ).fvs_nodes
    fvs_node_set = np.asarray(fvs_node_set, dtype=int)

    incidence = sparse.csc_array(hypergraph.get_binary_incidence_matrix())
    if not incidence.has_sorted_indices:
        incidence.sort_indices()

    update_mask = np.ones(incidence.nnz, dtype=bool)

    for v in fvs_node_set:
        pos = np.flatnonzero(incidence.indices == v)
        degree_v = len(pos)

        if degree_v == 0:
            continue

        if degree_v <= min_keep:
            update_mask[pos] = True
            continue

        local_keep = rng.random(degree_v) > dropout_FVS
        num_keep = int(local_keep.sum())

        if num_keep < min_keep:
            add_idx = rng.choice(
                np.flatnonzero(~local_keep),
                size=min_keep - num_keep,
                replace=False,
            )
            local_keep[add_idx] = True

        update_mask[pos] = local_keep

    masked_incidence = incidence.copy()
    masked_incidence.data = masked_incidence.data * update_mask
    masked_incidence.eliminate_zeros()

    hye_sizes = np.asarray(masked_incidence.sum(axis=0)).ravel()
    keep_hyes = np.flatnonzero(hye_sizes >= 2)
    sub_incidence = masked_incidence[:, keep_hyes]

    if sub_incidence.shape[1] == 0:
        raise ValueError("No hyperedges of size >= 2 remain after FVS sampling.")

    return IncidenceHypergraph(sub_incidence), fvs_node_set, update_mask




def _is_acyclic_after_removing_nodes(
    incidence_csc: sparse.csc_array,
    incidence_csr: sparse.csr_array,
    removed_nodes: set[int],
) -> bool:
    """
    Check whether the factor graph is acyclic after deleting removed_nodes.

    True means acyclic; False means at least one cycle remains.
    """

    n_nodes, n_hyes = incidence_csc.shape

    active_nodes = np.ones(n_nodes, dtype=bool)
    active_hyes = np.ones(n_hyes, dtype=bool)

    if removed_nodes:
        removed = np.asarray(list(removed_nodes), dtype=int)
        removed = removed[(removed >= 0) & (removed < n_nodes)]
        active_nodes[removed] = False

    in_core_nodes, in_core_hyes, _, _ = _factor_graph_2core(
        incidence_csc=incidence_csc,
        incidence_csr=incidence_csr,
        active_nodes=active_nodes,
        active_hyes=active_hyes,
    )

    return not np.any(in_core_nodes) and not np.any(in_core_hyes)


def reinsertion_prune_fvs_from_incidence(
    incidence_csc: sparse.csc_array,
    incidence_csr: sparse.csr_array,
    fvs_nodes: list[int],
    order: str = "reverse",
    seed: int | None = None,
) -> list[int]:
    """
    Postprocess the FVS from FcoreHD by reinsertion.

    Try restoring each FVS node. If the factor graph remains acyclic,
    the node is redundant and can be removed from the FVS.
    """

    fvs_list = list(dict.fromkeys(map(int, fvs_nodes)))
    fvs_set = set(fvs_list)

    if order == "reverse":
        candidates = list(reversed(fvs_list))

    elif order == "random":
        rng = np.random.default_rng(seed)
        candidates = fvs_list.copy()
        rng.shuffle(candidates)

    elif order == "degree":
        node_deg = np.asarray(incidence_csr.getnnz(axis=1)).ravel()
        candidates = sorted(fvs_list, key=lambda v: node_deg[v])

    else:
        raise ValueError(f"Unknown reinsertion order: {order}")

    for v in candidates:
        if v not in fvs_set:
            continue

        trial_fvs = fvs_set - {v}

        if _is_acyclic_after_removing_nodes(
            incidence_csc=incidence_csc,
            incidence_csr=incidence_csr,
            removed_nodes=trial_fvs,
        ):
            fvs_set.remove(v)

    return sorted(fvs_set)