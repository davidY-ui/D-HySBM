"""Multi-order Rust BP variant with soft-posterior community-prior learning."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Iterable

import numpy as np
from scipy import sparse, special

from src.data.representation.incidence_hypergraph import IncidenceHypergraph
from src.model.kappa import compute_C_prime_d,compute_C_third_d
from src.model.numerical import hyperedge_pi, sparse_reduce_lse
from .dynamic_updates import (
    compute_psi_dynamic_programming_loop,
    compute_psi_tilde_dynamic_programming_loop,
)
from .dynamic_updates import (
    compute_psi_dynamic_programming_loop_rust,
    compute_psi_tilde_dynamic_programming_loop_rust,
)
from .fvs_function import FVS_random_mask_proration,FcoreHD_FVS
from itertools import combinations
import time

# Define the type of sparse matrix that is utilized to store the messages during message
# passing. These can be different for messages from hyperedges to nodes and from nodes
# to hyperedges.
TYPE_HYE_TO_NODE: sparse.spmatrix = sparse.csc_array
TYPE_NODE_TO_HYE: sparse.spmatrix = sparse.csc_array

CLIP_MIN: float = -30
CLIP_MAX: float = -1e-15

class SharedCSCMessages:
    """
    Store K communities on one shared CSC sparsity structure.

    shape:
        Original incidence matrix shape (N, E).
    indices:
        Node index for each CSC nonzero; shape (M,).
    indptr:
        Start/end offsets of each hyperedge in data; shape (E+1,).
    data:
        Message values for K communities; shape (K, M).

    M = incidence.nnz.
    """

    def __init__(
        self,
        shape,
        indices,
        indptr,
        K,
        data=None,
    ):
        self.shape = tuple(shape)
        self.N, self.E = self.shape
        self.K = int(K)

        # Note:
        # Avoid copying so node->hye and hye->node
        # share the same indices / indptr arrays.
        self.indices = indices
        self.indptr = indptr

        self.M = len(indices)

        if data is None:
            self.data = np.empty(
                (self.K, self.M),
                dtype=np.float64,
                order="C"
            )
        else:
            data = np.asarray(
                data,
                dtype=np.float64
            )

            if data.shape != (self.K, self.M):
                raise ValueError(
                    f"data shape should be {(self.K, self.M)}, "
                    f"got {data.shape}"
                )

            self.data = np.ascontiguousarray(data)

    @classmethod
    def create_pair(cls, incidence, K):
        """
        Create node->hye and hye->node messages sharing exactly the same
        indices / indptr arrays.
        """

        incidence = sparse.csc_array(incidence)

        if not incidence.has_sorted_indices:
            incidence = incidence.copy()
            incidence.sort_indices()

        # Retain only one copy of the structure.
        indices = np.asarray(incidence.indices)
        indptr = np.asarray(incidence.indptr)

        node_to_hye = cls(
            incidence.shape,
            indices,
            indptr,
            K
        )

        hye_to_node = cls(
            incidence.shape,
            indices,
            indptr,
            K
        )

        return node_to_hye, hye_to_node

    def edge_slice(self, hye):
        """Return the data offset range of hyperedge hye."""
        return slice(
            self.indptr[hye],
            self.indptr[hye + 1]
        )

    def edge_data(self, hye):
        """Return K-dimensional messages for all nodes in a hyperedge; shape (K, d)."""
        s = self.edge_slice(hye)
        return self.data[:, s]

    def row_sum(
        self,
        position_weights=None
    ):

        out = np.empty(
            (self.K, self.N),
            dtype=np.float64
        )

        if position_weights is not None:

            position_weights = np.asarray(
                position_weights,
                dtype=np.float64
            )

            if position_weights.shape != (self.M,):

                raise ValueError(
                    "position_weights has wrong shape: "
                    f"{position_weights.shape}, "
                    f"expected {(self.M,)}"
                )


        for a in range(self.K):

            if position_weights is None:

                weights = self.data[a]

            else:

                weights = (
                    self.data[a]
                    *
                    position_weights
                )

            out[a] = np.bincount(
                self.indices,
                weights=weights,
                minlength=self.N
            )

        return out

    def normalize_log_inplace(self):
        """Log-normalize across K communities at each node-hyperedge incidence position."""

        norm = special.logsumexp(
            self.data,
            axis=0
        )

        self.data -= norm[None, :]

    def clip_inplace(self):
        np.clip(
            self.data,
            CLIP_MIN,
            CLIP_MAX,
            out=self.data
        )

    def as_csc(self, assignment, copy=False):
        """
        Provide compatibility with legacy functions by temporarily wrapping
        the data for community assignment in a CSC matrix.
        Avoid frequent use in the main message-passing loop.
        """

        return sparse.csc_array(
            (
                self.data[assignment],
                self.indices,
                self.indptr
            ),
            shape=self.shape,
            copy=copy
        )





class loop_HypergraphBlockModel:
    """Hypergraph version of the Stochastic Block Model, introduced in

    "Message Passing on Hypergraphs: Detectability, Phase Transitions, and Higher-Order
    Information", Ruggeri et al.


    This probabilistic model for hypergraphs partitions the nodes into K hard
    communities, specified by an array of assignments t. The communities interact
    through a symmetric affinity matrix p, with shape (K, K). Together, the community
    assignments t and the affinity matrix p define the Bernoulli probability of the
    single hyperedges to be observed or not.
    """

    def __init__(
        self,
        n: np.ndarray | None,
        p_list: dict[int, np.ndarray] | None,
        N: int,
        K: int,
        fix_hye_size: list[int] | None,
        selected_hye_sizes: list[int] | None = None,
        shared_hye_sizes: list[list[int]] | None = None,
        omega: dict[int, float] | None = None,
        beta_d: dict[int, float] | None = None,
        share_beta_d: bool = False,
        affinity_update: str = "paper-w",
    ) -> None:
        r"""Stochastic Block Model for Hypergraphs.
        This version of SBM considers, for every node i, hard community assignments
        :math::`t_i`, i.e. categorical assignments to one out of K communities.
        Together with a (K, K) affinity matrix, these two parameters define the
        likelihood for unweighted hypergraphs (i.e. hyperedges have weights in {0, 1}).
        A prior :math::`n=(n_1, \ldots, n_K)` for the community assignments can also be
        specified.

        Parameters 
        ----------
        n: array of prior parameters for the communities.
            If specified, this array is used as initialization for EM inference,
            otherwise it is initialized at random.
            The array has length K equal to the number of communities, and specifies the
            categorical prior probabilities.
        p_list: dictionary of symmetric matrices of community interaction probabilities, keyed by hyperedge size as integers.
            If specified, these matrices are used as initialization for EM inference,
            otherwise they are initialized at random.
            Each matrix has shape (K, K), where K is the number of communities, and
            contains the inter and intra-community interaction probabilities,
            constrained to the [0, 1] interval.
        N: number of nodes.
        K: number of communities.
        fix_hye_size: list of hyperedge sizes to be fixed.
            Notice that this quantity is used to infer probabilistic quantities in the
            model, but is not checked against input hypergraphs.
        selected_hye_sizes: list of hyperedge sizes to be selected.
        shared_hye_sizes: groups sharing the affinity shape W.
            Omitted selected orders are added as singleton groups.
        share_beta_d: if True, also pool the target mean c within each group.
            Pooling uses C_prime_d weights and preserves group intensity before
            clipping. It overrides unequal beta targets within that group.
        affinity_update: "paper-w" keeps the normalized shared W and layer scales.
            "direct-p" learns one complete p per group using the original
            multiplicative update, without W normalization or beta calibration.
        """

        if affinity_update not in ("paper-w", "direct-p"):
            raise ValueError("affinity_update must be 'paper-w' or 'direct-p'.")
        self.affinity_update = affinity_update

        # Model related attributes
        self._check_params(n, p_list, K, N, selected_hye_sizes,fix_hye_size,omega)
        self.n = n.copy() if n is not None else None
        if p_list is not None:
            self.p_list = p_list.copy() 
        else:
            self.p_list = {}
        self.fix_hyper_layer = fix_hye_size if fix_hye_size is not None else []
        self.N = N
        self.K = K
        self.selected_hye_sizes: list[int] = selected_hye_sizes if selected_hye_sizes is not None else list(p_list.keys()) # If omitted, use the orders present in p_list.
        self.max_hye_size: int = np.max(self.selected_hye_sizes)
        # Each selected layer belongs to exactly one group. Omitted layers
        # still need initialization and parameter updates of their own.
        selected = set(self.selected_hye_sizes)
        if len(selected) != len(self.selected_hye_sizes):
            raise ValueError("selected_hye_sizes must not contain duplicates.")
        self.shared_hye_sizes = []
        seen = set()
        for group in shared_hye_sizes if shared_hye_sizes is not None else []:
            values = [group] if np.isscalar(group) else list(group)
            if not values or any(
                isinstance(d, (bool, np.bool_))
                or not isinstance(d, (int, np.integer))
                for d in values
            ):
                raise ValueError("Each shared group must contain integer orders.")
            sizes = [int(d) for d in values]
            if any(d not in selected for d in sizes):
                raise ValueError("Shared groups must use selected orders only.")
            if len(set(sizes)) != len(sizes) or seen.intersection(sizes):
                raise ValueError("Shared groups must not overlap or repeat orders.")
            self.shared_hye_sizes.append(sizes)
            seen.update(sizes)
        self.shared_hye_sizes.extend(
            [int(d)] for d in self.selected_hye_sizes if d not in seen
        )
        self.omega = omega

        self.beta_d = None if beta_d is None else {**beta_d, 2: 1.0}
        self.share_beta_d = share_beta_d

        self.omega_per_message: np.ndarray | None = None

        # Quantities inferred after message passing.
        # log of the messages from hyperedges to nodes. Stored as lists of sparse
        # matrices. For every hyperedge e and node i, the matrix at position a in the
        # list contains the messages from e to i, for community assignment a.
        self.log_hye_to_node: SharedCSCMessages | None = None
        # log of the messages from nodes to hyperedges.
        # They are encoded similarly to the messages above.
        self.log_node_to_hye: SharedCSCMessages | None = None
        # Other quantities, log-marginals and external field
        self.log_marginals: np.ndarray | None = None
        self.external_field: np.ndarray | None = None
        self.c0 : float | None = None
        self.layer_scale: dict[int, float] | None = None
        # Resolved target ratios after optional pooling; beta_d stays the input.
        self.effective_beta_d: dict[int, float] | None = None
        

        # Training diagnostics.
        self.training_iter: int | None = None
        self.n_diff: list[float] = []
        self.c_diff: list[float] = []  
        self.log_marginal_diff: list[list[float]] = []
        self.update_vertex_ratio: float = 1.0
        self.update_message_ratio: float | None = None


        self.em_time_per_iter: list[float] = []
        self.em_total_time: float | None = None
        self.em_stop_reason: str = "max_iter"
        # self.mp_time_per_em_iter: list[float] = []
        # self.mp_stop_reason: list[str] = []

        # Random number generator.
        self.rng: np.random.Generator = np.random.default_rng()

    @property
    def c(self):
        """Return the rescaled affinity matrix c, defined as
        .. math::
            c = N p
        where N is the number of nodes and p the affinity matrix.
        """
        return self.p * self.N
    
    def average_degree_from_G_for_hye_size(
            self,
            hypergraph: IncidenceHypergraph,
            hye_size: int = 2,
    ):
        sub_hyg = hypergraph.hye_sizes_select([hye_size])
        
        # incidence = sub_hyg.get_binary_incidence_matrix()
        # degree_all = incidence.sum(axis=1)
        # average_degree_from_G = degree_all.mean()

        average_degree_from_G = hye_size  *( hye_size -1 ) * sub_hyg.E / sub_hyg.N

        return average_degree_from_G

    def em_inference(
        self,
        hypergraph: IncidenceHypergraph,
        em_iter: int = 20,
        em_thresh: float = 1e-5,
        mp_iter: int = 2000,
        mp_thresh: float = 1e-5,
        mp_patience: int = 50,
        seed: int | None = None,
        dirichlet_alpha: float | None = None,
        dropout: float = 0.99,
        dropout_vertex: float = 0.6,
        dropout_vertex_FVS: float | None = None,
        dropout_vertex_rest: float | None = None,
        dropout_FVS: float = 0,
        warm_p_prior: bool = False,
    ) -> None:
        """Run BP with fixed parameters; learn W or p directly according to affinity_update."""
        em_total_start = time.perf_counter()
        self.em_time_per_iter = []
        self.em_stop_reason = "max_iter"

        if dropout_vertex_FVS is None:
            dropout_vertex_FVS = dropout_vertex
        if dropout_vertex_rest is None:
            dropout_vertex_rest = dropout_vertex
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._check_hypergraph_vs_model_params(hypergraph)

        orders = [int(d) for d in self.selected_hye_sizes]
        fixed_n = self.n is not None
        fixed_p = all(d in self.fix_hyper_layer for d in orders)

        if fixed_n and fixed_p:
            # All generative parameters are known: run BP with fixed n and p, without estimating layer scales from edge counts.
            # beta and shared-W constraints belong to parameter calibration and do not apply in this branch.
            n = np.asarray(self.n, dtype=float)
            if (
                n.shape != (self.K,)
                or np.any(~np.isfinite(n))
                or np.any(n <= 0)
                or not np.isclose(n.sum(), 1.0)
            ):
                raise ValueError("Fixed n must be finite, positive and normalized.")

            matrices = []
            for d in orders:
                if d not in self.p_list:
                    raise ValueError(f"No fixed p is provided for order {d}.")
                p_fixed = np.asarray(self.p_list[d], dtype=float)
                if (
                    p_fixed.shape != (self.K, self.K)
                    or np.any(~np.isfinite(p_fixed))
                    or np.any(p_fixed < 0)
                    or np.any(p_fixed >= 1)
                    or not np.allclose(p_fixed, p_fixed.T)
                ):
                    raise ValueError(
                        f"Fixed p for order {d} must be a finite symmetric "
                        "K x K matrix with entries in [0, 1)."
                    )
                matrices.append(p_fixed)

            self.p = np.array(matrices, dtype=float, copy=True)
            # Clear scale state from earlier calls so it cannot be mistaken for this run's results.
            self.average_degree = None
            self.layer_scale = None
            self.c0 = None
            self.effective_beta_d = None
            logging.info(
                "Fixed n and all p: use supplied parameters directly; "
                "skip layer scales, beta calibration and shared-p synchronization."
            )
        elif self.affinity_update == "direct-p":
            self._initialize_direct_affinities(
                hypergraph, fixed_n=fixed_n, fixed_p=fixed_p,
                warm_p_prior=warm_p_prior,
            )
        else:
            # When parameters are learnable, retain empirical layer scales, beta, and shared-group logic.
            self.average_degree = np.array([
                self.average_degree_from_G_for_hye_size(
                    hypergraph, hye_size=d
                )
                for d in orders
            ], dtype=float)
            c_emp = self.average_degree

            if 2 not in orders:
                raise ValueError(
                    "Order 2 is required as the beta reference."
                )
            if np.any(~np.isfinite(c_emp)) or np.any(c_emp <= 0):
                raise ValueError(
                    "Every selected order needs positive empirical c."
                )

            # Empirical beta for the current training graph
            c2_emp = float(c_emp[orders.index(2)])
            beta_emp = c_emp / c2_emp

            # None first uses empirical layer scales; share_beta_d then pools them by group.
            # Preserve the user-supplied self.beta_d value of None.
            beta_target = beta_emp.copy()

            if self.beta_d is not None:
                beta_target = np.array([
                    self.beta_d.get(d, beta_emp[i])
                    for i, d in enumerate(orders)
                ], dtype=float)

            if (
                np.any(~np.isfinite(beta_target))
                or np.any(beta_target <= 0)
            ):
                raise ValueError(
                    "beta_d must be finite and positive."
                )

            # Adjustment factor relative to the empirical scale
            # None or the complete empirical beta both give ratio = 1.
            ratio = beta_target / beta_emp
            chi = np.array([
                compute_C_prime_d(d) for d in orders
            ])

            # Preserve the total expected hyperedge count before clipping.
            normalization = (
                np.dot(chi, c_emp)
                / np.dot(chi, c_emp * ratio)
            )

            # First retain the existing beta calibration and total-intensity rule.
            c_target = c_emp * ratio * normalization

            if self.share_beta_d:
                size_to_idx = {d: i for i, d in enumerate(orders)}
                for group in self.shared_hye_sizes:
                    if len(group) == 1:
                        continue
                    indices = [size_to_idx[d] for d in group]
                    group_chi = chi[indices]
                    c_shared = float(
                        np.dot(group_chi, c_target[indices]) / group_chi.sum()
                    )
                    c_target[indices] = c_shared

            # Reference the new order-2 scale without rescaling other layers.
            # A singleton keeps its absolute target c, even if its beta changes.
            self.c0 = float(c_target[orders.index(2)])
            self.effective_beta_d = {
                d: float(c_target[i] / self.c0)
                for i, d in enumerate(orders)
            }
            scales = c_target / self.N
            self.layer_scale = dict(
                zip(orders, scales.tolist())
            )

            logging.info("Layer scales: %s", self.layer_scale)
            logging.info("Effective beta targets: %s", self.effective_beta_d)

            if self.fix_hyper_layer and not fixed_n:
                raise ValueError(
                    "Fixed layers with fixed scales require a fixed n."
                )

            if not fixed_n:
                self._random_init_n()

            if fixed_p:
                self.p = np.array(
                    [self.p_list[d] for d in orders],
                    dtype=float,
                )
            else:
                self._random_init_p_list(warm_p_prior)

            # Apply layer scales once after initialization.
            self._synchronize_shared_p()

        need_fvs = (
            dropout_FVS > 0
            or not np.isclose(
                dropout_vertex_FVS,
                dropout_vertex_rest,
            )
        )

        if need_fvs:
            fvs_info = FcoreHD_FVS(
                hypergraph,
                random_tie=True,
                seed=seed,
            )
            fvs_node_set = fvs_info.fvs_nodes
            fvs_average_degree = fvs_info.fvs_average_degree
        else:
            fvs_node_set = None
            fvs_average_degree = None

        logging.info(
            "FVS nodes: %s, FVS average degree: %s",
            len(fvs_node_set) if fvs_node_set is not None else 0,
            fvs_average_degree,
        )

        size_to_idx = {
            d: i for i, d in enumerate(orders)
        }

        for it in range(em_iter):
            em_iter_start = time.perf_counter()
            logging.info("EM iteration %s", it)

            self.parallel_message_passing(
                hypergraph,
                mp_iter=mp_iter,
                mp_thresh=mp_thresh,
                patience=mp_patience,
                warm_start=True,
                seed=None,
                dirichlet_alpha=dirichlet_alpha,
                dropout=dropout,
                dropout_vertex=dropout_vertex,
                dropout_vertex_FVS=dropout_vertex_FVS,
                dropout_vertex_rest=dropout_vertex_rest,
                dropout_FVS=dropout_FVS,
                fvs_node_set=fvs_node_set,
            )

            if not fixed_n:
                old_n = self.n.copy()
                # Both affinity modes learn community proportions from soft posteriors.
                new_n = self.updated_community_prior_soft()
                self.n = new_n / new_n.sum()

                self.n_diff.append(
                    np.abs(old_n - self.n).sum()
                    / old_n.sum()
                )

            if not fixed_p:
                old_c = self.c.copy()

                for group in self.shared_hye_sizes:
                    sizes = (
                        [int(group)]
                        if np.isscalar(group)
                        else list(group)
                    )

                    shared_idx = [
                        size_to_idx[int(d)]
                        for d in sizes
                        if d not in self.fix_hyper_layer
                    ]

                    if shared_idx:
                        self.updated_affinity_matrix_list(
                            hypergraph,
                            shared_idx=shared_idx,
                        )

                # Each group has already been updated and clipped.
                # Do not call _synchronize_shared_p again here.
                changes = [
                    self.c_diff_relative(
                        old_c[i], self.c[i]
                    )
                    for i, d in enumerate(orders)
                    if d not in self.fix_hyper_layer
                    and self.omega[d] > 0
                ]

                self.c_diff.append(
                    float(np.mean(changes))
                    if changes else 0.0
                )

            # Refresh the external field from updated parameters for both input forms.
            if self.log_marginals is not None:
                self.external_field = (
                    self.compute_external_field_list()
                )

            self.training_iter = it + 1
            self.em_time_per_iter.append(
                time.perf_counter() - em_iter_start
            )

            if not fixed_n or not fixed_p:
                param_diff = (
                    (self.n_diff[-1] if not fixed_n else 0.0)
                    + (self.c_diff[-1] if not fixed_p else 0.0)
                )

                logging.info(
                    "EM parameter difference: %s",
                    param_diff,
                )

                if param_diff <= em_thresh:
                    self.em_stop_reason = "em_thresh"
                    break

        self.em_total_time = (
            time.perf_counter() - em_total_start
        )

        logging.info(
            "EM stopped by %s, total time = %.4f s",
            self.em_stop_reason,
            self.em_total_time,
        )




    def parallel_message_passing(
        self,
        hypergraph: IncidenceHypergraph,
        mp_iter: int = 2000,
        mp_thresh: float = 1.0e-5,
        dirichlet_alpha: float | None = None,
        dropout: float = 0.99,
        dropout_vertex: float = 0.6,
        dropout_vertex_FVS: float | None = None,
        dropout_vertex_rest: float | None = None,
        patience: int = 50,
        seed: int | None = None,
        warm_start: bool = True,
        dropout_FVS: float = 0,
        fvs_node_set: list[int] | None = None,
    ) -> None:
        """Perform message passing inference of the node assignments.

        Parameters
        ----------
        hypergraph: a hypergraph.
        mp_iter: maximum number of message passing iterations.
        mp_thresh: threshold for message passing convergence.
            The threshold is computed over the absolute difference of the log-marginals
            between two consecutive iterations.
        dirichlet_alpha: parameter for the Dirichlet distribution.
            Utilized for the initialization of the messages, which are drawn from a
            uniform Dirichlet distribution with parameter alpha.
            If None, alpha is chosen automatically.
        dropout: dropout rate.
            The dropout rate it the number of randomly discarded updates in the messages
            and marginals. At every iteration of message passing, these discarded values
            are kept at the previous iteration value.
        patience: number of steps below the mp_thresh.
            After a number of consecutive iterations, specified by patience, with an
            absolute change in log-marginals below the mp_thresh, the message passing
            procedure is stopped.
        seed: random seed.
        warm_start: whether to re-initialize the messages and marginal beliefs.
        """
        logging.info("\tMessage passing...")
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._check_hypergraph_vs_model_params(hypergraph)

        all_messages_init = (
            self.log_hye_to_node is not None
            and self.log_node_to_hye is not None
            and self.log_marginals is not None
            and self.external_field is not None
        )

        if not warm_start or not all_messages_init:
            alpha = 10.0 * self.K if dirichlet_alpha is None else dirichlet_alpha
            self._init_message_passing(hypergraph, dirichlet_alpha=alpha)
            logging.debug(
                f"\t\tInitialized hye to node:\n"
                f"{self.log_hye_to_node.data[0, :5]}"
            )

            logging.debug(
                f"\t\tInitialized node to hye:\n"
                f"{self.log_node_to_hye.data[0, :5]}"
            )
            logging.debug(f"\t\tInitialized marginals:\n{self.log_marginals[:5]}")
            logging.debug(f"\t\tInitialized external field:\n{self.external_field}")

        # ======================================================
        # Order-dependent omega for every incidence position
        # ======================================================

        edge_sizes = np.diff(
            self.log_hye_to_node.indptr
        ).astype(int)

        omega_per_edge = np.asarray(
            [
                self.omega[int(d)]
                for d in edge_sizes
            ],
            dtype=np.float64
        )

        self.omega_per_message = np.repeat(
            omega_per_edge,
            edge_sizes
        )

        if self.omega_per_message.shape != (
            self.log_hye_to_node.M,
        ):

            raise ValueError(
                "omega_per_message has wrong shape: "
                f"{self.omega_per_message.shape}, "
                f"expected {(self.log_hye_to_node.M,)}"
            )

        self.log_marginal_diff.append(list())
        patience_count = 0
        oscillation_iters = 0
        for i in range(mp_iter):
            ##Subgraph updates
            # if dropout_FVS > 0:
            #     hypergraph,_,_ = FVS_random_mask_proration(
            #         hypergraph,
            #         self.log_node_to_hye[0],
            #         dropout_FVS=dropout_FVS,
            #         seed=None,  # keep the current random number generator unaltered.
            #         fvs_node_set=fvs_node_set,
            #     )
            old_log_marginals = self.log_marginals.copy()

            self._parallel_message_passing_step(hypergraph,fvs_node_set, dropout,dropout_vertex,dropout_vertex_FVS,dropout_vertex_rest, dropout_FVS)
            update_log_marginals_numbers = self.log_marginals.shape[0]*self.update_vertex_ratio if self.update_vertex_ratio > 0 else self.log_marginals.shape[0]
            self.log_marginal_diff[-1].append(
                np.abs(old_log_marginals - self.log_marginals).sum()/update_log_marginals_numbers
            )
            # # Relative error
            # self.log_marginal_diff[-1].append(
            #     np.abs(old_log_marginals - self.log_marginals).sum()/np.abs(old_log_marginals).sum()
            # )
            # if i % 10 == 0 :
            logging.info(
                f"\t\tMP step {i} - difference in log-marginals from previous iter: "
                f"{self.log_marginal_diff[-1][-1]}"
            )

            if np.abs(self.log_marginal_diff[-1][-1]) <= mp_thresh:
                patience_count += 1
            else:
                patience_count = 0

            if patience_count == patience:
                logging.info(
                    "\tMessage passing threshold passed. Message passing terminated."
                )
                break

    def _parallel_message_passing_step(
        self,
        hypergraph: IncidenceHypergraph,
        fvs_node_set: list[int] | None,
        dropout: float = 0.99,
        dropout_vertex: float = 0.6,
        dropout_vertex_FVS: float | None = None,
        dropout_vertex_rest: float | None = None,
        dropout_FVS: float = 0,
    ) -> None:
        """
        Perform one step of message passing.

        SharedCSCMessages:
            data.shape = (K, M)

        M = incidence.nnz
        """

        # ======================================================
        # Basic structure
        # ======================================================

        M = self.log_node_to_hye.M
        indices = self.log_node_to_hye.indices



        # ======================================================
        # 1. node -> hye
        # ======================================================

        omega_per_message = (
            self.omega_per_message
        )

        # ------------------------------------------------------
        # Weighted incoming factor-message sum:
        #
        # sum_f omega_{d(f)} log qhat_{f -> i}
        #
        # shape = (K, N)
        # ------------------------------------------------------

        hye_message_sum = (
            self.log_hye_to_node.row_sum(
                position_weights=omega_per_message
            )
        )


        # shape = (K, M)
        new_node_to_hye = np.empty_like(
            self.log_node_to_hye.data
        )


        for assignment in range(self.K):

            new_node_to_hye[assignment] = (
                hye_message_sum[
                    assignment,
                    indices
                ]
                +
                np.log(
                    self.n[assignment]
                )
                -
                self.external_field[
                    assignment
                ]
                -
                (
                    omega_per_message
                    *
                    self.log_hye_to_node.data[
                        assignment
                    ]
                )
            )


        del hye_message_sum


        # ======================================================
        # node -> hye normalization
        # ======================================================

        norm = special.logsumexp(
            new_node_to_hye,
            axis=0
        )

        new_node_to_hye -= norm[None, :]

        del norm

        np.clip(
            new_node_to_hye,
            CLIP_MIN,
            CLIP_MAX,
            out=new_node_to_hye
        )


        # ======================================================
        # node -> hye update mask
        # ======================================================

        if dropout_FVS > 0:

            _, _, node_message_mask = (
                FVS_random_mask_proration(
                    hypergraph,

                    # compatibility adapter
                    self.log_node_to_hye.as_csc(
                        0,
                        copy=False
                    ),

                    dropout_FVS=dropout_FVS,
                    seed=None,
                    fvs_node_set=fvs_node_set,
                )
            )

            node_message_mask = np.asarray(
                node_message_mask,
                dtype=bool
            )

        elif dropout > 0:

            node_message_mask = (
                self.rng.random(M)
                >= dropout
            )

        else:

            # None updates all positions without allocating an all-True mask of length M.
            node_message_mask = None


        if node_message_mask is not None and node_message_mask.shape != (M,):
            raise ValueError(
                "node_message_mask has wrong shape: "
                f"{node_message_mask.shape}, "
                f"expected {(M,)}"
            )


        # ======================================================
        # Write back node -> hye.
        # ======================================================

        if node_message_mask is None:
            # Write back contiguous message blocks to avoid per-community Boolean indexing and temporary copies.
            self.log_node_to_hye.data[:] = new_node_to_hye
            updated_node_num = M
        else:
            for assignment in range(self.K):
                self.log_node_to_hye.data[
                    assignment, node_message_mask
                ] = new_node_to_hye[assignment, node_message_mask]
            updated_node_num = np.count_nonzero(node_message_mask)

        del new_node_to_hye


        logging.debug(
            f"\t\tUpdated node to hye:\n"
            f"{self.log_node_to_hye.data[0, :5]}"
        )


        # ======================================================
        # 2. hye -> node update mask
        # ======================================================

        if dropout_FVS > 0:

            # Reuse the incidence-position mask in FVS mode.
            hye_message_mask = node_message_mask

        elif dropout > 0:

            hye_message_mask = (
                self.rng.random(M)
                >= dropout
            )

        else:

            # The original Rust interface accepts mask=None to compute every incidence position.
            hye_message_mask = None


        # ======================================================
        # Rust hye -> node
        #
        # positions.shape = (R,)
        # new_hye_to_node.shape = (K, R)
        # ======================================================

        positions, new_hye_to_node = (
            compute_psi_dynamic_programming_loop_rust(
                hypergraph=hypergraph,
                model=self,
                mask=hye_message_mask,
            )
        )


        if new_hye_to_node.shape != (
            self.K,
            positions.size,
        ):
            raise ValueError(
                "Unexpected Rust output shape: "
                f"{new_hye_to_node.shape}, "
                f"expected "
                f"{(self.K, positions.size)}"
            )


        # ======================================================
        # hye -> node normalization
        # ======================================================

        norm = special.logsumexp(
            new_hye_to_node,
            axis=0
        )

        new_hye_to_node -= norm[None, :]

        del norm

        np.clip(
            new_hye_to_node,
            CLIP_MIN,
            CLIP_MAX,
            out=new_hye_to_node
        )


        # ======================================================
        # Write directly to SharedCSCMessages.
        # ======================================================

        # M strictly increasing integers starting at 0 and ending at M-1 form the full CSC position sequence.
        # Validate order before block writeback; scatter reordered or partial results by positions.
        full_ordered_output = (
            hye_message_mask is None
            and positions.size == M
            and (
                M == 0
                or (
                    positions[0] == 0
                    and positions[-1] == M - 1
                    and np.all(positions[1:] > positions[:-1])
                )
            )
        )
        if full_ordered_output:
            self.log_hye_to_node.data[:] = new_hye_to_node
        else:
            for assignment in range(self.K):
                self.log_hye_to_node.data[
                    assignment, positions
                ] = new_hye_to_node[assignment]


        updated_hye_num = positions.size


        self.update_message_ratio = (
            updated_node_num
            +
            updated_hye_num
        ) / (2.0 * M)


        del new_hye_to_node
        del positions


        logging.debug(
            f"\t\tUpdated hye to node:\n"
            f"{self.log_hye_to_node.data[0, :5]}"
        )


        # ======================================================
        # 3. Update marginals
        # ======================================================

        # Select nodes first, then compute only the marginal candidates written back this round.
        # No random calls intervene before the original mask-sampling point, so RNG order is preserved.
        # Call the original method even when dropout=0 to preserve its random-number consumption.
        vertex_mask = (
            self._make_vertex_update_mask(
                dropout_vertex,
                dropout_vertex_FVS,
                dropout_vertex_rest,
                fvs_node_set
            )
        )

        vertex_positions = np.flatnonzero(vertex_mask)
        updated_vertex_num = vertex_positions.size
        if updated_vertex_num:
            # Use slices for full-node updates; otherwise select only columns for updated nodes.
            selected_vertices = (
                slice(None) if updated_vertex_num == self.N else vertex_positions
            )
            hye_message_sum = self.log_hye_to_node.row_sum(
                position_weights=self.omega_per_message
            )
            new_marginals = np.empty(
                (updated_vertex_num, self.K), dtype=np.float64
            )
            for assignment in range(self.K):
                new_marginals[:, assignment] = (
                    hye_message_sum[assignment, selected_vertices]
                    + np.log(self.n[assignment])
                    - self.external_field[assignment]
                )
            del hye_message_sum

            norm = special.logsumexp(new_marginals, axis=1)
            new_marginals -= norm[:, None]
            del norm
            np.clip(
                new_marginals, CLIP_MIN, CLIP_MAX, out=new_marginals
            )
            self.log_marginals[selected_vertices] = new_marginals
            del new_marginals
        # Retain all old marginals when no nodes update; still recompute the external field.


        logging.debug(
            f"\t\tUpdated marginals:\n"
            f"{self.log_marginals[:5]}"
        )

        # 4. Update external field

        self.external_field = (
                self.compute_external_field_list()
            )

        logging.debug(
            f"\t\tUpdated external field:\n"
            f"{self.external_field}"
        )

    #Hard update
    def updated_community_prior(self) -> np.ndarray:
        """Parameter updates for the community priors n during EM inference.

        Returns
        -------
        The updated array of community priors.
        """
        assignments = self.community_assignments()
        comm, counts = np.unique(assignments, return_counts=True)

        n = np.zeros(self.K)
        n[comm] = counts / self.N
        return np.clip(n, a_min=1.0e-20, a_max=1.0)
    ##Soft update of n
    def updated_community_prior_soft(self) -> np.ndarray:
        """Soft EM update for community prior n."""
        log_q = self.log_marginals

        # Ensure each row is a normalized log probability.
        log_q = log_q - special.logsumexp(log_q, axis=1, keepdims=True)

        # q_i(a)
        q = np.exp(log_q)

        # n_a = 1/N sum_i q_i(a)
        n = q.mean(axis=0)
        # n_floor = max(2.0 / self.N, 1e-6)
        n =np.clip(n, a_min=1.0e-20, a_max=1.0)
        n = n / n.sum()

        return n

##
    def community_assignments(self):
        marginals = self.log_marginals
        return np.argmax(marginals, axis=1)
##
    def updated_affinity_matrix_list(
        self,
        hypergraph: IncidenceHypergraph,
        shared_idx: list[int],
    ) -> None:
        """Learn shared W or a complete common p within each group, according to the selected mode."""
        if len(shared_idx) == 0:
            return

        if self.affinity_update == "direct-p":
            self._updated_direct_affinity_matrix(hypergraph, shared_idx)
            return

        if self.layer_scale is None:
            raise RuntimeError(
                "Initialize layer_scale in em_inference first."
            )

        n = np.asarray(self.n, dtype=float)
        if np.any(~np.isfinite(n)) or np.any(n <= 0):
            raise ValueError(
                "n must be finite and positive."
            )

        R = np.zeros((self.K, self.K), dtype=float)

        for idx in shared_idx:
            pi, interactions = self.hye_pi_message_update(
                hypergraph,
                idx,
                return_interactions=True,
            )

            if np.any(~np.isfinite(pi)) or np.any(pi <= 0):
                raise ValueError(
                    "Observed hyperedges need positive finite pi."
                )

            # Each layer uses its own old p.
            R += self.p[idx] * np.einsum(
                "eab,e->ab",
                interactions,
                1.0 / pi,
            )

        shape = (
            R / np.outer(n, n)
            if R.sum() > 0
            else None
        )

        self.apply_beta_scaling(
            shared_idx,
            shape_matrix=shape,
        )


    def _updated_direct_affinity_matrix(self, hypergraph, shared_idx):
        """Update full shared p directly, without normalizing W or resetting mean intensity by order."""
        shared_idx = list(shared_idx)
        if not shared_idx:
            return
        if any(self.selected_hye_sizes[i] in self.fix_hyper_layer for i in shared_idx):
            raise ValueError("Fixed layers must not enter a direct-p parameter update.")
        n = np.asarray(self.n, dtype=float)
        if (n.shape != (self.K,) or np.any(~np.isfinite(n))
                or np.any(n <= 0) or not np.isclose(n.sum(), 1.0)):
            raise ValueError("n must be finite, positive and normalized.")
        p_old = np.array(self.p[shared_idx[0]], dtype=float, copy=True)
        if (p_old.shape != (self.K, self.K) or np.any(~np.isfinite(p_old))
                or np.any(p_old < 0) or np.any(p_old >= 1)
                or not np.allclose(p_old, p_old.T)):
            raise ValueError("The common direct-p matrix must be finite and symmetric in [0,1).")
        if any(not np.allclose(self.p[i], p_old, rtol=1e-12, atol=1e-15)
               for i in shared_idx):
            raise ValueError("Direct-p updates require identical shared p matrices first.")

        numerator = np.zeros((self.K, self.K), dtype=float)
        for i in shared_idx:
            pi, interactions = self.hye_pi_message_update(
                hypergraph, i, return_interactions=True)
            if np.any(~np.isfinite(pi)) or np.any(pi < 0):
                raise ValueError("Hyperedge pi must be finite and nonnegative.")
            numerator += np.einsum("eab,e->ab", interactions,
                                   1.0 / np.clip(pi, 1e-20, None))
        exposure = sum(compute_C_prime_d(self.selected_hye_sizes[i])
                       for i in shared_idx)
        denominator = self.N ** 2 * exposure * np.outer(n, n)
        p_new = np.clip(2.0 * p_old * numerator / denominator, 1e-20, .99)
        if np.any(~np.isfinite(p_new)):
            raise ValueError("Direct-p update produced nonfinite probabilities.")
        for i in shared_idx:
            self.p[i] = p_new.copy()
        if self.log_marginals is not None:
            self.external_field = self.compute_external_field_list()


    def apply_beta_scaling(
        self,
        shared_idx,
        *,
        shape_matrix=None,
    ):
        """
        Normalize the shared shape W, then generate p using each layer scale.
        Clip W directly at its bound without compensation or renormalization.
        Clipping may leave the realized mean intensity below target.
        """
        shared_idx = list(shared_idx)

        if not shared_idx:
            return

        if self.layer_scale is None:
            raise RuntimeError(
                "Initialize layer_scale in em_inference first."
            )

        n = np.asarray(self.n, dtype=float)

        if (
            np.any(~np.isfinite(n))
            or np.any(n <= 0)
            or not np.isclose(n.sum(), 1.0)
        ):
            raise ValueError(
                "n must be positive and normalized."
            )

        fixed = [
            self.selected_hye_sizes[i] in self.fix_hyper_layer
            for i in shared_idx
        ]

        if any(fixed) and not all(fixed):
            raise ValueError(
                "Fixed and learnable layers need separate groups."
            )

        W = np.array(
            self.p[shared_idx[0]]
            if shape_matrix is None else shape_matrix,
            dtype=float,
            copy=True,
        )

        if (
            W.shape != (self.K, self.K)
            or np.any(~np.isfinite(W))
            or np.any(W < 0)
            or not np.allclose(W, W.T)
        ):
            raise ValueError(
                "W must be finite, nonnegative and symmetric."
            )

        W = (W + W.T) / 2
        mean_W = float(n @ W @ n)

        if not np.isfinite(mean_W) or mean_W <= 0:
            raise ValueError(
                "W must have a positive finite mean."
            )

        W /= mean_W

        p_limit = 1.0 - 1e-12
        scales = {
            i: self.layer_scale[
                int(self.selected_hye_sizes[i])
            ]
            for i in shared_idx
        }

        if any(
            not np.isfinite(s)
            or not 0 < s <= p_limit
            for s in scales.values()
        ):
            raise ValueError(
                "Target mean p must lie in (0, p_limit]."
            )

        # Bound the common W; clip without compensation.
        cap = p_limit / max(scales.values())
        W = np.minimum(W, cap)

        candidates = {
            i: s * W
            for i, s in scales.items()
        }

        for i, p_new in candidates.items():
            if (
                np.any(~np.isfinite(p_new))
                or np.any(p_new < 0)
                or np.any(p_new >= 1)
            ):
                raise ValueError(
                    f"Layer index {i}: p must lie in [0, 1)."
                )

        # Validate fixed matrices without modifying them.
        if all(fixed):
            for i, p_new in candidates.items():
                p_old = self.p[i]

                if (
                    np.any(~np.isfinite(p_old))
                    or np.any(p_old < 0)
                    or np.any(p_old >= 1)
                    or not np.allclose(
                        p_old,
                        p_new,
                        rtol=1e-9,
                        atol=1e-14,
                    )
                ):
                    raise ValueError(
                        "Fixed p is incompatible with layer scales."
                    )
            return

        for i, p_new in candidates.items():
            self.p[i] = p_new

        if self.log_marginals is not None:
            self.external_field = (
                self.compute_external_field_list()
            )



##
    def compute_external_field_list(
        self,
        use_omega: bool = True,
    ) -> np.ndarray:
        """Compute the external field using community masses shared by all layers."""
        # Preserve existing behavior: use_omega does not select a branch; self.omega still supplies weights.
        weights = [float(self.omega[int(d)]) for d in self.selected_hye_sizes]
        external_field_ = np.zeros(self.K, dtype=np.float64)
        if not any(weights):
            return external_field_

        # m[b] = sum_i q_i(b): traverse nodes once, avoiding an N x K x K array per layer.
        # Use stored messages without extra normalization to preserve their post-clipping numerical meaning.
        mass = np.exp(self.log_marginals.reshape(self.N, self.K)).sum(axis=0)
        for layer_idx, (hye_size, omega_d) in enumerate(
            zip(self.selected_hye_sizes, weights)
        ):
            if omega_d == 0.0:
                continue
            # Original formula: w_d C'_d / N * c_d @ m; c_d = N p_d cancels N.
            # Read current p and w on every call to avoid stale caches after learning.
            external_field_ += (
                omega_d * compute_C_prime_d(hye_size) * (self.p[layer_idx] @ mass)
            )
        return external_field_

###
    def single_hye_pi_list(self, assignments: Iterable[int], hyperedge_size: int) -> float: 
        r"""Compute the hyperedge unnormalized probability.
        For a hyperedge e and community assignments t, the unnormalized probability is
        given by
        .. math::
            \pi_e := \sum_{i < j \in e} p_{t_i t_j}

        Parameters
        ----------
        assignments: community assignments.
            This array contains the community assignments :math::`t_i` (with values
            between 0 and K-1, where K is the number of communities) for all nodes i in
            the hyperedge.

        Returns
        -------
        The value of :math::`\pi_e`.
        """
        K = self.K
        hye_comm_counts = [0] * K
        counts = Counter(assignments)
        for comm, count in counts.items():
            hye_comm_counts[comm] = count

        return hyperedge_pi(hye_comm_counts, self.p_list[hyperedge_size]) 
##
    def hye_pi(
        self, hypergraph: IncidenceHypergraph, d : int, return_interactions: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        r"""Compute the hyperedge unnormalized probabilities for all the hyperedges in
        the hypergraph. For a hyperedge e, the unnormalized probability has form
        .. math::
             \pi_e := \sum_{i <j \in e} p_{t_i t_j}
        with p affinity matrix and :math::`t_i` community assignment of node i.

        Parameters
        ----------
        hypergraph: the input hypergraph.
        d: the index of the hyperedge size.
        return_interactions: whether to optionally return the tensor of community
            interactions within hyperedges, defined as, for any hyperedge e and
            communities a, b:
            .. math::
                \#_{ab}^{(e)} := \sum_{i <j \in e} \delta_{t_i a} \delta_{t_j b}
            where :math::`\delta_{xy}` is the Dirac delta, equal to 1 if :math::`x=y`,
            else 0.
            The tensor :math::`\#` has shape (E, K, K), with E number of hyperedges and
            K number of communities.
        Returns
        -------
        The array of :math::`\pi_e` values. Optionally, the tensor of :math::`\#`
        values.
        """
        E = hypergraph.E
        K = self.K
        p = self.p[d]
        incidence = hypergraph.get_binary_incidence_matrix()

        onehot_assignments = np.zeros((self.N, K))
        onehot_assignments[np.arange(self.N), self.community_assignments()] = 1

        counts = incidence.transpose() @ onehot_assignments
        assert counts.shape == (E, K)
        del onehot_assignments

        interactions = counts.reshape(E, 1, K) * counts.reshape(E, K, 1)
        interactions[:, np.arange(K), np.arange(K)] = counts * (counts - 1) / 2
        assert interactions.shape == (E, K, K)
        del counts

        pi = 0.5 * (
            np.sum(interactions * p.reshape(1, K, K), axis=(1, 2))
            + np.inner(interactions[:, np.arange(K), np.arange(K)], np.diagonal(p))
        )

        if return_interactions:
            return pi, interactions
        return pi
    
##Exact update
    def hye_pi_message_update(
        self,
        hypergraph: IncidenceHypergraph,
        d: int,
        return_interactions: bool = False,
    ) -> np.ndarray:
        r"""
        Message-based soft numerator for EM update.
        For a hyperedge e and communities a, b, define the soft pair count as
        .. math::   

            interactions[hye][a,b] +=
                sum_{i<j in e} soft_pair_count_{ab}^{ij} / Z_e

        This function returns the interactions without multiplying by p.
        The outside update should use:

            p_new = p_old * 2 * numerator / denominator

        """
        hye_size = self.selected_hye_sizes[d]
        K = self.K
        p = self.p[d]

        indptr = self.log_node_to_hye.indptr

        hye_sizes = np.diff(indptr)
        hyes = np.flatnonzero(hye_sizes == hye_size)
        sub_E = len(hyes)

        interactions = np.zeros((sub_E, K, K), dtype=float)

        for hye_idx, hye in enumerate(hyes):
            start = indptr[hye]
            end = indptr[hye + 1]
            size = end - start

            # Usually unnecessary because hyes has already been filtered.
            if size != hye_size:
                raise ValueError(
                    f"Hyperedge {hye} has size {size}, expected {hye_size}."
                )

            log_messages = (
                self.log_node_to_hye.data[
                    :,
                    start:end
                ].T
            )

            assert log_messages.shape == (
                size,
                K
            )
            # log-message -> probability-message
            log_messages = log_messages - special.logsumexp(
                log_messages, axis=1, keepdims=True
            )
            messages = np.exp(log_messages)
            #undirect
            # Use equivalent nonnegative accumulation to avoid cancellation error.
            prefix = np.zeros_like(messages)
            np.cumsum(messages[:-1], axis=0, out=prefix[1:])
            pair_sum = messages.T @ prefix
            pair_num = 0.5 * (pair_sum + pair_sum.T)
            # pair_num[np.diag_indices(K)] *= 0.5

            # #direct
            # suffix_sum = np.cumsum(messages[::-1], axis=0)[::-1] - messages
            # pair_num = messages.T @ suffix_sum

            #Slow implementation
            # pair_num = np.zeros_like(p, dtype=float)
            # # Ze = 0.0

            # for i, j in combinations(range(size), 2):
            #     qi = messages[i]
            #     qj = messages[j]

            #     # Undirected hypergraph: symmetrize (a,b) and (b,a).
            #     pair_num += np.outer(qi, qj)
            #     #Ze += float(qi @ p @ qj)
            #Ze = max(Ze, eps)
            #numerator += pair_num / Ze
            interactions[hye_idx] = pair_num

        assert interactions.shape == (sub_E, self.K, self.K)

        # pi = 0.5 * (
        #     np.sum(interactions * p.reshape(1, K, K), axis=(1, 2))
        #     + np.inner(interactions[:, np.arange(K), np.arange(K)], np.diagonal(p))
        # )

        pi = np.einsum("eab,ab->e", interactions, p)

        if return_interactions:
            return pi, interactions
        return pi

##
    def free_energy(self, hypergraph: IncidenceHypergraph) -> float:
        """Compute the free energy of a hypergraph utilizing the message passing
        cavity approximations. The free energy, often denoted as :math::`F = -log Z`,
        corresponds to the negative log-normalizing constant of the Boltzmann
        distribution. Z is also called the evidence of the probabilistic model.

        Parameters
        ----------
        hypergraph: hypergraph.

        Returns
        -------
        The log-likelihood value.
        """
        l = len(self.selected_hye_sizes)
        self._check_hypergraph_vs_model_params(hypergraph)
        K = self.K
        N = self.N
        external_field = self.compute_external_field_list()

        log_marginals = self.log_marginals
        # hye_dims = hypergraph.get_binary_incidence_matrix().sum(axis=0)

        # Use the same CSC ordering as the message positions returned by Rust.
        hye_dims = np.diff(self.log_node_to_hye.indptr).astype(int)

        # Effective weight of each hyperedge
        omega_edge = np.asarray(
            [self.omega[int(d)] for d in hye_dims],
            dtype=np.float64,
        )

        if np.any(~np.isfinite(omega_edge)) or np.any(omega_edge < 0):
            raise ValueError("Layer weights must be finite and nonnegative.")

        # Weight of each node-hyperedge incidence position
        omega_position = np.repeat(omega_edge, hye_dims)

        # ======================================================
        # Node-related addends
        # ======================================================

        positions, psi_vals = (
            compute_psi_dynamic_programming_loop_rust(
                hypergraph=hypergraph,
                model=self,
                mask=None,
            )
        )


        node_idx = (
            self.log_node_to_hye.indices[
                positions
            ]
        )


        # shape = (N, K)
        f_i = np.empty(
            (N, K),
            dtype=np.float64
        )


        # for k in range(K):

        #     f_i[:, k] = (
        #         np.bincount(
        #             node_idx,
        #             weights=psi_vals[k],
        #             minlength=N,
        #         )
        #         -
        #         external_field[k]
        #     )

        # positions contains the message offsets returned by this Rust call.
        omega_message = omega_position[positions]
        active_message = omega_message > 0

        for k in range(K):
            f_i[:, k] = (
                np.bincount(
                    node_idx[active_message],
                    weights=(
                        omega_message[active_message]
                        * psi_vals[k, active_message]
                    ),
                    minlength=N,
                )
                - external_field[k]
            )


        assert f_i.shape == (
            N,
            K
        )


        del positions
        del psi_vals
        del node_idx


        f_i = special.logsumexp(
            a=f_i,
            b=self.n.reshape(1, -1),
            axis=1
        )

        f_i_sum = f_i.sum()

        # Edge-related addends.
        # First addend.
        # first_addend = compute_psi_tilde_dynamic_programming_loop_rust(
        #     hypergraph=hypergraph, model=self
        # )
        # first_addend = ((hye_dims - 1) * first_addend).sum()

        log_z_edge = compute_psi_tilde_dynamic_programming_loop_rust(
                hypergraph=hypergraph,
                model=self,
            )

        active_edge = omega_edge > 0

        first_addend = np.sum(
            omega_edge[active_edge]
            * (hye_dims[active_edge] - 1)
            * log_z_edge[active_edge]
        )

        # Second addend.
        log_marginal_sum = special.logsumexp(log_marginals, axis=0)
        cross_log_marginal_sum = log_marginal_sum.reshape(
            (1, K)
        ) + log_marginal_sum.reshape((K, 1))
        assert cross_log_marginal_sum.shape == (K, K)

        cross_log_marginals = log_marginals.reshape((N, 1, K)) + log_marginals.reshape(
            (N, K, 1)
        )
        assert cross_log_marginals.shape == (N, K, K)
        cross_log_marginals = special.logsumexp(cross_log_marginals, axis=0)

        second_addend = 0
        # for i in range(l):
        #     second_addend_ = special.logsumexp(
        #         a=np.hstack([cross_log_marginal_sum, cross_log_marginals]),
        #         b=np.hstack([self.c[i], -self.c[i]]),
        #     )
        #     second_addend_ = np.exp(second_addend_)
        #     second_addend_ *= compute_C_third_d(self.selected_hye_sizes[i]) / (2 * N)
        #     second_addend += second_addend_

        for i, d in enumerate(self.selected_hye_sizes):
            omega_d = float(self.omega[int(d)])

            if omega_d == 0.0:
                continue

            log_term = special.logsumexp(
                a=np.hstack([
                    cross_log_marginal_sum,
                    cross_log_marginals,
                ]),
                b=np.hstack([
                    self.c[i],
                    -self.c[i],
                ]),
            )

            second_addend += (
                omega_d
                * np.exp(log_term)
                * compute_C_third_d(d)
                / (2 * N)
            )


        f_e_sum = first_addend + second_addend

        return -f_i_sum + f_e_sum


##
    @staticmethod
    def _check_params(
        n: np.ndarray, p: dict[int, np.ndarray], K: int, N: int,  selected_hye_sizes: list[int] | None, fix_hye_size: list[int] | None,omega:dict[int, float] | None = None
    ) -> None:
        """Check the correctness of the initialization parameters."""
        # Check coherence between n and p.
        if n is not None:
            if not np.allclose(n.sum(), 1):
                raise ValueError(
                    "The prior parameters n for the community distribution do not "
                    "sum to 1."
                )
            if np.any(n < 0):
                raise ValueError(
                    "The prior parameters n for the community distribution contain "
                    "negative values."
                )
            if len(n.shape) != 1:
                raise ValueError(
                    "The array of prior parameters n is not one-dimensional."
                )
            if n.shape != (K,):
                raise ValueError(
                    "The array of prior parameters n has dimension different from the "
                    "number of communities K."
                )

        if p is not None:
            # if len(p) != len(selected_hye_sizes):
            #     raise ValueError(
            #         "The length of the list p does not match the number of selected hyperedge sizes."
            #     )

            for p_ in p.values():
                if not np.all(p_ == p_.T):
                    raise ValueError("The probability matrix p is not symmetric.")

                if np.any(p_ > 1) or np.any(p_ < 0):
                    raise ValueError(
                        "The probability matrix p contains values outside "
                        "the (0, 1) interval."
                    )

                if p_.shape != (K, K):
                    raise ValueError("The matrix p has shape different from (K, K).")

        if p is not None and n is not None:
            # if len(p) != len(selected_hye_sizes):
            #     raise ValueError(
            #         "The length of the list p does not match the number of selected hyperedge sizes."
            #     )
            if not all(p_.shape == (K, K) for p_ in p.values()):
                raise ValueError(
                    "The shapes of n and p do not match. They need to be respectively "
                    "(K,) and (K, K) for some integer K."
                )
            
        if K < 2:
            raise ValueError("The number of communities K cannot be lower than 2.")

        # Check coherence between N and selected_hye_sizes.
        
        if selected_hye_sizes is not None:
            if any(size < 2 for size in selected_hye_sizes):
                raise ValueError("All selected hyperedge sizes must be at least 2.")
            if any(size > N for size in selected_hye_sizes):
                raise ValueError(
                    "All selected hyperedge sizes must be at most the number of nodes N."
                )
            
        if fix_hye_size is not None:
            if  not set(fix_hye_size).issubset(set(selected_hye_sizes)):
                print(f"fix_hye_size: {fix_hye_size}, selected_hye_sizes: {selected_hye_sizes}")
                raise ValueError(
                    "fix_hye_size must be a subset of selected_hye_sizes."
                )

        if any(value < 0 for value in omega.values()):

                raise ValueError(
                    "All omega_d must be non-negative."
                )
##
    def _check_hypergraph_vs_model_params(
        self, hypergraph: IncidenceHypergraph
    ) -> None:
        """Check that the model parameters are coherent with an input hypergraph."""
        if hypergraph.N != self.N:
            raise ValueError(
                "The input hypergraph has a different number of nodes "
                "than the value specified for the model."
            )

        if hypergraph.max_hye_size > self.max_hye_size:
            raise ValueError(
                "The input hypergraph contains hyperedges bigger than the max_hye_size "
                "specified in the model."
            )
        if any(size not in self.selected_hye_sizes for size in hypergraph.selected_hye_sizes):
            raise ValueError(
                "The input hypergraph contains hyperedges of size "
                f"{hypergraph.selected_hye_sizes}, which is not included in the list of selected hyperedge sizes."
            )
##
    def _random_init_n(self) -> None:
        """Random initialization of the community priors n."""
        self.n = self.rng.dirichlet(alpha=[100] * self.K)
##
    def _random_init_p_list(
        self,
        warm_p_prior: bool,
    ) -> None:
        """Initialize parameters for the current mode; paper-w applies layer scales afterward."""
        if self.affinity_update == "direct-p":
            self._random_init_direct_p_list(warm_p_prior)
            return
        K, N = self.K, self.N
        orders = [
            int(d) for d in self.selected_hye_sizes
        ]

        if self.layer_scale is None:
            raise RuntimeError(
                "Initialize layer_scale in em_inference first."
            )

        if warm_p_prior:
            p = np.empty(
                (len(orders), K, K),
                dtype=float,
            )

            for i, d in enumerate(orders):
                prior = self.p_list[d]
                eps = (
                    prior[0, 1] / prior[0, 0]
                    if prior[0, 0] != 0
                    else 0
                )

                scale = N * self.layer_scale[d]
                p_in = scale / (
                    N * (
                        eps
                        + (1.0 - eps) * np.sum(self.n ** 2)
                    )
                )
                p_out = eps * p_in

                p[i] = (
                    p_out * np.ones((K, K))
                    + (p_in - p_out) * np.eye(K)
                )

        else:
            p_s = np.ones((K, K)) / (10 * (K - 1))
            p_s += self.rng.random((K, K)) / 50
            p_s = (
                np.triu(p_s, 1)
                + np.triu(p_s, 1).T
            )

            np.fill_diagonal(
                p_s,
                1.0 + self.rng.random(K) / 50,
            )

            p_s = np.clip(
                p_s / N,
                1e-10,
                1.0,
            )
            p = np.repeat(
                p_s[None, :, :],
                len(orders),
                axis=0,
            )

        for d in self.fix_hyper_layer:
            if d not in orders:
                raise ValueError(
                    f"Fixed order {d} is not selected."
                )
            if d not in self.p_list:
                raise KeyError(
                    f"No fixed p is provided for order {d}."
                )

            p_fixed = np.asarray(
                self.p_list[d],
                dtype=float,
            )

            if (
                p_fixed.shape != (K, K)
                or not np.allclose(p_fixed, p_fixed.T)
            ):
                raise ValueError(
                    "Fixed p must be a symmetric K x K matrix."
                )

            p[orders.index(d)] = p_fixed.copy()

        self.p = p



##
    def _initialize_direct_affinities(self, hypergraph, *, fixed_n, fixed_p,
                                     warm_p_prior):
        """Direct-p mode uses empirical edge counts only for optional warm initialization, without constraining later intensity."""
        if self.beta_d is not None or self.share_beta_d:
            raise ValueError("direct-p does not use beta_d/share_beta_d; remove beta options.")
        self.layer_scale = None
        self.c0 = None
        self.effective_beta_d = None
        self.average_degree = np.array([
            self.average_degree_from_G_for_hye_size(hypergraph, hye_size=int(d))
            for d in self.selected_hye_sizes
        ], dtype=float)
        if not fixed_n:
            self._random_init_n()
        n = np.asarray(self.n, dtype=float)
        if (n.shape != (self.K,) or np.any(~np.isfinite(n))
                or np.any(n <= 0) or not np.isclose(n.sum(), 1.0)):
            raise ValueError("n must be finite, positive and normalized.")
        if fixed_p:
            # n may be learned, but all supplied p values must remain unchanged; do not force them equal.
            self.p = np.array([self.p_list[int(d)] for d in self.selected_hye_sizes],
                              dtype=float, copy=True)
            for matrix in self.p:
                self._validate_direct_fixed_p(matrix)
        else:
            self._random_init_p_list(warm_p_prior)
            self._synchronize_shared_p()
        logging.info("Direct grouped p update; no W/beta calibration or layer-scale constraints.")

    def _validate_direct_fixed_p(self, matrix):
        if (matrix.shape != (self.K, self.K) or np.any(~np.isfinite(matrix))
                or np.any(matrix < 0) or np.any(matrix >= 1)
                or not np.allclose(matrix, matrix.T)):
            raise ValueError("Fixed p must be a finite symmetric K x K matrix in [0,1).")

    def _random_init_direct_p_list(self, warm_p_prior):
        """Initialize full p directly; warm initialization uses the supplied assortativity ratio and empirical intensity."""
        orders = [int(d) for d in self.selected_hye_sizes]
        if warm_p_prior:
            p = np.empty((len(orders), self.K, self.K), dtype=float)
            for i, d in enumerate(orders):
                prior = np.asarray(self.p_list[d], dtype=float)
                eps = prior[0, 1] / prior[0, 0] if prior[0, 0] != 0 else 0.
                denominator = self.N * (eps + (1. - eps) * np.sum(self.n ** 2))
                p_in = self.average_degree[i] / denominator
                p_out = eps * p_in
                p[i] = p_out * np.ones((self.K, self.K)) + (p_in - p_out) * np.eye(self.K)
            p = np.clip(p, 1e-20, .99)
        else:
            p_s = np.ones((self.K, self.K)) / (10 * (self.K - 1))
            p_s += self.rng.random((self.K, self.K)) / 50
            p_s = np.triu(p_s, 1) + np.triu(p_s, 1).T
            np.fill_diagonal(p_s, 1.0 + self.rng.random(self.K) / 50)
            p = np.repeat(np.clip(p_s / self.N, 1e-10, .99)[None, :, :],
                          len(orders), axis=0)
        for d in self.fix_hyper_layer:
            fixed = np.asarray(self.p_list[d], dtype=float)
            self._validate_direct_fixed_p(fixed)
            p[orders.index(d)] = fixed.copy()
        self.p = p


    def _init_message_passing_old(
        self,
        hypergraph: IncidenceHypergraph,
        dirichlet_alpha: float = 10.0,
    ) -> None:
        r"""Random initialization of the messages, marginal beliefs, and external field.
        The initialization is performed to respect the fixed-point conditions given by
        the message passing equations.

        Parameters
        ----------
        hypergraph: a hypergraph.
        dirichlet_alpha: parameter to initialize the messages and marginal beliefs.
            These are drawn from a Dirichlet distribution with a uniform parameter array
            :math::`(\alpha, \ldots, \alpha)` with length the number of communities.
        """
        incidence = hypergraph.get_binary_incidence_matrix()

        def random_prob_init():
            beliefs = [incidence.copy().astype(float) for _ in range(self.K)]
            vals = self.rng.dirichlet(
                [dirichlet_alpha] * len(beliefs), size=len(beliefs[0].data)
            )
            for i, belief in enumerate(beliefs):
                belief.data *= vals[:, i]

            return beliefs

        # Random initialization of messages from nodes to hyperedges.
        log_node_to_hye = random_prob_init()
        for belief in log_node_to_hye:
            belief.data = np.log(belief.data)
        self.log_node_to_hye = [TYPE_NODE_TO_HYE(mat) for mat in log_node_to_hye]

        # Random initialization of the marginal beliefs.
        marginals = self.rng.dirichlet([dirichlet_alpha] * self.K, size=self.N)
        assert marginals.shape == (self.N, self.K)
        self.log_marginals = np.log(marginals)

        # Compute external field from marginals.
        self.external_field = self.compute_external_field_list()

        # Infer hye to node as ratio of marginals and noe to hye
        log_hye_to_node = []
        for assignment in range(self.K):
            log_hye_to_node.append(
                TYPE_HYE_TO_NODE(
                    incidence * self.log_marginals[:, assignment].reshape(self.N, 1)
                )
                - self.log_node_to_hye[assignment]
            )

        normalizer = sparse_reduce_lse(*log_hye_to_node)
        for assignment in range(self.K):
            log_hye_to_node[assignment].data -= normalizer.data
        self.log_hye_to_node = log_hye_to_node


    def _init_message_passing(
        self,
        hypergraph: IncidenceHypergraph,
        dirichlet_alpha: float = 10.0,
    ) -> None:

        incidence = sparse.csc_array(
            hypergraph.get_binary_incidence_matrix()
        )

        if not incidence.has_sorted_indices:
            incidence = incidence.copy()
            incidence.sort_indices()


        # 1. Create both message sets with shared indices / indptr.

        (
            self.log_node_to_hye,
            self.log_hye_to_node
        ) = SharedCSCMessages.create_pair(
            incidence,
            self.K
        )

        M = self.log_node_to_hye.M
        # 2. Initialize node -> hye.
        # vals.shape = (M, K)
        vals = self.rng.dirichlet(
            [dirichlet_alpha] * self.K,
            size=M
        )

        # data.shape = (K, M)
        self.log_node_to_hye.data[:] = np.log(
            vals.T
        )

        # 3. Initialize marginals.
        marginals = self.rng.dirichlet(
            [dirichlet_alpha] * self.K,
            size=self.N
        )

        assert marginals.shape == (
            self.N,
            self.K
        )

        self.log_marginals = np.log(marginals)
        
        # 4. external field
        self.external_field = (
            self.compute_external_field_list()
        )

        # 5. Initialize hye -> node.

        indices = self.log_hye_to_node.indices

        # No need to create K sparse matrices.
        for a in range(self.K):

            self.log_hye_to_node.data[a] = (
                self.log_marginals[
                    indices,
                    a
                ]
                -
                self.log_node_to_hye.data[a]
            )

        # Normalize over the K communities.
        self.log_hye_to_node.normalize_log_inplace()

        self.log_hye_to_node.clip_inplace()
        
    def c_diff_relative(self, c_old, c_new) -> float:
        rel = np.abs(c_new - c_old).sum()/c_old.sum()
        #rel = np.abs(c_new - c_old) / np.maximum(np.abs(c_old), 1e-20)
        return rel
    


    def _synchronize_shared_p(self):
        """Synchronize shared structure at initialization; direct-p only copies full p, without layer scaling."""
        size_to_idx = {
            int(d): i
            for i, d in enumerate(self.selected_hye_sizes)
        }

        for group in self.shared_hye_sizes:
            sizes = (
                [int(group)]
                if np.isscalar(group)
                else list(group)
            )

            indices = [
                size_to_idx[int(d)]
                for d in sizes
            ]

            if self.affinity_update == "direct-p":
                fixed = [self.selected_hye_sizes[i] in self.fix_hyper_layer for i in indices]
                if any(fixed) and not all(fixed):
                    raise ValueError("Fixed and learnable layers need separate shared groups.")
                if not all(fixed):
                    common = self.p[indices[0]].copy()
                    for i in indices:
                        self.p[i] = common.copy()
                continue

            self.apply_beta_scaling(indices)


 
    def _make_vertex_update_mask(
        self,
        dropout_vertex: float = 0.6,
        dropout_vertex_FVS: float | None = None,
        dropout_vertex_rest: float | None = None,
        fvs_node_set: list[int] | set[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Build an FVS-aware mask for node marginal updates.

        True updates a node marginal this round; False retains the old marginal.
        dropout_vertex is the default node dropout probability;
        dropout_vertex_FVS and dropout_vertex_rest apply to FVS and non-FVS nodes.

        Dropout is the fraction left unchanged; update ratio = 1 - dropout.
        """

        # Use dropout_vertex when separate FVS / rest dropout values are unspecified.
        if dropout_vertex_FVS is None:
            dropout_vertex_FVS = dropout_vertex

        if dropout_vertex_rest is None:
            dropout_vertex_rest = dropout_vertex

        # Keep inputs within [0, 1].
        dropout_vertex = float(np.clip(dropout_vertex, 0.0, 1.0))
        dropout_vertex_FVS = float(np.clip(dropout_vertex_FVS, 0.0, 1.0))
        dropout_vertex_rest = float(np.clip(dropout_vertex_rest, 0.0, 1.0))

        # 1. Build the FVS node mask.
        is_fvs_node = np.zeros(self.N, dtype=bool)

        if fvs_node_set is not None and len(fvs_node_set) > 0:
            fvs_nodes = np.asarray(list(fvs_node_set), dtype=int)

            # Guard against out-of-bounds indices.
            fvs_nodes = fvs_nodes[
                (fvs_nodes >= 0) & (fvs_nodes < self.N)
            ]

            is_fvs_node[fvs_nodes] = True

        is_rest_node = ~is_fvs_node

        # 2. Build the node update mask.
        vertex_update_mask = np.zeros(self.N, dtype=bool)

        num_fvs_nodes = np.count_nonzero(is_fvs_node)
        num_rest_nodes = np.count_nonzero(is_rest_node)

        # Use dropout_vertex_FVS for FVS nodes.
        if num_fvs_nodes > 0:
            vertex_update_mask[is_fvs_node] = (
                self.rng.random(num_fvs_nodes) >= dropout_vertex_FVS
            )

        # Use dropout_vertex_rest for non-FVS nodes.
        if num_rest_nodes > 0:
            vertex_update_mask[is_rest_node] = (
                self.rng.random(num_rest_nodes) >= dropout_vertex_rest
            )

        # 3. Record the actual update fraction.
        self.update_vertex_ratio = float(np.mean(vertex_update_mask))


        return vertex_update_mask
