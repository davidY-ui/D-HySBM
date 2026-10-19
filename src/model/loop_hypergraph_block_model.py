"""Infer multi-order hypergraph block models with order-specific affinities."""

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
        """

        # Model related attributes
        self._check_params(n, p_list, K, N, selected_hye_sizes,fix_hye_size)
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

        
        # Quantities inferred after message passing.
        # log of the messages from hyperedges to nodes. Stored as lists of sparse
        # matrices. For every hyperedge e and node i, the matrix at position a in the
        # list contains the messages from e to i, for community assignment a.
        self.log_hye_to_node: list[TYPE_HYE_TO_NODE] | None = None
        # log of the messages from nodes to hyperedges.
        # They are encoded similarly to the messages above.
        self.log_node_to_hye: list[TYPE_NODE_TO_HYE] | None = None
        # Other quantities, log-marginals and external field
        self.log_marginals: np.ndarray | None = None
        self.external_field: np.ndarray | None = None

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
        project: bool = False,
        cancel_oscillation : bool = None
    ) -> None:
        """Perform Expectation Maximization (EM) inference on a hypergraph.
        The inference routine consist of alternating message passing, where the
        community assignments :math::`t_i` are inferred, and updates to the global
        parameters, i.e. the affinity matrix w and community priors n.
        If the affinity w or priors n are provided at initialization of the model, these
        are not inferred, but kept fixed.

        Parameters
        ----------
        hypergraph: hypergraph to perform inference on.
        em_iter: maximum number of EM iterations.
            One iteration consists of the message passing routine plus the global
            parameter updates.
        em_thresh: threshold for EM convergence.
            The threshold is computed over the absolute difference of the community
            priors and the affinity matrix between two consecutive EM iterations.
        mp_iter: maximum number of message passing iterations.
        mp_thresh: threshold for message passing convergence.
            The threshold is computed over the absolute difference of the log-marginals
            between two consecutive iterations.
        mp_patience: number of steps below the mp_thresh.
            After a number of consecutive iterations, specified by patience, with an
            absolute change in log-marginals below the mp_thresh, the message passing
            procedure is stopped.
        seed: random seed.
        dirichlet_alpha: parameter for the Dirichlet distribution.
            Utilized for the initialization of the messages, which are drawn from a
            uniform Dirichlet distribution with parameter alpha.
            If None, alpha is chosen automatically.
        dropout: dropout rate.
            The dropout rate it the number of randomly discarded updates in the messages
            and marginals. At every iteration of message passing, these discarded values
            are kept at the previous iteration value.
        """
        em_total_start = time.perf_counter()
        self.em_time_per_iter = []
        # self.mp_time_per_em_iter = []
        # self.mp_stop_reason = []
        self.em_stop_reason = "max_iter"

        if dropout_vertex_FVS is None:
            dropout_vertex_FVS = dropout_vertex

        if dropout_vertex_rest is None:
            dropout_vertex_rest = dropout_vertex   


        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._check_hypergraph_vs_model_params(hypergraph)
        self.average_degree = np.zeros(len(self.selected_hye_sizes))
        for i in range(len(self.selected_hye_sizes)):
            self.average_degree[i] = self.average_degree_from_G_for_hye_size(hypergraph, hye_size=self.selected_hye_sizes[i])

        if self.n is None:
            fixed_n = False
            self._random_init_n()
            logging.info(f"Initialized n prior:\n{self.n}")
        else:
            fixed_n = True

        if len(self.fix_hyper_layer) < len(self.selected_hye_sizes):
            fixed_p = False
            self._random_init_p_list( warm_p_prior)
            logging.info(f"Initialized rescaled affinity c=N*p:\n{self.c}")
        else:
            fixed_p = True
            p_ = np.zeros((len(self.selected_hye_sizes), self.K, self.K), dtype=float)
            for idx, hye_size in enumerate(self.selected_hye_sizes):
                p_[idx] = self.p_list[hye_size]
            self.p = p_
        fvs_node = FcoreHD_FVS(
            hypergraph,
            random_tie=True,
            seed=seed,
        )
        fvs_node_set = fvs_node.fvs_nodes

        fvs_average_degree =  fvs_node.fvs_average_degree
        logging.info(f"fvs_node_set: {len(fvs_node_set)}, fvs_average_degree: {fvs_average_degree}")
        for it in range(em_iter):
            em_iter_start = time.perf_counter()
            logging.info(f"EM iteration {it}")


            # Local parameters: message passing.
            self.parallel_message_passing(
                hypergraph,
                mp_iter=mp_iter,
                mp_thresh=mp_thresh,
                patience=mp_patience,
                warm_start=True,
                seed=None,  # keep the current random number generator unaltered.
                dirichlet_alpha=dirichlet_alpha,
                dropout=dropout,
                dropout_vertex=dropout_vertex,
                dropout_vertex_FVS = dropout_vertex_FVS,
                dropout_vertex_rest = dropout_vertex_rest,
                dropout_FVS=dropout_FVS,
                fvs_node_set=fvs_node_set,
                cancel_oscillation = cancel_oscillation
            )
            damping = 0.01
            n_damping_update = 0

            # Global parameters: EM updates.
            if not fixed_n or not fixed_p:
                logging.info("\tUpdates of priors n and affinity p...")
            if not fixed_n:
                old_n = self.n.copy()
                #Soft update of n

                new_n = self.updated_community_prior_soft()
                # self.n = alpha_ * self.n + (1 - alpha_) * old_n
                # self.n = self.updated_community_prior()
                self.n_diff.append(np.abs(old_n - new_n).sum()/old_n.sum())
                # self.n_diff.append(np.abs(old_n - self.n).sum())
            else:
                old_n = self.n.copy()
                
            if not fixed_p:
                old_c = self.c.copy()
                c_diff_ = np.zeros(len(old_c))
                for idx, hye_size in enumerate(self.selected_hye_sizes):
                    if self.fix_hyper_layer is not None and hye_size in self.fix_hyper_layer:
                        continue   # Skip update if this hyperedge size is fixed
                    self.p[idx], real_p = self.updated_affinity_matrix_list(hypergraph, hyper_size_idx=idx,project = project)
                    c_diff_[idx] = self.c_diff_relative(old_c[idx],  real_p * self.N)
                    if c_diff_[idx] > 0.0007*self.N and (np.max(new_n) - np.max(old_n)) >= 0.2:
                        self.p[idx] = damping * self.p[idx] + (1 - damping) * old_c[idx] / self.N
                        if not fixed_n:
                            self.n = damping * new_n + (1 - damping) * old_n
                            n_damping_update = 1
                c_diff_all = c_diff_.sum()/(len(self.selected_hye_sizes) - len(self.fix_hyper_layer))

                self.c_diff.append(c_diff_all) #Mean difference
                
                logging.info(
                    f"\tRescaled affinity matrix c=N*p:\n{self.c}"
                    f"\n\tDifference from previous iteration:"
                    f"{self.c_diff[-1]}"
                )
            ## update n
            if n_damping_update == 0 and not fixed_n:
                self.n = new_n
            if not fixed_n:
                logging.info(
                        f"\tCommunity prior:\n{self.n}"
                        "\n\tDifference from previous iteration: "
                        f"{self.n_diff[-1]}"
                    )

            self.training_iter = it + 1

            em_iter_elapsed = time.perf_counter() - em_iter_start
            self.em_time_per_iter.append(em_iter_elapsed)

            if not fixed_n or not fixed_p:
                param_diff = 0.0
                if not fixed_n:
                    param_diff += self.n_diff[-1]
                if not fixed_p:
                    param_diff += self.c_diff[-1]
                logging.info(
                        f"em_thresh: {param_diff} "
                    )
                if param_diff <= em_thresh:
                    self.em_stop_reason = "em_thresh"
                    logging.info(
                        "Expectation-maximization threshold passed. "
                        "inference terminated."
                    )
                    break
        
        self.em_total_time = time.perf_counter() - em_total_start

        logging.info(
            f"EM stopped by {self.em_stop_reason}, "
            f"total time = {self.em_total_time:.4f} s"
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
        cancel_oscillation: bool =None
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
                f"\t\tInitialized hye to node:\n{self.log_hye_to_node[0].data[:5]}"
            )
            logging.debug(
                f"\t\tInitialized node to hye:\n{self.log_node_to_hye[0].data[:5]}"
            )
            logging.debug(f"\t\tInitialized marginals:\n{self.log_marginals[:5]}")
            logging.debug(f"\t\tInitialized external field:\n{self.external_field}")

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
            if cancel_oscillation:
                if self._is_cancel_oscillation(self.log_marginal_diff[-1]):
                    oscillation_iters += 1
                else:
                    oscillation_iters = 0
                if oscillation_iters >= 3:
                    logging.info(
                        "\tMessage passing oscillation detected. Message passing terminated."
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
        """Perform one step of message passing, updating the messages from nodes to
        factors, the messages from factors to nodes, the marginal probabilities and
        external field."""
        inc = hypergraph.get_binary_incidence_matrix()
        l = len(self.selected_hye_sizes) 

        ###Node-to-edge updates use the same rule and need no changes.
        # Update node to hye.
        new_node_to_hye = [None] * self.K
        for assignment in range(self.K):
            col_sum = self.log_hye_to_node[assignment].sum(axis=1)
            assert col_sum.shape == (self.N,)
            col_sum += np.log(self.n[assignment]) - self.external_field[assignment]
            col_sum = col_sum.reshape((self.N, 1))
            new_node_to_hye[assignment] = (
                TYPE_HYE_TO_NODE(inc * col_sum) - self.log_hye_to_node[assignment]
            )

        norm = sparse_reduce_lse(*new_node_to_hye)
        for assignment in range(self.K):
            new_node_to_hye[assignment].data -= norm.data
            new_node_to_hye[assignment].data = np.clip(
                new_node_to_hye[assignment].data, a_min=CLIP_MIN, a_max=CLIP_MAX
            )

        

        # # TODO dropout could be made more efficient here. Do it or not?
        # if dropout > 0:
        #     non_dropout_mask = (
        #         self.rng.random(len(self.log_node_to_hye[0].data)) >= dropout
        #     )
        #     for assignment in range(self.K):
        #         self.log_node_to_hye[assignment].data[
        #             non_dropout_mask
        #         ] = new_node_to_hye[assignment].data[non_dropout_mask]
        # else:
        #     for assignment in range(self.K):
        #         self.log_node_to_hye[assignment].data = new_node_to_hye[assignment].data

        # logging.debug(f"\t\tUpdated node to hye:\n{self.log_node_to_hye[0].data[:5]}")

        # TODO FVS-dropout could be made more efficient here. Do it or not?
        if dropout_FVS > 0:
            _,_,non_dropout_mask = FVS_random_mask_proration(
                hypergraph,
                self.log_node_to_hye[0],
                dropout_FVS=dropout_FVS,
                seed=None,  # keep the current random number generator unaltered.
                fvs_node_set=fvs_node_set,
            )
            for assignment in range(self.K):
                self.log_node_to_hye[assignment].data[
                    non_dropout_mask
                ] = new_node_to_hye[assignment].data[non_dropout_mask]
        elif dropout_FVS == 0 and dropout > 0:
            non_dropout_mask = (
            self.rng.random(len(self.log_node_to_hye[0].data)) >= dropout
            )
            for assignment in range(self.K):
                self.log_node_to_hye[assignment].data[
                    non_dropout_mask
                ] = new_node_to_hye[assignment].data[non_dropout_mask]

        else:
            non_dropout_mask = np.ones(
                len(self.log_node_to_hye[0].data),
                dtype=bool
            )
            for assignment in range(self.K):
                self.log_node_to_hye[assignment].data = new_node_to_hye[assignment].data


        total_num_ = len(self.log_node_to_hye[0].data)
        updated_num = np.count_nonzero(non_dropout_mask)
        self.update_message_ratio = updated_num / total_num_

        logging.debug(f"\t\tUpdated node to hye:\n{self.log_node_to_hye[0].data[:5]}")

        ##Edge-to-node updates differ; adapt compute_psi_dynamic_programming.
        # Update hye to node.
        # if dropout > 0:
        #     non_dropout_mask = (
        #         self.rng.random(len(self.log_hye_to_node[0].data)) >= dropout
        #     )
        # else:
        #     non_dropout_mask = None
        if dropout_FVS == 0 and dropout > 0:
            non_dropout_mask = (
                self.rng.random(len(self.log_hye_to_node[0].data)) >= dropout
            )
            updated_num += np.count_nonzero(non_dropout_mask)
            total_num_ += len(self.log_hye_to_node[0].data)
            self.update_message_ratio = updated_num / total_num_
        elif dropout_FVS == 0 and dropout == 0:
            non_dropout_mask = np.ones(
                    len(self.log_hye_to_node[0].data),
                    dtype=bool
                )
        new_hye_to_node = [
            TYPE_HYE_TO_NODE(x)
            for x in compute_psi_dynamic_programming_loop(
                hypergraph=hypergraph,
                model=self,
                mask=non_dropout_mask,
            )
        ]

        norm = sparse_reduce_lse(*new_hye_to_node)
        for assignment in range(self.K):
            new_hye_to_node[assignment].data -= norm.data
            new_hye_to_node[assignment].data = np.clip(
                new_hye_to_node[assignment].data, a_min=CLIP_MIN, a_max=CLIP_MAX
            )

        for assignment in range(self.K):
            self.log_hye_to_node[assignment].data[non_dropout_mask] = new_hye_to_node[
                assignment
            ].data

        logging.debug(f"\t\tUpdated hye to node:\n{self.log_hye_to_node[0].data[:5]}")

        # Update marginals.
        new_marginals = []
        for assignment in range(self.K):
            col_sum = self.log_hye_to_node[assignment].sum(axis=1)
            assert col_sum.shape == (self.N,)
            col_sum += np.log(self.n[assignment]) - self.external_field[assignment]
            new_marginals.append(col_sum)
        new_marginals = np.stack(new_marginals, axis=1)
        assert new_marginals.shape == (self.N, self.K)

        new_marginals = new_marginals - special.logsumexp(
            new_marginals, axis=1, keepdims=True
        )
        new_marginals = np.clip(new_marginals, a_min=CLIP_MIN, a_max=CLIP_MAX)

        # if dropout_vertex > 0:
        #     non_dropout_mask = self.rng.random(self.N) >= dropout_vertex
        #     self.log_marginals[non_dropout_mask] = new_marginals[non_dropout_mask]
        # else:
        #     self.log_marginals = new_marginals

        non_dropout_mask = self._make_vertex_update_mask(dropout_vertex,dropout_vertex_FVS,dropout_vertex_rest,fvs_node_set)
        self.log_marginals[non_dropout_mask] = new_marginals[non_dropout_mask]

        
        logging.debug(f"\t\tUpdated marginals:\n{self.log_marginals[:5]}")

        # Update external field.
        self.external_field = np.zeros(self.K)
        for l_ in range(l):
            lse_term = special.logsumexp(
                a=self.log_marginals.reshape((self.N, 1, self.K)),
                b=self.c[l_].reshape(1, self.K, self.K),
                axis=(0, 2),
            )
            assert lse_term.shape == (self.K,)
            C_prime = compute_C_prime_d(self.selected_hye_sizes[l_])
            self.external_field += C_prime / (self.N) * np.exp(lse_term)

        logging.debug(f"\t\tUpdated external field:\n{self.external_field}")
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
    def updated_affinity_matrix_list(self,  hypergraph: IncidenceHypergraph,hyper_size_idx: int,project: bool) -> np.ndarray:
        """Parameter updates for the affinity matrix p during EM inference.

        Parameters
        ----------
        hypergraph: a hypergraph.

        Returns
        -------
        The updated affinity matrix.
        """
        # Numerator.
        p_new = np.zeros_like(self.p[0])
        p_projected = np.zeros_like(self.p[0])

        # size_idx = self.selected_hye_sizes[i]
        ##Hard update
        # pi, interactions = self.hye_pi(hypergraph.hye_sizes_select([size_idx]), i, return_interactions=True)
        #Soft update
        pi, interactions =  self.hye_pi_message_update(hypergraph, hyper_size_idx, return_interactions=True)

        numerator = np.tensordot(
            interactions, 1 / np.clip(pi, a_min=1.0e-20, a_max=None), axes=(0, 0)
        )
        assert numerator.shape == (self.K, self.K)

        
        # Denominator.
        
        C_prime = compute_C_prime_d(self.selected_hye_sizes[hyper_size_idx])
        # denominator = (
        #     self.N * C_prime * (self.N * np.outer(self.n, self.n) - np.diag(self.n))
        # )
        denominator = (
            self.N * C_prime * (self.N * np.outer(self.n, self.n))
        )

        p_new = self.p[hyper_size_idx]  * 2 * numerator / denominator
        # p_update = self._project_to_planted_partition(p_update)  ##p_in and p_out projection
        if project :
            p_projected = self._project_to_planted_partition(p_new ,hyper_size_idx)

            return np.clip(p_projected, a_min=1e-20, a_max=0.99),np.clip(p_new, a_min=1e-20, a_max=0.99)
        else:
             return np.clip(p_new, a_min=1e-20, a_max=0.99),np.clip(p_new, a_min=1e-20, a_max=0.99)

##
    def compute_external_field_list(self) -> np.array:
        r"""Compute the approximate external field, defined as
        .. math::
            h(t_i) :=
                \sum_{l}\frac{C'_{d_l}}{N}
                \sum_{j \in V} \sum_{t_j} c_{t_i t_j} q_j(t_j)
        where
        .. math::
            C'_{d_l} = \binom{N-2}{d_l-2} \frac{1}{\kappa_{d_l}}

        Returns
        -------
        The external field h.
        """
        log_marginals = self.log_marginals
        c = self.c
        K = self.K
        N = self.N
        l = len(self.selected_hye_sizes) 

        external_field_ = np.zeros(K)
        for l_ in range(l):
            lse_term = special.logsumexp(
                a=log_marginals.reshape((N, 1, K)),
                b=c[l_].reshape(1, K, K),
                axis=(0, 2),
            )
            assert lse_term.shape == (K,)
            C_prime = compute_C_prime_d(self.selected_hye_sizes[l_])
            external_field_ += C_prime / (N) * np.exp(lse_term)

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

        incidence = hypergraph.get_binary_incidence_matrix()
        indptr = incidence.indptr
        indices = incidence.indices

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

            log_messages = np.empty((size, K), dtype=float)

            for a in range(K):
                message = self.log_node_to_hye[a]
                log_messages[:, a] = message.data[
                    message.indptr[hye] : message.indptr[hye + 1]
                ]

            # log-message -> probability-message
            log_messages = log_messages - special.logsumexp(
                log_messages, axis=1, keepdims=True
            )
            messages = np.exp(log_messages)
            #undirect
            S = messages.sum(axis=0)
            pair_num = 0.5 * (np.outer(S, S) - messages.T @ messages)
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
            hye_idx += 1

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
        ones = np.ones(hypergraph.E)
        log_marginals = self.log_marginals
        hye_dims = hypergraph.get_binary_incidence_matrix().sum(axis=0)

        # Node-related addends.
        f_i = [
            x.tocsc().dot(ones) - external_field[k]
            for k, x in enumerate(
                compute_psi_dynamic_programming_loop(hypergraph=hypergraph, model=self)
            )
        ]
        assert len(f_i) == K
        assert all(x.shape == (N,) for x in f_i)
        f_i = np.vstack(f_i).T
        assert f_i.shape == (N, K)
        f_i = special.logsumexp(a=f_i, b=self.n.reshape(1, -1), axis=1)
        f_i_sum = f_i.sum()

        # Edge-related addends.
        # First addend.
        first_addend = compute_psi_tilde_dynamic_programming_loop(
            hypergraph=hypergraph, model=self
        )
        first_addend = ((hye_dims - 1) * first_addend).sum()

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
        for i in range(l):
            second_addend_ = special.logsumexp(
                a=np.hstack([cross_log_marginal_sum, cross_log_marginals]),
                b=np.hstack([self.c[i], -self.c[i]]),
            )
            second_addend_ = np.exp(second_addend_)
            second_addend_ *= compute_C_third_d(self.selected_hye_sizes[i]) / (2 * N)
            second_addend += second_addend_
        f_e_sum = first_addend + second_addend

        return -f_i_sum + f_e_sum

##
    @staticmethod
    def _check_params(
        n: np.ndarray, p: dict[int, np.ndarray], K: int, N: int,  selected_hye_sizes: list[int] | None, fix_hye_size: list[int] | None
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
        self, warm_p_prior: bool) -> None:
        """Random initialization of the affinity matrix p."""
        def affinity(K, c, n, eps, N):
            p_in = c / (N * (eps + (1.0 - eps) * np.sum(n**2)))
            p_out = eps * p_in
            p = p_out * np.ones(K) + (p_in - p_out) * np.eye(K)
            return p
        
        K = self.K
        N = self.N
        E = len(self.selected_hye_sizes)
        
        p = np.ones((E, K, K)) / (10 * (K - 1))
        p += self.rng.random((E, K, K)) / 50

        if warm_p_prior:
            for  i, p_ in enumerate(self.p_list.values()):
                eps_ = p_[0, 1] / p_[0, 0] if p_[0, 0] != 0 else 0
                p[i] = affinity(K, self.average_degree[i], self.n, eps_, N)

        else:
            for e in range(E):
                p[e] = np.triu(p[e], 1) + np.triu(p[e], 1).T
                np.fill_diagonal(p[e], 1.0 + self.rng.random(K) / 50)
            p /= N
            p = np.clip(p, a_min=1e-10, a_max=1.0)

        if self.fix_hyper_layer is not None:
            for size in self.fix_hyper_layer:
                p_fixed = self.p_list[size]
                if size not in self.selected_hye_sizes:
                    raise ValueError(f"Hyperedge size {size} is not in selected_hye_sizes.")
                if p_fixed.shape != (K, K):
                    raise ValueError(
                        f"fixed p for size {size} must have shape {(K, K)}, got {p_fixed.shape}"
                    )

                if not np.allclose(p_fixed, p_fixed.T):
                    raise ValueError(f"fixed p for size {size} must be symmetric.")

                e = self.selected_hye_sizes.index(size)
                p[e] = np.clip(p_fixed, a_min=1e-10, a_max=1.0)

        self.p = p
##
    def _init_message_passing(
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

    def c_diff_relative(self, c_old, c_new) -> float:
        rel = np.abs(c_new - c_old).sum()/c_old.sum()
        #rel = np.abs(c_new - c_old) / np.maximum(np.abs(c_old), 1e-20)
        return rel
    
    def average_c_from_p(self, p_mat):
        n = np.asarray(self.n, dtype=float)
        p_mat = np.asarray(p_mat, dtype=float)
        # logging.info(f"Current p_mat: {p_mat}")
        if np.max(n) > 0.99:
            n = np.zeros_like(n) + n.mean()
        average_c_from_p = self.N * (n @ p_mat @ n)
        # average_c_from_p = self.N * (n @ p_mat @ n)
        return average_c_from_p 


    def _project_to_planted_partition(self, p_mat: np.ndarray,hyper_size_idx: int) -> np.ndarray:
        """
        Project a K x K affinity matrix to planted-partition form:

            diagonal entries are all equal: p_in
            off-diagonal entries are all equal: p_out
        """
        p_mat = np.asarray(p_mat, dtype=float).copy()

        K = p_mat.shape[0]
        eye_mask = np.eye(K, dtype=bool)
        tol = 1e-20

       
        def compute_replacement_value(vals, tol=1e-20, ratio_th=1.2):
            """
            vals: a group of values, e.g. all diagonal or all off-diagonal entries.

            Returns:
                should_replace: whether replacement is needed.
                value: the replacement value.
            """

            vals = np.asarray(vals, dtype=float)

            valid_mask = np.abs(vals) > tol
            has_zero = np.any(~valid_mask)

            nonzero_vals = vals[valid_mask]

            # If all values are zero
            if nonzero_vals.size == 0:
                return True, 0.0

            v_max = np.max(nonzero_vals)
            v_min = np.min(nonzero_vals)

            # Weighted mean
            if abs(v_max + v_min) <= tol:
                raise ValueError("v_max and v_min are too close to zero for a stable weighted average.")
            else:
                value = np.average([v_max, v_min], weights=[v_max, v_min])

            ratio = v_max / max(v_min, tol)

            should_replace = (
                has_zero
                or ratio > ratio_th
            )

            return should_replace, value

        # Process diagonal entries.
        diag_vals = p_mat[eye_mask]
        replace_diag, p_in = compute_replacement_value(diag_vals, tol=tol)

        if replace_diag:
            p_mat[eye_mask] = p_in

        # Process off-diagonal entries.
        off_vals = p_mat[~eye_mask]
        replace_off, p_out = compute_replacement_value(off_vals, tol=tol)

        if replace_off:
            p_mat[~eye_mask] = p_out

        #Preserve the average degree.
        # if self.ratio_extra is None:
        #     self.ratio_extra = 1
        # ratio_extra = self.ratio_extra
        average_c_from_p = self.average_c_from_p(p_mat)
        logging.info(f"hyper_size_layer:{hyper_size_idx} ,Average c from p: {average_c_from_p} Average degree from G: {self.average_degree[hyper_size_idx]}")

        p_mat *= self.average_degree[hyper_size_idx] / average_c_from_p

        return p_mat
    
    

    def _is_cancel_oscillation(
        self,
        history,
        min_window: int = 2,
        max_window: int =  3,
        rel_sum_tol: float = 0.05,
        amp_tol: float = 1e-5,
    ) -> bool:
        """
        Detect cancellation of one positive value and several negative values,
        or one negative value and several positive values, to approximately zero.

        history: self.log_marginal_diff[-1]
        """

        if history is None or len(history) < min_window or len(history) < 60:
            return False

        vals_all = np.asarray(history, dtype=float)

        # Use only the most recent max_window entries.
        vals_all = vals_all[-max_window:]

        for L in range(min_window, len(vals_all) + 1):
            vals = vals_all[-L:]

            abs_sum = np.sum(np.abs(vals))

            # Small overall magnitude indicates near-convergence rather than oscillation.
            if abs_sum < amp_tol:
                continue

            group_sum = np.sum(vals)
            rel_cancel = abs(group_sum) / abs_sum

            pos_count = np.sum(vals > amp_tol)
            neg_count = np.sum(vals < -amp_tol)

            # Both positive and negative values must be present.
            if pos_count == 0 or neg_count == 0:
                continue

            # Detect one positive with several negatives, or the reverse.
            grouped_pattern = (
                (pos_count == 1 and neg_count >= 1)
                or
                (neg_count == 1 and pos_count >= 1)
            )

            if grouped_pattern and rel_cancel < rel_sum_tol:
                return True

        return False
        
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