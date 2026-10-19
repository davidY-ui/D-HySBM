#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Search order-layer groupings in one Python 3.10-compatible file.

Requires the project src package and its BP runtime environment.
    python realdata_group_search.py --help
By default enumerate all partitions for <=6 orders and use greedy bisection above
6 (user-selected threshold; the paper uses 4). Each group requires at least
3*K*(K+1)/2 full-graph hyperedges (user multiplier 3; the paper uses 5). Retain
sparse layers, which may share groups with other layers. Greedy search only tries
adjacent boundaries in sorted order and accepts strict AUC gains >0.001.
Six orders have at most 203 partitions. Manually expanded exhaustive searches
default to at most 100000 raw candidates; exceeding this raises an error.
--search-mode greedy explicitly switches modes. --max-exhaustive-partitions 0
removes the cap and may require impractical computation.
--dry-run prepares five folds and a task plan without training. Preserve the
original BP/five-fold/order-positive-count-weighted AUC; the paper's matrix EM
backend is not implemented. All added code is in this file, without auxiliary
Python files, Fisher/responsibility statistics, or probe screening. Functions and
classes are defined directly, without code strings, exec, or dynamic modules.
Use a fresh --output directory. Results: best_groups.json, best_result.json,
and candidate_auc.csv.
The default training entry is main_message_passing_loop.py.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
import copy
from copy import deepcopy
import csv
from decimal import Decimal
import hashlib
from itertools import combinations
import json
import math
from numbers import Integral
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import threading
import time

import numpy as np


# ==================== Data loading, five-fold splitting, and AUC ====================

def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                   encoding='utf-8')
    os.replace(tmp, path)



def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()



def roc_auc(labels, scores):
    """Mann--Whitney AUC: positive/negative ties count as half a win; no SciPy dependency."""
    y, s = np.asarray(labels), np.asarray(scores, dtype=float)
    if y.ndim != 1 or y.shape != s.shape or not np.isin(y, [0, 1]).all():
        raise ValueError('AUC requires equally sized 1D binary labels and scores.')
    if not np.isfinite(s).all():
        raise ValueError('AUC scores must be finite.')
    pos, neg = s[y == 1], np.sort(s[y == 0])
    if not len(pos) or not len(neg):
        raise ValueError('AUC requires both positive and negative examples.')
    left = np.searchsorted(neg, pos, side='left')
    right = np.searchsorted(neg, pos, side='right')
    return float((left.astype(float) + right).sum() / (2 * len(pos) * len(neg)))



def normalized_marginals(log_marginals):
    logs = np.asarray(log_marginals, dtype=float)
    if logs.ndim != 2 or min(logs.shape) < 1:
        raise ValueError('log_marginals must have shape (N,K).')
    if np.isnan(logs).any() or np.isposinf(logs).any():
        raise ValueError('Invalid log_marginals.')
    maximum = logs.max(axis=1, keepdims=True)
    if not np.isfinite(maximum).all():
        raise ValueError('Every node needs at least one finite log probability.')
    q = np.exp(logs - maximum)
    return q / q.sum(axis=1, keepdims=True)



def edge_scores(q, p, edges):
    """Preserve notebook matrix multiplication order so rounding does not reorder nearly equal scores."""
    edges = np.asarray(edges, dtype=int)
    if edges.ndim != 2 or edges.shape[1] < 2:
        raise ValueError('Each scoring batch must contain same-order edges.')
    d = edges.shape[1]
    result = np.zeros(len(edges), dtype=float)
    pair_scores = q @ p @ q.T
    for i in range(d):
        for j in range(i + 1, d):
            result += pair_scores[edges[:, i], edges[:, j]]
    return result / (d * (d - 1) / 2)



def empirical_beta(counts):
    if counts.get(2, 0) <= 0 or any(m <= 0 for m in counts.values()):
        raise ValueError('Every selected training layer, including order 2, needs edges.')
    return {int(d): float(d * (d - 1) * m / (2 * counts[2]))
            for d, m in counts.items()}



def load_json_training_graph(path, orders):
    """Read edge-dict/node-data JSON and merge duplicate binary hyperedges before splitting."""
    obj = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(obj, dict) or not isinstance(obj.get('edge-dict'), dict):
        raise ValueError('JSON must contain an edge-dict object mapping edge IDs to node lists.')
    node_data = obj.get('node-data', {})
    if not isinstance(node_data, dict):
        raise ValueError('JSON node-data must be an object keyed by node ID.')
    metadata = obj.get('hypergraph-data', {})
    if (obj.get('network-type') == 'directed' or obj.get('directed') is True
            or (isinstance(metadata, dict) and metadata.get('directed') is True)):
        raise ValueError('Directed JSON is incompatible with this undirected hypergraph model.')

    def sort_id(value):
        try:
            return (0, int(value), value)
        except ValueError:
            return (1, value)

    all_nodes, records = set(node_data), []
    for edge_id, values in obj['edge-dict'].items():
        if not isinstance(values, list):
            raise ValueError(f'JSON edge {edge_id}: expected a list of node IDs.')
        if any(isinstance(v, bool) or not isinstance(v, (str, int)) for v in values):
            raise ValueError(f'JSON edge {edge_id}: each node ID must be a string or integer.')
        members = set(map(str, values))
        all_nodes.update(members)
        records.append((edge_id, members))
    node_ids = sorted(all_nodes, key=sort_id)
    if not node_ids:
        raise ValueError('JSON graph contains no nodes.')
    mapping = {node_id: i for i, node_id in enumerate(node_ids)}
    records.sort(key=lambda item: (len(item[1]), sort_id(item[0])))
    retained, seen, selected_records = [], set(), 0
    for _, members in records:
        edge = tuple(sorted(mapping[node] for node in members))
        if len(edge) < 2:
            continue
        if len(edge) in orders:
            selected_records += 1
            if edge not in seen:
                retained.append(edge)
        seen.add(edge)
    counts = {d: sum(len(edge) == d for edge in retained) for d in orders}
    return {'N': len(node_ids), 'node_ids': node_ids, 'edges': set(retained),
            'ordered_edges': retained, 'counts': counts,
            'affinity_orders': list(dict.fromkeys(len(e) for e in retained)),
            'beta_emp': empirical_beta(counts),
            'source_summary': {'format': 'json_edge_dict', 'record_count': len(records),
                               'unique_edges_size_at_least_2': len(seen),
                               'selected_unique_edges': len(retained),
                               'merged_selected_duplicates': selected_records - len(retained),
                               'interpretation': 'static_binary_presence; timestamps and edge weights are not used'}}



def load_training_graph(path, orders):
    """Read NPZ or project JSON, mapping JSON string node IDs to contiguous integers."""
    suffix = Path(path).suffix.lower()
    if suffix == '.json':
        return load_json_training_graph(path, orders)
    if suffix != '.npz':
        raise ValueError('Input graph must be a .npz or .json file.')
    with np.load(path, allow_pickle=True) as data:
        if 'B' in data:
            B = data['B']
            if B.dtype == object and B.size == 1:
                B = B.item()
            if hasattr(B, 'tocsc'):
                if not np.isfinite(B.data).all() or (B.data < 0).any():
                    raise ValueError(f'Invalid sparse incidence matrix: {path}')
                B = (B != 0).tocsc()
                B.eliminate_zeros()
                N, E = B.shape
                edges = [tuple(sorted(int(x) for x in B.indices[B.indptr[e]:B.indptr[e+1]]))
                         for e in range(E)]
            else:
                B = np.asarray(B)
                if B.ndim != 2 or not np.isfinite(B).all() or (B < 0).any():
                    raise ValueError(f'Invalid incidence matrix: {path}')
                N, E = B.shape
                edges = [tuple(np.flatnonzero(B[:, e]).tolist()) for e in range(E)]
        elif 'hyperedges' in data and 'N' in data:
            N = int(data['N'].item())
            edges = []
            for raw in data['hyperedges']:
                values = np.asarray(raw)
                if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
                    raise ValueError(f'Hyperedges must contain integer node IDs: {path}')
                edge = tuple(sorted(int(x) for x in values))
                if (not edge or len(set(edge)) != len(edge)
                        or edge[0] < 0 or edge[-1] >= N):
                    raise ValueError(f'Invalid hyperedge: {path}')
                edges.append(edge)
        else:
            raise ValueError(f'NPZ needs B or hyperedges plus N: {path}')
        if N < 1:
            raise ValueError(f'NPZ must retain a positive total node count: {path}')
        if 'N' in data and int(data['N'].item()) != N:
            raise ValueError(f'Inconsistent N in {path}')
    retained = [edge for edge in edges if len(edge) in orders]
    if len(set(retained)) != len(retained):
        raise ValueError(f'Duplicate training edges: {path}')
    actual_order = list(dict.fromkeys(len(e) for e in retained))
    counts = {d: sum(len(e) == d for e in retained) for d in orders}
    return {'N': int(N), 'edges': set(retained), 'counts': counts,
            'ordered_edges': retained, 'affinity_orders': actual_order,
            'beta_emp': empirical_beta(counts)}



def sample_negative_edges(N, d, count, observed, rng):
    """Uniformly sample distinct d-node sets absent from the complete graph."""
    total = math.comb(N, d)
    forbidden = {e for e in observed if len(e) == d}
    available = total - len(forbidden)
    if count > available:
        raise ValueError(f'Order {d}: need {count} unique negative edges but only '
                         f'{available} nonedges exist; reduce --negative-ratio.')
    # Enumerate small dense spaces directly to avoid prolonged rejection sampling for nonedges.
    if total <= 200_000 or (total <= 2_000_000 and available < 4 * count):
        pool = [e for e in combinations(range(N), d) if e not in forbidden]
        return [pool[int(i)] for i in rng.choice(len(pool), size=count, replace=False)]
    result, seen = [], set()
    attempts, limit = 0, max(10_000, 100 * count)
    while len(result) < count:
        edge = tuple(sorted(int(x) for x in rng.choice(N, size=d, replace=False)))
        attempts += 1
        if edge not in forbidden and edge not in seen:
            seen.add(edge)
            result.append(edge)
        if attempts >= limit and len(result) < count:
            raise ValueError(f'Order {d}: negative sampling exhausted its attempt budget; '
                             'reduce --negative-ratio for this dense layer.')
    return result



def make_fivefold_split(source, output, orders, *, seed, negative_ratio):
    """Create and cache five folds from a complete graph without altering the input graph."""
    source, output = Path(source).resolve(), Path(output).resolve()
    orders = sorted(int(d) for d in orders)
    if not math.isfinite(negative_ratio) or negative_ratio <= 0:
        raise ValueError('negative-ratio must be finite and positive.')
    config = {'version': 2, 'source': str(source), 'source_sha256': digest_file(source),
              'orders': orders, 'fold_count': 5, 'split_seed': int(seed),
              'negative_ratio': float(negative_ratio)}
    manifest_path = output / 'split_manifest.json'
    if manifest_path.is_file():
        saved = json.loads(manifest_path.read_text(encoding='utf-8'))
        if saved['configuration'] != config:
            raise ValueError('Saved fivefold split uses different data/settings; use a new --output.')
        for relative, expected in saved['files'].items():
            path = output / relative
            if not path.is_file() or digest_file(path) != expected:
                raise ValueError(f'Saved fold file is missing or changed: {path}; use a new --output.')
        print(f'Reusing fivefold split: {output}', flush=True)
        return {'folds': {int(f): {k: output / v for k, v in row.items()}
                          for f, row in saved['folds'].items()}, 'manifest_path': manifest_path}

    full = load_training_graph(source, orders)
    for d, count in full['counts'].items():
        if count < 5:
            raise ValueError(f'Order {d}: fivefold validation requires at least 5 observed '
                             f'edges; found {count}. Remove this order from --orders.')
    edges = full['ordered_edges']
    sizes = np.array([len(e) for e in edges], dtype=int)
    assignment = np.empty(len(edges), dtype=np.int8)
    fold_negatives = {f: [] for f in range(5)}
    for d in orders:
        rng = np.random.default_rng(np.random.SeedSequence([seed, d, 0]))
        groups = np.array_split(rng.permutation(np.flatnonzero(sizes == d)), 5)
        for f, group in enumerate(groups):
            assignment[group] = f
            count = int(math.ceil(len(group) * negative_ratio))
            neg_rng = np.random.default_rng(np.random.SeedSequence([seed, d, f, 1]))
            fold_negatives[f].extend(sample_negative_edges(full['N'], d, count, full['edges'], neg_rng))

    def edge_array(items):
        # Retain a one-dimensional ragged array even when all chosen edges have the same order.
        result = np.empty(len(items), dtype=object)
        for i, edge in enumerate(items):
            result[i] = np.asarray(edge, dtype=np.int64)
        return result

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'split_assignment.npz', N=full['N'],
                        hyperedges=edge_array(edges), edge_sizes=sizes, fold=assignment)
    folds, summary = {}, {}
    generated_files = [output / 'split_assignment.npz']
    if 'node_ids' in full:
        mapping_path = output / 'node_mapping.json'
        dump_json(mapping_path, {'index_to_node_id': full['node_ids']})
        generated_files.append(mapping_path)
    for f in range(5):
        directory = output / f'fold_{f:02d}'
        directory.mkdir(parents=True, exist_ok=True)
        train_edges = [e for i, e in enumerate(edges) if assignment[i] != f]
        positives = [e for i, e in enumerate(edges) if assignment[i] == f]
        validation = positives + fold_negatives[f]
        train_path, val_path = directory / 'train.npz', directory / 'validation.npz'
        hye_path = directory / 'train.txt'
        np.savez_compressed(train_path, N=full['N'], hyperedges=edge_array(train_edges))
        hye_path.write_text(''.join(' '.join(map(str, e)) + '\n' for e in train_edges), encoding='utf-8')
        np.savez_compressed(val_path, N=full['N'], fold=f, hyperedges=edge_array(validation),
                            edge_sizes=np.array([len(e) for e in validation], dtype=int),
                            y_true=np.r_[np.ones(len(positives), dtype=np.int8),
                                         np.zeros(len(fold_negatives[f]), dtype=np.int8)])
        folds[f] = {'train_path': train_path, 'validation_path': val_path, 'hye_path': hye_path}
        summary[f] = {d: {'train_positive': int(np.sum((sizes == d) & (assignment != f))),
                          'validation_positive': int(np.sum((sizes == d) & (assignment == f))),
                          'validation_negative': sum(len(e) == d for e in fold_negatives[f])}
                      for d in orders}
        generated_files.extend([train_path, val_path, hye_path])
    dump_json(manifest_path, {'configuration': config, 'N': full['N'], 'counts': full['counts'],
              'source_summary': full.get('source_summary', {'format': 'npz'}),
              'summary': summary,
              'folds': {f: {k: str(v.relative_to(output)) for k, v in row.items()}
                        for f, row in folds.items()},
              'files': {str(p.relative_to(output)): digest_file(p) for p in generated_files}})
    print(f'Created fivefold split: {output} (N={full["N"]}, edges={len(edges)})', flush=True)
    return {'folds': folds, 'manifest_path': manifest_path}



def load_validation(path, fold, orders):
    with np.load(path, allow_pickle=True) as data:
        N = int(data['N'].item())
        if 'fold' in data and int(data['fold'].item()) != fold:
            raise ValueError(f'Fold number mismatch: {path}')
        labels, sizes, raw = data['y_true'], data['edge_sizes'], data['hyperedges']
        if labels.ndim != 1 or sizes.shape != labels.shape or len(raw) != len(labels):
            raise ValueError(f'Inconsistent validation arrays: {path}')
        if not np.isin(labels, [0, 1]).all():
            raise ValueError(f'Invalid validation labels: {path}')
        edges = []
        for edge, d in zip(raw, sizes):
            values = np.asarray(edge)
            if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
                raise ValueError(f'Invalid node IDs: {path}')
            e = tuple(sorted(int(v) for v in values))
            if len(e) != d or len(set(e)) != d or d < 2 or e[0] < 0 or e[-1] >= N:
                raise ValueError(f'Invalid validation edge: {path}')
            edges.append(e)
    if len(set(edges)) != len(edges):
        raise ValueError(f'Duplicate validation edges: {path}')
    by_order = {}
    for d in orders:
        idx = np.flatnonzero(sizes == d)
        y = labels[idx]
        if not (np.any(y == 1) and np.any(y == 0)):
            raise ValueError(f'Fold {fold}, order {d}: need positive and negative examples.')
        by_order[d] = {'edges': np.asarray([edges[i] for i in idx], dtype=int),
                       'labels': y.astype(int)}
    return {'N': N, 'by_order': by_order,
            'positives': {e for e, y in zip(edges, labels) if y == 1},
            'negatives': {e for e, y in zip(edges, labels) if y == 0}}



def score_fit(fit, validation, affinity_orders, orders):
    q = normalized_marginals(fit['log_marginals'])
    if q.shape[0] != validation['N']:
        raise ValueError('Training/validation node count mismatch.')
    p = fit['p']
    if isinstance(p, np.ndarray) and p.dtype == object and p.size == 1:
        if isinstance(p.item(), dict):
            p = p.item()
    if isinstance(p, dict):
        matrices = {int(d): np.asarray(v, dtype=float) for d, v in p.items()}
    else:
        p = np.asarray(p, dtype=float)
        if p.ndim == 2:
            matrices = {d: p for d in orders}
        elif p.ndim == 3:
            if 'selected_hye_sizes' in fit:
                affinity_orders = [int(v) for v in np.asarray(fit['selected_hye_sizes']).ravel()]
            if len(affinity_orders) != len(p) or len(set(affinity_orders)) != len(p):
                raise ValueError('Cannot map fitted affinity matrices to orders.')
            matrices = dict(zip(affinity_orders, p))
        else:
            raise ValueError('Invalid fitted p dimensions.')
    rows = []
    for d in orders:
        matrix = matrices[d]
        if (matrix.shape != (q.shape[1], q.shape[1]) or not np.isfinite(matrix).all()
                or (matrix < 0).any() or not np.allclose(matrix, matrix.T)):
            raise ValueError(f'Invalid affinity matrix for order {d}.')
        batch = validation['by_order'][d]
        rows.append({'order': d, 'n_positive': int((batch['labels'] == 1).sum()),
                     'n_negative': int((batch['labels'] == 0).sum()),
                     'auc': roc_auc(batch['labels'], edge_scores(q, matrix, batch['edges']))})
    mean = np.average([r['auc'] for r in rows], weights=[r['n_positive'] for r in rows])
    return {'weighted_auc': float(mean), 'per_order': rows}


# ==================== Partition enumeration and greedy search ====================

def _order_types(values):
    values = tuple(values)
    if (not values or any(isinstance(d, bool) or not isinstance(d, Integral) or d < 2
                          for d in values) or len(set(values)) != len(values)):
        raise ValueError("order types must be nonempty distinct integers >=2")
    return tuple(sorted(int(d) for d in values))


def canonical_partition(groups, orders=None):
    """Canonical immutable groups with disjointness and optional coverage checks."""
    partition = tuple(sorted(_order_types(group) for group in groups))
    if not partition:
        raise ValueError("a partition needs at least one nonempty group")
    flattened = tuple(d for group in partition for d in group)
    if len(set(flattened)) != len(flattened):
        raise ValueError("partition groups overlap")
    if orders is not None and set(flattened) != set(_order_types(orders)):
        raise ValueError("partition must cover the order types exactly")
    return partition


def enumerate_partitions(orders):
    """Yield every set partition once, including noncontiguous blocks.

    Restricted-growth construction gives deterministic canonical blocks and
    yields the shared baseline first. This is a generator, so a budgeted
    explicit exhaustive request need not construct all Bell(n) partitions.
    """
    orders = _order_types(orders)
    blocks = []
    def visit(index):
        if index == len(orders):
            yield tuple(tuple(block) for block in blocks)
            return
        value = orders[index]
        for block in blocks:
            block.append(value)
            yield from visit(index+1)
            block.pop()
        blocks.append([value])
        yield from visit(index+1)
        blocks.pop()
    yield from visit(0)


def bell_number(n):
    """Count all set partitions using Bell's triangle, O(n^2) time/O(n) memory."""
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 0:
        raise ValueError("Bell number index must be a nonnegative integer")
    row = [1]
    for _ in range(int(n)):
        next_row = [row[-1]]
        for value in row:
            next_row.append(next_row[-1]+value)
        row = next_row
    return row[0]


def contiguous_splits(partition):
    """All partitions formed by one contiguous cut of one current group."""
    partition = canonical_partition(partition)
    candidates = set()
    for index, group in enumerate(partition):
        other = list(partition[:index]+partition[index+1:])
        for cut in range(1, len(group)):
            candidates.add(canonical_partition(other+[group[:cut], group[cut:]]))
    return sorted(candidates)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _checked_score(result):
    if not isinstance(result, dict) or "mean_auc" not in result:
        raise ValueError("evaluate must return a dict containing mean_auc")
    try:
        auc = float(result["mean_auc"])
    except (TypeError, ValueError) as exc:
        raise ValueError("mean_auc must be a finite number in [0,1]") from exc
    if isinstance(result["mean_auc"], bool) or not math.isfinite(auc) or not 0 <= auc <= 1:
        raise ValueError("mean_auc must be a finite number in [0,1]")
    return dict(result, mean_auc=auc)


def _rank(partition, score):
    return (-score["mean_auc"], len(partition), partition)


def _improves(candidate_auc, incumbent_auc, min_gain):
    # Compare the decimal values as printed, so .801-.8 does not pass an
    # exact .001 threshold merely because binary subtraction rounds upward.
    return (Decimal(str(candidate_auc))-Decimal(str(incumbent_auc))
            > Decimal(str(min_gain)))


def search_partitions(orders, evaluate, *, strategy="auto", exhaustive_max_orders=6,
                      admissible=None, candidate_jobs=1, min_gain=.001,
                      max_evals=None, on_step=None, max_exhaustive_partitions=100000):
    """Search validation AUC and return baseline, best, evaluations and history.

    ``evaluate(canonical_partition)`` returns a dict with finite ``mean_auc``
    and optional ``eligible_for_acceptance`` (default True). An ineligible
    baseline raises; other ineligible results are recorded and cannot win.
    ``admissible(partition)`` is checked before training and must admit the
    shared baseline. It can impose a controller-defined sparse-group rule.

    ``max_evals`` includes the shared baseline. A cut-off exhaustive search
    or greedy round reports ``search_complete=False`` and
    ``stop_reason='evaluation_budget'``. An interrupted greedy round can
    retain its best observed qualifying split, without claiming that all
    cuts were compared. Exhaustive selection has no min_gain gate.

    Before training, the exhaustive strategy checks the RAW Bell-number
    search space against ``max_exhaustive_partitions`` (default 100000).
    ``max_evals`` does not bypass this guard: an adversarial admissibility
    filter might scan the entire raw space to find one candidate. Explicitly
    raising the guard, or setting it to 0/None, authorizes the larger space.
    The strategy is never silently switched to greedy. Candidates stream
    in waves of at most ``candidate_jobs`` and stop scanning at the budget.

    Candidate evaluations use at most ``candidate_jobs`` threads. Callback
    snapshots and deterministic result ordering are handled by the calling
    coordinator thread. The controller owns the separate global fit limit.
    Evaluation progress notifications are limited to once per five seconds
    before copying the growing state; baseline, round and final notifications
    are unconditional.
    """
    orders = _order_types(orders)
    candidate_jobs = _positive_integer(candidate_jobs, "candidate_jobs")
    exhaustive_max_orders = _positive_integer(exhaustive_max_orders, "exhaustive_max_orders")
    if max_evals is not None:
        max_evals = _positive_integer(max_evals, "max_evals")
    if max_exhaustive_partitions is not None:
        if (isinstance(max_exhaustive_partitions, bool)
                or not isinstance(max_exhaustive_partitions, Integral)
                or max_exhaustive_partitions < 0):
            raise ValueError("max_exhaustive_partitions must be a nonnegative integer or None")
        max_exhaustive_partitions = int(max_exhaustive_partitions)
    try:
        min_gain = float(min_gain)
    except (TypeError, ValueError) as exc:
        raise ValueError("min_gain must be finite and nonnegative") from exc
    if not math.isfinite(min_gain) or min_gain < 0:
        raise ValueError("min_gain must be finite and nonnegative")
    if strategy not in ("auto", "exhaustive", "greedy"):
        raise ValueError("strategy must be auto, exhaustive or greedy")
    if not callable(evaluate) or (admissible is not None and not callable(admissible)):
        raise ValueError("evaluate and optional admissible must be callable")
    if on_step is not None and not callable(on_step):
        raise ValueError("on_step must be callable")
    chosen = ("exhaustive" if len(orders) <= exhaustive_max_orders else "greedy") if strategy == "auto" else strategy
    raw_count = bell_number(len(orders)) if chosen == "exhaustive" else None
    if (chosen == "exhaustive" and max_exhaustive_partitions
            and raw_count > max_exhaustive_partitions):
        raise ValueError(f"Exhaustive search over {len(orders)} order types has {raw_count} raw "
                         f"partitions, exceeding max_exhaustive_partitions={max_exhaustive_partitions}. "
                         "Explicitly raise this limit or disable it with 0/None to proceed.")
    allowed = admissible if admissible is not None else lambda partition: True
    shared = (orders,)
    if not allowed(shared):
        raise ValueError("the shared baseline is inadmissible")
    baseline = _checked_score(evaluate(shared))
    if not baseline.get("eligible_for_acceptance", True):
        raise RuntimeError("Shared baseline BP did not converge; increase the fitting budget "
                           "or explicitly permit unconverged budget-endpoint comparisons.")
    state = dict(baseline=baseline, best=baseline, best_partition=shared,
                 evaluations=[dict(partition=shared, score=baseline)], history=[],
                 stop_reason=None, search_complete=False, strategy=chosen,
                 raw_partition_count=raw_count)
    last_notification = None
    def notify(event):
        nonlocal last_notification
        if on_step is not None:
            now = time.monotonic()
            if event == "evaluation" and last_notification is not None and now-last_notification < 5.:
                return
            last_notification = now
            snapshot = deepcopy(state)
            snapshot["event"] = event
            on_step(snapshot)
    def evaluate_one(partition):
        return _checked_score(evaluate(partition))
    def evaluate_batch(partitions, executor=None):
        if candidate_jobs == 1:
            for partition in partitions:
                yield partition, evaluate_one(partition)
        elif executor is not None:
            for partition, score in zip(partitions, executor.map(evaluate_one, partitions)):
                yield partition, score
        else:
            with ThreadPoolExecutor(max_workers=candidate_jobs) as executor:
                # map preserves submission order even if jobs finish out of order.
                for partition, score in zip(partitions, executor.map(evaluate_one, partitions)):
                    yield partition, score
    def record(partition, score):
        state["evaluations"].append(dict(partition=partition, score=score))
    def remaining():
        return None if max_evals is None else max_evals-len(state["evaluations"])
    def finish(reason, complete):
        state.update(stop_reason=reason, search_complete=complete)
        notify("finished")
        return state
    notify("baseline")
    if chosen == "exhaustive":
        entry = dict(iteration=0, incumbent=shared, evaluated=[], accepted=False,
                     selected=shared, round_complete=False)
        state["history"].append(entry)
        partitions = iter(enumerate_partitions(orders))
        next(partitions)  # The already evaluated shared baseline is first.
        raw_visited, exhausted = 1, False
        pool = ThreadPoolExecutor(max_workers=candidate_jobs) if candidate_jobs > 1 else nullcontext(None)
        with pool as executor:
            while True:
                if exhausted or raw_visited == raw_count:
                    entry["round_complete"] = True
                    return finish("all_partitions_evaluated", True)
                limit = remaining()
                if limit == 0:
                    return finish("evaluation_budget", False)
                wave_size = candidate_jobs if limit is None else min(candidate_jobs, limit)
                wave = []
                while len(wave) < wave_size:
                    try:
                        partition = next(partitions)
                    except StopIteration:
                        exhausted = True
                        break
                    raw_visited += 1
                    if allowed(partition):
                        wave.append(partition)
                for partition, score in evaluate_batch(wave, executor):
                    record(partition, score)
                    eligible = bool(score.get("eligible_for_acceptance", True))
                    entry["evaluated"].append(dict(partition=partition, mean_auc=score["mean_auc"], eligible=eligible))
                    if eligible and _rank(partition, score) < _rank(state["best_partition"], state["best"]):
                        state.update(best_partition=partition, best=score)
                        entry.update(selected=partition, accepted=partition != shared)
                    notify("evaluation")
    current, fitted = shared, baseline
    while True:
        all_candidates = contiguous_splits(current)
        if not all_candidates:
            return finish("all_singletons", True)
        candidates = [partition for partition in all_candidates if allowed(partition)]
        if not candidates:
            return finish("no_admissible_candidates", True)
        limit = remaining()
        if limit == 0:
            return finish("evaluation_budget", False)
        truncated = limit is not None and limit < len(candidates)
        if truncated:
            candidates = candidates[:limit]
        winner, winning_score = current, fitted
        entry = dict(iteration=len(state["history"]), incumbent=current,
                     incumbent_auc=fitted["mean_auc"], evaluated=[], accepted=False,
                     selected=current, round_complete=not truncated)
        for partition, score in evaluate_batch(candidates):
            record(partition, score)
            eligible = bool(score.get("eligible_for_acceptance", True))
            entry["evaluated"].append(dict(partition=partition, mean_auc=score["mean_auc"],
                validation_gain=score["mean_auc"]-fitted["mean_auc"], eligible=eligible))
            if (eligible and _improves(score["mean_auc"], fitted["mean_auc"], min_gain)
                    and _rank(partition, score) < _rank(winner, winning_score)):
                winner, winning_score = partition, score
        accepted = winner != current
        entry.update(selected=winner, accepted=accepted,
                     selected_auc=winning_score["mean_auc"],
                     validation_gain=winning_score["mean_auc"]-fitted["mean_auc"])
        state["history"].append(entry)
        state.update(best_partition=winner, best=winning_score)
        notify("round")
        if truncated:
            return finish("evaluation_budget", False)
        if not accepted:
            return finish("no_validation_improvement", True)
        current, fitted = winner, winning_score


# ==================== Original BP training interface and fixed parameter closure ====================

_WORKER_DESCRIPTION = 'Run the original training API and export a compact closed-BP validation fit.\n\nThe external training project and its working environment remain required.\nThe original restart-selection artifact is preserved. Each fit receives one\nfixed-parameter BP closure and one independent complete residual update.\n'

#!/usr/bin/env python3



def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),encoding='utf-8')
    os.replace(temporary, path)

def _argument(arguments, flag, default=None):
    if flag not in arguments:
        if default is not None:
            return default
        raise ValueError(f'Training adapter requires training argument {flag}')
    if arguments.count(flag) != 1:
        raise ValueError(f'Duplicate training argument {flag}')
    i = arguments.index(flag)
    if i + 1 >= len(arguments):
        raise ValueError(f'Missing value for {flag}')
    return arguments[i + 1]

def _project_imports(project, entry):
    for path in (Path(project).resolve(), Path(entry).resolve().parent):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)

def _probability(logs, axis):
    logs = np.asarray(logs, dtype=float)
    if np.isnan(logs).any() or np.isposinf(logs).any():
        raise ValueError('Invalid log probabilities')
    maximum = logs.max(axis=axis, keepdims=True)
    if not np.isfinite(maximum).all():
        raise ValueError('Probability vector has no finite mass')
    p = np.exp(logs - maximum)
    return p / p.sum(axis=axis, keepdims=True)

def _structure(model, graph):
    """The real model uses SharedCSCMessages with one common CSC structure."""
    B = graph.get_binary_incidence_matrix()
    if hasattr(B, 'tocsc'):
        B = B.tocsc()
        if hasattr(B, 'has_sorted_indices') and not B.has_sorted_indices:
            B = B.copy()
            B.sort_indices()
    incoming = model.log_node_to_hye
    outgoing = model.log_hye_to_node
    for message in (incoming, outgoing):
        if (tuple(message.shape) != tuple(B.shape)
                or not np.array_equal(message.indices, B.indices)
                or not np.array_equal(message.indptr, B.indptr)):
            raise ValueError('Model messages and training incidence have different CSC ordering')
        if np.asarray(message.data).shape != (model.K, len(B.indices)):
            raise ValueError('Cavity data must have shape (K, incidence.nnz)')
    orders = [int(d) for d in model.selected_hye_sizes]
    if len(set(orders)) != len(orders) or any(d < 2 for d in orders):
        raise ValueError('Invalid fitted orders')
    sizes = np.diff(B.indptr)
    if not np.isin(sizes, orders).all():
        raise ValueError('Training incidence contains an unfitted order')
    if tuple(B.shape) != (model.N, len(B.indptr)-1):
        raise ValueError('Training incidence dimensions differ from model')
    return B, orders

def _controls(controls):
    controls = dict(controls or {})
    options = dict(mp_iter=int(controls.get('mp_iter',2000)),
                   mp_thresh=float(controls.get('mp_thresh',1e-5)),
                   patience=int(controls.get('patience',controls.get('mp_patience',50))),
                   seed=int(controls.get('seed',234)),warm_start=True,
                   dropout=0.,dropout_vertex=0.,dropout_vertex_FVS=0.,
                   dropout_vertex_rest=0.,dropout_FVS=0.,fvs_node_set=None)
    if options['mp_iter'] < 1 or options['patience'] < 1 or not np.isfinite(options['mp_thresh']) or options['mp_thresh'] <= 0:
        raise ValueError('BP controls need positive iteration budget, patience, and threshold')
    if not 0 <= options['seed'] < 2**32:
        raise ValueError('BP seed must be uint32')
    tolerance=float(controls.get('residual_thresh',options['mp_thresh']))
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('residual_thresh must be finite and positive')
    return options,tolerance

def _state_distance(first, second):
    arrays = [(first.log_marginals,second.log_marginals,1),
              (first.log_node_to_hye.data,second.log_node_to_hye.data,0),
              (first.log_hye_to_node.data,second.log_hye_to_node.data,0)]
    return max(float(np.max(np.abs(_probability(a,axis)-_probability(b,axis)),initial=0.))
               for a,b,axis in arrays)

def _close(model,graph,controls):
    """Warm fixed-parameter BP plus an independent complete one-step residual."""
    _structure(model,graph)
    options,tolerance = _controls(controls)
    before_n=np.asarray(model.n).copy()
    before_p=np.asarray(model.p).copy()
    model.p_list={int(d):model.p[i].copy() for i,d in enumerate(model.selected_hye_sizes)}
    model.external_field=model.compute_external_field_list()
    model.parallel_message_passing(graph,**options)
    if not np.array_equal(before_n,model.n) or not np.array_equal(before_p,model.p):
        raise ValueError('Fixed-parameter BP unexpectedly changed n or p')
    histories=getattr(model,'log_marginal_diff',[])
    history=np.asarray(histories[-1] if histories else [],dtype=float)
    patience=options['patience']
    stopped_below=(len(history)>=patience and np.isfinite(history[-patience:]).all()
                   and bool(np.all(np.abs(history[-patience:])<=options['mp_thresh'])))
    # The original stopping rule tracks only node beliefs. A full extra update
    # also detects oscillating cavity/factor messages hidden by stable nodes.
    check=copy.deepcopy(model)
    check_options=dict(options,mp_iter=1)
    check.parallel_message_passing(graph,**check_options)
    residual=_state_distance(model,check)
    converged=bool(stopped_below and np.isfinite(residual) and residual<=tolerance)
    if hasattr(model,'converged'):
        converged=converged and bool(model.converged)
    return dict(converged=converged,residual=residual,
                node_history_converged=bool(stopped_below),bp_iterations=int(len(history)),
                residual_threshold=tolerance,full_update=True)


def _fitted_arrays(model, requested_orders):
    """Validate the original fitted API without assuming a replacement model."""
    required = ('N', 'K', 'selected_hye_sizes', 'n', 'p', 'log_marginals',
                'log_node_to_hye', 'log_hye_to_node', 'compute_external_field_list',
                'parallel_message_passing')
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        model_type = type(model)
        module = sys.modules.get(model_type.__module__)
        source = getattr(module, '__file__', '<built-in or unavailable>')
        raise ValueError(
            'Training did not return a compatible fitted model: '
            f'returned_type={model_type.__module__}.{model_type.__qualname__}; '
            f'missing_attributes={missing}; model_source={source}. '
            'run_message_passing(argv) must return (best_model, restart_score), '
            'where best_model is the fitted BP object.')
    if (not callable(model.compute_external_field_list)
            or not callable(model.parallel_message_passing)):
        raise ValueError('Fitted model lacks the original BP methods')
    if (isinstance(model.N, (bool, np.bool_)) or not isinstance(model.N, (int, np.integer))
            or isinstance(model.K, (bool, np.bool_)) or not isinstance(model.K, (int, np.integer))
            or model.N < 1 or model.K < 1):
        raise ValueError('Fitted model N and K must be positive integers')
    raw_orders = np.asarray(model.selected_hye_sizes)
    if (raw_orders.ndim != 1 or not np.issubdtype(raw_orders.dtype, np.integer)
            or len(raw_orders) == 0 or np.any(raw_orders < 2)
            or len(set(raw_orders.tolist())) != len(raw_orders)):
        raise ValueError('Invalid fitted orders')
    orders = [int(d) for d in raw_orders]
    if set(orders) != set(requested_orders):
        raise ValueError('Fitted orders differ from the requested selected orders')
    n, p = np.asarray(model.n, dtype=float), np.asarray(model.p, dtype=float)
    if (n.shape != (model.K,) or not np.isfinite(n).all() or np.any(n <= 0)
            or not np.isclose(n.sum(), 1.)):
        raise ValueError('Fitted prior n must be positive and normalized')
    if (p.shape != (len(orders), model.K, model.K) or not np.isfinite(p).all()
            or np.any(p < 0) or np.any(p >= 1)
            or not np.allclose(p, p.transpose(0, 2, 1))):
        raise ValueError('Fitted affinities must be symmetric probability (L,K,K) matrices')
    try:
        weight_maps = [getattr(model, name) for name in ('w', 'omega')
                       if hasattr(model, name)]
        unit_weights = bool(weight_maps) and all(
            float(weights[d]) == 1. for weights in weight_maps for d in orders)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('Fitted model must provide unit order weights') from error
    if not unit_weights:
        raise ValueError('Training adapter requires unit order weights')
    logs = np.asarray(model.log_marginals, dtype=float)
    if logs.shape != (model.N, model.K):
        raise ValueError('Fitted log marginals must have shape (N,K)')
    q = _probability(logs, 1)
    return orders, n, p, q


def _selected_orders(path):
    values = np.atleast_1d(np.loadtxt(path, dtype=float))
    if (values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all()
            or np.any(values < 2) or np.any(values != np.floor(values))):
        raise ValueError('Selected orders must be a nonempty vector of integers >=2')
    orders = [int(value) for value in values]
    if len(orders) != len(set(orders)):
        raise ValueError('Selected orders must be distinct')
    return orders


def train(project: Path, entry: Path, arguments: list[str], snapshot_dir: Path,
          closure_controls: dict) -> dict:
    """Call the original training entrypoint, then close its chosen parameters."""
    project, entry, snapshot_dir = map(
        lambda path: Path(path).resolve(), (project, entry, snapshot_dir))
    if not project.is_dir() or not entry.is_file():
        raise FileNotFoundError('Training project or entrypoint is missing')
    arguments = list(map(str, arguments))
    sizes = _selected_orders(_argument(arguments, '--selected_hye_sizes'))
    node_count = int(_argument(arguments, '--N'))
    if node_count < 1:
        raise ValueError('Training node count must be positive')
    _project_imports(project, entry)
    namespace = runpy.run_path(str(entry), run_name='paper_adapter_import')
    function = namespace.get('run_message_passing')
    if not callable(function):
        raise ValueError('Training entrypoint must expose run_message_passing(argv)')
    trained = function(arguments)
    if not isinstance(trained, tuple) or len(trained) != 2:
        raise ValueError('run_message_passing must return (best_model, restart_score)')
    original, score = trained
    try:
        _fitted_arrays(original, sizes)
    except ValueError as error:
        raise ValueError(f'{error}; training_entrypoint={entry}') from error
    if original.N != node_count:
        raise ValueError('Fitted node count differs from the training request')
    from src.data.data_io import load_data
    graph = load_data(hye_file=_argument(arguments, '--hye_file'),
                      N=node_count).hye_sizes_select(sizes)
    _structure(original, graph)
    model = copy.deepcopy(original)
    diagnostics = _close(model, graph, closure_controls)
    orders, n, p, q = _fitted_arrays(model, sizes)
    with np.errstate(divide='ignore'):
        logs = np.log(q)
    values = dict(orders=np.asarray(orders, dtype=np.int64),
                  selected_hye_sizes=np.asarray(orders, dtype=np.int64),
                  n=n.copy(), p=p.copy(), q=q, log_marginals=logs,
                  converged=np.asarray(diagnostics['converged']),
                  bp_residual=np.asarray(diagnostics['residual']),
                  bp_iterations=np.asarray(diagnostics['bp_iterations'], dtype=np.int64))
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / 'snapshot.npz'
    temporary = snapshot_path.with_name(snapshot_path.name + f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **values)
    os.replace(temporary, snapshot_path)
    score = float(score)
    return dict(snapshot_path=str(snapshot_path), snapshot_dir=str(snapshot_dir),
                p_orders=orders, converged=diagnostics['converged'],
                bp_residual=diagnostics['residual'], bp_diagnostics=diagnostics,
                training_restart_score=score if np.isfinite(score) else None,
                training_restart_score_role='original_restart_selection_only',
                snapshot_role='fixed_parameter_BP_closure_for_validation')


def worker_main(argv=None):
    parser = argparse.ArgumentParser(description=_WORKER_DESCRIPTION)
    parser.add_argument('--train-request', type=Path, required=True)
    args = parser.parse_args(argv)
    request = json.loads(args.train_request.read_text(encoding='utf-8-sig'))
    try:
        result = train(Path(request['project']), Path(request['entry']), request['arguments'],
                       Path(request['snapshot_dir']), request.get('closure_controls', {}))
        result.update(fingerprint=request.get('fingerprint'), request_id=request.get('request_id'))
        _json(request['result_file'], result)
        return 0
    except Exception as error:
        _json(request['result_file'], dict(error=type(error).__name__, message=str(error),
                                         fingerprint=request.get('fingerprint'),
                                         request_id=request.get('request_id')))
        raise


# ==================== Scheduling, caching, outputs, and search settings ====================

_SEARCH_DESCRIPTION = 'Paper-style order grouping on the original BP backend and fair fivefold AUC.\n\nAuto: <=6 distinct order types exhaustive; >6 contiguous greedy splits.\nEvery group requires full-graph edges>=3*K*(K+1)/2 by user choice; paper used5.\nNo Fisher information, theory weights or responsibility screening are used.\n'

#!/usr/bin/env python3
VERSION = 2
GROUP_EDGE_MULTIPLIER = 3
HERE = Path(__file__).resolve().parent

def partition_id(partition):
    canonical = canonical_partition(partition)
    return 'g_' + hashlib.sha256(json.dumps(canonical).encode()).hexdigest()[:16]


def aggregate_scores(records, folds):
    fold_auc = []
    for fold in folds:
        selected = [row['weighted_auc'] for row in records if row['fold'] == fold]
        if not selected:
            raise ValueError(f'Missing results for fold {fold}')
        fold_auc.append(float(np.mean(selected)))
    return {'mean_auc': float(np.mean(fold_auc)),
            'std_auc': float(np.std(fold_auc, ddof=1)) if len(fold_auc) > 1 else 0.,
            'fold_auc': fold_auc, 'folds': list(folds)}


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def prepare(args):
    project = args.project_dir.resolve()
    entry = Path(args.entrypoint) if args.entrypoint else Path('main_message_passing_loop.py')
    if not entry.is_absolute():
        entry = project / entry
    if not entry.is_file():
        raise FileNotFoundError(f'Training entrypoint missing: {entry}. Set --project-dir or --entrypoint.')
    source = entry.read_text(encoding='utf-8-sig')
    for flag in ('--hye_file', '--N', '--selected_hye_sizes', '--shared_hye_sizes'):
        if flag not in source:
            raise ValueError(f'Training entrypoint does not advertise required flag {flag}: {entry}')
    if 'def run_message_passing(' not in source:
        raise ValueError('Entrypoint must expose run_message_passing(argv) returning (best_model, score).')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = project / 'data' / 'processed_input'
    full_path = (args.input_npz or root / f'{args.dataset}.npz').resolve()
    inputs = {str(entry): digest_file(entry)}
    generated = None
    if args.cv_root is None:
        generated = make_fivefold_split(full_path, output / 'cv5', args.orders,
                                               seed=args.split_seed,
                                               negative_ratio=args.negative_ratio)
        inputs[str(generated['manifest_path'])] = digest_file(generated['manifest_path'])
    folds, positives = {}, set()
    for fold in args.folds:
        if generated is not None:
            row = generated['folds'][fold]
            train_path, val_path, hye_path = row['train_path'], row['validation_path'], row['hye_path']
        else:
            train_path = root / f'{args.dataset}_train{fold}.npz'
            val_path = args.cv_root.resolve() / f'fold_{fold:02d}' / f'{args.dataset}_test{fold}.npz'
            hye_path = output / 'legacy_inputs' / f'train_{fold:02d}.txt'
        train = load_training_graph(train_path, args.orders)
        validation = load_validation(val_path, fold, args.orders)
        if train['N'] != validation['N'] or train['edges'] & validation['positives']:
            raise ValueError(f'Fold {fold}: inconsistent nodes or overlapping training/validation positives.')
        if positives & validation['positives']:
            raise ValueError('Validation positives overlap between folds.')
        positives |= validation['positives']
        if generated is None:
            hye_path.parent.mkdir(parents=True, exist_ok=True)
            hye_path.write_text(''.join(' '.join(map(str, e)) + '\n' for e in train['ordered_edges']),
                                encoding='utf-8')
        folds[fold] = {'train': train, 'validation': validation, 'hye_path': hye_path}
        for path in (train_path, val_path, hye_path):
            inputs[str(path.resolve())] = digest_file(path)
    known = positives | set().union(*(row['train']['edges'] for row in folds.values()))
    if full_path.is_file():
        full = load_training_graph(full_path, args.orders)
        if any(full['N'] != row['train']['N'] for row in folds.values()):
            raise ValueError('Full graph and folds have inconsistent node counts.')
        if not known.issubset(full['edges']):
            raise ValueError('Fold positives are not a subset of the full input graph.')
        known |= full['edges']
        inputs[str(full_path)] = digest_file(full_path)
    for fold, row in folds.items():
        if row['validation']['negatives'] & known:
            raise ValueError(f'Fold {fold}: a sampled negative is an observed edge.')
    counts = {d: sum(len(edge)==d for edge in known) for d in args.orders}
    sources = {Path(__file__).name: digest_file(Path(__file__))}
    for path in sorted((project / 'src').rglob('*.py')):
        sources[str(path.relative_to(project))] = digest_file(path)
    omitted = {'output', 'dry_run', 'max_evals', 'candidate_jobs', 'fold_jobs', 'max_parallel_fits'}
    configuration = {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items() if key not in omitted}
    protocol = {'version': VERSION, 'configuration': configuration, 'inputs': inputs,
                'sources': sources, 'numpy_version': np.__version__,
                'score_role': 'validation_used_for_group_selection',
                'score': 'within-order AUC, positive-count weighted; seeds averaged within each fold',
                'beta_policy': 'original empirical scales, no beta search and no share_beta_d',
                'grouping_policy': 'paper-style grouping with user multiplier3 and direct AUC search',
                'full_order_counts': {str(d): count for d, count in counts.items()},
                'group_edge_multiplier_default': GROUP_EDGE_MULTIPLIER,
                'group_edge_threshold_mode': 'multiplier_default' if args.min_group_edges is None else 'absolute_override',
                'min_group_edges': paper_min_group_edges(args.K) if args.min_group_edges is None else args.min_group_edges}
    fingerprint = _json_hash(protocol)
    prior = output / 'protocol.json'
    if prior.is_file() and json.loads(prior.read_text(encoding='utf-8')) != protocol:
        raise ValueError('Output contains a different protocol; use a new --output directory.')
    dump_json(prior, protocol)
    return project, entry.resolve(), folds, fingerprint, counts


class Evaluator:
    def __init__(self, args, project, entry, folds, fingerprint):
        self.args, self.project, self.entry = args, project, entry
        self.folds, self.fingerprint = folds, fingerprint
        self.output = args.output.resolve()
        self.slots = threading.BoundedSemaphore(args.max_parallel_fits)


    def closure_controls(self):
        return {'mp_iter': self.args.final_bp_iter, 'mp_thresh': self.args.final_bp_thresh,
                'patience': self.args.final_bp_patience, 'seed': self.args.final_bp_seed}


    def request(self, partition, fold, seed):
        partition = canonical_partition(partition, self.args.orders)
        directory = self.output / 'candidates' / partition_id(partition) / f'fold_{fold:02d}' / f'seed_{seed}'
        directory.mkdir(parents=True, exist_ok=True)
        row = self.folds[fold]
        sizes = directory / 'selected_hye_sizes.txt'
        sizes.write_text(''.join(f'{d}\n' for d in row['train']['affinity_orders']), encoding='utf-8')
        shared, omega = directory / 'shared_hye_sizes.json', directory / 'omega.json'
        dump_json(shared, partition)
        dump_json(omega, {d: 1. for d in self.args.orders})
        arguments = ['--hye_file', str(row['hye_path']), '--N', str(row['train']['N']),
                     '--selected_hye_sizes', str(sizes), '--shared_hye_sizes', str(shared),
                     '--omega', str(omega), '--save_dir', str(directory / 'mp_results'),
                     '--K', str(self.args.K), '--seed', str(seed),
                     '--n_jobs', str(self.args.inner_jobs)]
        for name in ('em_iter', 'em_thresh', 'mp_iter', 'mp_thresh', 'mp_patience', 'train_rounds',
                     'dropout', 'dropout_vertex', 'dropout_vertex_FVS', 'dropout_vertex_rest', 'dropout_FVS'):
            arguments += ['--' + name, str(getattr(self.args, name))]
        request = {'project': str(self.project), 'entry': str(self.entry), 'arguments': arguments,
                   'snapshot_dir': str(directory / 'snapshot'),
                   'closure_controls': self.closure_controls(),
                   'result_file': str(directory / 'worker_result.json'),
                   'request_id': _json_hash([self.fingerprint, partition, fold, seed]),
                   'fingerprint': self.fingerprint, 'partition': partition,
                   'fold': fold, 'seed': seed}
        path = directory / 'train_request.json'
        dump_json(path, request)
        request['request_file'] = str(path)
        return directory, request


    def launch(self, mode, request_file, log_file):
        command = [self.args.python, str(Path(__file__).resolve()), '--_paper-worker', mode, str(request_file)]
        env = os.environ.copy()
        env.update(PYTHONIOENCODING='utf-8', PYTHONHASHSEED=str(self.args.final_bp_seed))
        env['RAYON_NUM_THREADS'] = str(self.args.inner_jobs)
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[name] = '1'
        with self.slots:
            with Path(log_file).open('w', encoding='utf-8') as stream:
                run = subprocess.run(command, cwd=self.project, env=env, stdout=stream,
                                     stderr=subprocess.STDOUT, timeout=self.args.timeout)
        if run.returncode:
            tail = '\n'.join(Path(log_file).read_text(encoding='utf-8', errors='replace').splitlines()[-25:])
            raise RuntimeError(f'Worker failed ({run.returncode}); see {log_file}\n{tail}')


    def run_fit(self, partition, fold, seed):
        directory, request = self.request(partition, fold, seed)
        cache = directory / 'metrics.json'
        if cache.is_file():
            saved = json.loads(cache.read_text(encoding='utf-8'))
            if saved.get('fingerprint') != self.fingerprint:
                raise ValueError('Cached fit fingerprint differs.')
            path = Path(saved['snapshot_path'])
            if not path.is_file() or digest_file(path) != saved['snapshot_sha256']:
                raise ValueError(f'Cached snapshot missing or changed: {path}')
            return saved
        started = time.perf_counter()
        print(f'  train {partition_id(partition)} fold={fold} seed={seed}', flush=True)
        self.launch('--train-request', request['request_file'], directory / 'training.log')
        result = json.loads(Path(request['result_file']).read_text(encoding='utf-8'))
        if result.get('fingerprint') != self.fingerprint:
            raise ValueError('Training worker returned a different protocol fingerprint.')
        snapshot_path = Path(result.get('snapshot_path', Path(request['snapshot_dir']) / 'snapshot.npz'))
        with np.load(snapshot_path, allow_pickle=False) as snapshot:
            metrics = score_fit(snapshot, self.folds[fold]['validation'],
                                         snapshot['orders'].tolist(), self.args.orders)
        metrics.update(fingerprint=self.fingerprint, fold=fold, seed=seed,
                       snapshot_path=str(snapshot_path), snapshot_dir=request['snapshot_dir'],
                       snapshot_sha256=digest_file(snapshot_path),
                       worker_result=result, seconds=time.perf_counter()-started)
        dump_json(directory / 'training_complete.json', {'fingerprint': self.fingerprint,
                          'snapshot_sha256': metrics['snapshot_sha256']})
        dump_json(cache, metrics)
        return metrics


    def __call__(self, partition):
        partition = canonical_partition(partition, self.args.orders)
        records = []
        with ThreadPoolExecutor(max_workers=self.args.fold_jobs) as pool:
            futures = [pool.submit(self.run_fit, partition, fold, seed)
                       for fold in self.args.folds for seed in self.args.seeds]
            for future in as_completed(futures):
                records.append(future.result())
        records.sort(key=lambda row: (row['fold'], row['seed']))
        score = aggregate_scores(records, self.args.folds)
        score.update(partition=partition, candidate_id=partition_id(partition), fits=records,
                     fingerprint=self.fingerprint,
                     all_bp_closed=all(row['worker_result'].get('converged', False) for row in records))
        score['eligible_for_acceptance'] = score['all_bp_closed'] or self.args.allow_unconverged
        path = self.output / 'candidates' / partition_id(partition) / 'score.json'
        dump_json(path, score)
        print(f'{partition}: validation AUC={score["mean_auc"]:.6f} +/- {score["std_auc"]:.6f}', flush=True)
        # Full metrics remain in per-candidate score.json; keep search-state metadata compact.
        score['fits'] = [dict(fold=row['fold'], seed=row['seed'],
                              snapshot_path=row['snapshot_path'], weighted_auc=row['weighted_auc'])
                         for row in records]
        return score


def paper_min_group_edges(K):
    """Paper-shaped gate with user multiplier3 (the original paper uses5)."""
    if isinstance(K, bool) or int(K) != K or K < 2:
        raise ValueError('K must be an integer >=2')
    return GROUP_EDGE_MULTIPLIER * int(K) * (int(K)+1) // 2


def partition_is_admissible(partition, counts, min_edges):
    partition = canonical_partition(partition)
    counts = {int(d): int(count) for d, count in counts.items()}
    if min_edges < 0 or any(count < 0 for count in counts.values()):
        raise ValueError('Hyperedge counts and group threshold must be nonnegative')
    return all(sum(counts[d] for d in group) >= min_edges for group in partition)


def build_parser():
    p = argparse.ArgumentParser(description=_SEARCH_DESCRIPTION)
    p.add_argument('--project-dir', type=Path, default=HERE)
    p.add_argument('--entrypoint', help='Default: main_message_passing_loop.py')
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--dataset', default='high_school')
    p.add_argument('--input', '--input-npz', '--input-json', dest='input_npz', type=Path)
    p.add_argument('--cv-root', type=Path)
    p.add_argument('--output', type=Path, default=Path.cwd()/'paper_group_search_output')
    p.add_argument('--orders', nargs='+', type=int, default=[2,3,4,5])
    p.add_argument('--folds', nargs='+', type=int, default=[0,1,2,3,4])
    p.add_argument('--seeds', nargs='+', type=int, default=[234])
    p.add_argument('--split-seed', type=int, default=234)
    p.add_argument('--negative-ratio', type=float, default=1.)
    p.add_argument('--K', type=int, default=10)
    p.add_argument('--search-mode', choices=['auto','exhaustive','greedy'], default='auto')
    p.add_argument('--exhaustive-max-orders', type=int, default=6,
                   help='Auto exhaustive cutoff in DISTINCT ORDER TYPES, user override of paper4')
    p.add_argument('--max-exhaustive-partitions', type=int, default=100000,
                   help='Guard raw Bell count before training; 0 explicitly disables the guard')
    p.add_argument('--min-group-edges', type=int, default=None,
                   help='Default: user threshold3*K*(K+1)/2 on full graph; paper used5; 0 disables the constraint')
    p.add_argument('--min-gain', type=float, default=.001,
                   help='Strict AUC improvement for greedy splits; exhaustive chooses maximum AUC')
    p.add_argument('--max-evals', type=int,
                   help='Optional debug budget including shared baseline; marks truncated search incomplete')
    p.add_argument('--max-groups', type=int)
    p.add_argument('--candidate-jobs', type=int, default=2)
    p.add_argument('--fold-jobs', type=int, default=5)
    p.add_argument('--max-parallel-fits', type=int, default=5,
                   help='Global simultaneous fit-process cap across candidates, folds and seeds')
    p.add_argument('--inner-jobs', type=int, default=1)
    for suffix, default, kind in [('iter',2000,int),('thresh',1e-7,float),
                                   ('patience',10,int),('seed',567,int)]:
        p.add_argument('--final-bp-'+suffix, '--probe-'+suffix,
                       dest='final_bp_'+suffix, type=kind, default=default,
                       help='Final fixed-parameter BP closure; no Fisher probes are performed')
    for name, default, kind in [('train_rounds',3,int),('em_iter',20,int),('mp_iter',2000,int),
                              ('em_thresh',1e-5,float),('mp_thresh',1e-5,float),('mp_patience',50,int),
                              ('dropout',0.,float),('dropout_vertex',0.,float),
                              ('dropout_vertex_FVS',.8,float),('dropout_vertex_rest',.8,float),
                              ('dropout_FVS',0.,float)]:
        p.add_argument('--'+name.replace('_','-'), '--'+name, dest=name, type=kind, default=default)
    p.add_argument('--timeout', type=float)
    p.add_argument('--allow-unconverged', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p


def validate(args):
    for name in ('orders','folds','seeds'):
        values = getattr(args,name)
        if not values or len(set(values)) != len(values):
            raise ValueError(f'{name} must be nonempty and unique')
        setattr(args,name,sorted(values))
    if 2 not in args.orders or any(d < 2 for d in args.orders):
        raise ValueError('Orders must include reference2 and have sizes>=2')
    if any(f < 0 or (args.cv_root is None and f > 4) for f in args.folds):
        raise ValueError('Automatic fivefold uses fold indices0 through4')
    if any(s < 0 or s >= 2**32 for s in args.seeds+[args.split_seed,args.final_bp_seed]):
        raise ValueError('Seeds must be uint32')
    if args.input_npz is not None and args.cv_root is not None:
        raise ValueError('Use automatic input or --cv-root, not both')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.dataset):
        raise ValueError('Dataset must be a name, not a path')
    for name in ('K','exhaustive_max_orders','candidate_jobs','fold_jobs','max_parallel_fits',
                 'inner_jobs','final_bp_iter','final_bp_patience','train_rounds','em_iter',
                 'mp_iter','mp_patience'):
        if getattr(args,name) < 1:
            raise ValueError(f'{name} must be positive')
    if args.K < 2:
        raise ValueError('K must be >=2')
    if args.max_exhaustive_partitions < 0:
        raise ValueError('max-exhaustive-partitions must be >=0')
    if args.min_group_edges is not None and args.min_group_edges < 0:
        raise ValueError('min-group-edges must be >=0')
    if args.max_evals is not None and args.max_evals < 1:
        raise ValueError('max-evals must include at least the shared baseline')
    if args.max_groups is not None and not 1 <= args.max_groups <= len(args.orders):
        raise ValueError('max-groups must be between1 and the number of order types')
    for name in ('negative_ratio','final_bp_thresh','em_thresh','mp_thresh'):
        if not math.isfinite(getattr(args,name)) or getattr(args,name) <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if not math.isfinite(args.min_gain) or args.min_gain < 0:
        raise ValueError('min-gain must be finite and nonnegative')
    for name in ('dropout','dropout_vertex','dropout_vertex_FVS','dropout_vertex_rest','dropout_FVS'):
        if not math.isfinite(getattr(args,name)) or not 0 <= getattr(args,name) <= 1:
            raise ValueError(f'{name} must be within[0,1]')
    if args.timeout is not None and (not math.isfinite(args.timeout) or args.timeout <= 0):
        raise ValueError('timeout must be finite and positive')


def write_progress(output, state):
    dump_json(output/'search_state.json', state)
    baseline = state['baseline']['mean_auc']
    selected = canonical_partition(state['best_partition'])
    temporary = output/'candidate_auc.csv.tmp'
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['partition','mean_auc','std_auc',
            'gain_over_shared','eligible','selected','candidate_id'])
        writer.writeheader()
        for item in state['evaluations']:
            score, partition = item['score'], canonical_partition(item['partition'])
            writer.writerow(dict(partition=json.dumps(partition), mean_auc=score['mean_auc'],
                std_auc=score['std_auc'], gain_over_shared=score['mean_auc']-baseline,
                eligible=score.get('eligible_for_acceptance',True), selected=partition==selected,
                candidate_id=partition_id(partition)))
    os.replace(temporary, output/'candidate_auc.csv')
    dump_json(output/'search_history.json', state['history'])


def plan_search(args, counts):
    threshold = paper_min_group_edges(args.K) if args.min_group_edges is None else args.min_group_edges
    strategy = args.search_mode
    if strategy == 'auto':
        strategy = 'exhaustive' if len(args.orders) <= args.exhaustive_max_orders else 'greedy'
    allowed = lambda groups: (partition_is_admissible(groups,counts,threshold)
                             and (args.max_groups is None or len(groups) <= args.max_groups))
    raw_count = bell_number(len(args.orders)) if strategy == 'exhaustive' else None
    guard = (strategy == 'exhaustive' and args.max_exhaustive_partitions != 0
             and raw_count > args.max_exhaustive_partitions)
    examples, valid_count = [], None
    # Even an explicitly disabled training guard must not create a gigantic dry-run file.
    if strategy == 'exhaustive' and raw_count <= 100000:
        valid_count = 0
        for partition in enumerate_partitions(args.orders):
            if allowed(partition):
                valid_count += 1
                if len(examples) < 50:
                    examples.append(partition)
    elif strategy == 'greedy':
        examples = [p for p in contiguous_splits([args.orders]) if allowed(p)]
    return dict(strategy=strategy, order_type_count=len(args.orders), raw_partition_count=raw_count,
                exhaustive_max_orders=args.exhaustive_max_orders,
                max_exhaustive_partitions=args.max_exhaustive_partitions,
                exhaustive_guard_exceeded=guard, admissible_partition_count=valid_count,
                candidate_examples=examples, min_group_edges=threshold,
                group_edge_multiplier_default=GROUP_EDGE_MULTIPLIER,
                group_edge_threshold_mode='multiplier_default' if args.min_group_edges is None else 'absolute_override',
                paper_reference_multiplier=5,
                full_order_counts=counts, count_scope='complete observed graph; not training folds',
                sparse_orders=[d for d in args.orders if counts[d] < threshold],
                fits_per_partition=len(args.folds)*len(args.seeds),
                full_fit_count=(valid_count*len(args.folds)*len(args.seeds)
                                if valid_count is not None else None),
                candidate_jobs=args.candidate_jobs, fold_jobs=args.fold_jobs,
                max_parallel_fits=args.max_parallel_fits), allowed


def run(args):
    validate(args)
    project, entry, folds, fingerprint, counts = prepare(args)
    plan, allowed = plan_search(args,counts)
    output = args.output.resolve()
    dump_json(output/'search_plan.json',plan)
    if not allowed([args.orders]):
        raise ValueError(f'No admissible group: total selected edges={sum(counts.values())}, '
                         f'minimum={plan["min_group_edges"]}. Reduce K or explicitly override --min-group-edges.')
    print(f'Training entrypoint: {entry}\nOutput: {output}',flush=True)
    print(f'{len(args.orders)} order types; mode={plan["strategy"]}; '
          f'raw partitions={plan["raw_partition_count"]}; '
          f'admissible partitions={plan["admissible_partition_count"]}; '
          f'group edge minimum={plan["min_group_edges"]}.',flush=True)
    print(f'candidate-jobs={args.candidate_jobs}, fold-jobs={args.fold_jobs}, '
          f'GLOBAL fit cap={args.max_parallel_fits}, inner-jobs={args.inner_jobs}.',flush=True)
    evaluator = Evaluator(args,project,entry,folds,fingerprint)
    if args.dry_run:
        requests = [evaluator.request([args.orders],f,s)[1] for f in args.folds for s in args.seeds]
        dump_json(output/'dry_run.json',dict(
            plan=plan, initial_partition=[args.orders], initial_requests=requests,
            note='Preflight only; no training subprocess started. Guarded spaces require explicit action.'))
        print('Preflight complete: inspect search_plan.json and dry_run.json.',flush=True)
        return
    state = search_partitions(args.orders,evaluator,strategy=args.search_mode,
        exhaustive_max_orders=args.exhaustive_max_orders,
        max_exhaustive_partitions=args.max_exhaustive_partitions,
        admissible=allowed,candidate_jobs=args.candidate_jobs,min_gain=args.min_gain,
        max_evals=args.max_evals,on_step=lambda snapshot:write_progress(output,snapshot))
    best, baseline = state['best'], state['baseline']
    result = dict(recommended_groups=state['best_partition'],
        baseline_mean_auc=baseline['mean_auc'], selected_mean_auc=best['mean_auc'],
        selected_std_auc=best['std_auc'],
        paired_fold_gains=(np.asarray(best['fold_auc'])-baseline['fold_auc']).tolist(),
        num_evaluated_partitions=len(state['evaluations']), stop_reason=state['stop_reason'],
        search_strategy=state['strategy'], search_complete=state['search_complete'],
        raw_partition_count=state['raw_partition_count'],
        complete_exhaustive=state['strategy']=='exhaustive' and state['search_complete'],
        min_group_edges=plan['min_group_edges'], group_count_scope=plan['count_scope'],
        group_edge_multiplier_default=GROUP_EDGE_MULTIPLIER,
        group_edge_threshold_mode=plan['group_edge_threshold_mode'],
        full_order_counts=counts, sparse_orders=plan['sparse_orders'],
        selected_snapshot_paths=[row['snapshot_path'] for row in best['fits']],
        all_selected_bp_closed=best['all_bp_closed'],
        allow_unconverged_budget_mode=args.allow_unconverged,
        score_role='validation_used_for_group_selection',
        beta_policy='not searched; original per-order scales retained',
        method_scope='paper grouping strategy; original BP backend and fivefold weighted AUC retained')
    write_progress(output,state)
    dump_json(output/'best_result.json',result)
    dump_json(output/'best_groups.json',state['best_partition'])
    print(f'Selected groups: {state["best_partition"]}; AUC {baseline["mean_auc"]:.6f} -> '
          f'{best["mean_auc"]:.6f}; complete={state["search_complete"]}; '
          f'stop={state["stop_reason"]}',flush=True)


def search_main():
    args = build_parser().parse_args()
    try:
        run(args)
    except (ValueError,FileNotFoundError,RuntimeError,subprocess.TimeoutExpired) as error:
        print(f'ERROR: {error}',file=sys.stderr)
        return 1
    return 0


# ==================== Unified command-line entry ====================
def main():
    if sys.argv[1:2] == ['--_paper-worker']:
        return worker_main(sys.argv[2:])
    return search_main()

if __name__ == '__main__':
    raise SystemExit(main())
