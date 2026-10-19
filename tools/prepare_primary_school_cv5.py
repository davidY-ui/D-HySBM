"""Create static Primary School data and reproduce the High School CV5 schema.

No model is trained. Repeated contact events are aggregated by their node sets.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil

import numpy as np


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def edge_array(edges):
    result = np.empty(len(edges), dtype=object)
    result[:] = edges
    return result


def negative_edges(n, size, count, observed, rng):
    available = math.comb(n, size) - sum(len(edge) == size for edge in observed)
    if count > available:
        raise ValueError(f'Not enough order-{size} nonedges: {available} < {count}')
    sampled, attempts = set(), 0
    while len(sampled) < count:
        edge = tuple(sorted(map(int, rng.choice(n, size, replace=False))))
        attempts += 1
        if edge not in observed:
            sampled.add(edge)
        if attempts > max(10000, count * 1000):
            raise RuntimeError('Negative rejection sampling exceeded the reference limit.')
    return sorted(sampled)


def load_primary(source, metadata):
    document = json.loads(source.read_text(encoding='utf-8'))
    nodes = {str(node) for node in document['node-data']}
    with metadata.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    mapping = {str(row['json_nodeID']): int(row['nodeID']) for row in rows}
    if len(mapping) != len(rows) or set(mapping) != nodes:
        raise ValueError('Metadata and JSON nodes must have a one-to-one mapping.')
    n = len(nodes)
    if sorted(mapping.values()) != list(range(n)):
        raise ValueError('Metadata nodeID must be the complete range 0..N-1.')
    if set(document['edge-data']) != set(document['edge-dict']):
        raise ValueError('Event metadata and edge IDs differ.')
    contacts = Counter()
    for raw_nodes in document['edge-dict'].values():
        edge = tuple(sorted(mapping[str(node)] for node in raw_nodes))
        if len(edge) < 2 or len(set(edge)) != len(edge):
            raise ValueError('An event has fewer than two distinct nodes.')
        contacts[edge] += 1
    edges = sorted(contacts, key=lambda edge: (len(edge), edge))
    weights = np.array([contacts[edge] for edge in edges], dtype=np.int64)
    if int(weights.sum()) != len(document['edge-dict']):
        raise AssertionError('Contact counts were not preserved.')
    return n, edges, weights, rows


def reference_schema(reference):
    samples = {
        'train': reference / 'fold_00/high_school_train0.npz',
        'test': reference / 'fold_00/high_school_test0.npz',
        'split': reference / 'split_assignment.npz',
    }
    schema = {}
    for kind, path in samples.items():
        with np.load(path, allow_pickle=True) as saved:
            schema[kind] = {key: saved[key].dtype for key in saved.files}
    return schema


def prepare(source, metadata, reference, output, published_output=None):
    source, metadata, reference, output = map(lambda p: Path(p).resolve(),
                                              (source, metadata, reference, output))
    published = Path(published_output).resolve() if published_output else output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite a nonempty folder: {output}')
    reference_manifest = json.loads((reference / 'manifest.json').read_text(encoding='utf-8'))
    folds, seed = int(reference_manifest['folds']), int(reference_manifest['seed'])
    schema = reference_schema(reference)
    n, edges, weights, metadata_rows = load_primary(source, metadata)
    observed = set(edges)
    sizes = np.array(list(map(len, edges)), dtype=np.int64)
    orders = sorted(map(int, np.unique(sizes)))
    if any(int(np.sum(sizes == d)) < folds for d in orders):
        raise ValueError('Every retained order must have at least one positive in every fold.')
    output.mkdir(parents=True, exist_ok=True)

    # The full-data source follows high_school.npz: B contains the contact weight.
    full_b = np.zeros((n, len(edges)), dtype=np.int64)
    for column, edge in enumerate(edges):
        full_b[list(edge), column] = weights[column]
    full_path = output / 'primary_school.npz'
    np.savez_compressed(full_path, hyperedges=edge_array(edges), A=weights, B=full_b)
    shutil.copy2(metadata, output / 'primary_school_meta.csv')

    fold_id = np.full(len(edges), -1, dtype=np.int64)
    for d in orders:
        indices = np.flatnonzero(sizes == d)
        rng = np.random.default_rng(np.random.SeedSequence([seed, d, 0]))
        for fold, group in enumerate(np.array_split(rng.permutation(indices), folds)):
            fold_id[group] = fold
    assert np.all(fold_id >= 0)

    order_dir = output / 'orders'
    order_dir.mkdir()
    higher = [d for d in orders if d != 2]
    for length in range(len(higher) + 1):
        for chosen in itertools.combinations(higher, length):
            selected = ([2] if 2 in orders else []) + list(chosen)
            if selected:
                (order_dir / ('_'.join(map(str, selected)) + '.txt')).write_text(
                    ''.join(f'{d}\n' for d in selected), encoding='ascii')

    manifest = {
        'dataset': 'primary_school',
        'source': str(published / 'primary_school.npz'),
        'source_sha256': file_hash(full_path),
        'raw_source': str(source), 'raw_source_sha256': file_hash(source),
        'metadata_source': str(metadata), 'metadata_sha256': file_hash(metadata),
        'reference_cv5': str(reference),
        'reference_manifest_sha256': file_hash(reference / 'manifest.json'),
        'N': n, 'observed_unique_edges': len(edges),
        'raw_contact_events': int(weights.sum()),
        'counts_by_order': {str(d): int(np.sum(sizes == d)) for d in orders},
        'event_counts_by_order': {str(d): int(weights[sizes == d].sum()) for d in orders},
        'protocol': reference_manifest['protocol'], 'folds': folds, 'seed': seed,
        'numpy_version': np.__version__,
        'split_unit': reference_manifest['split_unit'],
        'node_policy': 'existing primary_school_meta.csv mapping 0..N-1; never renumber per fold',
        'negative_policy': reference_manifest['negative_policy'],
        'weight_policy': 'full NPZ preserves weighted B and A; fold train B is binary uint8 and A retains event counts, as in high_school_cv5',
        'temporal_policy': 'static unique-node-set aggregation; event counts retained, timestamps remain in the original JSON',
        'edge_order': 'sorted by hyperedge size, then lexicographically, as in high_school.npz',
        'rare_order_note': f'Order 5 has {int(np.sum(sizes == 5))} unique positives, yielding 1-2 per test fold.',
        'metadata_classes': len({row['classID'] for row in metadata_rows}),
        'initialization_seeds': reference_manifest['initialization_seeds'],
        'selection_policy': reference_manifest['selection_policy'],
        'cv_note': reference_manifest['cv_note'],
        'training_cli': reference_manifest['training_cli'],
        'fold_details': [],
    }

    def cast(kind, values):
        if set(values) != set(schema[kind]):
            raise ValueError(f'{kind} NPZ keys do not match the High School reference.')
        return {key: np.asarray(value, dtype=schema[kind][key]) for key, value in values.items()}

    for fold in range(folds):
        folder = output / f'fold_{fold:02d}'
        folder.mkdir()
        train_idx, test_idx = np.flatnonzero(fold_id != fold), np.flatnonzero(fold_id == fold)
        train, positive = [edges[i] for i in train_idx], [edges[i] for i in test_idx]
        rng = np.random.default_rng(np.random.SeedSequence([seed, fold, 1]))
        negative = []
        for d in orders:
            negative.extend(negative_edges(n, d, int(np.sum(sizes[test_idx] == d)), observed, rng))
        assert set(train).isdisjoint(positive) and observed.isdisjoint(negative)
        assert len(set(negative)) == len(negative) == len(positive)
        binary_b = np.zeros((n, len(train)), dtype=np.uint8)
        for column, edge in enumerate(train):
            binary_b[list(edge), column] = 1
        np.savez_compressed(folder / f'primary_school_train{fold}.npz', **cast('train', {
            'hyperedges': edge_array(train), 'A': weights[train_idx], 'B': binary_b,
            'N': n, 'edge_sizes': sizes[train_idx], 'original_indices': train_idx,
            'selected_hye_sizes': orders, 'fold': fold,
        }))
        (folder / 'train.txt').write_text(''.join(' '.join(map(str, edge)) + '\n' for edge in train), encoding='ascii')
        candidates = positive + negative
        np.savez_compressed(folder / f'primary_school_test{fold}.npz', **cast('test', {
            'hyperedges': edge_array(candidates),
            'y_true': np.r_[np.ones(len(positive), dtype=np.int8), np.zeros(len(negative), dtype=np.int8)],
            'edge_sizes': np.array(list(map(len, candidates)), dtype=np.int64),
            'N': n, 'fold': fold,
            'original_indices': np.r_[test_idx, np.full(len(negative), -1, dtype=np.int64)],
            'observed_weights': np.r_[weights[test_idx], np.zeros(len(negative), dtype=weights.dtype)],
        }))
        details = {
            'fold': fold, 'train_positives': len(train), 'test_positives': len(positive),
            'test_negatives': len(negative), 'train_fraction': len(train) / len(edges),
            'train_isolated_nodes': int(np.sum(binary_b.sum(axis=1) == 0)),
            'by_order': {str(d): {'train': int(np.sum(sizes[train_idx] == d)),
                                 'test_positive': int(np.sum(sizes[test_idx] == d)),
                                 'test_negative': sum(len(edge) == d for edge in negative)} for d in orders},
        }
        write_json(folder / 'counts.json', details)
        manifest['fold_details'].append(details)
        print(f'fold_{fold:02d}: train={len(train)}, test+={len(positive)}, test-={len(negative)}', flush=True)
    np.savez_compressed(output / 'split_assignment.npz', **cast('split', {
        'original_indices': np.arange(len(edges)), 'fold_id': fold_id, 'edge_sizes': sizes, 'N': n,
    }))
    write_json(output / 'manifest.json', manifest)

    counts_text = '\n'.join(f"| {item['fold']:02d} | {item['train_positives']} | {item['test_positives']} | {item['test_negatives']} |"
                            for item in manifest['fold_details'])
    readme = f'''# Primary School：与 High School 一致的五折数据

共 {n} 个节点、{len(edges)} 条唯一超边，来自 {int(weights.sum())} 次接触事件。
源 JSON 未修改。使用已有 primary_school_meta.csv 节点映射；所有折均保留 0..{n - 1} 节点。
本目录只有数据与生成脚本，没有训练模型或产生 AUC 结果。

## 文件结构

- primary_school.npz：完整静态网络，字段与 high_school.npz 相同：hyperedges、A、B。
  A 为接触事件次数，B 在超边端点处填 A。
- primary_school_meta.csv：原 metadata 文件的原样副本，可用于事后 NMI 评价。
- fold_00 至 fold_04：每折含 primary_school_train0.npz / primary_school_test0.npz
  （编号随折变化）、train.txt、counts.json。
- orders：与 high_school_cv5 一样的八种保留阶集合文件，每行一个阶数。
- split_assignment.npz：原始唯一超边索引、测试折号、阶数与 N。
- manifest.json：源数据哈希、节点政策、采样种子、计数与处理说明。

train NPZ 字段、dtype 与 High School 对应文件一致：hyperedges、A、B、N、
edge_sizes、original_indices、selected_hye_sizes、fold。
每折的训练 B 是二值 uint8 关联矩阵，与 High School 五折文件一致。
test NPZ 字段：hyperedges、y_true、edge_sizes、N、fold、original_indices、observed_weights。
负边的 y_true=0、original_indices=-1、observed_weights=0。
NPZ 超边为一维 object 数组，加载时使用 allow_pickle=True，仅加载可信文件。

## 划分规则

沿用 High School 的种子 {seed} 与 NumPy SeedSequence 规则，按阶数分别打乱并分成五份。
同一节点集合的重复接触先合并为权重，不能跨训练和测试拆开。
每条真实超边恰好测试一次，各折约 80% 训练、20% 测试。
每折、每阶按 1:1 抽取测试负边，负边排除完整观测网络，折内不重复。
不同模型或保留阶组合共用同一份测试样本。
数据为静态聚合，时间戳仍保留在源 JSON 中。

| 折 | 训练正边 | 测试正边 | 测试负边 |
| --- | ---: | ---: | ---: |
{counts_text}

## 使用

训练使用 fold_XX/train.txt，并指定 --N {n}；不把测试文件传给训练入口。
评估读取对应 primary_school_testX.npz。
现有 High School 专用脚本若写死 high_school_train/test 文件名，需改成 primary_school_train/test；
本目录保留相同内容 schema，但不伪装为 High School 数据。
类别标签仅用于事后分析，没有参与划分或负采样。

生成器拒绝覆盖非空目录；重现时将 --output 指向新目录。
'''
    (output / 'README.md').write_text(readme, encoding='utf-8')
    shutil.copy2(Path(__file__), output / 'prepare_primary_school_cv5.py')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--published-output', type=Path)
    args = parser.parse_args()
    result = prepare(args.input, args.metadata, args.reference, args.output, args.published_output)
    print(f"Prepared {result['N']} nodes and {result['observed_unique_edges']} unique hyperedges.")
