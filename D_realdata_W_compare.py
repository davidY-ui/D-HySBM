#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare W/p and differ/share on identical data/order combinations/seeds (Python 3.10).

Run in the hypergraph-message-passing project with an HSBM training entry
supporting --affinity-update. DCHSBM is unchanged.
--mode all runs four cases; --mode w-share p-share selects specific cases.
Retain training metrics; this script does not compute hyperedge prediction AUC.
Use --node-assignments to compute Overlap.
The default training entry is main_message_passing_loop.py.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
MODES = {
    'w-differ': ('paper-w', 'differ'),
    'w-share': ('paper-w', 'share'),
    'p-differ': ('direct-p', 'differ'),
    'p-share': ('direct-p', 'share'),
}
DEFAULT_TRAIN_LIST = [f'high_school_train{i}' for i in (1, 2, 3, 4, 0)]
DEFAULT_ORDER_SETS = [[2], [2, 3], [2, 4], [2, 5], [2, 3, 4],
                      [2, 3, 5], [2, 4, 5], [2, 3, 4, 5]]
SCALAR_KEYS = ['overlap', 'overlap_active', 'best_free_energy', 'em_total_time',
               'update_message_ratio', 'update_vertex_ratio']
DIAGNOSTIC_KEYS = ['all_free_energy', 'n_diff', 'c_diff', 'log_marginal_diff',
                   'mp_iter_per_em_iter', 'em_time_per_iter']
TRAIN_PARAMETERS = ['em_iter', 'em_thresh', 'mp_iter', 'mp_thresh', 'mp_patience',
                    'train_rounds', 'dropout', 'dropout_vertex', 'dropout_vertex_FVS',
                    'dropout_vertex_rest', 'dropout_FVS']


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def object_array(values):
    values = list(values)
    result = np.empty(len(values), dtype=object)
    for i, value in enumerate(values):
        result[i] = value
    return result


def scalar_from_npz(data, key):
    if key not in data.files:
        return float('nan')
    value = np.asarray(data[key], dtype=object).reshape(-1)
    if not len(value) or value[0] is None:
        return float('nan')
    try:
        return float(value[0])
    except (TypeError, ValueError):
        return float('nan')


def shared_groups(orders, sharing):
    return [list(orders)] if sharing == 'share' else [[int(d)] for d in orders]


def canonical_groups(groups):
    return tuple(sorted(tuple(sorted(map(int, group))) for group in groups))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--project-dir', type=Path, default=PROJECT_DIR)
    parser.add_argument('--entrypoint', default='main_message_passing_loop.py')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--mode', '--modes', nargs='+', choices=['all', *MODES],
                        default=['all'], help='all=四组；也可指定一组或多组')
    parser.add_argument('--train-list', nargs='+', default=DEFAULT_TRAIN_LIST)
    parser.add_argument('--order-sets', default=json.dumps(DEFAULT_ORDER_SETS),
                        help='JSON，例如 [[2,3],[2,3,4,5]]')
    parser.add_argument('--seeds', nargs='+', type=int, default=[234])
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--n-jobs', type=int, default=40,
                        help='全局训练进程上限；-1使用90%%逻辑CPU预算')
    parser.add_argument('--inner-jobs', type=int,
                        help='每个训练的n_jobs/Rayon线程；省略时按CPU预算自动分配')
    parser.add_argument('--node-assignments', type=Path,
                        help='可选真实节点标签，仅传给原入口用于计算Overlap')
    parser.add_argument('--dry-run', action='store_true')
    for name, default, kind in [('em_iter', 20, int), ('em_thresh', 1e-5, float),
                              ('mp_iter', 2000, int), ('mp_thresh', 1e-5, float),
                              ('mp_patience', 50, int), ('train_rounds', 10, int),
                              ('dropout', 0., float), ('dropout_vertex', 0., float),
                              ('dropout_vertex_FVS', .8, float),
                              ('dropout_vertex_rest', .8, float), ('dropout_FVS', 0., float)]:
        parser.add_argument('--' + name.replace('_', '-'), '--' + name,
                            dest=name, type=kind, default=default)
    return parser


def validate(args):
    if 'all' in args.mode:
        if args.mode != ['all']:
            raise ValueError('--mode all不能再和其他模式同时指定')
        args.mode = list(MODES)
    elif len(set(args.mode)) != len(args.mode):
        raise ValueError('模式不能重复')
    # Use a fixed ordering so argument order does not change task IDs or result paths.
    args.mode = [name for name in MODES if name in args.mode]
    if not args.train_list or len(set(args.train_list)) != len(args.train_list):
        raise ValueError('训练集列表不能为空或重复')
    for name in args.train_list:
        if not name or not name.replace('_', '').replace('-', '').isalnum():
            raise ValueError(f'训练集须为注册名称，不含路径或后缀：{name!r}')
    try:
        sets = json.loads(args.order_sets)
    except (ValueError, TypeError) as error:
        raise ValueError('--order-sets须为JSON二维整数列表') from error
    if not isinstance(sets, list) or not sets:
        raise ValueError('--order-sets须为非空二维整数列表')
    for sizes in sets:
        if (not isinstance(sizes, list) or not sizes
                or any(isinstance(d, bool) or not isinstance(d, int) or d < 2 for d in sizes)
                or len(set(sizes)) != len(sizes)):
            raise ValueError('每个阶数组合须包含不同的整数阶数>=2')
    args.order_sets = [sorted(sizes) for sizes in sets]
    if len({tuple(s) for s in args.order_sets}) != len(args.order_sets):
        raise ValueError('阶数组合不能重复')
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('种子不能为空或重复')
    if any(seed < 0 or seed + (args.train_rounds - 1) * 1024 >= 2**32 for seed in args.seeds):
        raise ValueError('种子及各重启的种子须在uint32范围内')
    if args.K < 2 or args.n_jobs == 0 or args.n_jobs < -1:
        raise ValueError('K须>=2；n-jobs须为正整数或-1')
    if args.inner_jobs is not None and args.inner_jobs < 1:
        raise ValueError('inner-jobs须为正整数')
    for key in ('em_iter', 'mp_iter', 'mp_patience', 'train_rounds'):
        if getattr(args, key) < 1:
            raise ValueError(f'{key}须为正整数')
    for key in ('em_thresh', 'mp_thresh'):
        value = getattr(args, key)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key}须为有限正数')
    for key in ('dropout', 'dropout_vertex', 'dropout_vertex_FVS', 'dropout_vertex_rest', 'dropout_FVS'):
        value = getattr(args, key)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{key}须在[0,1]内')


def prepare(args):
    validate(args)
    project = args.project_dir.resolve()
    entry = Path(args.entrypoint)
    if not entry.is_absolute():
        entry = project / entry
    entry = entry.resolve()
    inputs = {name: project / 'data/processed_input' / f'{name}.npz'
              for name in args.train_list}
    required = [entry, *inputs.values()]
    if args.node_assignments is not None:
        args.node_assignments = args.node_assignments.resolve()
        required.append(args.node_assignments)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError('尚未启动训练，以下文件不存在：\n' + '\n'.join(missing))
    source = entry.read_text(encoding='utf-8-sig')
    for flags in [('--affinity-update', '--affinity_update'), ('--shared_hye_sizes',), ('--model',)]:
        if not any(flag in source for flag in flags):
            raise ValueError(f'训练入口不支持{flags[0]}，请使用配套新版入口：{entry}')
    count = len(args.mode) * len(args.order_sets) * len(args.train_list) * len(args.seeds)
    cpu_budget = max(1, int((os.cpu_count() or 1) * .9))
    if args.inner_jobs is not None and args.inner_jobs > cpu_budget:
        raise ValueError(f'inner-jobs={args.inner_jobs}超过整机CPU预算{cpu_budget}')
    outer = min(cpu_budget if args.n_jobs == -1 else args.n_jobs, count, cpu_budget)
    inner = args.inner_jobs or max(1, cpu_budget // outer)
    if outer * inner > cpu_budget:
        outer = max(1, min(outer, cpu_budget // inner))
    configuration = {key: getattr(args, key) for key in TRAIN_PARAMETERS}
    configuration.update(K=args.K, modes=args.mode, order_sets=args.order_sets,
                         train_list=args.train_list, seeds=args.seeds,
                         inner_jobs=inner, model='hsbm',
                         node_assignments=str(args.node_assignments) if args.node_assignments else None)
    hashes = {str(path): digest_file(path) for path in required}
    sources = {str(entry): hashes[str(entry)], 'comparison_runner': digest_file(__file__)}
    for path in sorted((project / 'src').rglob('*.py')):
        sources[str(path.relative_to(project))] = digest_file(path)
    protocol = dict(configuration=configuration, inputs=hashes, sources=sources,
                    numpy_version=np.__version__, metric_scope='original training metrics; no AUC computed')
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True,
                                            ensure_ascii=False).encode()).hexdigest()
    output = (args.output or project / 'data/synthetic_m_realdata/W_compare').resolve()
    root = output / f'run_{fingerprint[:12]}'
    root.mkdir(parents=True, exist_ok=True)
    old = root / 'protocol.json'
    if old.is_file() and json.loads(old.read_text(encoding='utf-8')) != protocol:
        raise ValueError(f'输出协议冲突，请换--output目录：{root}')
    write_json(old, protocol)
    return project, entry, inputs, root, fingerprint, outer, inner


def prepare_tasks(args, project, entry, inputs, root, fingerprint, inner):
    tasks = []
    for mode in args.mode:
        affinity, sharing = MODES[mode]
        for idx_sizes, sizes in enumerate(args.order_sets):
            for idx_train, dataset in enumerate(args.train_list):
                for seed in args.seeds:
                    idx = len(tasks)
                    directory = (root / mode / dataset /
                                 f'case_{idx_sizes:03d}_sizes_{"_".join(map(str, sizes))}' /
                                 f'seed_{seed}')
                    directory.mkdir(parents=True, exist_ok=True)
                    sizes_file = directory / 'selected_hye_sizes.txt'
                    np.savetxt(sizes_file, sizes, fmt='%d')
                    groups = shared_groups(sizes, sharing)
                    groups_file = directory / 'shared_hye_sizes.json'
                    write_json(groups_file, groups)
                    mp_dir = directory / 'mp_results'
                    mp_dir.mkdir(exist_ok=True)
                    command = [sys.executable, str(entry), '--model', 'hsbm',
                               '--affinity-update', affinity, '--real_dataset', dataset,
                               '--selected_hye_sizes', str(sizes_file),
                               '--shared_hye_sizes', str(groups_file), '--save_dir', str(mp_dir),
                               '--K', str(args.K), '--seed', str(seed), '--n_jobs', str(inner)]
                    for key in TRAIN_PARAMETERS:
                        command += ['--' + key, str(getattr(args, key))]
                    if args.node_assignments is not None:
                        command += ['--node_assignments', str(args.node_assignments)]
                    task = dict(idx=idx, idx_train=idx_train, idx_sizes=idx_sizes,
                                mode=mode, affinity_update=affinity, sharing=sharing,
                                seed=seed, selected_hye_sizes=sizes, shared_hye_sizes=groups,
                                real_dataset=dataset, dataset=dataset, train_file=str(inputs[dataset]),
                                directory=str(directory), inferred_params_path=str(mp_dir / 'inferred_params.npz'),
                                command=command, fingerprint=fingerprint, K=args.K)
                    write_json(directory / 'task.json', task)
                    tasks.append(task)
    return tasks


def read_result(task):
    path = Path(task['inferred_params_path'])
    with np.load(path, allow_pickle=True) as data:
        if ('affinity_update' not in data.files
                or str(np.asarray(data['affinity_update']).item()) != task['affinity_update']):
            raise ValueError(f'训练结果未确认请求的更新模式：{path}')
        if ('shared_hye_sizes' not in data.files
                or canonical_groups(np.asarray(data['shared_hye_sizes'], dtype=object).tolist())
                != canonical_groups(task['shared_hye_sizes'])):
            raise ValueError(f'训练结果中的分组不符：{path}')
        selected = np.asarray(data['selected_hye_sizes'], dtype=int).ravel().tolist()
        if set(selected) != set(task['selected_hye_sizes']) or len(selected) != len(set(selected)):
            raise ValueError(f'训练结果中的阶数不符，检查是否有空阶层：{path}')
        p, n = np.asarray(data['p']), np.asarray(data['n']).reshape(-1)
        if p.shape != (len(selected), task['K'], task['K']) or n.shape != (task['K'],):
            raise ValueError(f'训练结果的p/n形状不符：{path}')
        if task['mode'] == 'p-share' and not all(np.allclose(matrix, p[0]) for matrix in p):
            raise ValueError(f'p-share结果中的完整p不相同：{path}')
        result = {key: value for key, value in task.items() if key != 'command'}
        result.update(p=p, n=n, log_marginals=np.asarray(data['log_marginals']))
        for key in SCALAR_KEYS:
            result[key] = scalar_from_npz(data, key)
        for key in DIAGNOSTIC_KEYS:
            result[key] = np.asarray(data[key]) if key in data.files else None
    return result


def run_task(task, project, inner):
    directory = Path(task['directory'])
    completion = directory / 'complete.json'
    path = Path(task['inferred_params_path'])
    if completion.is_file():
        saved = json.loads(completion.read_text(encoding='utf-8'))
        if (saved['fingerprint'] != task['fingerprint'] or not path.is_file()
                or saved['inferred_params_sha256'] != digest_file(path)):
            raise ValueError(f'已有结果的缓存记录不一致：{directory}')
        result = read_result(task)
        result.update(cached=True, wall_seconds=saved['wall_seconds'])
        print(f'[cache] {task["mode"]} {task["dataset"]} D={task["selected_hye_sizes"]} seed={task["seed"]}', flush=True)
        return result
    env = os.environ.copy()
    env.update(PYTHONIOENCODING='utf-8', PYTHONHASHSEED=str(task['seed']),
               RAYON_NUM_THREADS=str(inner))
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '1'
    prefix = f'{task["mode"]} {task["dataset"]} D={task["selected_hye_sizes"]} seed={task["seed"]}'
    print(f'[start {task["idx"]}] {prefix}', flush=True)
    started = time.perf_counter()
    log_file = directory / 'mp_results/run_log.txt'
    # Outputs without a complete marker are incomplete; remove stale final NPZ before retrying the entry point.
    if path.is_file():
        path.unlink()
    with log_file.open('w', encoding='utf-8') as stream:
        process = subprocess.Popen(task['command'], cwd=project, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace', bufsize=1)
        try:
            for number, line in enumerate(process.stdout, 1):
                stream.write(line)
                if number <= 20 or number % 50 == 0:
                    print(f'[{prefix}] {line}', end='', flush=True)
            process.wait()
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                process.wait()
    if process.returncode:
        tail = '\n'.join(log_file.read_text(encoding='utf-8', errors='replace').splitlines()[-50:])
        raise RuntimeError(f'训练失败：{prefix}\n日志：{log_file}\n{tail}')
    result = read_result(task)
    seconds = time.perf_counter() - started
    write_json(completion, dict(fingerprint=task['fingerprint'], wall_seconds=seconds,
                               inferred_params_sha256=digest_file(path)))
    result.update(cached=False, wall_seconds=seconds)
    print(f'[done] {prefix}; overlap={result["overlap"]:.6f}; F={result["best_free_energy"]:.6f}', flush=True)
    return result


def write_csv(path, fields, rows):
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_summary(root, args, records, outer, inner):
    directory = root / 'summary'
    directory.mkdir(exist_ok=True)
    records = sorted(records, key=lambda row: row['idx'])
    summary = {key: np.array([row[key] for row in records], dtype=float) for key in SCALAR_KEYS}
    summary.update(n=np.stack([row['n'] for row in records]),
                   selected_hye_sizes_list=object_array(row['selected_hye_sizes'] for row in records),
                   p_list=object_array(row['p'] for row in records),
                   log_marginals_list=object_array(row['log_marginals'] for row in records),
                   case_results=object_array(records), train_list=np.array(args.train_list),
                   seeds=np.array(args.seeds), mode_list=np.array(args.mode),
                   K=np.asarray(args.K), outer_n_jobs=np.asarray(outer), inner_n_jobs=np.asarray(inner),
                   train_rounds=np.asarray(args.train_rounds))
    for key in DIAGNOSTIC_KEYS:
        summary[key + '_list'] = object_array(row[key] for row in records)
    for destination, key in [('real_dataset', 'dataset'), ('train_file', 'train_file'),
                             ('inferred_params_paths', 'inferred_params_path'), ('mode', 'mode'),
                             ('affinity_update', 'affinity_update'), ('sharing', 'sharing'),
                             ('seed', 'seed'), ('task_indices', 'idx'), ('train_indices', 'idx_train'),
                             ('size_indices', 'idx_sizes'), ('wall_seconds', 'wall_seconds')]:
        summary[destination] = np.asarray([row[key] for row in records])
    for key in TRAIN_PARAMETERS:
        summary[key] = np.asarray(getattr(args, key))
    summary['seed_mp'] = np.asarray(args.seeds[0] if len(args.seeds) == 1 else args.seeds)
    temporary = directory / 'summary.npz.tmp'
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **summary)
    os.replace(temporary, directory / 'summary.npz')
    fields = ['mode', 'affinity_update', 'sharing', 'dataset', 'seed', 'selected_hye_sizes',
              *SCALAR_KEYS, 'wall_seconds', 'cached', 'inferred_params_path']
    rows = []
    for result in records:
        row = {key: result[key] for key in fields}
        row['selected_hye_sizes'] = json.dumps(row['selected_hye_sizes'])
        rows.append(row)
    write_csv(directory / 'summary.csv', fields, rows)
    paired = {}
    for result in records:
        key = (result['idx_sizes'], result['idx_train'], result['seed'])
        paired.setdefault(key, {})[result['mode']] = result
    fields = ['dataset', 'seed', 'selected_hye_sizes']
    for metric in SCALAR_KEYS:
        fields += [f'{metric}_{mode.replace("-", "_")}' for mode in MODES]
        fields += [f'delta_{metric}_w_minus_p_differ', f'delta_{metric}_w_minus_p_share',
                   f'delta_{metric}_share_minus_differ_w', f'delta_{metric}_share_minus_differ_p']
    rows = []
    for key in sorted(paired):
        group = paired[key]
        sample = next(iter(group.values()))
        row = dict(dataset=sample['dataset'], seed=sample['seed'],
                   selected_hye_sizes=json.dumps(sample['selected_hye_sizes']))
        for metric in SCALAR_KEYS:
            values = {mode: group.get(mode, {}).get(metric, float('nan')) for mode in MODES}
            for mode, value in values.items():
                row[f'{metric}_{mode.replace("-", "_")}'] = value
            row[f'delta_{metric}_w_minus_p_differ'] = values['w-differ'] - values['p-differ']
            row[f'delta_{metric}_w_minus_p_share'] = values['w-share'] - values['p-share']
            row[f'delta_{metric}_share_minus_differ_w'] = values['w-share'] - values['w-differ']
            row[f'delta_{metric}_share_minus_differ_p'] = values['p-share'] - values['p-differ']
        rows.append(row)
    write_csv(directory / 'comparison.csv', fields, rows)
    return directory


def run(args):
    project, entry, inputs, root, fingerprint, outer, inner = prepare(args)
    tasks = prepare_tasks(args, project, entry, inputs, root, fingerprint, inner)
    plan = dict(run_dir=str(root), entrypoint=str(entry), modes=args.mode,
                num_tasks=len(tasks), num_training_restarts=len(tasks) * args.train_rounds,
                outer_n_jobs=outer, inner_n_jobs=inner, tasks=tasks)
    write_json(root / 'plan.json', plan)
    print(f'模式：{args.mode}\n任务：{len(tasks)}；每任务重启：{args.train_rounds}\n'
          f'全局并行：{outer}；每任务n_jobs/Rayon：{inner}；BLAS/OMP：1\n'
          f'实验目录：{root}\n任务计划：{root / "plan.json"}', flush=True)
    if args.dry_run:
        print('Dry run complete; no training subprocess started.', flush=True)
        return
    records, failures = [], []
    with ThreadPoolExecutor(max_workers=outer) as pool:
        futures = {pool.submit(run_task, task, project, inner): task for task in tasks}
        for future in as_completed(futures):
            try:
                records.append(future.result())
            except Exception as error:
                failures.append(dict(task=futures[future], error=type(error).__name__, message=str(error)))
                print(f'ERROR: {error}', file=sys.stderr, flush=True)
    if records:
        directory = write_summary(root, args, records, outer, inner)
        print(f'汇总：{directory / "summary.npz"}\n逐任务表：{directory / "summary.csv"}\n'
              f'配对对照表：{directory / "comparison.csv"}', flush=True)
    write_json(root / 'status.json', dict(expected=len(tasks), completed=len(records),
                                         failed=len(failures), complete=not failures, failures=failures))
    if failures:
        raise RuntimeError(f'{len(failures)}个任务失败；已保存成功部分，详情见{root / "status.json"}')


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
