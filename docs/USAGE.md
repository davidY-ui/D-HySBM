# Command-line usage reference

Run commands from the repository root. See [README.md](../README.md#installation) for the separate plotting and inference environments, and [NOTEBOOKS.md](NOTEBOOKS.md) for default figure regeneration. The compact plotting inputs and saved fits are included under `plot_data/`; the full research `data/` tree is excluded.

## Message passing with fixed parameters

The bundled `examples/tiny/` inputs contain ten hyperedges on eight nodes and two communities. After installing the inference environment and Rust extension, run this small fixed-parameter example from the repository root:

```sh
python main_message_passing_loop.py --hye_file examples/tiny/edges.txt --N 8 --K 2 --n examples/tiny/n_vals.txt --p examples/tiny/p_vals.json --selected_hye_sizes examples/tiny/orders.txt --fix_hye_size examples/tiny/orders.txt --em_iter 1 --mp_iter 100 --mp_thresh 1e-5 --mp_patience 5 --dropout 0 --dropout_vertex 0 --dropout_FVS 0 --train_rounds 1 --n_jobs 1 --seed 123 --save_dir outputs/tiny
```

`--n` fixes proportions; `--p` alone does not fix affinities. `--fix_hye_size` selects fixed orders. `--N` preserves isolated nodes in edge-list inputs. Optional `--node_assignments` enables overlap evaluation.

## Expectation-maximization inference

Omit fixed parameters to learn proportions, affinities, and node memberships. A small example using the included graph is:

```sh
python main_message_passing_loop.py --hye_file examples/tiny/edges.txt --N 8 --K 2 --model hsbm --affinity-update paper-w --selected_hye_sizes examples/tiny/orders.txt --em_iter 5 --mp_iter 100 --mp_patience 5 --mp_thresh 1e-5 --dropout 0.75 --dropout_vertex 0.75 --dropout_FVS 0 --train_rounds 1 --n_jobs 1 --seed 123 --save_dir outputs/tiny_em
```

### High School training example

The bundled plotting graph is `plot_data/realdata/high_school.npz`. The registered `--real_dataset high_school` route reads `data/processed_input/high_school.npz`, so prepare that separate training location first. The following command creates it when missing and leaves an existing file unchanged:

```sh
python -c "from pathlib import Path; import shutil; p=Path('data/processed_input'); p.mkdir(parents=True,exist_ok=True); t=p/'high_school.npz'; t.exists() or shutil.copy2('plot_data/realdata/high_school.npz',t)"
```

Then train with the original BP-EM entry point:


```sh
python main_message_passing_loop.py --real_dataset high_school --K 10 --model hsbm --affinity-update paper-w --max_hye_size 5 --em_iter 20 --em_thresh 1e-5 --mp_iter 2000 --mp_thresh 1e-5 --mp_patience 50 --train_rounds 3 --seed 234 --save_dir outputs/high_school
```

This starts a new fit. Notebook plotting uses the supplied saved fits and does not require running this training command.

`paper-w` learns normalized shapes `W` with order-specific scales. HSBM-only `direct-p` learns complete affinity matrices. `--shared_hye_sizes` reads disjoint JSON groups such as `[[2, 3], [4]]`, sharing `W` in `paper-w` or complete `p` in `direct-p`. Omitted orders form singleton groups.

`--model dchsbm` adds order-specific node activities. `--theta` supplies frozen activity vectors; `--learn_theta` enables their estimation.

| Main inference argument | Meaning |
| --- | --- |
| `--real_dataset`, `--hye_file`, `--pickle_file` | Choose exactly one hypergraph input route. |
| `--N`, `--K` | Node count for edge-list inputs; number of communities (`K >= 2`). |
| `--selected_hye_sizes`, `--max_hye_size` | Retain listed orders; discard edges above a maximum order. |
| `--n`, `--p`, `--fix_hye_size` | Community proportions, affinity matrices, and orders with fixed affinities. |
| `--model`, `--affinity-update` | `hsbm` or `dchsbm`; `paper-w` or HSBM-only `direct-p`. |
| `--shared_hye_sizes` | JSON groups of orders sharing an affinity shape or matrix. |
| `--em_iter`, `--em_thresh` | EM iteration limit and parameter-change threshold. |
| `--mp_iter`, `--mp_thresh`, `--mp_patience` | BP limit, convergence threshold, and consecutive qualifying iterations. |
| `--dropout`, `--dropout_vertex` | Fractions of edge-message and node-marginal updates dropped. |
| `--dropout_FVS` | Set to `0` for ordinary incidence-mask updates. |
| `--train_rounds` | Independent restarts; select the lowest training free-energy score. |
| `--n_jobs` | Parallel workers; `1` uses one worker, `-1` uses available cores. |
| `--seed`, `--save_dir` | Random seed and output directory. |

## Sampling synthetic data

The eight illustrative labels in `examples/tiny/node_assignments.txt` assign nodes 0–3 to community 0 and nodes 4–7 to community 1. These labels are a toy sampling example, separate from the research datasets.

```sh
python main_sampling_dc.py --model hsbm --p examples/tiny/p_vals.json --selected_hye_sizes examples/tiny/orders.txt --node_assignments examples/tiny/node_assignments.txt --allow_repeated False --count_method exact --seed 123 --save_dir outputs/tiny_sampling
```

Labels determine `N` and community indices; this sampling CLI has no `--N` or `--K` flags. `--selected_hye_sizes` determines the sampled orders. `--allow_repeated False` generates simple Bernoulli hyperedges; default `True` generates Poisson copies. `--count_method` is `exact` (default) or `approx`.

DCHSBM sampling requires `--model dchsbm --theta PATH`, where the JSON file maps each retained order to a nonnegative activity vector with one value per node. The HSBM command above needs the Python numerical packages but does not call the Rust extension.

## Input formats

Order `d` is the number of nodes in a hyperedge. Nodes use IDs `0, ..., N-1`, and communities use `0, ..., K-1`.

Paths under `./input/` in the format descriptions below illustrate user-supplied files. The runnable examples above use the included `examples/tiny/` inputs or the bundled High School graph.

- **Text hypergraph:** `--hye_file ./input/edges.txt`, one edge per line, integer IDs separated by single spaces, without headers or blank lines. Repeated edges and nodes within edges are removed:

  ```text
  0 1
  0 2 3
  ```

- **Pickle hypergraph:** `--hye_file ./input/edges.pkl` reads lists such as `[(0, 1), (0, 2, 3)]`, preserving repeated edges. Edges need at least two distinct integer IDs. `--pickle_file` reads serialized `IncidenceHypergraph` objects.
- **Registered dataset:** `--real_dataset high_school` reads `data/processed_input/high_school.npz`. Its `B` field is a dense or wrapped SciPy sparse incidence matrix of shape `(N, M)`, with positive membership entries binarized. Dataset names are registered in `src/data/data_io.py`.

| Parameter file | Contents |
| --- | --- |
| `orders.txt` (`--selected_hye_sizes`, `--fix_hye_size`) | Integer orders, one per line. Fixed orders must be retained orders. |
| `n_vals.txt` (`--n`) | `K` positive proportions summing to one, loaded with `numpy.loadtxt`. |
| `p_vals.json` (`--p`) | Order keys mapped to finite symmetric `K x K` matrices, with entries in `[0, 1)`. Provide every retained order. |
| `node_assignments.txt` | `N` integer community labels in node-ID order. |
| `theta.json` (`--theta`) | Order keys mapped to nonnegative length-`N` activity vectors; inference requires mean one per vector. |

For `K=2`, an affinity file containing orders 2 and 3 is:

```json
{"2": [[0.02, 0.005], [0.005, 0.02]], "3": [[0.03, 0.01], [0.01, 0.03]]}
```

These are pairwise affinities `p^(d) = c^(d)/N = c0 * beta_d * W^(d)/N`. The shape matrix `W^(d)` satisfies `n.T @ W^(d) @ n = 1`.

The bundled held-out files are `plot_data/realdata/high_school_cv5/fold_{f:02d}/high_school_test{f}.npz`, for `f=0,...,4`; they contain `hyperedges` and binary `y_true`. The notebook uses the included saved CV fits. Regenerating the training pipeline additionally requires the matching training-fold files and settings. Preserve node numbering and keep held-out files separate from training inputs.

## Other drivers and results

`D_realdata_W_compare.py` compares `w-differ`, `w-share`, `p-differ`, and `p-share`. `realdata_group_search.py` selects order-sharing groups by five-fold AUC:

```sh
python realdata_group_search.py --input-npz plot_data/realdata/high_school.npz --K 10 --orders 2 3 4 5 --seeds 234 235 236 --split-seed 234 --output outputs/group_search
```

Group-search NPZ input accepts `B`, or `hyperedges` plus scalar `N`. `--dry-run` prepares its plan. AUC selects the model; independent evaluation needs separate held-out data.

`D_function_sampling_multiseed.py`, `sampling_multiseed_dchsbm.py`, and `DC_m2_multiseed.py` provide `--help`. Their `--seed-base` generates graph-sampling seeds; inference in `DC_m2_multiseed.py` uses a separate `--seed-mp`. Planted-label generation uses `--label-seed` (default `100`).

`D_function_m2_collapse_multiseed.py` and `D_function_m2_signal_multiseed.py` configure their experiment grids in their `__main__` blocks and launch sampling and inference when run.

Inference saves `inferred_params.npz` and `args.txt`: log-marginals, parameters, orders, sharing groups, free energy, and convergence/runtime diagnostics. DCHSBM adds activities. Sampling saves `results.pkl` and `args.pkl`. All five publication notebooks resolve their default inputs from the included `plot_data/` tree or self-contained analytical calculations; no author-machine input paths need to be configured for their default runs.

## Original environment record

The following table is copied from the original project environment record; it is not the environment used for the current notebook checks and is not a minimum-version table. The current plotting environment is pinned by [requirements-plotting.txt](../requirements-plotting.txt).

| Component | Recorded version |
| --- | --- |
| Python | 3.10.4 |
| NumPy | 1.22.3 |
| SciPy | 1.8.1 |
| Joblib | 1.5.3 |
| Matplotlib | 3.10.8 |
| pandas | 1.4.2 |
| IPython | 8.38.0 |
| Jupyter | 1.1.1 |
| JupyterLab | 4.5.6 |
| Notebook | 7.5.5 |
| ipykernel | 7.2.0 |
| tqdm | 4.67.3 |
| rust_hysbm | 0.1.0 |
| Maturin | 1.15.0 |
| rustc | 1.98.0 |
| Cargo | 1.98.0 |
