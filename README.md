# D-HySBM: Hypergraph Message Passing

Belief propagation (BP), expectation-maximization (EM), and sampling for order-resolved hypergraph stochastic block models (HSBM) and degree-corrected models (DCHSBM).

**Code and extensions:** Dawei Ye.  
**Companion manuscript:** *Statistical physics of Information Propagation in Hypergraph Block Models*, Dawei Ye and Pan Zhang.

## Origin of the architecture and required acknowledgment

**The main software architecture and the underlying hypergraph message-passing framework are based on the work of Ruggeri, Lonardi, and De Bacco and their original implementation.** This project adapts and extends that framework with order-dependent model parameters, degree-corrected variants, Rust-accelerated updates, and experiment/analysis drivers. The upstream architecture and algorithms should be credited when using this project.

> Nicolò Ruggeri, Alessandro Lonardi, and Caterina De Bacco. "Message-passing on hypergraphs: detectability, phase transitions and higher-order information." *Journal of Statistical Mechanics: Theory and Experiment* **2024**, 043403 (2024). [DOI: 10.1088/1742-5468/ad343b](https://doi.org/10.1088/1742-5468/ad343b).

Original implementation: [nickruggeri/hypergraph-message-passing](https://github.com/nickruggeri/hypergraph-message-passing).

```bibtex
@article{ruggeri2024messagepassing,
  author  = {Ruggeri, Nicol\`{o} and Lonardi, Alessandro and {De Bacco}, Caterina},
  title   = {Message-passing on hypergraphs: detectability, phase transitions and higher-order information},
  journal = {Journal of Statistical Mechanics: Theory and Experiment},
  year    = {2024},
  volume  = {2024},
  number  = {4},
  pages   = {043403},
  doi     = {10.1088/1742-5468/ad343b}
}
```

See [third-party attribution](THIRD_PARTY_NOTICES.md) for the upstream notice. The code authorship field in [CITATION.cff](CITATION.cff) describes this project's code and extensions; the upstream contributors are acknowledged separately above and in its references.

## Installation

Use **Python 3.10**. Run the commands below from the extracted repository root. The two environments below serve different workflows; select the corresponding interpreter/kernel when running a command or notebook.

### Plotting the included results

```sh
python -m venv .venv-plotting
```

Activate it with `.\.venv-plotting\Scripts\Activate.ps1` in PowerShell, or `source .venv-plotting/bin/activate` on Linux/macOS. Then install:

```sh
python -m pip install -r requirements-plotting.txt
```

This installs the numerical packages, plotting libraries, and kernel tools used for notebook verification. The default five notebooks replot the included numeric inputs and do not require Rust. To use a browser-based JupyterLab interface, optionally install and start it:

```sh
python -m pip install jupyterlab
python -m jupyter lab
```

VS Code can use the same `.venv-plotting` kernel. The command-line verification tool below does not need JupyterLab.

### Running new BP/EM fits or sampling

For the original inference setup, create a separate Python 3.10 environment:

```sh
python -m venv .venv-inference
```

Activate `.venv-inference` using the corresponding command for your shell, then install the recorded numerical dependencies:

```sh
python -m pip install numpy==1.22.3 scipy==1.8.1 joblib==1.5.3 "maturin>=1.15,<2"
```

BP/EM uses the local Rust extension. Install a [Rust toolchain](https://www.rust-lang.org/tools/install) supporting edition 2024 and the platform's native build tools, then build the extension into this same environment:

```sh
python -m pip install --no-build-isolation ./rust_hysbm
```

Synthetic sampling uses the Python numerical dependencies and does not require the Rust build. A separate inference environment keeps the recorded NumPy version distinct from the plotting requirements.

### Environment records

The notebook execution checks used Python **3.10.19**, NumPy **1.23.1**, SciPy **1.8.1**, Matplotlib **3.10.8**, and pandas **2.3.3**; complete plotting pins are in [requirements-plotting.txt](requirements-plotting.txt).

[environment_py310.yml](environment_py310.yml) and [package_versions_py310.txt](package_versions_py310.txt) preserve the original project's environment record, including the original NumPy 1.22.3 setup. The Conda export contains Linux-specific packages and is not a portable Windows lockfile. [Cargo.lock](rust_hysbm/Cargo.lock) records Rust dependencies. These records are environment snapshots, not minimum-version declarations or evidence of a fresh build on every platform.

## Reproducing the notebook figures

Open a notebook from the repository root, select the plotting environment, and run all cells in order. Outputs are written under `outputs/`. The plotting functions and their required compact numerical inputs are included.

| Notebook | Included input / role | Verified code cells |
| --- | --- | ---: |
| `main_plot.ipynb` | `plot_data/main_plot/`; order mixtures, runtime, fixed budgets and signal compensation | 11/11 |
| `main_plot_DCHSBM.ipynb` | `plot_data/dchsbm/`; two empirical and three analytical plots | 9/9 |
| `realdata_plot.ipynb` | `plot_data/realdata/`; AUC, affinities, partitions and membership plots from saved fits | 31/31 |
| `transition_detectivity.ipynb` | Self-contained analytical calculations | 8/8 |
| `BP_growth_validation_multigeneration.ipynb` | `plot_data/bp_growth/`; single- and multigeneration BP measurements | 15/15 |

All **74 code cells** were executed in separate fresh Jupyter kernels and clean checkout copies, with access to the original author's project directories blocked. The runs generated 61 PNG files, including the real-data figures for different order sets. See [the verification report](docs/NOTEBOOK_VERIFICATION.md) and [notebook guide](docs/NOTEBOOKS.md).

To repeat the check:

```sh
python tools/verify_notebooks.py --work-dir ../D-HySBM-notebook-check --timeout 600
```

The work directory must be outside the repository and empty/nonexistent. It receives executed notebooks, per-cell results, figures and a JSON report. Add `--notebook main_plot` to check one notebook. A real-data run can take a few minutes because it exports many matrices and tables.

The BP notebook defaults to verified saved measurements. Its single-generation cache contains 100,000 roots per ordinary case and **30,000 roots** per A/critical/B path case; it is explicitly labeled as that historical campaign. Its multigeneration cache is the completed main experiment with population 20,000, depth 12 and 10 independent runs per A/B case. The original 200,000-path and other simulation controls remain available for an explicit new run. Figure regeneration does not constitute a fresh execution of all large simulations or a newly compiled Rust build.

The main notebook also supports `SHOW_SIGNAL_DIAGNOSTICS=True` and `REBUILD_SIGNAL_SUMMARY=True`. The latter uses the 15 included individual signal files; its reconstructed arrays were checked for exact equality with the existing aggregate.

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

## Data and repository layout

```text
src/                       Python models, hypergraph representations and I/O
rust_hysbm/                Native extension source and Cargo dependency records
examples/tiny/             Small BP/EM and sampling inputs
plot_data/main_plot/       Main-figure aggregates and individual signal summaries
plot_data/dchsbm/          DCHSBM aggregate
plot_data/realdata/        Small High School inputs, held-out folds and saved fits
plot_data/bp_growth/       Verified BP measurement summaries and trajectories
tools/verify_notebooks.py  Independent real-kernel notebook execution checks
tools/prepare_primary_school_cv5.py  Retained preparation utility
docs/                     Usage, data, provenance and verification notes
```

`plot_data/realdata/` includes 160 CV fit files, 32 full-data fit files, the five held-out test folds, the processed graph and node metadata. Input values are preserved; `SOURCE.json` and BP provenance records identify their origins and hashes. See [docs/DATA.md](docs/DATA.md).

The full approximately 51.2 GiB research `data/` tree, unrelated raw datasets, full particle populations, unused experiment outputs, local environments and compiler products are excluded. Root `data/` is the separate location used by registered training inputs; it is ignored by Git. `plot_data/`, `examples/`, and the complete `src/data/` source package are included. Generated `outputs/` are also ignored.

Legacy research scripts are retained with these limitations:

- `D_function_m5_sharec_multiseed.py` references the absent `sampling_multiseed_dc` module; the verified notebooks do not call it.
- `main_sampling_loop.py` creates the original hard-coded `C:\tmp_joblib` directory at import time; use `main_sampling_dc.py` for the documented sampling workflow.
- `docs/data-notes/` contains historical notes and layouts; use the current [notebook guide](docs/NOTEBOOKS.md) for plotting.

## Citation and licensing

[CITATION.cff](CITATION.cff) lists **Dawei Ye** as the software author. Cite this code and acknowledge the Ruggeri et al. framework above when using the implementation; cite the companion manuscript separately where relevant. Add the actual public repository URL, release version/date and Zenodo version DOI to the CFF after publication.

The original upstream MIT copyright and permission notice is retained in [LICENSES/hypergraph-message-passing-MIT.txt](LICENSES/hypergraph-message-passing-MIT.txt); see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). A separate license for this project's new contributions has not been selected.

For GitHub upload and Zenodo release steps, see [PUBLISHING.md](PUBLISHING.md). The current data and validation scope are described in [docs/RELEASE_NOTES.md](docs/RELEASE_NOTES.md).
