# Notebook plotting guide

All five notebooks have their default plotting inputs in the release. Run each from the repository root, top to bottom, in the Python 3.10 environment described by `requirements-plotting.txt`. The default plotting paths do not need the author's original D: drive or home-directory checkout. No external local drawing scripts are required: the plotting functions are embedded in the notebooks, and the repository also retains its research source and preprocessing utility.

```sh
python -m pip install -r requirements-plotting.txt
python tools/verify_notebooks.py --work-dir ../D-HySBM-notebook-check --timeout 600
```

The chosen work directory must be empty/nonexistent and outside this source directory. A full real-data notebook run saves many matrix/partition figures and can take a few minutes. The runner creates one separate checkout and Jupyter kernel per notebook, does not reuse user variables, does not skip source cells, and records errors and newly generated figure files. It rejects reads/listing/scanning in the original source-machine directories. Executed copies and the JSON report stay in the work directory; the upload notebooks retain empty saved output.

## Main plots

`main_plot.ipynb` uses `plot_data/main_plot/`. Default execution creates five figures: overlap by order set, runtime by order set, theoretical collapse, fixed-budget boundaries and signal compensation. `SHOW_SIGNAL_DIAGNOSTICS=True` enables five additional diagnostic figures from the same supplied data. `REBUILD_SIGNAL_SUMMARY=True` reconstructs a summary from the 15 individual signal files; that reconstruction has been checked for exact array equality with the original aggregate. It does not run new BP/EM inference.

## DCHSBM plots

`main_plot_DCHSBM.ipynb` uses `plot_data/dchsbm/`. It recomputes and checks theory against the saved NPZ, then makes two empirical plots and three analytical plots. The original configuration omitted `synthetic_loop` in its absolute input path; the publication copy uses the correct local file. Output is under `outputs/dchsbm/` unless explicitly overridden by its documented environment variables.

## Real-data plots

`realdata_plot.ipynb` uses `plot_data/realdata/`. It recomputes held-out AUC from saved parameters and the five test folds, checks summary/raw fit equality and layer order, then reconstructs full-data affinity, partition and membership figures. There are 160 CV fits and 32 full-data fits; these are saved original results, not fresh training runs. Full-data class agreement is post-hoc analysis and is separate from held-out AUC.

The original notebook disabled diagnostic export but later tried to display a confusion figure unconditionally. The publication copy creates that requested figure on demand using the unchanged confusion-plot function and original fitted values. Metrics and affinity formulas have not been changed. Historical absolute path strings in saved experiment metadata are not used for file lookup.

## Analytical threshold plots

`transition_detectivity.ipynb` is self-contained and produces four figures from its original analytical calculations, including the pairwise boundary, gamma decomposition and relative-gap plot. It writes under `outputs/transition_detectivity/`.

## BP growth plots

`BP_growth_validation_multigeneration.ipynb` defaults to `USE_CACHED_PLOT_DATA=True`. It calls the original plotting functions on verified numeric measurements to make four figures. It does not display a previously rendered image in place of recalculation, and it does not launch large fresh simulations by default.

The single-generation cache is the correctly labeled historical campaign with 100,000 roots per ordinary case and 30,000 roots per A/critical/B case. It is not a corrected 200,000-path rerun; those campaigns are never pooled. The multigeneration cache is the completed main campaign: population 20,000, depth 12, 10 independent runs per A/B case, amplitude 1e-4. Recomputed summaries are checked against the saved trajectories and configuration. See the adjacent provenance files for the original raw-file checks.

To perform a new experiment, explicitly set `USE_CACHED_PLOT_DATA=False` and enable the one-generation and/or multigeneration run controls. The original algorithms and 100k/200k/main settings remain available. Such simulations are a separate workload from the plotting verification.

## Verification scope

The verification report records every code cell, errors, displayed PNGs, saved figure files and the executed notebook. Default Run All is the acceptance check. Optional main-plot signal diagnostics and reconstruction were also checked separately. This verifies the plotting workflow against the supplied numeric inputs, not a fresh run of all paper simulations or a newly compiled Rust extension.
