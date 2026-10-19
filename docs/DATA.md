# Data included for plotting

The working project's full root `data/` tree is about 51.2 GiB and remains excluded. The publication copy now retains a selected, much smaller **original plotting input set** under `plot_data/`. The source package `src/data/` is also included in full. The `.gitignore` rule `/data/` does not exclude either of those directories.

## Included inputs

| Directory | Contents |
| --- | --- |
| `plot_data/main_plot/` | Four original aggregate NPZs plus 15 original signal-compensation realization summaries; enough for the main plots, optional diagnostics and optional signal-summary reconstruction |
| `plot_data/dchsbm/` | One original five-sample DCHSBM aggregate NPZ |
| `plot_data/realdata/` | Small processed High School graph and node metadata; five held-out CV test files; two experiment summaries/protocols; 160 CV fit NPZs and 32 full-data fit NPZs |
| `plot_data/bp_growth/` | Historical single-generation measurements, completed multigeneration main summary, independent-run trajectories, manifests and provenance; no large particle populations |

The main-plot, DCHSBM and real-data inputs are byte-for-byte copies of the corresponding original files. Each directory includes a `SOURCE.json` inventory with SHA256 values. The BP `PROVENANCE.json` describes the original measurements, compressed trajectory representation, source settings and checks. Input metadata may retain historical path strings as provenance; the notebooks resolve runtime inputs locally.

The single-generation BP cache has 100,000 roots in each ordinary group and 30,000 in A/critical/B path groups. It is explicitly labeled historical and is not represented as the current 200,000-path configuration. The multigeneration main cache uses population 20,000, 12 generations, 10 independent runs per A/B case and amplitude 1e-4. No new samples are synthesized to stand in for missing measurements.

## High School source

The companion manuscript cites R. Mastrandrea, J. Fournet and A. Barrat, *PLOS ONE* **10**, e0136497 (2015), [doi:10.1371/journal.pone.0136497](https://doi.org/10.1371/journal.pone.0136497). This is the source-dataset reference; the derived graph, metadata, held-out folds and fitted arrays in this publication snapshot are identified separately by their hashes.

## Excluded inputs

Large simulation trees, raw event archives for unrelated datasets, unused inference files, per-particle BP populations, compiler products and saved execution outputs are not part of the upload archive. The retained plotting inputs do not supply every training dataset for every legacy driver. Registered real-data training routes in `src/data/data_io.py` continue to use the separate root `data/processed_input/` layout.

## Small inference and preprocessing examples

`examples/tiny/` is the independent eight-node installation example. `tools/prepare_primary_school_cv5.py` is the unchanged preparation utility retained from the excluded tree; its source JSON, matching metadata and reference CV inputs are not supplied. Historical notes under `docs/data-notes/` are labeled accordingly.

## Regenerating figures

Install `requirements-plotting.txt` on Python 3.10 and run the notebooks in order from the repository root. `tools/verify_notebooks.py` checks the same workflow in clean copies and real kernels. See [NOTEBOOKS.md](NOTEBOOKS.md). All newly generated outputs are written under `outputs/`, which remains excluded from Git.
