# Publication plotting release

Prepared as an independent copy; the original working project remains unchanged. This is a local source/data package, not a published GitHub/Zenodo release.

## Resulting behavior

All five notebooks now resolve their default plotting inputs within the extracted repository and write new figures/tables under `outputs/`. The compact original summaries, saved fits, test folds and BP measurements needed to redraw the figures are retained under `plot_data/`; the large simulation tree and compiled artifacts remain excluded.

## Changes

- Merged the original detailed BP/EM/sampling usage into the publication README and synchronized `docs/USAGE.md`; separated the plotting and inference environments, corrected the High School preparation step, and supplied toy sampling labels. Four documented small-example/preparation commands were executed in an isolated copy.

- README, root `.gitignore`, software CFF and publishing instructions are included. The only software author is Dawei Ye, as specified by the author.
- The main architecture/framework is credited to Ruggeri, Lonardi and De Bacco (2024), DOI `10.1088/1742-5468/ad343b`; the upstream MIT notice is preserved. A separate license for new contributions is not selected automatically.
- Replaced notebook input/output configuration paths with checkout-relative plotting paths; the DCHSBM input is the correct existing file.
- Added source hashes/provenance for the selected plot inputs. Numerical arrays from the main, DCHSBM and real-data archives are unchanged.
- BP defaults to replotting verified cached measurements rather than launching a fresh simulation or skipping the single-generation plot. Its real 30k-path sample size is explicit. Original BP/analytical algorithms and plot functions are retained.
- Real-data fit lookup uses local saved fits instead of original-machine paths. Its requested confusion figure is generated on demand when optional diagnostic export is disabled. Metric formulas are unchanged.
- Added `requirements-plotting.txt` and `tools/verify_notebooks.py`: real Jupyter execution in separate clean copies, source-machine path rejection, cell/figure/error reporting and executed copies. The tool's regression tests use actual notebook kernels.
- Removed Notebook saved outputs, execution counts, timing metadata and widget state from upload sources. Core research `.py`, Rust and build files are preserved unchanged, except for the additional publication verification tool/tests.
- Retained detailed original usage in `docs/USAGE.md`, removed the Conda export's local prefix, retained historical data notes with warnings, and moved the small primary-school preparation utility unchanged to `tools/`.

## Validation and limits

The supplied verification report gives the executed code-cell counts and actual figure generation for each notebook. CFF is checked against the official 1.2.0 schema, plotting inputs against source hashes, exclusion/staging rules against a scratch Git index, and the final archive against its file manifest.

Large simulation campaigns, fresh Rust compilation and a new corrected 200,000-path BP campaign are outside this plotting check. Public repository, software version/release date, DOI and the new-contribution license remain to be filled from the eventual actual publication.
