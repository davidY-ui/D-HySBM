# Third-party attribution

## Hypergraph Message Passing

This project builds on the main software architecture, message-passing framework, and implementation of:

- Nicolò Ruggeri, Alessandro Lonardi, and Caterina De Bacco, *Message-passing on hypergraphs: detectability, phase transitions and higher-order information*, Journal of Statistical Mechanics: Theory and Experiment **2024**, 043403 (2024), [doi:10.1088/1742-5468/ad343b](https://doi.org/10.1088/1742-5468/ad343b).
- Original repository: [nickruggeri/hypergraph-message-passing](https://github.com/nickruggeri/hypergraph-message-passing).
- Upstream revision consulted for attribution and license recovery: [`03c8b95b07b5367ff8fca237a79f69895618c581`](https://github.com/nickruggeri/hypergraph-message-passing/tree/03c8b95b07b5367ff8fca237a79f69895618c581). This is the inspected upstream revision, not a claim that the working project was originally forked at that exact revision.

The original notice is:

> Copyright (c) 2023 Nicolò Ruggeri

The complete, unchanged upstream MIT license is included at [LICENSES/hypergraph-message-passing-MIT.txt](LICENSES/hypergraph-message-passing-MIT.txt) and applies to the upstream material covered by that notice. At least `src/model/numerical.py` and `src/data/representation/incidence_hypergraph.py` contain copied or adapted upstream code; this is not an exhaustive file-level provenance map.

The architecture and algorithms from that work are credited separately from Dawei Ye's project-specific code and extensions. The latter include order-dependent model parameters, degree-corrected variants, Rust-accelerated updates, and experiment/analysis drivers. This attribution does not assign the upstream work to the current code author and does not select a license for new contributions.

## Other dependencies

Python/Rust dependencies retain their own licenses and notices. They are installed separately and are not vendored into this source package. Original third-party source archives are excluded. Selected processed High School inputs required for plotting are included with their source reference and hashes; see [docs/DATA.md](docs/DATA.md).
