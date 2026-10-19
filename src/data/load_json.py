"""Convert JSON hypergraphs and available node labels to reusable input files."""

import json
import re
import numpy as np
from scipy import sparse
from pathlib import Path
from collections import Counter
from typing import Any
from src.data.representation.incidence_hypergraph import IncidenceHypergraph

def _sort_key(x):
    """Sort numeric strings such as '1', '2', and '10' numerically; sort other strings lexically."""
    try:
        return int(x)
    except Exception:
        return str(x)


def _safe_name(s):
    """Convert a label field name to a safe filename, e.g. 'metadata.party' -> 'metadata_party'."""
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(s))


def _get_by_path(d, path, default=None):
    """Read nested fields, e.g. path='metadata.party'."""
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def infer_label_fields(
    node_data,
    exclude_fields=("name", "id"),
    max_num_classes=20,
    min_coverage=0.5,
):
    """
    Find node-data fields suitable for community labels automatically.
    Examples include affiliation, party, class, group, and category.

    A candidate must not be a unique identifier such as name/id, must have
    enough nonempty values, and must have a limited number of distinct values.
    """
    field_values = {}

    for node_id, attrs in node_data.items():
        if not isinstance(attrs, dict):
            continue

        for key, value in attrs.items():
            if key in exclude_fields:
                continue

            if isinstance(value, (str, int, float, bool)):
                field_values.setdefault(key, []).append(value)

    label_fields = []

    num_nodes = max(len(node_data), 1)

    for key, values in field_values.items():
        values = [v for v in values if v is not None and v != ""]
        unique_values = sorted(set(values), key=lambda x: str(x))

        coverage = len(values) / num_nodes

        if coverage >= min_coverage and 2 <= len(unique_values) <= max_num_classes:
            label_fields.append(key)

    return label_fields


def encode_node_labels(
    node_ids,
    node_data,
    label_field,
    label_order=None,
    unknown_label=-1,
):
    """
    Map arbitrary node labels to integer community IDs.

    For example, Democrat -> 0 and Republican -> 1.
    """
    raw_labels = []

    for node_id in node_ids:
        attrs = node_data.get(node_id, {})
        label = _get_by_path(attrs, label_field, default=None)

        if label is None or label == "":
            raw_labels.append(None)
        else:
            raw_labels.append(label)

    if label_order is None:
        unique_labels = sorted(
            {x for x in raw_labels if x is not None},
            key=lambda x: str(x)
        )
    else:
        unique_labels = list(label_order)

    label_to_id = {
        label: i
        for i, label in enumerate(unique_labels)
    }

    assignments = np.array(
        [
            label_to_id.get(label, unknown_label)
            for label in raw_labels
        ],
        dtype=int
    )

    return assignments, raw_labels, label_to_id

def json_hypergraph_to_npz_universal(
    json_file,
    npz_file,
    hye_file=None,
    node_assignment_dir=None,
    label_fields="auto",
    edge_key="edge-dict",
    node_key="node-data",
    min_hye_size=2,
    max_num_classes=20,
    save_mapping=True,
    **kwargs: Any,
)-> IncidenceHypergraph:
    """
    Convert a JSON hypergraph to incidence and node-label files.

    Parameters
    ----------
    json_file:
        Input JSON file.
    npz_file:
        Output NPZ file containing incidence matrix B.
    hye_file:
        Output text file with one hyperedge per line.
    node_assignment_dir:
        Directory for node assignment files.
    label_fields:
        "auto" detects usable label fields. Explicit lists such as
        ["affiliation"] or ["party", "class"] are also accepted.
    edge_key:
        JSON field containing the hyperedge dictionary; defaults to "edge-dict".
    node_key:
        JSON field containing node attributes; defaults to "node-data".
    min_hye_size:
        Discard hyperedges smaller than this size.
    max_num_classes:
        Maximum number of classes allowed when detecting labels automatically.
    save_mapping:
        Whether to save the node ID mapping.

    Returns
    -------
    B:
        SciPy sparse incidence matrix.
    """
    json_file = Path(json_file)
    npz_file = Path(npz_file)
    npz_file.parent.mkdir(parents=True, exist_ok=True)

    if hye_file is not None:
        hye_file = Path(hye_file)
        hye_file.parent.mkdir(parents=True, exist_ok=True)

    if node_assignment_dir is not None:
        node_assignment_dir = Path(node_assignment_dir)
        node_assignment_dir.mkdir(parents=True, exist_ok=True)

    with open(json_file, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if edge_key not in obj:
        raise KeyError(
            f"Cannot find edge key '{edge_key}' in json. "
            f"Available keys are: {list(obj.keys())}"
        )

    edge_dict = obj[edge_key]
    node_data = obj.get(node_key, {})

    # 1. Collect all nodes.
    node_ids = set(node_data.keys())

    for hye in edge_dict.values():
        node_ids.update(map(str, hye))

    node_ids = sorted(node_ids, key=_sort_key)

    # 2. Renumber nodes as 0, 1, ..., N-1.
    node_to_idx = {
        node_id: i
        for i, node_id in enumerate(node_ids)
    }

    idx_to_node = np.array(node_ids, dtype=object)

    # 3. Construct incidence matrix B.
    rows = []
    cols = []
    kept_edge_ids = []
    hyperedges_reindexed = []

    # Collect all hyperedges first.
    edge_items = []

    for e_id, hye in edge_dict.items():
        hye = sorted(set(map(str, hye)), key=_sort_key)

        if len(hye) < min_hye_size:
            continue

        edge_items.append(
            {
                "e_id": e_id,
                "hye": hye,
                "size": len(hye),
            }
        )

    # Sort by hyperedge size, then by original hyperedge ID within each size.
    edge_items = sorted(
        edge_items,
        key=lambda x: (x["size"], _sort_key(x["e_id"]))
    )

    # Construct the incidence matrix from the sorted hyperedges.
    for item in edge_items:
        e_id = item["e_id"]
        hye = item["hye"]

        hye_idx = [
            node_to_idx[node_id]
            for node_id in hye
        ]

        e_idx = len(kept_edge_ids)

        kept_edge_ids.append(e_id)
        hyperedges_reindexed.append(hye_idx)

        for node_idx in hye_idx:
            rows.append(node_idx)
            cols.append(e_idx)

    N = len(node_ids)
    E = len(kept_edge_ids)

    data = np.ones(len(rows), dtype=np.int8)

    B = sparse.csc_matrix(
        (data, (rows, cols)),
        shape=(N, E)
    )
    # 4. Detect label fields automatically.
    if label_fields == "auto":
        label_fields = infer_label_fields(
            node_data,
            max_num_classes=max_num_classes
        )

    if label_fields is None:
        label_fields = []

    if isinstance(label_fields, str):
        label_fields = [label_fields]

    # 5. Generate node assignments for each label field.
    extra_arrays = {}

    for label_field in label_fields:
        assignments, raw_labels, label_to_id = encode_node_labels(
            node_ids=node_ids,
            node_data=node_data,
            label_field=label_field,
        )

        safe_label = _safe_name(label_field)

        extra_arrays[f"node_assignments"] = assignments
        extra_arrays[f"raw_labels_{safe_label}"] = np.array(raw_labels, dtype=object)
        extra_arrays[f"label_map_{safe_label}"] = np.array(
            list(label_to_id.items()),
            dtype=object
        )

        if node_assignment_dir is not None:
            assignment_file = node_assignment_dir / f"node_assignment_{json_file.stem}.txt"
            np.savetxt(
                assignment_file,
                assignments,
                fmt="%d"
            )

            print("saved node assignment:", assignment_file)
            print("label field:", label_field)
            print("label map:", label_to_id)
            print("assignment counts:", dict(sorted(Counter(assignments).items())))

    # 6. Save B in the format expected by load_real_hypergraph().
    B_obj = np.empty((), dtype=object)
    B_obj[()] = B

    np.savez_compressed(
        npz_file,
        B=B_obj,
        node_ids=idx_to_node,
        edge_ids=np.array(kept_edge_ids, dtype=object),
        **extra_arrays,
    )

    # 7. Save the hyperedge file.
    if hye_file is not None:
        with open(hye_file, "w", encoding="utf-8") as f:
            for hye in hyperedges_reindexed:
                f.write(" ".join(map(str, hye)) + "\n")

        print("saved hyperedges:", hye_file)

    # 8. Save the node ID mapping.
    if save_mapping:
        mapping_file = npz_file.with_name(npz_file.stem + "_node_mapping.txt")

        with open(mapping_file, "w", encoding="utf-8") as f:
            header = ["new_id", "old_id"]

            for label_field in label_fields:
                header.append(label_field)

            f.write("\t".join(header) + "\n")

            for new_id, old_id in enumerate(node_ids):
                row = [str(new_id), str(old_id)]

                attrs = node_data.get(old_id, {})
                for label_field in label_fields:
                    row.append(str(_get_by_path(attrs, label_field, default="")))

                f.write("\t".join(row) + "\n")

        print("saved node mapping:", mapping_file)

    # hye_lengths = np.asarray(B.sum(axis=0)).ravel().astype(int)
    # hye_count = dict(sorted(Counter(hye_lengths).items()))

    print("saved npz:", npz_file)
    # print("N =", N)
    # print("E =", E)
    # print("B.shape =", B.shape)
    # print("hye_count =", hye_count)
    # print("label_fields =", label_fields)

    return IncidenceHypergraph(B, **kwargs)