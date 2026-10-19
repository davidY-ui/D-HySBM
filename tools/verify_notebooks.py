#!/usr/bin/env python3
"""Execute release notebooks in independent clean copies using real Jupyter kernels.

Example (run from the release root)::

    python tools/verify_notebooks.py --work-dir ../notebook-verification
    python tools/verify_notebooks.py --work-dir ../one-notebook \
        --notebook main_plot --timeout 600 --python /path/to/python

Install nbformat, nbclient, ipykernel and the notebook's scientific dependencies
in the chosen Python environment first. No kernelspec or user settings are
installed. A JSON report and executed notebooks are kept in --work-dir.
An exit status of 1 means at least one notebook failed; 2 is a usage error.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import uuid

import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager


EXCLUDED_DIRECTORIES = {
    ".git", "outputs", "target", "__pycache__", ".ipynb_checkpoints",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", "cache", ".cache",
}
FIGURE_SUFFIXES = {".png", ".pdf", ".svg", ".jpg", ".jpeg", ".eps", ".webp", ".tif", ".tiff"}
FORBIDDEN_ROOTS = [r"D:\D_HSBM", r"D:\loop_SBM", "/work/home"]


BOOTSTRAP = r'''
import os as _verify_os
import sys as _verify_sys
import ntpath as _verify_ntpath

def _verify_normalize_path(path):
    value = _verify_os.fsdecode(path)
    value = _verify_ntpath.normcase(_verify_ntpath.normpath(value)).replace('\\', '/')
    if value.startswith('//?/'):
        value = value[4:]
    return value.rstrip('/')

_verify_forbidden = [_verify_normalize_path(root) for root in __FORBIDDEN_ROOTS__]

def _verify_audit(event, args):
    if event not in ('open', 'os.listdir', 'os.scandir') or not args:
        return
    path = args[0]
    if not isinstance(path, (str, bytes, _verify_os.PathLike)):
        return
    candidates = [_verify_normalize_path(path),
                  _verify_normalize_path(_verify_os.path.abspath(path))]
    for value in candidates:
        if any(value == root or value.startswith(root + '/') for root in _verify_forbidden):
            raise PermissionError('Notebook verification denied source-machine path: ' + str(path))

_verify_sys.addaudithook(_verify_audit)
get_ipython().run_line_magic('matplotlib', 'inline')
'''


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def figures_in(root):
    """Inventory actual files, so pre-existing release figures do not count."""
    return {
        str(path.relative_to(root)).replace("\\", "/"): {
            "bytes": path.stat().st_size, "sha256": sha256(path)
        }
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in FIGURE_SUFFIXES
    }


def copy_release(source, destination):
    """Exclude build/output caches at every level and refuse external symlinks."""
    def ignore(directory, names):
        ignored = []
        for name in names:
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError("Release contains a symlink; provide a self-contained copy: " + str(path))
            if path.is_dir() and name in EXCLUDED_DIRECTORIES:
                ignored.append(name)
        return ignored

    shutil.copytree(source, destination, ignore=ignore)


def kernel_environment(runtime):
    environment = os.environ.copy()
    directories = {
        "MPLCONFIGDIR": "matplotlib", "IPYTHONDIR": "ipython",
        "JUPYTER_DATA_DIR": "jupyter-data", "JUPYTER_RUNTIME_DIR": "jupyter-runtime",
        "JUPYTER_CONFIG_DIR": "jupyter-config", "TEMP": "temp", "TMP": "temp",
        "TMPDIR": "temp",
    }
    for variable, name in directories.items():
        path = runtime / name
        path.mkdir(parents=True, exist_ok=True)
        environment[variable] = str(path)
    environment["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    # Do not inherit an unrelated local import path into the clean release.
    environment.pop("PYTHONPATH", None)
    return environment


def verify_notebook(source, notebook_name, work, python, timeout):
    start = time.monotonic()
    notebook_path = source / notebook_name
    run_dir = work / "runs" / notebook_path.stem
    release_copy = run_dir / "release"
    runtime = run_dir / "runtime"
    executed_path = work / "executed" / notebook_name
    executed_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "notebook": notebook_name, "success": False,
        "source_sha256": sha256(notebook_path),
        "working_copy": str(release_copy), "executed_notebook": str(executed_path),
        "code_cells": 0, "executed_code_cells": 0, "empty_code_cells": 0,
        "cells": [], "errors": [], "png_outputs": 0,
        "saved_figures": [], "saved_figure_count": 0,
    }
    notebook = None
    injected = False
    before_figures = {}
    try:
        copy_release(source, release_copy)
        before_figures = figures_in(release_copy)
        notebook = nbformat.read(release_copy / notebook_name, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.execution_count = None
                cell.outputs = []
                cell.metadata.pop("execution", None)
        bootstrap = nbformat.v4.new_code_cell(
            BOOTSTRAP.replace("__FORBIDDEN_ROOTS__", repr(FORBIDDEN_ROOTS)),
            metadata={"tags": ["verification-bootstrap"]},
        )
        notebook.cells.insert(0, bootstrap)
        injected = True
        environment = kernel_environment(runtime)
        kernel_name = "release-verification"
        kernel_dir = runtime / "kernels" / kernel_name
        kernel_dir.mkdir(parents=True, exist_ok=True)
        (kernel_dir / "kernel.json").write_text(json.dumps({
            "argv": [str(python), "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Temporary release verification", "language": "python",
        }), encoding="utf-8")
        specs = KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)], ensure_native_kernel=False)
        manager = AsyncKernelManager(
            kernel_name=kernel_name, kernel_spec_manager=specs,
            connection_file=str(runtime / "jupyter-runtime" / "kernel.json"),
        )
        client = NotebookClient(
            notebook, km=manager, kernel_name=kernel_name, timeout=timeout,
            startup_timeout=120, allow_errors=True,
            skip_cells_with_tag="__never_skip_" + uuid.uuid4().hex,
            record_timing=True,
        )
        # allow_errors records an error and continues, so every later cell is
        # still attempted. Infrastructure failures/timeouts remain failures.
        client.execute(cwd=str(release_copy), env=environment, cleanup_kc=True)
        bootstrap_errors = [out for out in bootstrap.outputs if out.output_type == "error"]
        for error in bootstrap_errors:
            row["errors"].append({"cell_index": None, "phase": "bootstrap",
                                  "ename": error.ename, "evalue": error.evalue,
                                  "traceback": error.get("traceback", [])})
    except Exception as error:
        row["errors"].append({
            "cell_index": None, "phase": "execution", "ename": type(error).__name__,
            "evalue": str(error), "traceback": traceback.format_exc().splitlines(),
        })
    finally:
        if notebook is not None:
            if injected:
                notebook.cells.pop(0)
            for index, cell in enumerate(notebook.cells):
                if cell.cell_type != "code":
                    continue
                row["code_cells"] += 1
                empty = not cell.source.strip()
                row["empty_code_cells"] += int(empty)
                executed = cell.execution_count is not None
                row["executed_code_cells"] += int(executed)
                cell_errors = []
                png_count = 0
                for output in cell.outputs:
                    png_count += int("image/png" in output.get("data", {}))
                    if output.output_type == "error":
                        error = {"cell_index": index, "phase": "cell", "ename": output.ename,
                                 "evalue": output.evalue, "traceback": output.get("traceback", [])}
                        cell_errors.append(error)
                        row["errors"].append(error)
                row["png_outputs"] += png_count
                row["cells"].append({
                    "cell_index": index, "cell_id": cell.get("id"),
                    "execution_count": cell.execution_count, "empty": empty,
                    "png_outputs": png_count, "errors": cell_errors,
                })
            notebook.metadata["release_verification"] = {
                "source_sha256": row["source_sha256"], "python": str(python),
                "forbidden_source_roots": FORBIDDEN_ROOTS,
                "bootstrap": BOOTSTRAP.replace("__FORBIDDEN_ROOTS__", repr(FORBIDDEN_ROOTS)),
            }
            nbformat.write(notebook, executed_path)
        if release_copy.exists():
            for name, details in figures_in(release_copy).items():
                if details != before_figures.get(name):
                    row["saved_figures"].append({"path": name, **details})
        row["saved_figure_count"] = len(row["saved_figures"])
        row["all_nonempty_code_cells_executed"] = all(
            cell["empty"] or cell["execution_count"] is not None for cell in row["cells"]
        )
        row["success"] = bool(notebook is not None and not row["errors"]
                              and row["all_nonempty_code_cells_executed"])
        row["elapsed_seconds"] = round(time.monotonic() - start, 3)
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--notebook", action="append", metavar="NAME",
                        help="Root-level notebook filename or stem; repeat to select several (default: all).")
    parser.add_argument("--work-dir", required=True, type=Path,
                        help="New or empty directory outside the release for all copies, kernels and results.")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds for each cell (default: 600).")
    parser.add_argument("--python", type=Path, default=Path(sys.executable),
                        help="Python interpreter for the temporary Jupyter kernel (default: current Python).")
    args = parser.parse_args(argv)
    source = Path(__file__).resolve().parents[1]
    work = args.work_dir.resolve()
    python = args.python.expanduser().resolve()
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    if not python.is_file():
        parser.error("--python must point to an existing Python executable")
    if work == source or source in work.parents:
        parser.error("--work-dir must be outside the release directory")
    if work.exists() and any(work.iterdir()):
        parser.error("--work-dir must be new or empty (existing results are never overwritten)")
    names = args.notebook or [path.name for path in sorted(source.glob("*.ipynb"))]
    selected = []
    for name in names:
        if not name.endswith(".ipynb"):
            name += ".ipynb"
        if Path(name).name != name or not (source / name).is_file():
            parser.error("not a root-level release notebook: " + name)
        if name not in selected:
            selected.append(name)
    if not selected:
        parser.error("no notebooks found")
    work.mkdir(parents=True, exist_ok=True)
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(), "source_release": str(source),
        "work_dir": str(work), "python": str(python), "timeout_seconds_per_cell": args.timeout,
        "forbidden_source_roots": FORBIDDEN_ROOTS, "notebooks": [], "success": False,
    }
    for name in selected:
        print("Executing " + name + " ...", flush=True)
        row = verify_notebook(source, name, work, python, args.timeout)
        report["notebooks"].append(row)
        report["success"] = all(item["success"] for item in report["notebooks"])
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (work / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print("  {}: {}/{} code cells, {} PNG outputs, {} saved figures".format(
            "PASS" if row["success"] else "FAIL", row["executed_code_cells"],
            row["code_cells"], row["png_outputs"], row["saved_figure_count"]), flush=True)
    print("Report: " + str(work / "report.json"), flush=True)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
