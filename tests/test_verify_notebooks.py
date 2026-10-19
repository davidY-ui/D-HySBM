"""Integration tests for the clean-copy notebook verifier (real Jupyter kernels)."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import nbformat


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "verify_notebooks.py"


class VerifyNotebooksTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="verify-notebooks-test-")
        self.root = Path(self.temp.name)
        self.source = self.root / "release"
        (self.source / "tools").mkdir(parents=True)
        if SCRIPT.exists():
            shutil.copy2(SCRIPT, self.source / "tools" / SCRIPT.name)
        self.work = self.root / "verification"

    def tearDown(self):
        self.temp.cleanup()

    def notebook(self, name, sources):
        nb = nbformat.v4.new_notebook(
            cells=[nbformat.v4.new_code_cell(source) for source in sources]
        )
        # A stale output must never be mistaken for a newly executed figure.
        nb.cells[0].execution_count = 99
        nb.cells[0].outputs = [nbformat.v4.new_output("stream", text="STALE OUTPUT")]
        path = self.source / (name + ".ipynb")
        nbformat.write(nb, path)
        return path

    def run_verifier(self, *arguments):
        return subprocess.run(
            [sys.executable, str(self.source / "tools" / SCRIPT.name),
             "--work-dir", str(self.work), "--python", sys.executable,
             "--timeout", "45", *arguments],
            capture_output=True, text=True, timeout=180,
        )

    def test_executes_every_cell_and_reports_real_figures_and_failures(self):
        for excluded in ("outputs", ".git", "__pycache__", "target"):
            (self.source / excluded).mkdir()
            (self.source / excluded / "must-not-copy.txt").write_text("excluded")
        (self.source / "input.txt").write_text("real fixture")
        good = self.notebook("good", [
            "from pathlib import Path\n"
            "assert Path('input.txt').read_text() == 'real fixture'\n"
            "assert all(not Path(p).exists() for p in "
            "['outputs', '.git', '__pycache__', 'target'])\nvalue = 41",
            "import matplotlib.pyplot as plt\n"
            "plt.plot([1, 2], [value, value + 1])\n"
            "plt.savefig('fresh.png')\nplt.show()",
            "assert value + 1 == 42\nprint('ALL CELLS RAN')",
        ])
        before = hashlib.sha256(good.read_bytes()).hexdigest()
        self.notebook("bad", ["raise ValueError('fixture failure')", "print('AFTER ERROR')"])
        self.notebook("unselected", ["raise AssertionError('must not run')"])
        result = self.run_verifier("--notebook", "good", "--notebook", "bad.ipynb")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        report = json.loads((self.work / "report.json").read_text(encoding="utf-8"))
        self.assertFalse(report["success"])
        rows = {item["notebook"]: item for item in report["notebooks"]}
        self.assertEqual(set(rows), {"good.ipynb", "bad.ipynb"})
        successful = rows["good.ipynb"]
        self.assertTrue(successful["success"], successful)
        self.assertEqual(successful["code_cells"], 3)
        self.assertEqual(successful["executed_code_cells"], 3)
        self.assertTrue(all(row["execution_count"] is not None for row in successful["cells"]))
        self.assertGreaterEqual(successful["png_outputs"], 1)
        self.assertEqual(successful["saved_figure_count"], 1)
        self.assertEqual(rows["bad.ipynb"]["executed_code_cells"], 2)
        self.assertEqual(rows["bad.ipynb"]["errors"][0]["ename"], "ValueError")
        self.assertEqual(rows["bad.ipynb"]["errors"][0]["cell_index"], 0)
        executed = nbformat.read(successful["executed_notebook"], as_version=4)
        self.assertEqual(len(executed.cells), 3)
        self.assertNotIn("STALE OUTPUT", json.dumps(executed))
        self.assertIn("ALL CELLS RAN", json.dumps(executed))
        self.assertEqual(hashlib.sha256(good.read_bytes()).hexdigest(), before)
        self.assertFalse((self.source / "fresh.png").exists())

    def test_denies_source_machine_reads_and_directory_enumeration(self):
        self.notebook("forbidden", [
            "open(r'D:\\D_HSBM\\secret.csv', 'r')",
            "import os\nos.listdir(r'D:\\loop_SBM')",
            "import os\nlist(os.scandir('/work/home'))",
            "print('GUARD DID NOT SKIP LATER CELLS')",
        ])
        result = self.run_verifier()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        report = json.loads((self.work / "report.json").read_text(encoding="utf-8"))
        row = report["notebooks"][0]
        self.assertEqual(row["executed_code_cells"], 4)
        self.assertEqual([error["ename"] for error in row["errors"]],
                         ["PermissionError", "PermissionError", "PermissionError"])
        self.assertTrue(all("source-machine path" in error["evalue"] for error in row["errors"]))

    def test_success_exit_and_skip_tags_do_not_skip_computation(self):
        path = self.notebook("success", ["answer = 6 * 7", "", "assert answer == 42"])
        notebook = nbformat.read(path, as_version=4)
        notebook.cells[0].metadata["tags"] = ["skip-execution"]
        nbformat.write(notebook, path)
        result = self.run_verifier()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads((self.work / "report.json").read_text(encoding="utf-8"))
        self.assertTrue(report["success"])
        row = report["notebooks"][0]
        self.assertEqual(row["code_cells"], 3)
        self.assertEqual(row["executed_code_cells"], 2)
        self.assertEqual(row["empty_code_cells"], 1)
        self.assertTrue(row["all_nonempty_code_cells_executed"])


if __name__ == "__main__":
    unittest.main()
