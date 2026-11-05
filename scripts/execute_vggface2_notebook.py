"""Execute a portable, seeded COPY of the existing EDA notebook; never edit input."""
from __future__ import annotations

import json
import os
import asyncio
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / ".venv" / "research-deps"
os.environ["PYTHONPATH"] = str(DEPS) + os.pathsep + os.environ.get("PYTHONPATH", "")
if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import nbformat
from nbclient import NotebookClient


def main() -> None:
    source = ROOT / "data" / "VGGFace2" / "vggface2.ipynb"
    if not (source.parent / "vggface2.duckdb").is_file():
        raise FileNotFoundError("Read-only execution requires the existing VGGFace2 DuckDB catalog.")
    target = ROOT / "reports" / "recognition" / "research"
    target.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(source, as_version=4)
    replacements = 0
    for cell in notebook.cells:
        if cell.cell_type == "code" and "BASE_DIR = Path(" in cell.source:
            cell.source, replacements = re.subn(
                r"BASE_DIR = Path\([^\n]+\)",
                lambda match: "BASE_DIR = Path(" + repr(str(source.parent)) + ")",
                cell.source,
            )
    if replacements != 1:
        raise RuntimeError(f"Expected exactly one BASE_DIR adaptation, got {replacements}")
    seed_cell = nbformat.v4.new_code_cell(
        "import os\n"
        "os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')\n"
        "import random, numpy as np, torch\n"
        "random.seed(42)\nnp.random.seed(42)\ntorch.manual_seed(42)\n"
        "import duckdb\n"
        "_original_connect = duckdb.connect\n"
        "def _seeded_connect(*args, **kwargs):\n"
        "    connection = _original_connect(*args, **kwargs)\n"
        "    connection.execute('SELECT setseed(0.42)')\n"
        "    return connection\n"
        "duckdb.connect = _seeded_connect\n"
    )
    notebook.cells.insert(1, seed_cell)
    notebook.metadata["research_execution"] = {
        "source": str(source.relative_to(ROOT)),
        "adaptations": ["portable BASE_DIR", "Python/NumPy/PyTorch/DuckDB random seeds", "Apply original notebook OpenMP workaround before seeded imports, kernel process only"],
        "original_sql_preserved": True,
        "database_access": "read-only (existing notebook connection)",
    }
    start = time.perf_counter()
    error = None
    try:
        NotebookClient(notebook, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        nbformat.write(notebook, target / "vggface2_executed.ipynb")
    result = {"status": "failed" if error else "passed", "elapsed_seconds": round(time.perf_counter() - start, 2),
              "output": "reports/recognition/research/vggface2_executed.ipynb", "error": error,
              "known_limitations": ["Original JOIN-based split means are image-weighted, not per-identity arithmetic means.",
                                    "Original markdown summary contains old hard-coded results; consult audit_summary.json."]}
    (target / "notebook_execution.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
