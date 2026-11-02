"""Execute bounded real-image GPU notebook checks using the shared main environment.

Source notebooks remain output-free. Executed copies and training artifacts use a
new timestamped artifact root by default and are not scientific accuracy results.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIRECTORY = ROOT / "notebooks" / "metrics-learn"


def runtime_cell(destination: Path):
    """Record the actual kernel runtime separately from the runner's interpreter."""
    return nbformat.v4.new_code_cell(
        "from pathlib import Path\n"
        "import importlib.metadata\n"
        "import json\n"
        "import platform\n"
        "import sys\n"
        "import torch\n"
        "runtime = {\n"
        "    'python': platform.python_version(),\n"
        "    'executable': sys.executable,\n"
        "    'working_directory': str(Path.cwd().resolve()),\n"
        "    'packages': {name: importlib.metadata.version(name) for name in\n"
        "                 ('torch', 'torchvision', 'numpy', 'duckdb', 'pydantic')},\n"
        "    'cuda_runtime': torch.version.cuda,\n"
        "    'cuda_available': torch.cuda.is_available(),\n"
        "    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,\n"
        "}\n"
        f"Path({str(destination)!r}).write_text(json.dumps(runtime, indent=2), encoding='utf-8')\n"
        "print(json.dumps(runtime, indent=2))\n"
        "if not ((3, 12) <= sys.version_info[:2] <= (3, 14)):\n"
        "    raise RuntimeError('The smoke test requires supported Python 3.12 through 3.14.')\n"
        "if runtime['packages']['torch'] != '2.10.0+cu130' or runtime['cuda_runtime'] != '13.0':\n"
        "    raise RuntimeError('The kernel must use the main PyTorch 2.10.0 CUDA 13.0 stack.')\n"
        "if runtime['packages']['torchvision'] != '0.25.0+cu130':\n"
        "    raise RuntimeError('The kernel must use the matching main torchvision 0.25.0 CUDA 13.0 wheel.')\n"
        "if not runtime['cuda_available']:\n"
        "    raise RuntimeError('The main environment cannot access a CUDA GPU.')"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", choices=["triplet_loss", "arcface", "both"], default="both")
    parser.add_argument("--kernel", default="mtrlearn", help="Registered Jupyter kernel name.")
    parser.add_argument("--artifacts-root", type=Path,
                        help="New comparison root; relative paths resolve from the repository.")
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    artifacts_root = args.artifacts_root or ROOT / "artifacts" / "smoke-main-cu130" / timestamp
    if not artifacts_root.is_absolute():
        artifacts_root = ROOT / artifacts_root
    artifacts_root = artifacts_root.resolve()
    os.environ["MTRLEARN_SMOKE"] = "1"
    os.environ["MTRLEARN_ARTIFACTS_ROOT"] = str(artifacts_root)
    os.environ["MPLBACKEND"] = "Agg"
    if os.name == "nt":
        import asyncio
        if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    output = artifacts_root / "notebooks"
    output.mkdir(parents=True, exist_ok=True)
    names = ["triplet_loss", "arcface"] if args.notebook == "both" else [args.notebook]
    results = []
    for name in names:
        source = NOTEBOOK_DIRECTORY / f"{name}.ipynb"
        notebook = nbformat.read(source, as_version=4)
        runtime_path = output / f"{name}_runtime.json"
        notebook.cells.insert(0, runtime_cell(runtime_path))
        start = time.perf_counter()
        error = None
        try:
            NotebookClient(notebook, timeout=1200, kernel_name=args.kernel,
                           resources={"metadata": {"path": str(NOTEBOOK_DIRECTORY)}}).execute()
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"
        finally:
            executed_path = output / f"{name}_executed.ipynb"
            nbformat.write(notebook, executed_path)
        result = {
            "notebook": name,
            "passed": error is None,
            "elapsed_seconds": round(time.perf_counter() - start, 2),
            "error": error,
            "kernel_name": args.kernel,
            "runtime": json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.exists() else None,
            "source_notebook": str(source),
            "executed_notebook": str(executed_path),
            "working_directory": str(NOTEBOOK_DIRECTORY),
            "artifacts_root": str(artifacts_root),
            "training_artifacts_root": str(artifacts_root / "smoke"),
            "shared_data_manifests": str(ROOT / "artifacts" / "shared" / "manifests"),
            "scope": "Bounded real-image GPU integration check, not full model training.",
        }
        (output / f"{name}_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)
    summary = {"passed": all(result["passed"] for result in results), "results": results}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
