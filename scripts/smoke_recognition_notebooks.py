"""Execute both notebooks on tiny real-image subsets using the mtrlearn GPU kernel.

Source notebooks stay output-free. Executed copies and bounded training artifacts
are saved under artifacts/smoke and are not scientific accuracy results.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", choices=["triplet_loss", "arcface", "both"], default="both")
    args = parser.parse_args()
    os.environ["MTRLEARN_SMOKE"] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    if os.name == "nt":
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    output = ROOT / "artifacts" / "smoke" / "notebooks"
    output.mkdir(parents=True, exist_ok=True)
    names = ["triplet_loss", "arcface"] if args.notebook == "both" else [args.notebook]
    for name in names:
        notebook = nbformat.read(ROOT / "notebooks" / f"{name}.ipynb", as_version=4)
        start = time.perf_counter()
        error = None
        try:
            NotebookClient(notebook, timeout=1200, kernel_name="mtrlearn",
                           resources={"metadata": {"path": str(ROOT)}}).execute()
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"
        finally:
            nbformat.write(notebook, output / f"{name}_executed.ipynb")
        result = {"notebook": name, "passed": error is None,
                  "elapsed_seconds": round(time.perf_counter() - start, 2), "error": error,
                  "scope": "Bounded real-image GPU integration check, not full model training."}
        (output / f"{name}_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        if error:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
