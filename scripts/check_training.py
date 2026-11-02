"""Format/lint the owned code and run behavioral checks without editing teammates."""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import black

ROOT = Path(__file__).resolve().parents[1]
OWNED = [
    "examples",
    "src/training",
    "src/datasets",
    "src/liveness",
    "src/recognition/classification",
    "src/recognition/verification",
    "tests/test_training.py",
    "tests/test_datasets.py",
    "tests/test_recognition.py",
    "tests/test_liveness.py",
    "tests/test_liveness_workflow.py",
    "scripts/build_liveness_notebook.py",
    "scripts/bundle_liveness.py",
    "scripts/check_training.py",
    "scripts/check_docs.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", action="store_true")
    parser.add_argument("--lint-only", action="store_true")
    parser.add_argument("--include-liveness", action="store_true")
    args = parser.parse_args()
    failures = []
    for target in OWNED:
        path = ROOT / target
        files = sorted(path.rglob("*.py")) if path.is_dir() else [path]
        for file in files:
            source = file.read_text(encoding="utf-8")
            try:
                formatted = black.format_file_contents(
                    source,
                    fast=False,
                    mode=black.Mode(target_versions={black.TargetVersion.PY312}),
                )
            except black.NothingChanged:
                continue
            if args.format:
                file.write_text(formatted, encoding="utf-8")
            else:
                failures.append(str(file.relative_to(ROOT)))
    if failures:
        raise SystemExit("Black formatting required: " + ", ".join(failures))
    subprocess.run(
        [sys.executable, "-m", "ruff", "check", *OWNED], cwd=ROOT, check=True
    )
    if not args.lint_only:
        environment = os.environ.copy()
        environment.setdefault("OMP_NUM_THREADS", "2")
        environment.setdefault("MKL_NUM_THREADS", "2")
        environment["MPLCONFIGDIR"] = str(ROOT / "runs/.matplotlib")
        tests = [
            "tests/test_training.py",
            "tests/test_recognition.py",
            "tests/test_datasets.py",
        ]
        if args.include_liveness:
            tests.extend(["tests/test_liveness.py", "tests/test_liveness_workflow.py"])
        # Never clean an existing directory: every verification gets a unique base.
        from uuid import uuid4

        # Native OpenCV file APIs cannot reliably use non-ASCII Windows paths.
        temporary = Path(tempfile.gettempdir()) / f"faceguard-pytest-{uuid4().hex}"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                *tests,
                "-q",
                "--tb=short",
                f"--basetemp={temporary}",
            ],
            cwd=ROOT,
            env=environment,
            check=True,
        )


if __name__ == "__main__":
    main()
