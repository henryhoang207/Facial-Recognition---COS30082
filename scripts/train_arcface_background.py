"""Run full-image ArcFace training and final evaluation as one background job.

Launch with the repository's CUDA Python. The launcher may detach this process;
status.json, logs/, and report.md remain available throughout the run. No sample
or epoch limits are applied. No detector or face bounding boxes are used.
Reusing --job-dir resumes completed preparation and
the training engine's last complete epoch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LOCAL_ZONE = timezone(timedelta(hours=7))


def now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_report(job: Path, state: dict) -> None:
    options = read_json(job / "job_options.json")
    batch_size = options["batch_size"]
    config_path = job / "experiment_config.json"
    config = read_json(config_path) if config_path.exists() else {"amp_dtype": "bf16"}
    rows = [
        "# ArcFace training report",
        "",
        f"Status: **{state['status']}**. Stage: **{state['stage']}**.",
        f"Updated: {datetime.now(LOCAL_ZONE).isoformat(timespec='seconds')} (UTC+7).",
        "",
        "iResNet-34, 128-D embedding; ArcFace margin 0.3, scale 64; seed 42.",
        f"Batch {batch_size} (P={batch_size // 4}, K=4), "
        f"CUDA augmentation, {config.get('amp_dtype', 'fp16').upper()} AMP, "
        "4 training workers, "
        "2 evaluation workers. Maximum 100 epochs; early-stopping patience 10.",
        "Full source catalog; no smoke subset or batch limits.",
        "Pipeline: original processed 112x112 image -> CPU decode -> pinned "
        "uint8 batch -> CUDA augmentation/normalization -> iResNet-34. "
        "No detection, bbox crop, or detector-based image filtering.",
        "",
        "Preparation, logs, checkpoints, graphs, metrics, and source snapshots "
        "are stored below this job directory. Original images remain unchanged.",
        "",
    ]
    profile_path = job / "batch_profile.json"
    if profile_path.exists():
        profile = read_json(profile_path)
        rows += [
            "## Measured GPU batch profile",
            "",
            "Bounded real-image pilot with ArcFace, AdamW and AMP. "
            "This measures memory and step throughput, not model accuracy.",
            "",
            "| Batch | Peak allocated MiB | Peak reserved MiB | Device used MiB |",
            "| --- | --- | --- | --- |",
        ]
        for candidate in profile.get("candidates", []):
            rows.append(
                f"| {candidate['batch_size']} | "
                f"{candidate.get('peak_allocated_mib', 0):.0f} | "
                f"{candidate.get('peak_reserved_mib', 0):.0f} | "
                f"{candidate.get('after', {}).get('device_used_mib', 0):.0f} |"
            )
        rows += ["", profile.get("recommendation", {}).get("reason", ""), ""]
    progress_path = job / "training_progress.json"
    if progress_path.exists():
        progress = read_json(progress_path)
        rows += ["## Dataset", "", "```json", json.dumps(progress, indent=2), "```", ""]
    if state.get("latest_epoch"):
        rows += [
            "## Last completed epoch",
            "",
            "```json",
            json.dumps(state["latest_epoch"], indent=2),
            "```",
            "",
        ]
    result_path = job / "final_results.json"
    if result_path.exists():
        rows += [
            "## Results",
            "",
            "The best checkpoint is selected on dev. Test thresholds are frozen "
            "from dev; test is evaluated after training, without a margin sweep.",
            "",
            "```json",
            json.dumps(read_json(result_path), indent=2),
            "```",
            "",
        ]
    else:
        rows += ["Final training and test metrics are not available yet.", ""]
    if state.get("error"):
        rows += ["## Failure", "", state["error"], ""]
    rows += [
        "## Files",
        "",
        "- `status.json`: job state, process IDs, heartbeat, latest epoch.",
        "- `logs/`: separate logs for preparation and training.",
        f"- Base manifests: `{config.get('face_protocol', 'preparation/base/')}`.",
        "- `arcface/runs/margin_0.3_seed_42/weights/`: "
        "best, last, embedding checkpoints.",
        "- `arcface/runs/margin_0.3_seed_42/trainer_runs/`: "
        "live history and epoch checkpoints.",
        "- `arcface/test_results.csv` and `arcface/test/`: "
        "final evaluation and graphs.",
        "- `final_results.json`: machine-readable final summary.",
        "",
        "Resume an interrupted job using the same Python interpreter, script, "
        "and `--job-dir`. An interrupted partial epoch is repeated.",
    ]
    temporary = job / "report.md.tmp"
    temporary.write_text("\n".join(rows) + "\n", encoding="utf-8")
    temporary.replace(job / "report.md")


def update_status(job: Path, state: dict, **changes) -> None:
    state.update(changes, updated_at_utc=now())
    if (job / "phase.json").exists():
        state["training_phase"] = read_json(job / "phase.json")["phase"]
    histories = list(job.glob("arcface/runs/*/trainer_runs/*/history.csv"))
    if histories:
        history = max(histories, key=lambda path: path.stat().st_mtime_ns)
        try:
            with history.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            if rows:
                state["latest_epoch"] = rows[-1]
                state["history_path"] = str(history)
        except (OSError, csv.Error):
            pass
    write_json(job / "status.json", state)
    write_report(job, state)


def run_stage(job: Path, state: dict, name: str, args: list[str]) -> None:
    log_path = job / "logs" / f"{name}.log"
    print(f"[{now()}] Starting {name}; log: {log_path}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{now()}] command: {args!r}\n")
        log.flush()
        process = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        update_status(
            job,
            state,
            status="running",
            stage=name,
            stage_started_at_utc=now(),
            child_pid=process.pid,
            log_path=str(log_path),
            command=args,
        )
        while process.poll() is None:
            time.sleep(10)
            update_status(job, state)
        if process.returncode:
            raise RuntimeError(
                f"Stage {name} exited with code {process.returncode}. See {log_path}."
            )
    state.setdefault("completed_stages", {})[name] = now()
    update_status(job, state, child_pid=None)
    print(f"[{now()}] Completed {name}", flush=True)


def snapshot_sources(job: Path) -> None:
    snapshot = job / "source_snapshot"
    if snapshot.exists():
        return
    files = [Path(__file__).resolve(), ROOT / "src/common/face_crop.py"]
    for relative in (
        "src/datasets",
        "src/recognition/identification",
        "src/training",
    ):
        files.extend((ROOT / relative).glob("*.py"))
    files.extend(
        [
            ROOT / "configs/datasets.yaml",
        ]
    )
    hashes = {}
    for source in files:
        relative = source.relative_to(ROOT)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    write_json(job / "source_sha256.json", hashes)


def train(job: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.recognition.identification.config import ExperimentConfig
    from src.recognition.identification.notebook_utils import processing_preview
    from src.recognition.identification.protocol import load_image_protocol
    from src.recognition.identification.training import (
        configure_runtime,
        environment_details,
        evaluate_selected,
        run_experiment,
    )

    options = read_json(job / "job_options.json")
    config_path = job / "experiment_config.json"
    if config_path.is_file():
        config = ExperimentConfig(**read_json(config_path))
        if config.method != "arcface" or Path(config.artifacts_root).resolve() != job:
            raise ValueError("Recorded ArcFace experiment does not belong to this job")
    else:
        config = ExperimentConfig.arcface(
            data_root=ROOT / "data/VGGFace2",
            face_protocol=job / "preparation/base",
            artifacts_root=job,
            identities_per_batch=options["batch_size"] // 4,
        )
    device = configure_runtime(config)
    write_json(config_path, config.to_dict())
    write_json(job / "environment.json", environment_details(device))
    write_json(job / "phase.json", {"phase": "loading_full_image_manifests"})
    print(
        f"[{now()}] Loading completed base manifests without bbox filtering", flush=True
    )
    prepared = load_image_protocol(
        config.data_root,
        config.face_protocol,
        seed=config.seed,
        images_per_identity=config.images_per_identity,
        probe_per_identity=4,
    )
    write_json(job / "training_progress.json", prepared.metadata)
    print(json.dumps(prepared.metadata, indent=2), flush=True)
    figure = processing_preview(
        prepared, config.data_root, config.output_dir / "graphs/processing.png"
    )
    plt.close(figure)
    write_json(job / "phase.json", {"phase": "training_full_images"})
    print(f"[{now()}] Starting full ArcFace training", flush=True)
    runs = run_experiment(config, prepared)
    write_json(job / "training_result.json", runs.iloc[0].to_dict())
    write_json(job / "phase.json", {"phase": "final_test"})
    print(f"[{now()}] Training complete; running fixed-protocol final test", flush=True)
    tests = evaluate_selected(config, prepared, runs)
    result = {
        "completed_at_utc": now(),
        "training": runs.iloc[0].to_dict(),
        "test": tests.to_dict("records"),
        "test_used_for_selection": False,
        "dataset": prepared.metadata,
    }
    write_json(job / "final_results.json", result)
    write_json(job / "phase.json", {"phase": "completed"})
    print(json.dumps(result, indent=2, default=str), flush=True)


def orchestrate(job: Path, workers: int, options: dict) -> None:
    job.mkdir(parents=True, exist_ok=True)
    (job / "logs").mkdir(exist_ok=True)
    os.environ.update(
        PYTHONUNBUFFERED="1",
        PYTHONIOENCODING="utf-8",
        MPLBACKEND="Agg",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        TQDM_MININTERVAL="30",
    )
    state = {
        "status": "starting",
        "stage": "initializing",
        "pid": os.getpid(),
        "started_at_utc": now(),
        "job_dir": str(job),
        "python": sys.executable,
        "pipeline": "full_images",
        "require_bboxes": False,
        "batch_size": options["batch_size"],
    }
    update_status(job, state)
    try:
        snapshot_sources(job)
        space = shutil.disk_usage(job)
        if space.free < 15 * 2**30:
            raise RuntimeError(
                "At least 15 GiB free disk space is required for this job."
            )
        base = job / "preparation/base"
        config_path = job / "experiment_config.json"
        if config_path.is_file():
            base = Path(read_json(config_path)["face_protocol"])
        base_files = (
            "dataset_report.json",
            "vggface2_classification.csv",
            "vggface2_verification.csv",
            "recognition_summary.json",
            "verification_pairs_val.csv",
            "verification_pairs_test.csv",
        )
        if config_path.is_file() and not all(
            (base / name).is_file() for name in base_files
        ):
            raise FileNotFoundError(f"Incomplete recorded manifests: {base}")
        if options.get("base_manifests") and not all(
            (base / name).is_file() for name in base_files
        ):
            source = Path(options["base_manifests"])
            if not all((source / name).is_file() for name in base_files):
                raise FileNotFoundError(f"Incomplete source base manifests: {source}")
            update_status(job, state, status="running", stage="reuse_base_manifests")
            base.mkdir(parents=True, exist_ok=True)
            copied = {}
            for source_file in source.iterdir():
                if source_file.is_file() and source_file.suffix in {".csv", ".json"}:
                    target = base / source_file.name
                    shutil.copy2(source_file, target)
                    with target.open("rb") as stream:
                        copied[source_file.name] = hashlib.file_digest(
                            stream, "sha256"
                        ).hexdigest()
            write_json(
                job / "manifest_reuse.json", {"source": str(source), "sha256": copied}
            )
        if not all((base / name).is_file() for name in base_files):
            run_stage(
                job,
                state,
                "prepare_manifests",
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "src.datasets.prepare",
                    "--config",
                    str(ROOT / "configs/datasets.yaml"),
                    "--only",
                    "recognition",
                    "--set",
                    f"output_dir={base.as_posix()}",
                    "--set",
                    f"workers={workers}",
                ],
            )
        run_stage(
            job,
            state,
            "train_and_evaluate",
            [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--job-dir",
                str(job),
                "--train-stage",
            ],
        )
        update_status(
            job, state, status="completed", stage="completed", completed_at_utc=now()
        )
    except BaseException as exc:
        (job / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        update_status(job, state, status="failed", error=str(exc), failed_at_utc=now())
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True, type=Path)
    parser.add_argument("--prepare-workers", type=int, default=4)
    parser.add_argument(
        "--base-manifests", type=Path, help="Reuse completed base manifests"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Physical batch; multiple of 4"
    )
    parser.add_argument(
        "--batch-profile", type=Path, help="JSON from a measured GPU batch pilot"
    )
    parser.add_argument("--train-stage", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.prepare_workers < 1:
        parser.error("--prepare-workers must be positive")
    if args.batch_size < 8 or args.batch_size % 4:
        parser.error("--batch-size must be a multiple of four and at least eight")
    job = args.job_dir.resolve()
    if not job.is_relative_to(ROOT / "artifacts"):
        parser.error("--job-dir must be inside this repository's artifacts directory")
    if args.train_stage:
        train(job)
    else:
        options = {
            "pipeline": "full_images",
            "batch_size": args.batch_size,
            "base_manifests": (
                str(args.base_manifests.resolve()) if args.base_manifests else None
            ),
        }
        job.mkdir(parents=True, exist_ok=True)
        options_path = job / "job_options.json"
        if options_path.exists():
            previous = read_json(options_path)
            if previous["batch_size"] != options["batch_size"]:
                parser.error("Changing batch size requires a fresh --job-dir")
            options = previous
        else:
            if (job / "experiment_config.json").exists():
                parser.error("An older run needs a fresh --job-dir for this pipeline")
            write_json(options_path, options)
        if args.batch_profile:
            shutil.copy2(args.batch_profile.resolve(), job / "batch_profile.json")
        orchestrate(job, args.prepare_workers, options)


if __name__ == "__main__":
    main()
