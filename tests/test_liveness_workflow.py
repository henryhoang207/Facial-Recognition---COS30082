"""Behavioral checks for archive staging, leakage, recovery and Colab delivery."""

import ast
import json
import shutil
from pathlib import Path
from zipfile import ZipFile

import cv2
import duckdb
import nbformat
import numpy as np
import pandas as pd
import pytest
import torch

from src.liveness.archive import stage_archive, stage_directory
from src.liveness.config import LivenessConfig, defaults, load_config
from src.liveness.metrics import eer_threshold, summarize
from src.liveness.models import build_model
from src.liveness.predictor import LivenessPredictor
from src.liveness.provenance import audit_images, make_provenance, verify_provenance
from src.liveness.select import select_candidates
from src.liveness.train import train
from src.training.trainer import Trainer

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


@pytest.fixture
def images(tmp_path):
    rows = []
    for index, split in enumerate(("train", "val", "test")):
        for attack_index, attack in enumerate(("live", "print", "replay")):
            for frame in range(2):
                relative = f"{split}/{attack}_{frame}.png"
                path = tmp_path / "images" / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                pixels = np.random.default_rng(
                    100 * index + 10 * attack_index + frame
                ).integers(0, 256, (48, 48, 3), dtype=np.uint8)
                cv2.imwrite(str(path), pixels)
                rows.append(
                    dict(
                        path=relative,
                        label=int(attack == "live"),
                        attack=attack,
                        subject=f"person_{index}",
                        source="fixture",
                        split=split,
                        x1=4,
                        y1=4,
                        x2=44,
                        y2=44,
                    )
                )
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return tmp_path / "images", manifest


def test_portable_provenance_rejects_missing_changed_and_leaked_data(images, tmp_path):
    root, manifest = images
    frame = audit_images(pd.read_csv(manifest), root)
    provenance = make_provenance(frame)
    moved = tmp_path / "moved"
    shutil.copytree(root, moved)
    verified = audit_images(frame, moved)
    verify_provenance(verified, provenance, "test")
    with pytest.raises(ValueError, match="lacks verifiable"):
        verify_provenance(verified, None, "test")
    leaked = verified.copy()
    leaked.loc[leaked.split == "test", "subject"] = "person_0"
    with pytest.raises(ValueError, match="subjects leak"):
        make_provenance(leaked)
    leaked = verified.copy()
    leaked.loc[leaked.index[-1], "sha256"] = leaked.loc[0, "sha256"]
    with pytest.raises(ValueError, match="duplicate image bytes"):
        make_provenance(leaked)
    (moved / frame.iloc[0].path).write_bytes(b"broken")
    with pytest.raises(ValueError, match="content changed"):
        audit_images(frame, moved)


@pytest.mark.parametrize("name", ["mobilenetv3", "deeppixbis"])
def test_real_model_losses_scores_and_frozen_statistics(name):
    model = build_model(
        name,
        pretrained=False,
        freeze_blocks=4 if name == "mobilenetv3" else 0,
        img_size=32,
    )
    model.train()
    frozen = {
        key: value.clone()
        for key, value in model.state_dict().items()
        if key.startswith("net.features.0.")
    }
    output = model(torch.rand(3, 3, 32, 32))
    loss = model.loss(output, torch.tensor([0.0, 1.0, 0.0]))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.score(output).shape == (3,)
    assert bool(((model.score(output) >= 0) & (model.score(output) <= 1)).all())
    if name == "deeppixbis":
        assert output["map_logit"].shape == (3, 1, 2, 2)
    for key, value in frozen.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)


def test_liveness_resume_matches_uninterrupted_weights_and_threshold(
    images, tmp_path, monkeypatch
):
    root, manifest = images
    raw = defaults("mobilenetv3")
    raw.update(
        pretrained=False,
        img_size=32,
        manifest=[str(manifest)],
        data_root=str(root),
        output_dir=str(tmp_path / "full"),
        fig_dir=str(tmp_path / "fig-full"),
    )
    raw["training"].update(
        epochs=2,
        batch_size=3,
        accumulation_steps=2,
        num_workers=0,
        amp=False,
        deterministic=True,
        progress=False,
        device="cpu",
    )
    full = train(LivenessConfig.model_validate(raw))
    original_epoch = Trainer._epoch
    epochs = 0

    def interrupt(self, loader, *, train):
        nonlocal epochs
        if train:
            epochs += 1
            if epochs == 2:
                raise RuntimeError("simulated disconnect")
        return original_epoch(self, loader, train=train)

    raw.update(
        output_dir=str(tmp_path / "partial"), fig_dir=str(tmp_path / "fig-partial")
    )
    with monkeypatch.context() as context:
        context.setattr(Trainer, "_epoch", interrupt)
        with pytest.raises(RuntimeError, match="simulated disconnect"):
            train(LivenessConfig.model_validate(raw))
    last = next((tmp_path / "partial").glob("*/last.pt"))
    raw.update(
        output_dir=str(tmp_path / "resume"), fig_dir=str(tmp_path / "fig-resume")
    )
    resumed = train(LivenessConfig.model_validate(raw), resume=last)
    first = torch.load(full.last_checkpoint, weights_only=True)
    second = torch.load(resumed.last_checkpoint, weights_only=True)
    assert first["history"] == second["history"]
    for key, value in first["task"].items():
        torch.testing.assert_close(value, second["task"][key], rtol=0, atol=0)
    inference_path = next(resumed.run_dir.glob("liveness_*.pt"))
    checkpoint = torch.load(inference_path, weights_only=True)
    assert (
        checkpoint["threshold"]
        == second["history"][second["best_epoch"] - 1]["val_threshold"]
    )
    predictor = LivenessPredictor(inference_path, device="cpu")
    image = cv2.imread(str(root / "val/live_0.png"))
    from src.common.face_crop import crop_face

    face = crop_face(image, (4, 4, 44, 44), size=32, scale=2.0)
    assert predictor.predict(face) == predictor.predict_frame(image, (4, 4, 44, 44))


def test_constant_scores_keep_the_roc_threshold_boundary():
    scores, labels = np.ones(4, dtype=np.float32), np.array([0, 0, 1, 1])
    threshold, eer = eer_threshold(scores, labels)
    rates = summarize(
        scores, labels, np.array(["print", "replay", "live", "live"]), threshold
    )
    assert threshold > 1 and rates["apcer"] == 0 and rates["bpcer"] == 1
    assert eer == rates["acer"] == 0.5
    with pytest.raises(ValueError, match="both live"):
        eer_threshold([0.1, 0.2], [0, 0])


def build_archive(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    rows = []
    for index, (attack, spoof_type, is_spoof) in enumerate(
        (("live", 0, 0), ("print", 1, 1), ("replay", 9, 1), ("mask", 10, 1))
    ):
        relative = f"Data/train/1/{attack}/face.png"
        path = source / relative
        path.parent.mkdir(parents=True)
        cv2.imwrite(str(path), np.full((32, 32, 3), index * 30, np.uint8))
        path.with_name("face_BB.txt").write_text("16 16 128 128 0.99")
        rows.append((relative, 1, "train", spoof_type, is_spoof))
    with duckdb.connect(str(source / "celeba_spoof.duckdb")) as con:
        con.execute(
            "CREATE TABLE celeba_labels(path VARCHAR, subject_id INTEGER, "
            "split VARCHAR, spoof_type INTEGER, is_spoof INTEGER)"
        )
        con.executemany("INSERT INTO celeba_labels VALUES (?,?,?,?,?)", rows)
    archive = tmp_path / "source.zip"
    with ZipFile(archive, "w") as zip_file:
        for path in source.rglob("*"):
            if path.is_file():
                zip_file.write(
                    path,
                    "CelebA_Spoof_processed/" + path.relative_to(source).as_posix(),
                )
    return archive


def test_archive_stages_only_eligible_files_and_rejects_zip_slip(tmp_path):
    archive = build_archive(tmp_path)
    root = tmp_path / "staged"
    result = stage_archive(archive, root, reserve_bytes=0)
    assert result["selected_files"] == 7
    assert not (root / "Data/train/1/mask/face.png").exists()
    assert stage_archive(archive, root, reserve_bytes=0) == result
    with ZipFile(archive, "a") as zip_file:
        zip_file.writestr("../escape", "bad")
    with pytest.raises(ValueError, match="unsafe ZIP"):
        stage_archive(archive, tmp_path / "unsafe", reserve_bytes=0)


def test_extracted_folder_staging_is_read_only_and_recovers_completed_copies(
    tmp_path, monkeypatch
):
    build_archive(tmp_path)
    source, target = tmp_path / "source", tmp_path / "staged_folder"
    before = {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    staged = stage_directory(source, target, reserve_bytes=0)
    assert staged["source_type"] == "directory" and staged["selected_files"] == 7
    assert not (target / "Data/train/1/mask/face.png").exists()
    for path in target.rglob("*"):
        if path.is_file() and path.name != "staging.json":
            assert path.read_bytes() == before[path.relative_to(target)]
    assert stage_directory(source, target, reserve_bytes=0) == staged
    missing = target / "Data/train/1/print/face.png"
    missing.unlink()
    copied = []
    original = shutil.copyfileobj

    def copy(reader, writer, **kwargs):
        copied.append(reader.name)
        return original(reader, writer, **kwargs)

    monkeypatch.setattr(shutil, "copyfileobj", copy)
    assert stage_directory(source, target, reserve_bytes=0) == staged
    assert copied == [str(source / "Data/train/1/print/face.png")]
    assert before == {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    with pytest.raises(ValueError, match="must not overlap"):
        stage_directory(source, source / "nested", reserve_bytes=0)
    assert not (source / "nested").exists()


def test_extracted_folder_staging_rejects_missing_files_and_low_disk(
    tmp_path, monkeypatch
):
    build_archive(tmp_path)
    source = tmp_path / "source"
    (source / "Data/train/1/live/face_BB.txt").unlink()
    with pytest.raises(FileNotFoundError, match="dataset file unavailable"):
        stage_directory(source, tmp_path / "missing", reserve_bytes=0)
    failure = json.loads((tmp_path / "missing/staging_failure.json").read_text())
    assert failure["expected_file"].endswith("face_BB.txt")
    assert failure["visible_directories"][0]["entries_sample"] == ["face.png"]
    from types import SimpleNamespace

    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=1))
    with pytest.raises(OSError, match="not enough local disk"):
        stage_directory(source, tmp_path / "full", reserve_bytes=0)


def test_extracted_folder_staging_rejects_catalog_path_traversal(tmp_path):
    build_archive(tmp_path)
    source = tmp_path / "source"
    with duckdb.connect(str(source / "celeba_spoof.duckdb")) as con:
        con.execute("INSERT INTO celeba_labels VALUES ('../private.png',1,'train',1,1)")
    with pytest.raises(ValueError, match="unsafe catalog path"):
        stage_directory(source, tmp_path / "invalid", reserve_bytes=0)


def test_validation_selection_needs_actual_laptop_latency(tmp_path):
    paths = []
    for model, acer in (("mobilenetv3", 0.09), ("deeppixbis", 0.08)):
        path = tmp_path / (model + ".pt")
        torch.save(
            {
                "config": {"model": model, "provenance": {"manifest_sha256": "same"}},
                "threshold": 0.5,
                "val_metrics": {"val_acer": acer},
            },
            path,
        )
        paths.append(path)
    pending = select_candidates(paths)
    assert pending["status"] == "pending_demo_laptop_latency"
    timings = [
        {
            "checkpoint_sha256": candidate["sha256"],
            "machine_role": "colab",
            "median_ms": 1,
        }
        for candidate in pending["candidates"]
    ]
    assert select_candidates(paths, timings)["selected_sha256"] is None
    for item in timings:
        item["machine_role"] = "demo_laptop"
        item["median_ms"] = (
            90
            if item["checkpoint_sha256"] == pending["candidates"][1]["sha256"]
            else 270
        )
    assert (
        select_candidates(paths, timings)["selected_sha256"]
        == pending["candidates"][1]["sha256"]
    )


def test_notebook_is_valid_output_free_and_every_code_cell_compiles():
    notebook = nbformat.read(ROOT / "notebooks/liveness_training.ipynb", as_version=4)
    nbformat.validate(notebook)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            assert cell.outputs == [] and cell.execution_count is None
            compile(cell.source, f"cell-{index}", "exec")


@pytest.mark.parametrize("group_path", ["COS30082 - Project/AI/FaceGuard", "FaceGuard"])
def test_notebook_resolves_group_outputs_and_separate_dataset_shortcut(
    tmp_path, group_path
):
    notebook = nbformat.read(ROOT / "notebooks/liveness_training.ipynb", as_version=4)
    cell = next(
        cell for cell in notebook.cells if "def resolve_drive_locations(" in cell.source
    )
    function = next(
        node
        for node in ast.parse(cell.source).body
        if isinstance(node, ast.FunctionDef) and node.name == "resolve_drive_locations"
    )
    namespace = {"GROUP_FOLDER_URL": "group-link", "DATASET_URL": "dataset-link"}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "drive-setup", "exec"),
        namespace,
    )
    group = tmp_path / group_path
    group.mkdir(parents=True)
    archive = tmp_path / "CelebA_Spoof_processed.zip"
    with pytest.raises(FileNotFoundError, match="dataset-link"):
        namespace["resolve_drive_locations"](tmp_path)
    archive.touch()
    assert namespace["resolve_drive_locations"](tmp_path) == (group, archive)
    if group_path != "FaceGuard":
        assert not (tmp_path / "FaceGuard").exists()


@pytest.mark.parametrize("shortcut", [False, True])
def test_notebook_prefers_the_extracted_dataset_folder(tmp_path, shortcut):
    notebook = nbformat.read(ROOT / "notebooks/liveness_training.ipynb", as_version=4)
    cell = next(
        cell for cell in notebook.cells if "def resolve_drive_locations(" in cell.source
    )
    function = next(
        node
        for node in ast.parse(cell.source).body
        if isinstance(node, ast.FunctionDef) and node.name == "resolve_drive_locations"
    )
    namespace = {"GROUP_FOLDER_URL": "group-link", "DATASET_URL": "dataset-link"}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "drive-setup", "exec"),
        namespace,
    )
    group = tmp_path / "COS30082 - Project/AI/FaceGuard"
    group.mkdir(parents=True)
    folder = tmp_path / (
        "CelebA_Spoof_processed"
        if shortcut
        else "COS30082 - Project/Data/anti_spoof/CelebA_Spoof_processed"
    )
    (folder / "Data").mkdir(parents=True)
    (folder / "celeba_spoof.duckdb").touch()
    (tmp_path / "CelebA_Spoof_processed.zip").touch()
    assert namespace["resolve_drive_locations"](tmp_path) == (group, folder)


def test_yaml_and_legacy_cli_preserve_model_defaults(tmp_path):
    for model in ("mobilenetv3", "deeppixbis"):
        config, resume = load_config(
            ["--model", model, "--manifest", str(tmp_path / "manifest.csv")]
        )
        assert config.training.batch_size * config.training.accumulation_steps == 64
        assert resume is None


def test_completed_notebook_job_is_reused_without_training(tmp_path):
    notebook = nbformat.read(ROOT / "notebooks/liveness_training.ipynb", as_version=4)
    cell = next(cell for cell in notebook.cells if "def run_job(" in cell.source)
    functions = [
        node
        for node in ast.parse(cell.source).body
        if isinstance(node, ast.FunctionDef)
    ]
    namespace = {
        "Path": Path,
        "json": json,
        "defaults": defaults,
        "LivenessConfig": LivenessConfig,
        "OUT": tmp_path,
        "MANIFEST": tmp_path / "manifest.csv",
        "LOCAL_DATA": tmp_path / "images",
        "FIG_DIR": tmp_path / "figures",
    }
    exec(
        compile(
            ast.Module(body=functions, type_ignores=[]), "notebook-functions", "exec"
        ),
        namespace,
    )
    raw = namespace["settings"]("mobilenetv3", "full_mobilenetv3")
    complete = Path(raw["output_dir"]) / "completed"
    complete.mkdir(parents=True)
    (complete / "resolved_config.json").write_text(
        LivenessConfig.model_validate(raw).model_dump_json()
    )
    (complete / "export.json").write_text(json.dumps({"checkpoint": "finished.pt"}))
    assert namespace["run_job"]("full_mobilenetv3", raw) == Path("finished.pt")
    (tmp_path / "frozen_candidates.json").write_text("{}")
    with pytest.raises(RuntimeError, match="frozen"):
        namespace["run_job"](
            "tune_mobilenetv3", namespace["settings"]("mobilenetv3", "tune_mobilenetv3")
        )


def test_processed_public_archive_to_two_models_and_measured_report(tmp_path):
    from src.datasets.config import PreparationConfig
    from src.datasets.liveness import LivenessPreparer
    from src.datasets.validate import ManifestValidator
    from src.liveness.benchmark import benchmark
    from src.liveness.evaluate import main as evaluate
    from src.liveness.report import write_report
    from src.training.config import read_yaml

    source = tmp_path / "public_source"
    source.mkdir()
    catalog = []
    for split, subjects in (("train", [1, 2, 3]), ("test", [3, 4])):
        for subject in subjects:
            for attack, spoof_type, is_spoof in (
                ("live", 0, 0),
                ("print", 2, 1),
                ("replay", 9, 1),
                ("mask", 10, 1),
            ):
                relative = f"Data/{split}/{subject}/{attack}/face.png"
                path = source / relative
                path.parent.mkdir(parents=True)
                seed = 1000 * int(split == "test") + subject * 20 + spoof_type
                pixels = np.random.default_rng(seed).integers(
                    0, 256, (48, 48, 3), dtype=np.uint8
                )
                cv2.imwrite(str(path), pixels)
                path.with_name("face_BB.txt").write_text("16 16 160 160 0.99")
                catalog.append((relative, subject, split, spoof_type, is_spoof))
    with duckdb.connect(str(source / "celeba_spoof.duckdb")) as con:
        con.execute(
            "CREATE TABLE celeba_labels (path VARCHAR, subject_id INTEGER, "
            "split VARCHAR, spoof_type INTEGER, is_spoof INTEGER)"
        )
        con.executemany("INSERT INTO celeba_labels VALUES (?,?,?,?,?)", catalog)
    archive = tmp_path / "public.zip"
    with ZipFile(archive, "w") as zipped:
        for path in source.rglob("*"):
            if path.is_file():
                zipped.write(
                    path,
                    "CelebA_Spoof_processed/" + path.relative_to(source).as_posix(),
                )
    root = tmp_path / "staged_public"
    stage_directory(source, root, reserve_bytes=0)
    settings = read_yaml(ROOT / "configs/datasets.yaml")
    settings.update(output_dir=str(tmp_path / "manifests"), workers=2)
    settings["liveness"]["root"] = str(root)
    prepared = PreparationConfig.model_validate(settings)
    prepared.output_dir.mkdir()
    summary = LivenessPreparer(prepared).prepare()
    assert summary["splits"] == {"train": 3, "val": 3, "test": 6}
    assert ManifestValidator(prepared).liveness()["rows"] == 12
    manifest = prepared.output_dir / "celeba_liveness.csv"
    fig = tmp_path / "results"
    paths = []
    for model in ("mobilenetv3", "deeppixbis"):
        raw = defaults(model)
        raw.update(
            pretrained=False,
            img_size=32,
            manifest=[str(manifest)],
            data_root=str(root),
            output_dir=str(tmp_path / "models"),
            fig_dir=str(fig),
        )
        raw["training"].update(
            epochs=1,
            batch_size=3,
            num_workers=0,
            progress=False,
            device="cpu",
            deterministic=True,
        )
        result = train(LivenessConfig.model_validate(raw))
        paths.append(next(result.run_dir.glob("liveness_*.pt")))
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps(select_candidates(paths)))
    for path in paths:
        evaluate(
            [
                "--checkpoint",
                str(path),
                "--manifest",
                str(manifest),
                "--data-root",
                str(root),
                "--fig-dir",
                str(fig),
                "--num-workers",
                "0",
                "--frozen",
                str(frozen),
            ]
        )
    report = write_report(
        fig,
        frozen,
        prepared.output_dir / "liveness_summary.json",
        tmp_path / "report.md",
    )
    assert all(
        title in report.read_text()
        for title in ("## Methodology", "## Results", "## Discussion")
    )
    assert (fig / "liveness_comparison_roc.png").exists()
    timing = benchmark(
        paths[0], [manifest], root, repeats=2, warmup=1, threads=2, machine_role="test"
    )
    assert timing["median_ms"] > 0 and timing["scope"].startswith("crop")
