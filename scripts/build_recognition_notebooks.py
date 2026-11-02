"""Regenerate the two English recognition notebooks without stored training outputs."""

from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]


def build(method: str, filename: str) -> None:
    title = "Triplet Loss" if method == "triplet" else "ArcFace"
    markdown = nbformat.v4.new_markdown_cell
    code = nbformat.v4.new_code_cell
    cells = [
        markdown(f"# {title}: iResNet-34 face embeddings\n\n"
                 "**Flow: image processing → model training → loss graphs and saved artifacts.**\n\n"
                 "Use the **Python (FaceGuard CUDA 13.0)** kernel. Both experiments use random iResNet-34, "
                 "112×112 RGB crops and 256-dimensional normalized embeddings. "
                 "No pretrained face weights are downloaded. Full runs can take many hours. "
                 "Set `SMOKE_TEST=True` for a bounded real-image code check; its outputs are isolated below the configured artifact root."),
        markdown("## 1. Environment and experiment settings\n\n"
                 "Use the repository's shared `.venv` environment with Python 3.14 and PyTorch CUDA 13.0. "
                 "Run `scripts/setup_mtrlearn.ps1` from the repository root to install the shared requirements "
                 "and register the `mtrlearn` kernel. Do not install packages into a running notebook kernel."),
        code('''from pathlib import Path
import os
import sys
import json

START_DIRECTORY = Path.cwd().resolve()
ROOT = next((candidate for candidate in (START_DIRECTORY, *START_DIRECTORY.parents)
             if (candidate / "src" / "recognition").is_dir()), None)
if ROOT is None:
    raise RuntimeError("Open this notebook from a directory inside the repository.")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image as DisplayImage
from src.recognition.config import ExperimentConfig
from src.recognition.data import prepare_manifests, FaceDataset, PKBatchSampler, make_loader
from src.recognition.models import IResNet34
from src.recognition.losses import TripletMetricLoss, ArcFaceHead
from src.recognition.training import configure_runtime, run_sweep, evaluate_selected
from src.recognition.notebook_utils import processing_preview, pilot_runtime, smoke_prepared, smoke_config

print("Python:", sys.executable)
print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
if not torch.cuda.is_available():
    raise RuntimeError("Select the Python (FaceGuard CUDA 13.0) kernel before running training.")
print("GPU:", torch.cuda.get_device_name(0))'''),
        code(f'''SMOKE_TEST = os.environ.get("MTRLEARN_SMOKE", "0") == "1"
RUN_FINAL_TEST = False  # Enable only after the margin-selection design is fixed.
ARTIFACTS_ROOT = Path(os.environ.get("MTRLEARN_ARTIFACTS_ROOT", str(ROOT / "artifacts" / "main-cu130")))
if not ARTIFACTS_ROOT.is_absolute():
    ARTIFACTS_ROOT = ROOT / ARTIFACTS_ROOT
ARTIFACTS_ROOT = ARTIFACTS_ROOT.resolve()

CONFIG = ExperimentConfig(
    method="{method}",
    data_root=ROOT / "data" / "VGGFace2",
    artifacts_root=ARTIFACTS_ROOT,
    identities_per_batch=8,
    images_per_identity=4,
    epochs=60,
    num_workers=0 if os.name == "nt" else 2,
    device="cuda",
    amp=True,
)
configure_runtime(CONFIG)
display(pd.Series(CONFIG.to_dict(), name="Experiment settings"))'''),
        markdown("## 2. Image processing and identity-disjoint manifests\n\n"
                 "Verify the source catalog, hash exact duplicates, and keep one representative without deleting originals. "
                 "Split original train identities into 432 train and 48 dev; original val identities become the 60-person final test. "
                 "The two notebooks reuse the same seed-42 manifests, evaluation pairs and training probe. "
                 "Near duplicates and identity aliases still require review; byte hashing cannot detect them."),
        code('''prepared = prepare_manifests(
    CONFIG.data_root,
    ROOT / "artifacts" / "shared" / "manifests",
    split_seed=42,
    dev_identities=48,
    genuine_per_identity=200,
    impostor_per_identity_pair=100,
    probe_per_identity=4,
)
if SMOKE_TEST:
    prepared = smoke_prepared(prepared)
    CONFIG = smoke_config(CONFIG)
    print("SMOKE TEST: tiny real-image subsets, altered curriculum gate, bounded epochs, no accuracy claims.")

display(pd.DataFrame([
    {"split": name, "identities": frame.identity_id.nunique(), "images": len(frame)}
    for name, frame in [("train", prepared.train), ("dev", prepared.dev), ("test", prepared.test)]
]))
print("Dev pairs:", len(prepared.dev_pairs), "Test pairs:", len(prepared.test_pairs))
print("Training probe triplets:", len(prepared.probe_triplets))'''),
        code('''figure = processing_preview(prepared, CONFIG.data_root, CONFIG.output_dir / "graphs" / "image_processing.png")
display(figure)
plt.close(figure)

sampler = PKBatchSampler(prepared.train.label.to_numpy(), P=CONFIG.identities_per_batch,
                         K=CONFIG.images_per_identity, batches_per_epoch=1, seed=42)
batch = next(iter(make_loader(prepared.train, CONFIG.data_root, train=True,
                              batch_sampler=sampler, num_workers=0, seed=42)))
print("Image tensor:", tuple(batch["image"].shape), batch["image"].dtype)
print("Normalized range:", float(batch["image"].min()), float(batch["image"].max()))
print("Images per identity:", torch.unique(batch["label"], return_counts=True)[1].tolist())
del batch'''),
        markdown("## 3. Model and objective\n\n"
                 "The reusable source is in `src/recognition/models.py` and `losses.py`. "
                 "The model uses iResNet stages [3,4,6,3], a 256-feature linear layer and final BatchNorm, "
                 "followed by explicit FP32 L2 normalization. Each margin run starts from the same encoder initialization for its seed."),
        code('''model = IResNet34(embedding_dim=256)
print(model)
print("Total encoder parameters:", f"{sum(p.numel() for p in model.parameters()):,}")
print("Trainable encoder parameters:", f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
del model'''),
    ]
    if method == "triplet":
        cells += [markdown("B0 uses squared Euclidean Triplet Loss on unit embeddings, margins **0.2, 0.3, 0.4**. "
                           "All B(K−1) directed positive pairs form the denominator, including missing semi-hard candidates as zero. "
                           "Random negatives are used first. After at least five random epochs, three consecutive fixed TRAIN probe checks "
                           "must pass ranking rate ≥0.75, mean distance gap ≥0.05 and feature spread >0.001. "
                           "The switch is one-way. Strict semi-hard selection has no hard-negative fallback; empty batches skip optimizer updates."),
                  code('''objective = TripletMetricLoss(margin=0.2)
print(objective)
print("Margin sweep:", CONFIG.margins)
print("Squared distance: d2(a,p) = ||normalize(a) - normalize(p)||^2")''')]
    else:
        cells += [markdown("B1 uses **ArcFace training classification** with angular margins **0.3, 0.4, 0.5 radians**, "
                           "scale **64**, normalized features/weights and the non-easy monotonic fallback. "
                           "The full training-identity head is removed for inference. Dev/test identities are unseen, "
                           "so no dev classification loss or head accuracy is used for selection."),
                  code('''objective = ArcFaceHead(num_classes=len(prepared.train_labels), embedding_dim=256,
                        margin=0.3, scale=64, seed=100042)
print(objective)
print("Training class weights:", tuple(objective.weight.shape))
print("Angular margin sweep (radians):", CONFIG.margins)
del objective''')]
    cells += [
        markdown("## 4. Common GPU memory pilot\n\n"
                 "Measure two bounded optimization steps for both objectives using disposable models, then choose a common batch profile. "
                 "The preferred profile is P=8, K=4, B=32; if it runs out of GPU memory, both use P=4, K=4, B=16. "
                 "The pilot never supplies learned weights to the experiment. Close other GPU jobs before running it."),
        code('''CONFIG = pilot_runtime(CONFIG, prepared)
print("Selected common profile:", CONFIG.identities_per_batch, "x", CONFIG.images_per_identity,
      "=", CONFIG.batch_size)
display(pd.Series(json.loads((Path(CONFIG.artifacts_root) / "shared" / "runtime_pilot.json").read_text())))'''),
        markdown("## 5. Model training and margin sweep\n\n"
                 "AdamW peak LR 3e-4, five-epoch warmup, cosine decay, maximum 60 epochs and dev-only early stopping. "
                 "Three margins are screened at seed 42; the selected margin is repeated at seeds 123 and 2026. "
                 "Completed runs are reused; interrupted runs resume at the last completed epoch. "
                 "The default artifact root is `artifacts/main-cu130/`; set `MTRLEARN_ARTIFACTS_ROOT` before launching "
                 "the kernel to choose another root. Existing CUDA 12.8 runs remain separate. "
                 "Change the artifact root when changing scientific settings; mismatched resume configurations are rejected. "
                 "The final test is not used during this cell."),
        code('''sweep_val = run_sweep(CONFIG, prepared)
display(sweep_val[["method", "margin", "seed", "stage", "best_epoch", "dev_tar", "dev_auc", "dev_eer"]])
print("Saved sweep:", CONFIG.output_dir / "sweep_val.csv")'''),
        markdown("## 6. Loss graphs and saved artifacts\n\n"
                 "Each run saves `history.csv`, loss/verification graphs, resumable `best.pt` and `last.pt`, "
                 "and a standalone `embedding.pt` with transform metadata and a dev-calibrated threshold. "
                 "The experiment folder saves `sweep_val.csv`, `selection.json` and the margin-comparison graph. "
                 "Smoke-test outputs must never be reported as trained-model performance."),
        code('''display(DisplayImage(filename=str(CONFIG.output_dir / "graphs" / "sweep_val.png")))
for run in sweep_val.itertuples():
    history_path = Path(run.history_path)
    history = pd.read_csv(history_path)
    display(history.tail())
    display(DisplayImage(filename=str(history_path.parent / "graphs" / "loss.png")))
print("Artifact root:", CONFIG.output_dir)
print("Weights:", *sorted(str(path) for path in CONFIG.output_dir.glob("runs/*/weights/embedding.pt")), sep="\\n")'''),
        markdown("## 7. Final test after dev selection\n\n"
                 "Set `RUN_FINAL_TEST=True` only after fixing the selected margin/protocol. "
                 "This evaluates only the selected margin, using each seed's unchanged dev threshold. "
                 "Open-set gallery/probe protocols are shared, with a separate dev-calibrated search threshold. "
                 "The training-class head is never used for unseen-person verification."),
        code('''if RUN_FINAL_TEST:
    test_results = evaluate_selected(CONFIG, prepared, sweep_val)
    display(test_results)
    print("Saved:", CONFIG.output_dir / "test_results.csv")
else:
    print("Final test remains unused. Set RUN_FINAL_TEST=True after completing dev selection.")'''),
        markdown("References: [iResNet source](https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/backbones/iresnet.py), "
                 "[FaceNet](https://arxiv.org/abs/1503.03832), [ArcFace](https://arxiv.org/abs/1801.07698), "
                 "[OpenMMLab angular fallback](https://github.com/open-mmlab/mmpretrain/blob/main/mmpretrain/models/heads/margin_head.py)."),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "mtrlearn", "display_name": "Python (FaceGuard CUDA 13.0)", "language": "python"},
        "language_info": {"name": "python", "version": "3.14"},
    })
    path = ROOT / "notebooks" / "metrics-learn" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = nbformat.read(path, as_version=4)
        if len(previous.cells) == len(notebook.cells):
            for old_cell, new_cell in zip(previous.cells, notebook.cells):
                if old_cell.cell_type == new_cell.cell_type and "id" in old_cell:
                    new_cell.id = old_cell.id
    nbformat.write(notebook, path)
    print(path)


if __name__ == "__main__":
    build("triplet", "triplet_loss.ipynb")
    build("arcface", "arcface.ipynb")
