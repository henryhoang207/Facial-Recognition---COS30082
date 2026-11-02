# Run the recognition notebooks

Use [triplet_loss.ipynb](triplet_loss.ipynb) for B0 and [arcface.ipynb](arcface.ipynb) for B1. Both notebooks follow image processing, model training, loss graphs, and artifact export. Code, comments, and notebook explanations are in English.

The notebook implementation lives in `src/recognition/verification/`. Import configuration, data preparation, models, losses, training, and notebook helpers from `src.recognition.verification`. Batched embedding metrics live in `src.recognition.verification.embedding_evaluation`; `src.recognition.verification.evaluator` provides the separate shared `embed(face)` pair-evaluation API. See the [package map](../../src/recognition/README.md) for the complete module layout.

The A/C classification package uses a ResNet-18 encoder and its own data and configuration. B0/B1 retain iResNet-34, P-by-K sampling, identity-disjoint development data, and the notebook sweep protocol. Both packages now use `src.training.Trainer.fit`, the `TrainingTask` interface, and the same version-1 checkpoint envelope. `MetricLearningTask` owns the B0/B1 backbone, objective, optional ArcFace head, and mining state. `ExperimentConfig` stays separate and adapts shared engine settings with `trainer_config(seed)`. Experiment-specific scheduling, selection, and stopping use the shared `TrainingPolicy` hooks.

From the repository root in PowerShell:

```powershell
./scripts/setup_environment.ps1
. ./.venv/Scripts/Activate.ps1
python -m jupyterlab
```

Select **Python (FaceGuard CUDA 13.0)** as the notebook kernel (kernel ID `mtrlearn`). The shared environment from `main` uses `.venv`, Python 3.14, PyTorch `2.10.0+cu130`, and torchvision `0.25.0+cu130`. The setup script installs `requirements/training-cu130.lock.txt` and registers the kernel against `.venv`. `requirements.txt` contains the shared direct dependencies, including these notebooks' libraries; `requirements-lock.txt` includes the same shared lockfile.

The earlier `scripts/setup_mtrlearn.ps1` command is a compatibility wrapper for the shared setup. An existing Conda `mtrlearn` environment is left unchanged, while the notebook kernel now points to `.venv`. For a fresh optional Conda setup, `environment.yml` creates Python 3.14 and installs the same lockfile; register its interpreter with a different kernel ID if you want both interpreters available. Do not install these dependencies into Conda base.

Run one notebook at a time so the two training jobs do not compete for GPU memory. Each screens three margins at seed 42, then repeats its selected margin at seeds 123 and 2026. Each run starts from a fresh random encoder. The two notebooks reuse the same identity-disjoint manifests and evaluation pairs. Full training can take many hours; the notebook does not automatically estimate scientific results from its smoke run.

The default physical batch size is 32, P=8 and K=4. A disposable GPU pilot tests both objectives and falls back to the common batch-16 profile if necessary. Windows notebooks use zero data workers to avoid notebook spawn issues; importable scripts can use the default two workers. Source images and the original DuckDB catalog are read-only. Manifest preparation only removes duplicate entries from new CSVs.

Outputs are organized as follows:

```text
artifacts/shared/manifests/         shared splits, exact-hash cache, pairs, probe

artifacts/main-cu130-shared-trainer/
  shared/runtime_pilot.json         measured GPU memory profile
  triplet/                         B0 outputs
  arcface/                         B1 outputs
    sweep_val.csv                  dev-only margin and seed results
    selection.json                 selected margin and criterion
    graphs/                        processing preview and margin plot
    runs/margin_0.3_seed_42/
      config.json                  settings, environment, initialization hash
      history.csv                  training and dev metrics by epoch
      result.json                  best epoch, threshold, curriculum status
      graphs/                      loss, ROC, and score-distribution plots
      trainer_runs/<session>/     unique shared-Trainer fit/resume sessions
      weights/best.pt              selected common inference checkpoint
      weights/last.pt              common epoch-boundary resume checkpoint
      weights/embedding.pt         common inference schema, encoder-only task

artifacts/smoke-main-cu130/<timestamp>/
  notebooks/                       executed copies and smoke status
  smoke/
    shared/runtime_pilot.json      measured smoke GPU memory profile
    triplet/                       isolated B0 software-check outputs
    arcface/                       isolated B1 software-check outputs
```

Rerunning a notebook resumes interrupted runs from the last complete epoch and reuses completed matching runs. Each `Trainer.fit` call owns a unique directory under the experiment run's `trainer_runs/`. The orchestrator discovers the latest completed session checkpoint after an interruption and republishes the stable `weights/` files after a successful fit. An interrupted partial epoch is repeated.

All checkpoint files use `format_version=1`. `last.pt` has `checkpoint_kind="training"` and includes task, optimizer, AMP scaler, shared training configuration, experiment metadata, RNG, loader RNG, and policy state. Use it for resume. `best.pt` has `checkpoint_kind="inference"` and contains the selected task state and validation metrics; it is not resumable. `embedding.pt` uses the same inference envelope with only `model.*` encoder keys in `task`, excluding the ArcFace head and training-only task state.

Runs record environment versions, configuration, dataset fingerprints, and a `training_contract` marker. Changes to these contracts reject reuse or resume; choose a fresh `artifacts_root` for a different experiment. Existing private-loop checkpoints, including earlier outputs under `artifacts/main-cu130/`, are not resumed or converted automatically. They remain untouched. The new default root keeps the shared-trainer runs separate. Checkpoints load with `weights_only=True`.

`RUN_FINAL_TEST=False` keeps the held-out test unused during model selection. Enable it only after fixing the protocol and selecting margins on dev. ArcFace's training-class head is never evaluated against unseen dev/test identities and is excluded from `embedding.pt`. The Triplet result records whether the exported checkpoint actually received semi-hard updates, even if a later point in the same run completed the curriculum.

For the app or the shared pair CLI, use the provided inference adapter:

```python
from src.recognition.verification.predictor import MetricEmbedder

embedder = MetricEmbedder("/absolute/path/to/weights/embedding.pt", device="cpu")
vector = embedder.embed(face)  # BGR uint8 shared crop, shape (112, 112, 3).
```

The output is a normalized 256-dimensional NumPy float32 vector. In a shared verification YAML model entry, set `factory: src.recognition.verification.predictor:MetricEmbedder`, `size: 112`, and `crop_scale: 1.0`. The adapter accepts metric `best.pt` and `embedding.pt` inference checkpoints and ignores any loss head in the selected task snapshot. The common checkpoint schema does not make A/C and B0/B1 encoder weights interchangeable.

To verify the entire notebook flow with a tiny real-image subset:

```powershell
.venv/Scripts/python.exe scripts/smoke_recognition_notebooks.py --kernel mtrlearn
```

The smoke run requires a shared Python 3.12-3.14/CUDA 13.0 kernel (the standard Windows setup uses Python 3.14), changes the training budget and curriculum gate, creates a fresh timestamped directory under `artifacts/smoke-main-cu130/`, and leaves the source notebooks output-free. It reuses the verified data manifests under `artifacts/shared/manifests/`; training outputs remain isolated under the timestamp's `smoke/` directory. Its weights and metrics verify software behavior and are not final recognition models or scientific accuracy results. Use `--notebook triplet_loss` or `--notebook arcface` to check one notebook.
