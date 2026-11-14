# Run the recognition notebooks

Use [triplet_loss.ipynb](triplet_loss.ipynb) for B0 and [arcface.ipynb](arcface.ipynb) for B1. Both notebooks follow image processing, model training, loss graphs, and artifact export. Code, comments, and notebook explanations are in English.

The notebook implementation lives in `src/recognition/identification/`. Import configuration, data preparation, models, losses, training, and notebook helpers from `src.recognition.identification`. Batched embedding metrics live in `src.recognition.identification.embedding_evaluation`; `src.recognition.verification.evaluator` provides the separate shared `embed(face)` pair-evaluation API. See the [package map](../../src/recognition/README.md) for the complete module layout.

The A/C classification package uses a ResNet-18 encoder and its own data and configuration. B0/B1 retain iResNet-34, P-by-K sampling, identity-disjoint development data, and the fixed-setting notebook protocol. Both packages now use `src.training.Trainer.fit`, the `TrainingTask` interface, and the same version-1 checkpoint envelope. `MetricLearningTask` owns the B0/B1 backbone, objective, optional ArcFace head, and mining state. `ExperimentConfig` stays separate and adapts shared engine settings with `trainer_config(seed)`. Experiment-specific scheduling, selection, and stopping use the shared `TrainingPolicy` hooks.

From the repository root in PowerShell:

```powershell
./scripts/setup_environment.ps1
. ./.venv/Scripts/Activate.ps1
python -m jupyterlab
```

Select **Python (FaceGuard CUDA 13.0)** as the notebook kernel (kernel ID `mtrlearn`). The shared environment from `main` uses `.venv`, Python 3.14, PyTorch `2.10.0+cu130`, and torchvision `0.25.0+cu130`. The setup script installs `requirements/training-cu130.lock.txt` and registers the kernel against `.venv`. `requirements.txt` contains the shared direct dependencies, including these notebooks' libraries; `requirements-lock.txt` includes the same shared lockfile.

The earlier `scripts/setup_mtrlearn.ps1` command is a compatibility wrapper for the shared setup. An existing Conda `mtrlearn` environment is left unchanged, while the notebook kernel now points to `.venv`. For a fresh optional Conda setup, `environment.yml` creates Python 3.14 and installs the same lockfile; register its interpreter with a different kernel ID if you want both interpreters available. Do not install these dependencies into Conda base.

Run one notebook at a time so the two training jobs do not compete for GPU memory. Each trains one run at seed 42, using fixed loss margin `0.3`. The notebooks do not sweep margins and do not launch confirmation-seed runs. Verification and open-set search thresholds are calibrated separately on dev and frozen for test. Each run starts from a fresh random encoder. The two notebooks reuse the same identity-disjoint manifests and evaluation pairs. Full training can take many hours; the notebook does not automatically estimate scientific results from its smoke run.

The ArcFace notebook and background runner share `ExperimentConfig.arcface()`. The standard B1 profile uses batch 128 (P=32, K=4), CUDA BF16 AMP, channels-last layout, and fused AdamW. Training uses FP32 master weights and AdamW moments. Dev/test scoring and inference exports use FP32. The Triplet notebook uses its existing FP16 AMP and disposable GPU memory pilot to select its batch size.

Four CPU data-loader workers decode the existing complete images as RGB `uint8`. Pinned batches move to CUDA for augmentation, normalization, and training. Evaluation uses batch 128, two workers, and deterministic preprocessing. Each worker prefetches two batches. Runs are capped at 100 epochs, with patience 10 and a 15-epoch minimum. The Triplet curriculum gate still applies to B0. The smoke check uses smaller batches, a bounded epoch count, and zero workers.

Prepare the data before opening the training cells:

```powershell
python -m src.datasets.prepare --config configs/datasets.yaml --only recognition
```

For an unattended ArcFace run, use the background runner with a fresh job directory and a completed base-manifest directory:

```powershell
python scripts/train_arcface_background.py --job-dir artifacts/arcface-images-128d-run --batch-size 128 --base-manifests data/manifests
```

The runner writes its status, logs, checkpoints, and final report below the selected job directory. On reuse, it loads the saved `experiment_config.json` for that job, including the original manifest path and precision settings. Launch it through the operating system's background-process facility when the terminal must remain free.

`CONFIG.face_protocol` retains its earlier name for compatibility and selects `data/manifests/` by default. `load_image_protocol` reads the base manifests; no detector model, bbox columns, or completed bbox scan is needed. It uses classification **train** rows only, removes repeated image hashes within each training identity, and excludes identities with fewer than K distinct images. Published verification val/test become identity-disjoint dev/test with the existing pair lists. Classification val/test rows are excluded. The original physical `train/id_*/` layout works without a physical `val/` directory. Images outside the source DuckDB catalog, including the separate `test/` directory, are not implicitly added to this protocol.

The current VGGFace2 source already contains tightly cropped 112x112 images. CPU workers decode the complete RGB frame and pad or resize it when needed. Pinned uint8 batches move to CUDA for augmentation and normalization before iResNet-34 produces a 128-D unit embedding. The four-worker prefetch queue holds up to eight batches, roughly 37 MiB of uint8 image pixels at batch 128 before worker metadata, active tensors, and model memory.

Switching from the bbox protocol to complete existing images requires a fresh training run; the default artifact root is separate from earlier bbox and 256-D runs. Inference still reads existing compatible 128-D or 256-D checkpoint dimensions from metadata. The backbone stages [3,4,6,3] and loss implementations remain the same. The optional `load_face_protocol` adapter remains available for previous completed bbox generations; A/C can continue using the standalone YuNet preparation workflow.

Outputs are organized as follows:

```text
data/manifests/                   base recognition manifests and pairs

artifacts/iresnet34-images-128d/
  shared/runtime_pilot.json         measured GPU memory profile
  triplet/                         B0 outputs
  arcface/                         B1 outputs
    runs.csv                       fixed-setting run and dev results
    selection.json                 fixed margin and evaluation protocol
    graphs/                        processing preview
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

`RUN_FINAL_TEST=False` keeps the held-out test unused during model selection. Enable it only after fixing the evaluation protocol. Held-out verification reuses the selected checkpoint's dev-calibrated threshold. Open-set search calibrates its own threshold on dev and freezes it for test. ArcFace's training-class head is never evaluated against unseen dev/test identities and is excluded from `embedding.pt`. The Triplet result records whether the exported checkpoint actually received semi-hard updates, even if a later point in the same run completed the curriculum.

For the app or the shared pair CLI, use the provided inference adapter:

```python
from src.recognition.identification.predictor import MetricEmbedder

embedder = MetricEmbedder("/absolute/path/to/weights/embedding.pt", device="cpu")
vector = embedder.embed(face)  # BGR uint8 shared crop, shape (112, 112, 3).
```

The output is a normalized 128-dimensional NumPy float32 vector. In a shared verification YAML model entry, set `factory: src.recognition.identification.predictor:MetricEmbedder`, `size: 112`, and `crop_scale: 1.0`. The adapter accepts metric `best.pt` and `embedding.pt` inference checkpoints and ignores any loss head in the selected task snapshot. The common checkpoint schema does not make A/C and B0/B1 encoder weights interchangeable.

To verify the entire notebook flow with a tiny real-image subset:

```powershell
.venv/Scripts/python.exe scripts/smoke_recognition_notebooks.py --kernel mtrlearn
```

The smoke run requires a shared Python 3.12-3.14/CUDA 13.0 kernel (the standard Windows setup uses Python 3.14), changes the training budget and curriculum gate, creates a fresh timestamped directory under `artifacts/smoke-main-cu130/`, and leaves the source notebooks output-free. It reuses the verified base manifests under `data/manifests/`; training outputs remain isolated under the timestamp's `smoke/` directory. Its weights and metrics verify software behavior and are not final recognition models or scientific accuracy results. Use `--notebook triplet_loss` or `--notebook arcface` to check one notebook.
