# FaceGuard AI

Face recognition attendance system with liveness (anti-spoofing) and emotion detection.
COS30082 Applied Machine Learning group project, Swinburne.

A webcam frame goes through four models that we train ourselves:

```
frame -> detector -> face crop -> embedder  -> who is it?   (compare with registered embeddings)
                               -> liveness  -> real or spoof?
                               -> emotion   -> which emotion?
```

Attendance is only logged when the face is recognised **and** live.

## Team

| Member | Owns | Code |
| --- | --- | --- |
| Hoàng Hữu Hoan (lead) | Anti-spoofing; repo, notebook template, checkpoints, submission | `src/liveness/` |
| Giáp Hoàng Gia Khánh | Face recognition Model A (classification), optional ArcFace; shared ROC/AUC evaluation and threshold | `src/recognition/` |
| Lê Đức Anh | Face recognition Model B (triplet loss) | `src/recognition/` |
| Trần Mạnh Đức | Emotion detection | `src/emotion/` |
| Lê Đức Thành | Datasets, shared face crop + loaders, spoof recordings, group report | `src/common/` |
| Nguyễn Nam Khánh | Web app and integration | `app/` |

## Repository layout

```
app/                 web app (webcam, registration, attendance log)
notebooks/           training notebooks, one per model; start from 00_template.ipynb
src/common/          face crop, data loaders, shared model interfaces
src/recognition/     classification/ (A/C), identification/ (B0/B1), verification/, preprocessing/
src/liveness/        anti-spoofing model
src/emotion/         emotion model
reports/figures/     result tables and plots used in the report
data/                local datasets (not committed, see data/README.md)
checkpoints/         local model weights (not committed, see below)
```

## Shared rules

- **PyTorch only.** Train on Colab or Kaggle GPUs, each owner on their own account.
- **One face crop.** Use `src.common.face_crop.crop_face` for training *and* in the app, so models see the same input.
- **One interface per model**, defined in `src/common/interfaces.py`, so the app can plug any model in:

  ```python
  detector.detect(frame)  -> list[box]
  embedder.embed(face)    -> np.ndarray   # L2-normalised
  liveness.predict(face)  -> (label, prob)
  emotion.predict(face)   -> (label, probs)
  ```

- **Train your own models.** Code from papers or GitHub is fine as a reference (cite it in the notebook and report).
  Downloaded trained weights are not a final model. ImageNet weights as a starting point are expected to be fine (pending tutor confirmation).
- **Checkpoints** go in the shared Drive folder `FaceGuard/checkpoints/`, named `<task>_<model>_<date>.pt`,
  e.g. `liveness_mobilenetv2_2026-11-01.pt`. Never commit weights to git.
- **Results** (tables and figures) go in `reports/figures/`, named `<task>_<model>_<what>.png`.

## Setup

### Colab / Kaggle (training)

Open `notebooks/00_template.ipynb`, save a copy as `notebooks/<task>_<model>.ipynb`, and run the Setup cell.
Review the setup cell before running it on a hosted service: `requirements.txt` now describes the shared CUDA 13.0 environment, including pinned PyTorch builds. The local lockfile is the reference for reproducible training. Hosted services supply their own Python, GPU driver, and PyTorch versions, so their compatibility must be checked separately.

The shared Drive folder appears under *Shared with me*. Right-click it → *Organise → Add shortcut* → *My Drive*
so every notebook can find it at `/content/drive/MyDrive/FaceGuard`.

### Local (app, notebooks, and training)

From the repository root in PowerShell, use the shared environment from `main`:

```powershell
./scripts/setup_environment.ps1
. ./.venv/Scripts/Activate.ps1
python -m jupyterlab
```

The setup script creates `.venv` with Python 3.14, installs the shared CUDA 13.0 lockfile, checks CUDA, and registers **Python (FaceGuard CUDA 13.0)** under kernel ID `mtrlearn`. PyTorch is `2.10.0+cu130` and torchvision is `0.25.0+cu130`. The metric-learning notebooks and the existing application use the same dependencies. See the [training guide](docs/training-guide.md) for manual setup, supported Python versions, and lockfile regeneration. An existing Conda `mtrlearn` environment is left in place; the registered kernel points to `.venv` after setup.

## Workflow

1. Branch from `main`: `git checkout -b <your-name>/<what>` (e.g. `hoan/liveness-baseline`).
2. Commit small, push your branch, open a pull request.
3. One other member reviews it before it is merged to `main`.
4. Clear notebook outputs before committing if they are large (keep the final results cell).

## Training and inference

Start with the [shared training guide](docs/training-guide.md) for environment setup, YAML configuration, CLI commands, warmup, accumulation, optimizer groups, and resume.
The [documentation index](docs/README.md) links to dataset, recognition, and design details.
Task-specific commands are listed below.

### Face recognition: Model A (classification)

Gia Khánh: Model A (ResNet-18 + cross-entropy), Model C (ArcFace), and shared verification evaluation.
See [training, YAML configuration, inference and ROC/AUC](docs/recognition-training.md).
The reusable loop is in `src/training/`; A/C are isolated in `src/recognition/classification/`. See the [recognition package map](src/recognition/README.md) for module responsibilities.
Local dataset manifests are connected; real-data accuracy, AUC and CPU latency are pending full training and evaluation.

### Face recognition: Model B (triplet loss)

Run [triplet_loss.ipynb](notebooks/metrics-learn/triplet_loss.ipynb) for B0 or [arcface.ipynb](notebooks/metrics-learn/arcface.ipynb) for B1 using the shared CUDA kernel. Both train an iResNet-34 encoder from random initialization and export normalized 128-dimensional embeddings. Their implementation is in `src/recognition/identification/`, with `MetricLearningTask` using the same `Trainer.fit` and checkpoint schema as A/C. The notebooks keep their own `ExperimentConfig`, P-by-K sampling, mining, and identity-disjoint evaluation protocol. Each trains one seed-42 run with four CPU loading workers, CUDA augmentation and training, patience 10, and fixed loss margin 0.3. Verification and open-set search thresholds are calibrated separately on dev and frozen for test. B1 uses batch 128, BF16 AMP, channels-last, and fused AdamW. The default notebook output root is `artifacts/iresnet34-images-128d/`. See the [metric-learning notebook guide](notebooks/metrics-learn/RECOGNITION.md) for artifacts, resume rules, inference, and bounded GPU smoke tests.

### Anti-spoofing

Owner: Hoan. Code in `src/liveness/`, run end to end with `notebooks/liveness_training.ipynb`.

Two candidates, trained on the same data, crop and schedule:

| Model | Idea | Loss | Trainable part |
| --- | --- | --- | --- |
| `mobilenetv3` | ImageNet MobileNetV3-Large, one live/spoof output | BCE | all but the first 4 blocks |
| `deeppixbis` | DenseNet-161 up to block 2, a 14×14 live/spoof map plus one output ([George & Marcel, 2019](https://arxiv.org/abs/1907.04047)) | ½ BCE on the map + ½ BCE on the output | all |

Input: a square crop of 2.0× the face box resized to 224 px, so phone edges and paper borders stay in view.
Training draws live and spoof images equally often. Data: CelebA-Spoof (print and replay only) plus our recordings, split by person.

**Train and evaluate** (from the repo root):

```bash
python -m src.liveness.manifest celeba --data-root DATA --root DATA/spoof_public/CelebA_Spoof --attacks print,replay --max-per-class 20 --out DATA/liveness/celeba.csv
python -m src.liveness.manifest folder --data-root DATA --root DATA/spoof_recorded --source recorded --test-subjects anh,duc --out DATA/liveness/recorded.csv
python -m src.liveness.train --model mobilenetv3 --manifest DATA/liveness/celeba.csv DATA/liveness/recorded.csv --data-root DATA --ckpt-dir CKPT
python -m src.liveness.evaluate --checkpoint CKPT/liveness_mobilenetv3_<date>.pt --manifest DATA/liveness/celeba.csv DATA/liveness/recorded.csv --data-root DATA
```

Repeat the last two with `--model deeppixbis`. `evaluate` adds both models to `reports/figures/liveness_results.csv` / `.md`
(one row per model × test source) and saves ROC and score-distribution plots.

**Use in the app:**

```python
from src.liveness.predictor import LivenessPredictor
liveness = LivenessPredictor("checkpoints/liveness_mobilenetv3_<date>.pt")
label, prob_live = liveness.predict_frame(frame, box)   # crops with its own size and scale
```

**Metrics** (ISO/IEC 30107-3). The threshold is picked on val at the equal-error point, saved in the checkpoint, and never tuned on test.

- **APCER**: attacks accepted as live, per attack type; we report the worst type.
- **BPCER**: live faces rejected as spoof.
- **ACER** = (APCER + BPCER) / 2: the number we compare models on.
- Also reported: AUC, EER, accuracy, BPCER at APCER = 5%.

**Targets.** Test people are never in training. "Recorded" is our own faces, prints and phones, the closest to the demo.

| Check | Minimum | Good |
| --- | --- | --- |
| ACER, CelebA-Spoof test | ≤ 10% | ≤ 5% |
| ACER, recorded test people | ≤ 15% | ≤ 10% |
| Live demo: print and phone attacks rejected (20 tries each) | 18 / 20 | 20 / 20 |
| Live demo: real faces accepted (20 tries) | 18 / 20 | 19 / 20 |
| Liveness time per face, demo laptop CPU | ≤ 300 ms (check every 3rd frame) | ≤ 100 ms (every frame) |

Measured on a 10-thread laptop CPU at 224 px: MobileNetV3 about 90 ms, DeepPixBiS about 270 ms per face.
The model with the lower ACER on recorded test people goes in the app; if they are within 2 points, take the faster one.

**Tests:** `python -m pytest tests/test_liveness.py` (CPU, about a minute) trains both models for one epoch on random images.

### Emotion detection

_TODO (Đức): how to train, how to run inference, accuracy and confusion matrix._

### Web app

_TODO (Nam Khánh): how to run the app, registration, attendance log._
