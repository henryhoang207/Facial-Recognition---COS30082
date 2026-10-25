# FaceGuard AI

Face recognition attendance system with anti-spoofing and emotion detection.
COS30082 Applied Machine Learning group project, Swinburne.

A webcam frame goes through four models that we train ourselves:

```
frame -> detector -> face crop -> embedder  -> who is it?   (compare with registered embeddings)
                               -> anti-spoofing -> real or spoof?
                               -> emotion   -> which emotion?
```

Attendance is only logged when the face is recognised **and** live.

## Team

| Member | Owns | Code |
| --- | --- | --- |
| Hoàng Hữu Hoan (lead) | Anti-spoofing; repo, notebook template, checkpoints, submission | `src/antispoofing/` |
| Giáp Hoàng Gia Khánh | Face recognition Model A (classification), optional ArcFace | `src/recognition/` |
| Lê Đức Anh | Face recognition Model B (triplet loss), ROC/AUC verification script | `src/recognition/` |
| Trần Mạnh Đức | Emotion detection | `src/emotion/` |
| Lê Đức Thành | Datasets, shared face crop + loaders, spoof recordings, group report | `src/common/` |
| Nguyễn Nam Khánh | Web app and integration | `app/` |

## Repository layout

```
app/                 web app (webcam, registration, attendance log)
notebooks/           training notebooks, one per model; start from 00_template.ipynb
src/common/          face crop, data loaders, shared model interfaces
src/recognition/     face embedding models (Model A, Model B) + verification eval
src/antispoofing/        anti-spoofing model
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
  antispoofing.predict(face)  -> (label, prob)
  emotion.predict(face)   -> (label, probs)
  ```

- **Train your own models.** Code from papers or GitHub is fine as a reference (cite it in the notebook and report).
  Downloaded trained weights are not a final model. ImageNet weights as a starting point are expected to be fine (pending tutor confirmation).
- **Checkpoints** go in the shared Drive folder `FaceGuard/checkpoints/`, named `<task>_<model>_<date>.pt`,
  e.g. `antispoofing_mobilenetv2_2026-11-01.pt`. Never commit weights to git.
- **Results** (tables and figures) go in `reports/figures/`, named `<task>_<model>_<what>.png`.

## Setup

### Colab / Kaggle (training)

Open `notebooks/00_template.ipynb`, save a copy as `notebooks/<task>_<model>.ipynb`, and run the Setup cell.
It clones this repo, installs `requirements.txt` and mounts Drive.
Colab and Kaggle already ship with PyTorch, so the setup cell does not reinstall it.

The shared Drive folder appears under *Shared with me*. Right-click it → *Organise → Add shortcut* → *My Drive*
so every notebook can find it at `/content/drive/MyDrive/FaceGuard`.

### Local (app and quick tests)

```bash
python -m venv .venv
source .venv/Scripts/activate     # Windows Git Bash; use .venv\Scripts\activate in cmd / PowerShell
pip install -r requirements.txt
```

## Workflow

1. Branch from `main`: `git checkout -b <your-name>/<what>` (e.g. `hoan/antispoofing-baseline`).
2. Commit small, push your branch, open a pull request.
3. One other member reviews it before it is merged to `main`.
4. Clear notebook outputs before committing if they are large (keep the final results cell).

## Training and inference

Each owner fills in their section when their model is ready.

### Face recognition: Model A (classification)

_TODO (Gia Khánh): how to train, how to run inference, final AUC._

### Face recognition: Model B (triplet loss)

_TODO (Đức Anh): how to train, how to run inference, final AUC, chosen threshold._

### Anti-spoofing

Owner: Hoan. Code in `src/antispoofing/`, run end to end with `notebooks/antispoofing_training.ipynb`.

Two candidates, trained on the same data, crop and schedule:

| Model | Idea | Loss | Trainable part |
| --- | --- | --- | --- |
| `mobilenetv3` | ImageNet MobileNetV3-Large, one live/spoof output | BCE | all but the first 4 blocks |
| `deeppixbis` | DenseNet-161 up to block 2, a 14×14 live/spoof map plus one output ([George & Marcel, 2019](https://arxiv.org/abs/1907.04047)) | ½ BCE on the map + ½ BCE on the output | all |

Input: a square crop of 2.0× the face box resized to 224 px, so phone edges and paper borders stay in view.
Training draws live and spoof images equally often. Data: CelebA-Spoof (print and replay only) plus our recordings, split by person.

**Train and evaluate** (from the repo root):

```bash
python -m src.antispoofing.manifest celeba --data-root DATA --root DATA/spoof_public/CelebA_Spoof --attacks print,replay --max-per-class 20 --out DATA/antispoofing/celeba.csv
python -m src.antispoofing.manifest folder --data-root DATA --root DATA/spoof_recorded --source recorded --test-subjects anh,duc --out DATA/antispoofing/recorded.csv
python -m src.antispoofing.train --model mobilenetv3 --manifest DATA/antispoofing/celeba.csv DATA/antispoofing/recorded.csv --data-root DATA --ckpt-dir CKPT
python -m src.antispoofing.evaluate --checkpoint CKPT/antispoofing_mobilenetv3_<date>.pt --manifest DATA/antispoofing/celeba.csv DATA/antispoofing/recorded.csv --data-root DATA
```

Repeat the last two with `--model deeppixbis`. `evaluate` adds both models to `reports/figures/antispoofing_results.csv` / `.md`
(one row per model × test source) and saves ROC and score-distribution plots.

**Use in the app:**

```python
from src.antispoofing.predictor import AntiSpoofingPredictor
antispoofing = AntiSpoofingPredictor("checkpoints/antispoofing_mobilenetv3_<date>.pt")
label, prob_live = antispoofing.predict_frame(frame, box)   # crops with its own size and scale
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
| Anti-spoofing time per face, demo laptop CPU | ≤ 300 ms (check every 3rd frame) | ≤ 100 ms (every frame) |

Measured on a 10-thread laptop CPU at 224 px: MobileNetV3 about 90 ms, DeepPixBiS about 270 ms per face.
The model with the lower ACER on recorded test people goes in the app; if they are within 2 points, take the faster one.

**Tests:** `python -m pytest tests/test_antispoofing.py` (CPU, about a minute) trains both models for one epoch on random images.

### Emotion detection

_TODO (Đức): how to train, how to run inference, accuracy and confusion matrix._

### Web app

_TODO (Nam Khánh): how to run the app, registration, attendance log._
