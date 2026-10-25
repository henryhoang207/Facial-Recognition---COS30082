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

1. Branch from `main`: `git checkout -b <your-name>/<what>` (e.g. `hoan/liveness-baseline`).
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

_TODO (Hoan): how to train, how to run inference, APCER / BPCER / ACER._

### Emotion detection

_TODO (Đức): how to train, how to run inference, accuracy and confusion matrix._

### Web app

_TODO (Nam Khánh): how to run the app, registration, attendance log._
