# ASL hand gesture implementation plan

Phase 1 delivers a reproducible recording-level comparison. Phase 2 will integrate the selected dynamic classifier with the static branch and webcam. The [feature specification](spec.md) defines the vocabulary, feature mathematics and acceptance cases.

## 1. Freeze the recording inventory and split

Extend the existing dataset CLI with an opt-in `hand_gestures` source in `configs/datasets.yaml`. Use a root-only source configuration for the AVI directory. The existing `SourceConfig` requires a database. Add the optional field to `PreparationConfig`, register a preparer in `src/datasets/prepare.py` and add the corresponding method to `ManifestValidator`. Preserve preparation with existing YAMLs and the current default selection of datasets. Resolve the ASL root relative to the YAML and select it explicitly with `--only hand_gestures`. [Dataset configuration](../../src/datasets/config.py) · [Preparation CLI](../../src/datasets/prepare.py) · [Validation CLI](../../src/datasets/validate.py)

The scanner in `src/hand-gestures/manifest.py` uses the catalog's deterministic RNG and atomic CSV/JSON writers. It hashes AVI bytes with SHA-256 and validates the stream with OpenCV. The review table accepts `clip_id`, clockwise `orientation`, `start_sec` and `end_sec`, `other_intervals` as a JSON array of time ranges, `signer_id`, `session_id`, `exclude` and `notes`. The supplied table records the observed orientation of SORRY. For recordings without annotated times, the loader estimates static holds and motion windows from the landmark track. The reports count those inferred windows. [Catalog helpers](../../src/datasets/catalog.py) · [Review table](../../configs/hand_gestures/asl_dynamic_review.csv)

**Deliverable.** Write `clips.csv`, `labels.json` and an inventory/split report under `data/manifests/hand_gestures/`. For the inspected snapshot, account for all 300 paths and 297 distinct hashes before quality exclusions. Check the ten S videos, the three duplicate pairs, and five recordings each for YES and NO. Validate that hashes, known groups and all derivative windows belong to a single split. Rerunning with the same inputs, review table and seed must reproduce the same assignments.

## 2. Build one landmark and feature path

The feature lives in `src/hand-gestures/`. The OpenCV/MediaPipe adapter and cache CLI are in `landmarks.py`, the geometry, temporal transforms and flip are in `features.py`, and manifest-backed sample loading is in `data.py`. Extraction and training settings in `configs/hand_gestures/asl_dynamic.yaml` use YAML-relative paths. [Training configuration](../../src/training/config.py)

Smoke-test MediaPipe Tasks and XGBoost with the repo's supported Python environment before pinning compatible packages. The shared requirements currently declare neither. Record the actual Python/package versions and MediaPipe model-asset hashes. The repo's general training rule uses PyTorch. The requested XGBoost classifiers use XGBoost's native trainer, the neural branch uses PyTorch, and MediaPipe supplies the pretrained landmark extractor. [Project training conventions](../../README.md#shared-rules) · [Shared environment](../../requirements.txt)

Decode each canonical AVI in time order, apply its reviewed orientation, and create fresh tracker state per clip. Save a raw `.npz` per clip under `data/hand_gestures/landmarks/`, keyed by clip hash, orientation and extractor version. Store timestamps, raw arrays and validity masks as defined in the spec. Write completed entries atomically. Re-extract stale or incomplete entries. Report hand/anchor coverage, valid interval length and rejected clips by class and split. Use training coverage and visual review to choose the body-anchor schema before the comparison.

**Exit check.** Use a short clip to compare cached and streaming feature values at matching timestamps within float tolerance. On synthetic trajectories, translating a rigid hand must leave local pose unchanged while moving the wrist channel. Two horizontal flips must restore geometry, handedness and anchors. One flip must reverse signed x motion for every timestep. Also test irregular frame intervals, a missing shoulder, degenerate palm scale, a short internal gap and a lost hand.

## 3. Train the shared static model and both dynamic candidates

Use one `src.hand-gestures.train` entry point with the three model selections below. `models.py` defines the frame MLP and GRU. `task.py` uses the shared trainer for optimizer steps, early stopping, checkpoint selection and resume. Both XGBoost models train directly on fixed-width rows. [Task interface](../../src/training/task.py) · [Trainer](../../src/training/trainer.py) · [Training guide](../training-guide.md)

| Selection | First run | Output in the run directory |
| --- | --- | --- |
| `static_xgb` | Sample verified stable frames from each training clip. Fit a 24-class shallow tree ensemble. Select on aggregate validation predictions per clip. | Native `model.json`, ordered static classes and feature columns. |
| `motion_xgb` | Build the ordered trajectory summaries in the spec for positive and `OTHER` windows. Start at tree depth 3 and select boosting rounds on validation. | Native `model.json`, ordered dynamic classes and pooled columns. |
| `mlp_gru` | Use the exact same window IDs, timestamps and flip policy. Fit the frame encoder and GRU with AdamW, clip-balanced sampling and early stopping. | Shared trainer's `best.pt` for prediction and `last.pt` for resume. |

Keep generated run directories under `artifacts/hand_gestures/`. Save the resolved config, seed, manifest digest, window IDs, feature schema/order, normalization parameters, class maps and extractor fingerprint with every run. Store the selected static/dynamic run paths and decision thresholds in a small deployment manifest. The loader checks label order, dimensions and schema versions before serving.

**Exit check.** On a tiny landmark fixture, exercise both input shapes, check finite outputs of width 24 or 8, and compare predictions before and after saving/loading each artifact. Verify every evaluation window's parent against the frozen split. Report the two candidates' validation scores and class support side by side.

## 4. Replay the full decision path and select a deployment in Phase 2

Use the Phase 1 artifact loader in `models.py` and add the timestamp-driven state machine in `decision.py`. Measure successful single commits on full clips in addition to Phase 1 accuracy on isolated intervals. This catches a gate that misses a correctly classifiable J or emits I before it. Include rejected clips and all no-hand intervals in the end-to-end accounting.

On validation, choose the dynamic model by the rule in the spec and tune gate onset/offset, pre-roll, static dwell, release and branch confidence. Compare candidates on the same hardware and input replays. Freeze the complete configuration and run the test report once. Write per-clip predictions, per-class support/recall, confusion matrices, extraction coverage, false commits and p50/p95 timings. Keep counts next to percentages for YES and NO. Distinguish unavailable real left-hand or negative footage from a measured zero error rate.

**Exit check.** Replay the spec's acceptance cases, including one J after an I-shaped onset and two commits for two signs separated by a release. A second session must start idle. Verify artifact/schema mismatch fails explicitly. Publish the chosen run IDs and the measurements behind the choice.

## 5. Connect the webcam feature in Phase 2

- **Backend.** Add an optional gesture model specification to `app/backend/config.py` and `app/config.yaml`. Load configured classifier artifacts through `app/backend/models/registry.py`. For `/ws/stream?mode=hand_gestures`, dispatch to a gesture pipeline in `app/backend/pipeline/hand_gestures.py` before constructing the face pipeline. Share read-only classifiers, give each socket its own MediaPipe tracker, time base, buffer and state machine, and close those resources on disconnect. Keep the original four-byte big-endian frame ID + JPEG format and device-independent elapsed time. No gesture action writes an attendance event. [WebSocket handler](../../app/backend/api/stream.py) · [Face pipeline](../../app/backend/pipeline/pipeline.py)
- **Response.** Retain the `ready`, `result` and `error` envelope and one result per valid frame. Include `frame_id`, `mode`, `frame_size`, `latency_ms`, `faces: []` and a `gesture` payload. Define `gesture.state`, nullable `candidate`, nullable `confidence` and nullable `committed_label`. Set the latter only on the emission frame. Missing assets make gesture mode unavailable with a terminal, actionable error. Existing face-only deployments can start without gesture assets.
- **Frontend.** Add the mode and result type in `app/frontend/src/types.ts`, a Gestures page under `pages/`, and navigation in `App.tsx`. Reuse the existing camera, preview and single-frame-in-flight WebSocket hook. Handle a terminal configuration error separately from a retryable bad JPEG or connection loss. Render the pending sign and last confirmed sign separately and clear stale state after reconnect. [Capture hook](../../app/frontend/src/hooks/useCamera.ts) · [Stream hook](../../app/frontend/src/hooks/useFaceStream.ts) · [Camera view](../../app/frontend/src/components/CameraView.tsx)

**Exit check.** Extend `tests/test_app_backend.py` for a mock gesture stream, two independent connections, absent assets, malformed frames and zero attendance writes. Keep data/feature, model round-trip and decision tests in `tests/test_hand_gestures.py` with small synthetic fixtures. Run the backend regression suite and frontend build, then repeat the webcam signs with the left and right hands in turn. Record hardware, coverage, fps and commit delay in the evaluation report.

## Phase 1 CLI from the repo root

With the shared `.venv` active, install the optional MediaPipe and XGBoost packages. Download the versioned MediaPipe task assets into the paths configured in `asl_dynamic.yaml`. The MediaPipe 0.10.31 Windows wheel has an incorrect internal platform tag reported by `pip check`. Both Task APIs were exercised in the project's Windows Python 3.14 environment.

```powershell
python -m pip install -r requirements/hand-gestures.txt
New-Item -ItemType Directory -Force artifacts/hand_gestures/assets
Invoke-WebRequest 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task' -OutFile artifacts/hand_gestures/assets/hand_landmarker.task
Invoke-WebRequest 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task' -OutFile artifacts/hand_gestures/assets/pose_landmarker_lite.task
python -m src.datasets.prepare --config configs/datasets.yaml --only hand_gestures
python -m src.datasets.validate --config configs/datasets.yaml --only hand_gestures
python -m src.hand-gestures.landmarks --config configs/hand_gestures/asl_dynamic.yaml
python -m src.hand-gestures.train --config configs/hand_gestures/asl_dynamic.yaml --model static_xgb
python -m src.hand-gestures.train --config configs/hand_gestures/asl_dynamic.yaml --model motion_xgb
python -m src.hand-gestures.train --config configs/hand_gestures/asl_dynamic.yaml --model mlp_gru
python -m src.hand-gestures.evaluate --config configs/hand_gestures/asl_dynamic.yaml --split test --run <static-run> --run <motion-run> --run <gru-run> --output artifacts/hand_gestures/comparison_test.json
```

Replace each run placeholder with the path printed by training. Each run contains validation metrics, a model artifact and the matching manifest and feature metadata. Evaluation also writes per-clip predictions to each run directory. Use `--split val` for a combined validation report. Extraction records hand and pose coverage in `data/hand_gestures/landmarks/extraction_report.json`. The default model schema is hand-only. Set `include_body: true` and retrain both dynamic models to compare them with the same optional anchor features.

### Test one classifier with DroidCam

Connect the phone in the DroidCam Windows client and activate **DroidCam Video**. From the repo root, open the preview with the latest Alphabet MLP checkpoint. [DroidCam Windows setup](https://droidcam.app/windows/)

```powershell
.\.venv\Scripts\python.exe -m src.hand-gestures.test-cam --camera 0
```

On Windows, the preview tries Media Foundation first, then DirectShow and the
default backend. If `0` opens the wrong camera, try `--camera 1`, `2`, and so
on until the correct DroidCam image appears.

Use `--model xgb` to compare the Alphabet XGBoost classifier. The Alphabet classifiers use CUDA from the default YAML, with native Windows MediaPipe hand tracking on the CPU. The preview shows the top label, confidence, hand detection time and classifier time for the latest prediction. Use `--run` with a training-run directory to load a specific checkpoint. The older `mlp_gru`, `motion_xgb` and `static_xgb` runs remain selectable. Dynamic models classify a rolling window of up to two seconds by default. Adjust its duration with `--window`. Press **Q** or **Esc** to quit.

## Phase 1 results on the local ASLDynamic copy

The hand-only runs used seed 42, the same frozen recording split, and 20 fps landmark extraction. Of the 297 unique recordings, three E training clips yielded no hand landmarks. Both dynamic classifiers scored the same 60 clips per holdout and every holdout clip produced a prediction. Each dynamic holdout contains 48 static clips as `OTHER` and 12 target clips.

| Classifier | Validation accuracy | Validation macro-F1 | Test accuracy | Test macro-F1 | Test p95 CPU model latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Static XGBoost | 85.42% | 0.846 | 77.08% | 0.754 | 0.125 ms |
| Motion XGBoost | 90.00% | 0.578 | 88.33% | 0.450 | 0.148 ms |
| MLP + GRU | 93.33% | 0.802 | 90.00% | 0.593 | 0.541 ms |

MLP + GRU ranked higher on validation macro-F1. On test it recognized 7 of the 12 target clips, compared with 5 of 12 for Motion XGBoost. Both missed the two HELLO clips and the single YES clip. Dynamic windows were inferred from detections. This recording holdout does not establish performance on unseen signers. The timings cover model calls on one CPU thread. Combined metrics are saved locally as `artifacts/hand_gestures/comparison_val.json` and `comparison_test.json`.

## ASL Alphabet follow-up

The follow-up compares static landmark MLP and XGBoost classifiers on A–Z images under `data/ASL`. The J and Z labels describe the poses in the published stills. Testing their trajectories requires labelled video. The three extra classes `del`, `nothing` and `space` are excluded. [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

The manifest groups images by numbered blocks shared across classes and excludes boundary neighbors between different splits. It also removes exact SHA-256 duplicates. All 26 publisher test images duplicate training-source files. Their exact matches were removed from the blocked splits and the publisher images are kept under the `external` split name as a reference. The main comparison uses the blocked test split. Signer identities are unavailable, so these splits can establish file-level separation.

MediaPipe detects 21 landmarks per image. The cache records the usable landmarks and the report includes extraction failures. Training converts wrist-relative points, apertures and joint angles to 86 features. The compact MLP uses layers of size 128 and 64 with dropout, following the three-layer landmark classifier pattern used by Borda and Holm. It samples horizontal flips, rotations within 8 degrees and small palm-scale jitter on GPU for training batches. XGBoost fits one fixed GPU-augmented copy alongside the original training features. Both models use capped class weights and select on validation macro-F1. The MLP logs clean train and validation metrics under the same inference conditions. The YAML sets batch size 512, CUDA BF16 AMP for MLP, FP32 landmark geometry, 100 epochs, patience 80, AdamW and cosine decay. [Landmark MLP reference](https://cs231n.stanford.edu/papers/text_file_840589357-CS_231N_Final_Report%20%282%29.pdf)

The configured extractor uses the MediaPipe GPU delegate in Ubuntu WSL. On the Windows host, use the prepared WSL environment under `artifacts/hand_gestures/wsl-gpu-venv`. It uses MediaPipe 0.10.35 and the WSLg OpenGL renderer. The runtime needs EGL and GLES, with the local GLES libraries under `artifacts/hand_gestures/wsl-gpu-libs` on this machine. Before running on a new host, check that MediaPipe logs a hardware renderer. The extractor keeps resumable chunks keyed to its delegate and fingerprint. [MediaPipe delegate documentation](https://developers.google.com/edge/api/mediapipe/python/mp/tasks/BaseOptions)

```bash
# Run in Ubuntu WSL from the same repository root.
export GALLIUM_DRIVER=d3d12
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
export EGL_PLATFORM=x11
export LD_LIBRARY_PATH="$PWD/artifacts/hand_gestures/wsl-gpu-libs/usr/lib/x86_64-linux-gnu"
artifacts/hand_gestures/wsl-gpu-venv/bin/python -m src.hand-gestures.alphabet_data --config configs/hand_gestures/asl_alphabet.yaml
```

```powershell
# Run training and evaluation in the activated Windows .venv.
$config = 'configs/hand_gestures/asl_alphabet.yaml'
python -m src.hand-gestures.alphabet train --config $config --model mlp
python -m src.hand-gestures.alphabet train --config $config --model xgb
python -m src.hand-gestures.alphabet evaluate --config $config --split test --run 'MLP_RUN_PATH' --run 'XGB_RUN_PATH' --output artifacts/hand_gestures/alphabet/comparison_test.json
```

Use the active shared `.venv` with the optional gesture dependencies and versioned hand model described above. Replace the two run paths with the paths printed during training. Results include per-class support, confusion matrices, full probability vectors and test accuracy with detection failures included in the denominator.

### GPU run results

MediaPipe processed 72,280 eligible images on the RTX 3070 Ti and returned landmarks for 60,708. In the blocked test, it detected a hand in 4,384 of 6,240 eligible images (70.26%). The classifier-only columns score those detected images. The last column counts extraction failures as missed signs.

| Classifier | Validation macro-F1 | Test macro-F1 on detected images | Test accuracy on detected images | Test accuracy on all eligible images |
| --- | ---: | ---: | ---: | ---: |
| Landmark MLP | 0.9582 | 0.9632 | 96.46% | 67.77% |
| Landmark XGBoost | 0.9582 | 0.9490 | 95.10% | 66.81% |

The MLP completed 100 CUDA BF16 epochs and selected epoch 70. Its clean train and validation accuracies at that checkpoint were 98.35% and 95.97%. XGBoost completed 100 CUDA boosting rounds and selected round 91. For the MLP, missing detections accounted for 1,856 of 2,011 image-level test misses. Results are under `artifacts/hand_gestures/alphabet/` alongside `mlp_learning_curves.png`. The test measures still-image performance within this dataset.
