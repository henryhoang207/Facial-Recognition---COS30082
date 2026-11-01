# Train and evaluate the anti spoofing models on Colab

Open `notebooks/liveness_training.ipynb` in Google Colab. This is the main entry point
for Hoan's public-data phase. It trains MobileNetV3-Large and DeepPixBiS in PyTorch,
exports checkpoints and produces comparison tables, plots and report sections.
Face collection, recorded-data adaptation, webcam trials and attendance integration
are deferred. This repository contains no fabricated training metrics or final weights.

## Prepare Drive and run the notebook

1. The notebook uses the group's [FaceGuard folder](https://drive.google.com/drive/folders/1vOxqufQJaj1zPhzzUZMdHj6VF3vLCnGX)
   at `My Drive/COS30082 - Project/AI/FaceGuard`. Other members can add a shortcut
   to that folder named `FaceGuard` directly under their own My Drive. The
   notebook checks the existing group location first and never creates a separate
   personal FaceGuard folder when the group folder cannot be found.
2. Open [CelebA_Spoof_processed.zip](https://drive.google.com/file/d/1jJ1gbDXIX-TMj-9dUp006aRiMxqMVrpB/view)
   and use **Organize > Add shortcut > My Drive**, retaining its filename. This
   makes the existing ZIP accessible to Colab without copying it. The notebook
   also accepts an existing shortcut under the group's `data/` folder or the
   FaceGuard folder itself. The published branch supplies the code; if using
   `faceguard_colab_source.zip` instead, place it in the group's FaceGuard folder.
3. Run setup, archive staging and manifest preparation. These stages may run on
   a CPU runtime to conserve free GPU quota. They require substantial disk and I/O.
4. Inspect the crop grid and exclusion log. Set `CROPS_REVIEWED=True` only after
   confirming the boxes. Explain any differences from the reference split counts.
5. Select a GPU runtime, rerun setup after reconnecting, and run both pilots.
   Inspect loss, memory and timings before running full training.
6. Run full training, optional bounded validation tuning, freeze candidates, then
   evaluate on test and export the report. Leave the frozen set unchanged after testing.

The source ZIP includes the code and configuration needed by the notebook. It is
independent of the older local checkout and does not include datasets, weights,
credentials or the local test environment. The notebook's source revision file
records its exact commit and content hashes.

## Data protocol

The archive is 42.7 GB. Staging checks its central directory and available disk,
extracts the DuckDB catalog, and selects live, print and replay images plus their
bounding boxes. It copies the ZIP locally only if the compressed and selected
uncompressed files fit together with a reserve. Otherwise it extracts selected
files directly from the mounted archive. Insufficient space fails with a clear
message; evaluation data is never silently subsampled.

The processed four-entry annotation vectors are not the original 44-entry vectors.
The exporter reads named DuckDB columns. `is_spoof=0` becomes training label 1
(live), and `is_spoof=1` becomes label 0 (spoof). Source spoof types 1/2/3 map to
print and 7/8/9 to replay. Mask images are excluded.

Official-test subjects remain in test. Their source-training rows are excluded.
Validation holds out 15% of remaining training subjects with seed 42. Images are
fully decoded and hashed; invalid images/boxes and conflicting labels are rejected.
For identical bytes across splits, test takes precedence over validation, which
takes precedence over training. All rejected rows and reasons are saved. This
detects exact duplicates, not every possible near-duplicate or recording session.

Historical reference counts are 75,780 train, 13,367 validation and 49,646 test.
They are not assumed to describe the supplied ZIP until recomputed. New integrity
checks can change these counts; explain changes in `data_review.json`.

## Training configuration and recovery

The two YAML files in `configs/liveness/` define the complete starting settings.
Example from the repository root:

```text
python -m src.liveness.train --config configs/liveness/mobilenetv3.yaml --manifest DATA/manifests/celeba_liveness.csv --data-root DATA/CelebA_Spoof_processed --ckpt-dir DRIVE/runs/mobilenetv3
python -m src.liveness.train --config configs/liveness/deeppixbis.yaml --manifest DATA/manifests/celeba_liveness.csv --data-root DATA/CelebA_Spoof_processed --ckpt-dir DRIVE/runs/deeppixbis
```

Both use crop scale 2.0 and input size 224, mild training augmentation, balanced
sampling, AdamW, weight decay 0.0001, cosine decay, maximum 15 epochs, patience 4,
mixed precision on CUDA and seed 42. MobileNet LR is 0.0003 with microbatch 64.
DeepPixBiS LR is 0.0001 with microbatch 32 and two accumulation steps. Initial
ImageNet weights are permitted; downloaded trained anti-spoofing weights are not
used. Frozen MobileNet blocks retain fixed batch-normalization statistics.

Pilots use at most 2,000 stratified training and 600 validation images for one
epoch. Final runs use the complete eligible splits. A high-validation-error run
can receive one additional run at one-third its original LR after an audit.
No tuning uses test scores. One seed per model is a practical minimum and must
be acknowledged in the report.

Unique run directories contain atomic `best.pt` and `last.pt`, history, resolved
configuration, environment, dataset provenance and an exported
`liveness_<model>_<date>.pt`. The exported file is for inference; `last.pt` is
for recovery. Resume to a new directory with the same scheduled epoch count:

```text
python -m src.liveness.train --config CONFIG.yaml --resume OLD_RUN/last.pt
```

The notebook detects completed jobs and interrupted runs automatically. Changing
the experiment ID creates a separate experiment. Keep the same data and settings
when recovering. Exact recovery is tested on CPU; CUDA reproducibility also
depends on deterministic kernels and an unchanged runtime/hardware environment.

## Evaluation and deployment selection

Each epoch calibrates the threshold on validation at the empirical equal-error
operating point. The best epoch minimizes validation ACER. Final exported weights
carry that epoch's threshold and portable manifest/subject/hash provenance.
Evaluation rechecks actual image content; it fails when provenance is absent or
the dataset differs, even if the original training paths no longer exist.

```text
python -m src.liveness.select --checkpoint MODEL1.pt MODEL2.pt --out frozen_candidates.json
python -m src.liveness.evaluate --checkpoint MODEL1.pt --manifest MANIFEST.csv --data-root DATA --fig-dir RESULTS --frozen frozen_candidates.json
```

Compare accuracy, print/replay APCER, worst-attack APCER, BPCER, ACER, AUC and EER.
ACER averages worst-attack APCER and BPCER. BPCER at pooled APCER 5% and EER are
descriptive ROC statistics; they do not replace the saved deployment threshold.
Test outcomes do not select the candidate. The proposed public-test ACER target
is 10%, with failures reported honestly.

When validation ACER differs by at most two percentage points, selection remains
pending until both checkpoints have a demo-laptop benchmark. Colab CPU timing
cannot stand in for the laptop. Download the private benchmark ZIP, extract it
to `benchmark_inputs/`, then run locally:

```text
python -m src.liveness.benchmark --checkpoint MODEL1.pt MODEL2.pt --manifest benchmark_inputs/manifest.csv --data-root benchmark_inputs --out cpu_benchmark.json
```

Measurements cover crop, preprocessing and CPU forward, using 5 warmups, 30
measurements and 10 Torch threads. Median and p95 include machine identification.
Upload the JSON to the experiment folder and run the notebook's last cell. This
updates the decision without changing the frozen candidates or consulting test.

## Outputs and handoff

The experiment folder on Drive contains manifests and rejection logs; both model
run histories and checkpoint files; frozen candidate hashes and validation
thresholds; per-image test scores; comparison CSV/Markdown; training/ROC/score
plots; confusion matrices and representative error crops; `liveness_report.md`
with measured Methodology, Results and Discussion sections; and the private
benchmark input pack.

The crop grid and error images contain dataset faces. Keep them private on Drive;
commit only permitted aggregate figures and tables to the public repository.
Read the [dataset agreement](https://github.com/ZhangYuanhan-AI/CelebA-Spoof#dataset-agreement)
before sharing any source or derived images.

```python
from src.liveness.predictor import LivenessPredictor

antispoofing = LivenessPredictor("MODEL.pt", device="cpu")
label, probability_live = antispoofing.predict_frame(frame, box)
```

`predict(face)` accepts a correctly sized uint8 BGR crop and returns the same
`("live" | "spoof", P(live))` contract. Inference does not download weights.
The app owner must eventually check liveness before recognition and attendance;
those live checks are outside this public-data phase.

## Verification

Install the training/test dependencies, including `nbformat`, then run:

On Windows, keep the Python environment on an ASCII-only path so OpenCV can
load its built-in cascade file. The liveness image loader itself supports
Unicode dataset paths. The check script uses the system temporary directory
for test images rather than the OneDrive workspace.

```text
python scripts/check_training.py --include-liveness
python scripts/check_docs.py
```

Tests exercise real randomly initialized architectures, the complete small-image
pipeline, loss/score shapes, frozen normalization, exact CPU recovery, portable
leakage checks, ZIP safety, validation-only selection and notebook validity.
They do not establish accuracy on the real dataset. Source notebook cells remain
output-free until actual Colab execution.

References: [MobileNetV3](https://arxiv.org/abs/1905.02244),
[DeepPixBiS](https://arxiv.org/abs/1907.04047),
[CelebA-Spoof](https://github.com/ZhangYuanhan-AI/CelebA-Spoof), and
[Colab guidance](https://research.google.com/colaboratory/faq.html).
