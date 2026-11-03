# Train and evaluate recognition models

Model A uses ResNet-18, a projection layer, and cross-entropy. Model C uses the same encoder with an ArcFace head. Both expose L2-normalized embeddings. Gia Khánh owns A, C, and the shared verification evaluation. Model B can use the same evaluator through the `embed(face)` interface.

Start with the [shared training guide](training-guide.md) for environment setup, YAML settings, optimizer groups, warmup, accumulation, and resume. This guide covers recognition-specific commands and contracts.

For Model C collapse experiments, use the separate [margin warmup and recovery guide](model-c-recovery-training.md). It documents the implemented gradient, regularization, normalization, and diagnostic controls without changing Model A's baseline.

Local VGGFace2 manifests are connected. See [dataset preparation](dataset-preparation.md) for split rules and generation commands. Full training on the real dataset has not been completed. Software tests and CUDA smoke checks verify pipeline behavior and do not measure recognition quality.

## Run Model A or C

Activate the training environment and run from the repository root:

```bash
python -m src.recognition.classification.train --config configs/recognition/model_a.yaml
python -m src.recognition.classification.train --config configs/recognition/model_c.yaml --lr 0.0003 --epochs 20 --batch-size 32
```

The YAML defines the baseline. Only explicitly supplied CLI arguments override it, and the command never writes those changes back to the YAML.

Common flags include `--lr`, `--epochs`, `--batch-size`, `--weight-decay`, `--device`, `--num-workers`, `--seed`, `--patience`, `--embedding-dim`, `--data-root`, `--output-dir`, `--monitor`, `--mode`, `--evaluator`, `--optimizer`, `--scheduler`, `--warmup-epochs`, `--accumulation-steps`, `--eta-min`, `--cache`, and `--amp-dtype`.

Use `--set SECTION.KEY=VALUE` for other fields, including lists and booleans:

```bash
python -m src.recognition.classification.train --config configs/recognition/model_a.yaml --device cuda --set training.clip_grad_norm=5.0 --set training.deterministic=true
```

Relative data and output paths, including path overrides, are resolved from the YAML directory. `--resume` is resolved from the command's working directory. Use absolute paths for datasets stored elsewhere.

The supplied YAML files prioritize `val_tar` with `mode: max`, then `val_auc`, then `val_loss`. Verification runs after classification validation in each epoch. To train without verification, select the alternative policy explicitly:

```bash
python -m src.recognition.classification.train --config configs/recognition/model_a.yaml --set verification=null --set "training.checkpoint_tiebreakers=[]" --monitor val_loss --mode min
```

A missing monitored metric raises an error. Random initialization is the baseline. Use `--set model.initialization=imagenet` only after the group confirms the course requirements with the tutor.

The local A/C baseline uses a maximum of 100 epochs with patience 5, BF16 mixed precision, and batch size 64 on the RTX 3060 12 GiB. Early stopping can end training before epoch 100. The cosine scheduler spans the configured 100-epoch budget unless overridden. Group backups remain on Google Drive; this command writes local run artifacts and does not upload them.

Checkpoint priority is TAR at the configured empirical FPR target, then AUC, then classification validation loss. TAR measures acceptance of same-person pairs at the selected operating point; AUC summarizes ranking across thresholds. When TAR and AUC tie exactly, a loss decrease greater than 0.0001 can update `best.pt` and reset patience. A decrease in loss cannot override a worse TAR or AUC with the default zero tolerances. The ordered policy can prefer a new TAR best even if AUC decreases, so choose a different order if AUC must remain the primary objective. See [multi-metric checkpoint selection](training-guide.md#select-checkpoints-with-multiple-metrics) for the complete YAML schema and tolerance semantics.

`early_stopping_monitor: null` follows this full ordered policy. Patience 5 now means five consecutive eligible epochs without improvement under that policy, rather than five epochs without AUC improvement. A/C validation losses use different objectives, so they serve as within-run tie-breakers and should not rank A against C. FPR values and score thresholds are recorded for diagnosis; neither is a monotonic quality metric to maximize blindly. Use only validation metrics for selection; keep test results out of the checkpoint policy.

## Configure the encoder and objective

| Setting | Baseline |
| --- | --- |
| Backbone | ResNet-18 with its standard stem and global average pooling |
| Input | 112 by 112 BGR crop, converted to RGB |
| Normalization | Mean `[0.5, 0.5, 0.5]`, standard deviation `[0.5, 0.5, 0.5]` |
| Projection | 512 backbone features to a 128-dimensional embedding |
| A objective | Cross-entropy over unnormalized classifier logits |
| C objective | Cross-entropy over ArcFace target-margin logits |
| ArcFace settings | Scale 30, angular margin 0.3 |
| Inference | Projection followed by L2 normalization, without the classifier |

Input size, normalization, augmentation, initialization, embedding dimension, ArcFace scale, and margin are configured in YAML. Model C classification accuracy uses cosine logits without the label-dependent margin. Its loss uses target-margin logits during both training and validation.

Recognition optimizer groups are task-relative module paths:

- `backbone`: `model.encoder.backbone`.
- `head`: `model.encoder.projection` and `model.head`, including the ArcFace class weights for C.

Set separate rates with:

```bash
python -m src.recognition.classification.train --config configs/recognition/model_a.yaml --warmup-epochs 3 --accumulation-steps 4 --eta-min 0.000001 --group-lr backbone=0.0001 --group-lr head=0.001
```

See the [shared optimizer-group instructions](training-guide.md#set-separate-learning-rates) for inheritance, validation, and per-group logging. Default warmup is off, accumulation is 1, and both groups inherit the shared LR.

AMP applies only on CUDA. Both supplied configurations use `amp: true` and `amp_dtype: bf16`. Eligible encoder and classifier operations use BF16 autocast; model weights, optimizer state, normalization, ArcFace angular calculations, and loss calculations stay FP32. This is mixed precision, not a conversion of all weights to BF16. Standalone classification evaluation, embedding inference, and the verification callback currently use FP32.

BF16 requires native support on the selected GPU and disables GradScaler. `train_optimizer_steps` records optimizer updates, and the scheduler advances only when an epoch has an update. FP16 remains available with `--amp-dtype fp16` and enables GradScaler overflow checks. Its initial scale is configured by `amp_init_scale`, default 65536. This field has no effect in BF16 or FP32. Use `--set training.amp=false` for FP32 training. CPU runs use FP32 regardless of the AMP settings.

Select BF16 explicitly for either model:

```bash
python -m src.recognition.classification.train --config configs/recognition/model_a.yaml --device cuda --amp-dtype bf16
python -m src.recognition.classification.train --config configs/recognition/model_c.yaml --device cuda --amp-dtype bf16
```

Check `environment.json` for `precision: bf16-mixed` and `gradient_scaling: false`. Keep `amp_dtype` unchanged when resuming; FP16 checkpoints from older configurations require FP16 settings to resume.

## Supply classification data

The default configuration uses `data/VGGFace2` and the bbox protocol in `data/manifests/recognition_faces/protocol_bf22763a4f78/`. Verification pairs use unseen source-validation identities, with separate verification validation and test sets. Team photos in `data/faces/` are gallery data.

The selected generation contains 480 classifier identities with 131,280 training, 16,496 classification-validation, and 16,422 classification-test images. Verification retains 30 identities per split and 3,000 positive plus 3,000 negative pairs per split. Its validation pairs reference 6,861 unique images. Another 13,832 source images remain in `review.csv` and are excluded from the active protocol. No classifier identity was lost. See [face bbox preparation](recognition-face-preparation.md) for generation, conservative filtering, review overrides, and independent validation.

The classification CSV accepts these columns:

```csv
path,identity,split,session,x1,y1,x2,y2
frames/alice_01.jpg,alice,train,session_1,10,20,110,120
faces/alice_02.jpg,alice,val,session_2,,,,
```

`path` is relative to `data.root`. `split` is `train`, `val`, or `test`. Session and bbox are optional. Supply all four bbox coordinates or leave all four empty. Without a bbox, the adapter treats the entire image as the face region.

The base local VGGFace2 manifests and source DuckDB do not contain face bboxes or landmarks. The [original VGGFace2 release](https://github.com/ox-vgg/vgg_face2#pre-processing) provides loosely cropped faces and separate bbox/five-keypoint metadata in those crop coordinates. The selected local protocol uses offline YuNet detections to supply the missing bboxes without replacing the source images. Automatic acceptance requires exactly one plausible face; ambiguous and undetected images are retained for review rather than selecting the largest face.

Classification reads boxes from its CSV. Both generic and batched verification read matching boxes through `verification.bbox_manifest` and require complete coverage of the selected pairs. They use the same shared square-crop function as the demo interface. Predicted keypoints remain in the detector audit; landmark alignment is not applied. Bboxes are automatic proposals rather than human-verified identity annotations. Use the same generation for A/C comparisons and start a fresh run when changing crop protocols.

Both input modes decode images and call `src.common.face_crop.crop_face` on CPU. They emit RGB `uint8` crops. The trainer moves each batch to the model device before augmentation and normalization. App inference receives a BGR crop from that same shared crop function and applies equivalent RGB conversion and normalization without random augmentation.

## Configure GPU augmentation and VRAM caching

The supplied A and C configurations use these settings:

```yaml
preprocess:
  size: 112
  crop_scale: 1.0
  mean: [0.5, 0.5, 0.5]
  std: [0.5, 0.5, 0.5]
  horizontal_flip: 0.5
  brightness: 0.1
  contrast: 0.1
  saturation: 0.1
  rotation_degrees: 10.0
  translate: 0.05
data:
  cache: auto
  cache_max_fraction: 0.8
  cache_reserve_gib: 2.0
verification:
  batch_size: 64
  cache: auto
```

These are partial sections. Keep the other fields in the complete configuration.

`FaceBatchTransform` runs on the model device. CUDA training uses GPU tensor operations for augmentation and normalization. CPU training uses the same operations on CPU. Each image receives independent random parameters, sampled again each time the batch is used. Validation, classification test, and verification have no random augmentation.

Augmentation applies horizontal flip, affine rotation and translation, brightness, contrast, then saturation. Color factors are sampled uniformly from `[1 - value, 1 + value]`. Rotation is sampled within the configured positive and negative degree limits. Translation is bounded by the configured fraction of image width and height, independently on each axis. Affine sampling uses bilinear interpolation with black padding. Normalization follows augmentation and stays in float32 under AMP. Set any augmentation field to zero to disable it.

With `data.cache: auto`, the entry point tries to cache **both classification train and validation splits** as unaugmented RGB `uint8` crops in VRAM after allocating the model. JPEG decoding, crop, and resize happen once on CPU during cache construction. Only the current augmented batch becomes float32. The program does not materialize all augmented images in VRAM or RAM. Test images and team gallery photos are excluded.

With `verification.cache: auto`, the built-in training evaluator separately caches the unique verification-validation crops. It reserves this smaller cache first and subtracts its allocation from the classification cache budget. All caches share the configured memory fraction and training reserve. Insufficient space falls back to decoding and transferring batches. `verification.cache` and `verification.batch_size` control the built-in A/C training evaluator; standalone generic verification still supports the single-image embedder interface for other model owners.

Verification normalizes on GPU without augmentation and computes fresh FP32 embeddings in batches each epoch. It caches images, not embeddings, because encoder weights change during training. The selected protocol's 6,861 verification-validation images occupy about 246.23 MiB as RGB `uint8` crops. Together with the classification train/validation crops, raw cache storage is about 5.42 GiB at 112 by 112. Model state, optimizer state, activations, normalized batches, and CUDA workspace require additional memory.

The crop storage requires `image_count * 3 * size * size` bytes. The cache budget is the smaller of `free_vram * cache_max_fraction` and `free_vram - cache_reserve_gib`. Free VRAM comes from [PyTorch's memory query](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.mem_get_info.html). The reserve leaves room for activations, gradients, optimizer state, and CUDA workspace; it is a configurable estimate of training needs.

If both classification splits do not fit, the program uses CPU data-loader batches, pinned-memory transfer, and GPU augmentation. It also falls back after a CUDA cache-allocation error. If training or verification runs out of CUDA memory while either cache is active, it releases both caches and retries with batch streaming from the last completed epoch. The retry uses a new run directory and preserves the failed attempt. Without a saved epoch, it restarts from the original seed or the supplied resume checkpoint. Partial-epoch updates are discarded. A second OOM in streaming mode raises an error; reduce the training or verification batch size as appropriate.

Run with automatic caching:

```bash
python -m src.recognition.classification.train --config configs/recognition/model_a.yaml --device cuda --cache auto --num-workers 4
```

Force batch streaming while retaining GPU augmentation:

```bash
python -m src.recognition.classification.train --config configs/recognition/model_c.yaml --device cuda --cache none --set verification.cache=none --num-workers 4 --set preprocess.rotation_degrees=5.0
```

Adjust the cache budget through CLI overrides:

```bash
python -m src.recognition.classification.train --config configs/recognition/model_a.yaml --device cuda --set data.cache_reserve_gib=3.0 --set data.cache_max_fraction=0.7
```

`num_workers` controls CPU decoding during cache construction or streaming. A completed CUDA cache uses zero loader workers and no pinned-memory copy because its images already reside on GPU. Shuffle still uses the checkpointed loader generator. Augmentation randomness uses the checkpointed device RNG. Cached crop tensors are rebuilt on startup and are not saved in checkpoints.

The command prints the selected mode and writes `data_pipeline.json` in the run directory. It records classification cache information, a nested `verification` cache report, and `total_cached_bytes`. Reports include the crop byte estimate, cache budget when queried, actual device, cached image count, and the reason for streaming. The cache policy is task-local; other task owners retain their existing loaders and preprocessing.

A real-data CUDA smoke check before bbox filtering cached all 141,068 original training and 17,682 original classification-validation crops, using 5,974,080,000 bytes, about 5.56 GiB. A and C each completed one forward, backward, and optimizer update with batch size 64 and finite losses and gradients. Batch streaming with pinned CPU input and CUDA augmentation was also checked. These checks do not establish training convergence or recognition accuracy.

A later check before bbox filtering cached all 7,026 original verification-validation crops and embedded them in 110 batches without decoding again during evaluation. It ran while holding an additional 5,974,080,000-byte CUDA allocation to represent classification crop storage. That allocation check did not decode the full classification dataset again or run a full training epoch.

With the selected bbox protocol, A and C each completed a BF16 smoke run with 64 real training crops and 64 classification-validation crops. Projection outputs were BF16, weights updated and remained finite, and all 6,861 verification-validation images had matching bbox entries. Independent validation checked all 183,851 accepted boxes against original-image dimensions and matched them to the classification and verification catalogs. Full training convergence remains unmeasured.

The folder adapter accepts detected face images in this layout when `data.manifest` is null:

```text
classification_data/
  train_data/alice/*.jpg
  train_data/bob/*.jpg
  val_data/alice/*.jpg
  val_data/bob/*.jpg
  test_data/alice/*.jpg
  test_data/bob/*.jpg
```

Train and validation must contain the same set of at least two identities. The training identity names define the class mapping saved with the checkpoint. Training does not open test images.

The loader rejects repeated train/validation paths, incompatible identity sets, and identity/session combinations shared across those splits. If the CSV provides `group` and `sha256`, it also rejects group or exact-content overlap. The VGGFace2 exporter supplies those fields after checking source images. External manifests without hashes do not trigger automatic image hashing in the loader.

Thành's shared loader can replace this task-local adapter when its interface is available. Model B can use the same manifest and class mapping with its own identity sampler and triplet mining.

## Save checkpoints and resume

Each invocation creates a unique directory under `runs/recognition/` with:

- `best.pt`, `last.pt`, and the export `recognition_<a or c>_<date>.pt`.
- `config.yaml`, `resolved_config.json`, `environment.json`, and `summary.json`.
- `history.csv` and `training.png`.
- `data_pipeline.json`, including the selected cache mode and augmentation device.

`last.pt` stores the model, optimizer, scheduler, AMP scaler, epoch, best metric, patience counter, history, preprocessing, class mapping, and RNG states. It also stores the best-weight snapshot so resume can recreate a missing inference artifact. `best.pt` and the dated export contain selected inference weights, metadata, and validation metrics.

Metadata records image paths, splits, bboxes, sessions, groups, and hashes. Keep the data unchanged between resume attempts.

```bash
python -m src.recognition.classification.train --config configs/recognition/model_a.yaml --resume /absolute/run/path/last.pt
```

Resume creates a new directory. Use the same YAML and hyperparameter overrides as the original run. Device, progress, and worker count may change. Epochs may increase only when no scheduler exists, including no warmup scheduler. Accumulated gradients are flushed before epoch-boundary checkpoints. A checkpoint that has reached its stopping condition does not continue under that policy.

Reproducible resume requires the same runtime/device behavior, explicit loader generators, nonpersistent workers, and samplers without unsaved state. Distributed training and resume partway through a batch are not currently supported. Call `seed_everything` before constructing models and loaders in a custom entry point.

## Evaluate classification and run inference

Run classification test evaluation only after the configuration is fixed:

```bash
python -m src.recognition.classification.evaluate --checkpoint /absolute/model_a.pt --split test --device cpu --output reports/figures/recognition/model_a_test.json
```

The output path must not already exist. Test data is not used for checkpoint selection. Use `--data-root` when images have moved, and keep the configured manifest available on the destination machine.

Use the selected checkpoint in the app:

```python
from src.common.face_crop import crop_face
from src.recognition.classification.predictor import RecognitionEmbedder

embedder = RecognitionEmbedder("checkpoint.pt", device="cpu")
face = crop_face(
    frame,
    box,
    size=embedder.preprocess_config.size,
    scale=embedder.preprocess_config.crop_scale,
)
vector = embedder.embed(face)
```

The output is a finite one-dimensional NumPy float32 vector with L2 norm approximately 1. The input must be a BGR uint8 crop with the saved dimensions. `model_version` identifies the checkpoint by SHA-256. Rebuild the gallery and calibration when the checkpoint or preprocessing changes.

CPU latency on the demo laptop has not been measured. Passing inference tests does not establish the under-50-ms target.

## Run shared verification

`PairEvaluator` accepts any object with `embed(face) -> np.ndarray` that returns a finite L2-normalized vector. A, B, and C are scored using the same pairs, crop, and metrics. Each image is embedded once per evaluation call.

Whitespace pair files contain three fields, with forward slashes in paths:

```text
alice/01.jpg alice/02.jpg 1
alice/01.jpg bob/01.jpg 0
```

Quote paths that contain spaces. Label 1 means the same identity; label 0 means different identities. CSV files require the header `path1,path2,label`. The evaluator rejects repeated or reversed pairs, a missing label class, and invalid embeddings. It does not guess LFW formats or label polarity.

Compare A and C:

```bash
python -m src.recognition.verification.run --config configs/recognition/verification.yaml --checkpoint A=/absolute/model_a.pt --checkpoint C=/absolute/model_c.pt
```

Each invocation creates a unique directory under `reports/figures/recognition/` with `results.csv`, `scores.csv`, `roc.png`, `metrics.json`, and its configuration. A raw-pixel baseline is enabled by default. Disable it with `--set baseline=false`.

Other flags include `--pairs`, `--data-root`, `--target-fpr`, `--split`, `--device`, `--output-dir`, `--calibration`, and `--set KEY=VALUE`.

For Model B or a different checkpoint format, use a YAML model entry with a custom factory:

```yaml
models:
  B:
    checkpoint: /absolute/model_b.pt
    factory: your_package.predictor:TripletEmbedder
```

The factory receives `(checkpoint, device=...)` and returns an object implementing `embed(face)`. The evaluator itself does not change. To supply a custom evaluator during recognition training, `--evaluator module:factory` calls `factory(recognition_config)`, which returns a callable `(task, device) -> dict[str, float]`.

Cosine uses similarity as its score. Euclidean uses negative distance so a higher score consistently indicates a more likely match. Unit-norm embeddings produce the same ranking and expected AUC for both metrics, apart from numerical differences and ties.

Validation chooses the threshold with the highest TAR among ROC points whose empirical FPR does not exceed `target_fpr`. Ties use the stricter threshold. Reports include `similarity_min`, `distance_max`, and positive/negative pair counts. A low negative-pair count gives coarse FPR resolution and does not establish a population-level FPR guarantee. ROC/AUC uses [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html).

Evaluate test pairs with fixed validation thresholds:

```bash
python -m src.recognition.verification.run --config configs/recognition/verification.yaml --checkpoint A=/absolute/model_a.pt --split test --pairs /absolute/verification_pairs_test.csv --calibration /absolute/validation_report/metrics.json
```

Model names, checkpoints, and crop settings must match calibration. Reusing the validation pair file as test data is rejected. A different filename alone does not establish independent sessions or a leak-free split. These thresholds describe 1:1 verification. The app's 1:N gallery search needs separate evaluation with registered and unknown people.

## Check the implementation

```bash
python scripts/check_training.py
python scripts/check_training.py --include-liveness
python scripts/check_training.py --format --lint-only
```

The script checks Black, Ruff, and the owned tests. CUDA tests run when a GPU is available and are otherwise skipped. `--include-liveness` adds Hoan's existing tests. Temporary files and caches are under `runs/`.

| Package | Responsibility |
| --- | --- |
| `src/training/` | Shared training mechanics |
| `src/recognition/classification/` | A, C, task-local data adapter, and inference |
| `src/recognition/verification/` | ROC/AUC, model comparison, and threshold calibration |
| `src/datasets/` | Reproducible dataset manifest exports |
| `configs/recognition/` | Recognition baseline configurations |
| `requirements/` | Isolated training environment input and lockfile |

The original common, liveness, emotion, and app implementations keep their own interfaces. The shared trainer can be adopted by each task owner through a task adapter and data loaders.
