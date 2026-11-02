# Prepare dataset mappings and manifests

The local datasets already contain DuckDB metadata, but their schemas and splits do not directly match the project loaders. `src.datasets.prepare` exports CSV and JSON files to `data/manifests/`. Source images and databases are read-only inputs. Dataset notebooks and pipelines retain their original content. Team photos remain separate gallery/demo data.

See the [shared training guide](training-guide.md) for environment setup and training commands.

## Generate and validate manifests

Run from the repository root with the training environment active:

```bash
python -m src.datasets.prepare --config configs/datasets.yaml
python -m src.datasets.validate --config configs/datasets.yaml
```

Use `--only recognition`, `--only liveness`, `--only emotion`, or `--only team` to select a dataset. Multiple names can follow `--only`. Override configuration through arguments:

```bash
python -m src.datasets.prepare --config configs/datasets.yaml --only recognition --set recognition.positive_pairs=1500 --set recognition.negative_pairs=1500
```

Relative paths are resolved from the YAML directory. The configuration defines seed, holdout fractions, pair counts, workers, and gallery detector settings. CSV/JSON output is deterministic for the same data and seed.

`fingerprints.sqlite3` caches image size, mtime, SHA-256, and header checks. To force a complete byte recheck, remove only that generated cache file before preparation. Do not change image content while deliberately preserving both size and mtime.

`dataset_report.json` records export results. The independent validation command writes `validation_report.json`, checking mappings, leakage, pair labels, class sets, clip/frame associations, and gallery bboxes. Images, databases, gallery photos, and generated manifests remain under ignored `data/` paths.

Local export results on November 1, 2026:

| Dataset | Exported data |
| --- | --- |
| VGGFace2 classification | 480 classes; 141,068 train, 17,682 validation, 17,648 test images |
| VGGFace2 verification | Validation and test each have 30 identities, 3,000 positive pairs, and 3,000 negative pairs |
| CelebA liveness | 75,780 train, 13,367 validation, 49,646 test images |
| DFEW | 11,697 benchmark clips, 187,152 frames, all five folds |
| Team gallery | 36 photos from six people; 10 reviewed crop candidates and 26 photos requiring detector/bbox review |

VGGFace2 had no rejected images from header checks or conflicting-identity hashes. CelebA excluded 61 source-training rows belonging to three subjects also present in test. Another 51,917 rows were omitted because their attack types were not selected. Source images and databases remain unchanged.

## Use VGGFace2 for A, B, and C

The source catalog contains 197,693 images: 480 identities in `train/` and 60 different identities in `val/`. Its global DuckDB mapping assigns indices 0 through 539. A classifier trained on the 480 source-training identities cannot classify the 60 unseen validation identities as known classes.

The generated manifests separate classification from unseen-identity verification:

- `vggface2_classification.csv` uses source-training identities and creates train, validation, and test image groups within each identity. All three splits contain the same 480 classes. The default group proportions are approximately 80/10/10, so image counts can differ slightly from those proportions.
- `vggface2_verification.csv` uses the unseen source-validation identities, split into separate verification validation and test identity pools, with 30 identities each by default.
- `verification_pairs_val.csv` and `verification_pairs_test.csv` each contain 3,000 same-identity pairs with label 1 and 3,000 different-identity pairs with label 0 by default.
- `vggface2_class_to_idx.json` stores contiguous classifier indices 0 through 479. The CSV also retains source indices for comparison.
- `vggface2_rejected.csv` records missing images, invalid headers, or identical bytes assigned to conflicting identities.

Images with the same SHA-256 or filename prefix within an identity stay in the same classification split. A filename prefix is a conservative grouping key, not a known recording session. Cross-identity byte duplicates are excluded on both sides rather than assigned a guessed label. Verification pairs are unique, contain different image bytes, and have labels checked against their identities.

This is a project protocol generated from the local subset, not an official VGGFace2 test result. Session metadata is unavailable. Filename and hash grouping cannot establish that all near-duplicates or same-session images have been separated.

The A, C, and verification YAML files already point to these manifests. The unseen verification identities and team photos do not enter the classifier. B can use the same classification CSV and mapping with its own P-by-K identity sampler.

```bash
python -m src.recognition.classification.train --config configs/recognition/model_a.yaml
python -m src.recognition.classification.train --config configs/recognition/model_c.yaml
```

These commands start full training. Manifest preparation does not execute them.

## Use processed CelebA-Spoof with liveness

`celeba_liveness.csv` matches Hoan's existing loader schema:

```text
path,label,attack,subject,source,split,x1,y1,x2,y2
```

Paths are relative to `data/CelebA_Spoof_processed/`. In the liveness manifest, live is 1 and spoof is 0. The source DuckDB `is_spoof` field and YOLO labels use the opposite convention. `celeba_label_mapping.json` records the mapping.

The processed metadata vector has four entries. The older source builder expects the original 44-entry CelebA vector, so the exporter reads named DuckDB columns instead. It reuses `src.liveness.manifest` bbox parsing, converting `_BB.txt` coordinates from a 224-pixel reference scale to original-image pixels.

The default protocol uses live images, print attacks, and replay attacks. Mask attacks are filtered by configuration and are never relabeled as print or replay. Three subjects overlap between source train and test. The exporter retains the selected official-test rows, excludes source-training rows for those subjects, and holds validation subjects out of the remaining training pool.

`celeba_rejected.csv` records exclusions and reasons. The source YAML that points both validation and test at the same data is not used for the project split.

```bash
python -m src.liveness.train --model mobilenetv3 --manifest data/manifests/celeba_liveness.csv --data-root data/CelebA_Spoof_processed --ckpt-dir checkpoints --num-workers 0
```

The liveness trainer, models, and evaluation remain unchanged. VGGFace2 receives full-file hash and header checks. Every exported CelebA row receives an image-header/dimension and bbox check. These checks do not decode every pixel of every image.

## Use processed DFEW with emotion models

DFEW contains video clips rather than independent FER2013 images. Its source metadata already defines five folds. The bundled source pipeline refers to `EmoLabel_DataSplit`, while the local directory is actually `5-fold_cross_val_split`.

- `dfew_fold1.csv` through `dfew_fold5.csv` contain one row per clip. They retain official test clips and derive validation from official training clips, stratified by emotion.
- `dfew_frames.csv` contains one row per frame and a `clip_id` for joining to a fold. All frames from a clip share its split. Frames are not randomly divided between splits.
- `dfew_class_to_idx.json` maps Happy to 0, Sad to 1, Neutral to 2, Angry to 3, Surprise to 4, Disgust to 5, and Fear to 6. Source labels 1 through 7 become training labels 0 through 6.

The exporter compares DuckDB against all ten official split CSV files and checks the 16-frame sequence for each benchmark clip. It exports single-labeled benchmark clips from those folds. Ambiguous or missing clips are not given guessed labels. Actor/session separation cannot be established from the available metadata.

The emotion owner can join frames to one fold by `clip_id` for an image model or use each clip directory for a video model. These exports do not change the emotion architecture or load the supplied pretrained checkpoints.

## Review team gallery photos

`team_gallery.csv` records folder-based `person_id`, a separate `person_index`, role `gallery`, optional session, bbox, and detector status. `team_person_to_idx.json` is a gallery mapping and does not share classifier indices with VGGFace2.

Current photos have no known session labels, so the session field stays empty. The OpenCV Haar detector marks an image `ready` only when it has one sufficiently large face candidate. `min_face_area_fraction`, default 0.05, rejects small candidates that may belong to people in the background.

Review these statuses:

| Status | Meaning |
| --- | --- |
| `ready` | One candidate passes the configured size check; review its crop before registration. |
| `no_face` | The frontal-face detector found no candidate, which can happen with angled faces. |
| `multiple_faces` | More than one candidate was found, including possible background faces. |
| `small_face_review` | The only candidate is too small to select as the foreground person. |
| `invalid_image` | The image could not be read or failed its header check. |

Angled photos and photos with background people remain in the manifest. They are not deleted or added to training. No actual gallery embeddings exist until a trained checkpoint is available.

The 36 photos are excluded from recognition train, validation, and test. They are not automatically treated as a same-session test set. Demo evaluation needs query images or videos from a new session, covering both registered and unknown people.

## Check dependencies and results

DuckDB 1.4.4 reads source catalogs in read-only mode. The training environment pins OpenCV 4.13.0.92 because the existing detector needs `CascadeClassifier`, which was unavailable in the previously installed OpenCV 5 build. See the [official OpenCV package](https://pypi.org/project/opencv-python/4.13.0.92/).

```bash
python scripts/check_training.py --include-liveness
python -m src.datasets.validate --config configs/datasets.yaml
```

Tests use small real image files and databases to check deterministic exports, label mappings, duplicate grouping, identity holdout, source subject overlap, clip/frame grouping, and gallery roles. Independent validation also checks the generated local manifests.

The dataset work passed a 31-test suite including existing liveness tests, followed by seven dataset tests after the background-crop check was added. Black and Ruff passed. Real-data checks loaded VGGFace2 batches of shape `[4,3,112,112]`, a liveness batch of `[6,3,224,224]` covering both labels in each split, and all 16 frames of a DFEW clip.

All four manifest validations passed. Ten gallery crops were inspected visually, and 26 photos remain flagged for review. These mapping checks do not provide accuracy or AUC results from a fully trained recognition model.
