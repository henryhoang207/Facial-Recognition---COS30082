# Generate face bboxes and recognition manifests

This command runs YuNet on the local VGGFace2 subset and publishes a separate recognition protocol with face bboxes. It does not download the full VGGFace2 image dataset, edit source images, or replace the original manifests.

The pretrained detector is a preprocessing tool. Recognition A and C retain their configured random initialization. Generated boxes are automatic detections, not human-verified identity annotations.

## Run detection

Activate the training environment and run from the repository root:

```bash
python -m src.recognition.preprocessing.prepare --config configs/recognition/face-preparation.yaml
```

The first run downloads the 232,589-byte [YuNet 2023mar model](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), which is compatible with the locked OpenCV 4 runtime. Its SHA-256 is pinned in YAML and verified before use. The model is licensed under the upstream MIT license. No new Python dependency is required.

The command reads the existing classification and verification catalogs, then scans their union once. Each worker owns its own detector instance. Detection uses CPU workers; this does not require recognition training to run at the same time.

Override configuration through arguments:

```bash
python -m src.recognition.preprocessing.prepare --config configs/recognition/face-preparation.yaml --set workers=4 --set detector.score_threshold=0.95
```

Image paths and output paths are resolved from the configuration file. A new policy creates a separate protocol directory. Worker count and chunk size do not change the selected boxes.

To inspect the command on a limited set before a full run:

```bash
python -m src.recognition.preprocessing.prepare --config configs/recognition/face-preparation.yaml --scan-limit 128
```

An incomplete scan writes its cache and scan report but does not publish training manifests. Run the command without `--scan-limit` to finish.

## Resume an interrupted scan

Run the same command again. `detections.sqlite3` stores results by image path, source fingerprint, detector checksum, detector settings, and preprocessing version. Results are committed after each chunk. An interrupted chunk may run again; completed unchanged images are reused.

Run one preparation job per output directory. Keep source images and the original manifests unchanged while detection or training is running.

## Understand face selection

The detector receives a square letterbox image without aspect-ratio distortion. Boxes and five predicted keypoints are mapped back to original-image coordinates. Only bboxes are used for cropping; landmark alignment is not applied.

The default settings are:

| Setting | Value |
| --- | --- |
| Detector input | 320 by 320 |
| Confidence threshold | 0.9 |
| NMS threshold | 0.3 |
| Minimum face side in the original image | 20 pixels |
| Minimum bbox area relative to the original image | 0.05 |
| Maximum normalized distance from image center | 0.4 |

An automatic acceptance requires exactly one detected face and all size/position checks to pass. Every additional detected face makes the image ambiguous, including smaller background faces. The command does not assume that the largest face belongs to the labeled person.

| Status | Meaning |
| --- | --- |
| `accepted` | One face passes the automatic checks. |
| `no_face` | No candidate passes the detector threshold. |
| `multiple_faces` | More than one face was detected. |
| `small_face` | The single candidate fails the minimum size or area check. |
| `off_center` | The single candidate is too far from the crop center. |
| `invalid_image` | The image could not be decoded. |
| `manual_accept` | A reviewed bbox was supplied in an override file. |
| `manual_reject` | A reviewer explicitly excluded the image. |

Automatic detection cannot prove the identity label. Inspect samples before training and review uncertain cases. Filtering can remove difficult poses or occlusions, so report the resulting filtered protocol rather than presenting it as an official VGGFace2 benchmark.

## Find generated files

Outputs live under `data/manifests/recognition_faces/`. `current.json` identifies the last completed protocol directory, named `protocol_<hash>`. Each published generation contains:

- `vggface2_classification.csv`, with bbox columns and contiguous class labels.
- `vggface2_class_to_idx.json`, matching those labels.
- `vggface2_verification.csv`, retaining the original unseen validation/test identity partitions.
- `verification_pairs_val.csv` and `verification_pairs_test.csv`.
- `bbox_manifest.csv`, used by the verification loader.
- `detector_catalog.csv`, with every result, candidate bbox, confidence, and keypoints.
- `review.csv`, with excluded or ambiguous images.
- `summary.json` and `resolved_config.json`.

Classification train/validation/test assignments remain unchanged for retained images. An identity is retained only if all three splits still contain accepted images; any identity dropped by this rule is listed in the summary. `source_label` preserves the original class index.

Verification identities remain disjoint from classifier identities and from the opposite verification split. Pairs are regenerated deterministically from accepted images, with 3,000 positive and 3,000 negative pairs per split by default. Identical image bytes do not form a positive pair. This preserves pair counts without reintroducing excluded faces.

Validate a generated directory independently:

```bash
python -m src.datasets.validate --config configs/datasets.yaml --only recognition --set output_dir=/absolute/path/to/protocol_directory
```

Generated files and detector weights are ignored by Git. Include the selected generation and model checksum in the group's Google Drive backup.

The selected local generation is `protocol_bf22763a4f78`. The full scan covered 197,683 catalog images, accepted 183,851 automatic boxes, and retained 13,832 images for review. Classification keeps all 480 identities with 131,280 train, 16,496 validation, and 16,422 test images. Verification keeps 19,653 images across the two unseen-identity pools, with 6,000 pairs and 30 identities per split. Original base manifests and image files remain available.

## Review and override ambiguous images

Create a CSV containing only reviewed decisions. Coordinates use the original image, with `x2` and `y2` as exclusive right/bottom bounds:

```csv
path,decision,x1,y1,x2,y2
train/n000002/example.jpg,accept,20,30,140,170
train/n000004/example.jpg,reject,,,,
```

The filenames above are placeholders. Select the bbox of the person matching the source identity folder. Use the candidates in `review.csv` as proposals; do not accept one solely because it has the largest area.

Apply the reviewed file:

```bash
python -m src.recognition.preprocessing.prepare --config configs/recognition/face-preparation.yaml --set overrides=/absolute/path/to/reviewed_boxes.csv
```

This reuses detector results and publishes a new protocol generation. Invalid, duplicate, or out-of-bounds overrides raise an error. The previous generation remains available.

## Use the selected protocol for training

Recognition A/C configurations select a generation through `data.manifest`, `verification.pairs`, and `verification.bbox_manifest`. `data.root` and `verification.root` continue to point at the original VGGFace2 image directory. Update all three manifest paths together when changing generations.

Classification reads bboxes from its CSV. Verification requires bbox coverage for every image used in its pairs and applies the same shared square-crop function. GPU caches store those resulting crops. The verification catalog checksum is recorded in calibration reports and training metadata; changing bboxes invalidates an old calibration or incompatible resume.

The demo receives crops through the existing shared crop interface. The offline `YuNetDetector.detect(frame)` also implements the common bbox-returning detector interface if the demo owner chooses to use it. No shared detector, liveness, emotion, or app code is modified by this command.

Start a new training run after changing from loose-image crops to detector bboxes. Existing checkpoints from the loose-crop protocol do not provide results for this new protocol.
