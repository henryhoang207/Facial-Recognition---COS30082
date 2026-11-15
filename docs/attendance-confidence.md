# Train an attendance confidence model

The attendance MLP scores whether the candidate proposed by Model A is correct. One sample is a window of five observations of one person. The default gallery contains six identities, each represented by the normalized mean of three enrollment embeddings.

The candidate has the highest mean cosine over the window. The MLP takes that candidate's mean, minimum, and standard deviation of cosine, mean cosine gap to its strongest alternative in each observation, and fraction of observations where the candidate ranks first. The same feature function is used for training and inference.

## Prepare the dataset

Run from the repository root:

```powershell
.venv/Scripts/python.exe -m src.recognition.attendance.prepare --config configs/recognition/attendance.yaml
```

Set `encoder_checkpoint` to your trained Model A checkpoint. The configuration defaults to the local epoch-27 checkpoint used to generate the provided manifests. Use `--set encoder_checkpoint=/absolute/path/best.pt` for a different file. The selection-pair CSV must match the checkpoint's recorded verification input. Those images are excluded from the attendance dataset.

Preparation freezes Model A and batches image embeddings on the configured device. The VGGFace2 verification validation identities are split into attendance train and dev identities. Verification test identities supply the attendance test set. Each identity contributes distinct enrollment and query images. Probe windows use separate images and never share an image with enrollment. All images in a window come from the same identity.

Half of the windows per identity include the true person in the gallery. The other windows use galleries without the true person. Every other enrolled training window uses the nearest distractor templates to provide harder cases. Dev and test use randomly sampled galleries. The label is 1 when the candidate is correct and 0 otherwise. The `kind` column separates wrong enrolled candidates from unknown people.

The generator writes these files under `data/manifests/attendance_vggface2/`:

| File | Content |
| --- | --- |
| `attempts.csv` | Split, true identity, query paths, gallery identities, candidate, label, and feature vector per window |
| `enrollment.csv` | Enrollment paths and identity indices per split |
| `identity_to_idx.json` | Contiguous identity indices for auditing |
| `dataset.json` | Protocol, counts, and SHA-256 fingerprints |

Use a new `dataset_dir` to generate another export. Update the checkpoint and regenerate features together when changing Model A. The identity index belongs to this dataset and is separate from Model A's classifier labels.

### Prepare VidTIMIT video attempts

Download the 43 subject ZIP files from the official release into `data/VidTIMIT/archives/`. Keep the archives intact. Cite Sanderson and Lovell (2009) when reporting results from this dataset. [VidTIMIT documentation](https://conradsanderson.id.au/vidtimit/) · [Zenodo release](https://zenodo.org/records/158963)

```powershell
.venv/Scripts/python.exe -m src.recognition.attendance.prepare_vidtimit --config configs/recognition/attendance-vidtimit.yaml
```

The preparer extracts the frames it needs, crops single faces with YuNet, and scores the windows with frozen Model A weights. Enrollment uses distinct speech clips from session 1. Each probe window uses ordered frames from one clip in session 2 or 3. The default stride of five at 25 fps samples every 200 ms. Train, dev, and test use disjoint identities. The training split includes hard galleries for enrolled and unknown probes. The output under `data/manifests/attendance_vidtimit/` has the same training schema as the VGGFace2 export, with additional session and clip columns in `attempts.csv`.

### Catalog YouTube Faces clips

```powershell
.venv/Scripts/python.exe -m src.recognition.attendance.catalog_youtube_faces --root data/YoutubeFaces --metadata youtube_faces_with_keypoints_full.csv --output data/manifests/youtube_faces
```

The catalog matches `videoID` to each NPZ archive and uses `personName` as the identity. It exports `clips.csv`, `identity_to_idx.json`, and `dataset.json`. The default split is approximately 60% train, 20% dev, and 20% test by identity. Each identity with multiple clips has one enrollment clip and separate probe clips. A single-clip identity can supply an unknown probe. `clips.csv` is the source catalog for generating labeled five-frame attempts and Model A features.

Generate the YouTube Faces attendance attempts and train with the VidTIMIT hard-mined export:

```powershell
.venv/Scripts/python.exe -m src.recognition.attendance.prepare_youtube_faces --config configs/recognition/attendance-youtube-faces.yaml
.venv/Scripts/python.exe -m src.recognition.attendance.train --config configs/recognition/attendance-youtube-faces.yaml
```

YouTube Faces uses the annotated face box and references frames as `archive.npz#frame_index`. Galleries alternate between nearest distractors and random draws within each split. A probe window can be scored with and without the true identity enrolled. The 500-epoch training configuration selects one dev threshold that meets the false-accept target on each source and reports held-out results in `metrics_by_source.json`.

### Compare models on fixed galleries

Build a manifest with stable rosters for the two video sources. Each roster receives windows from its enrolled identities and from visitors. The hard rosters and their visitors are selected using similarities between enrollment templates. The command reuses the dev and test identities and the probe windows from the source manifests.

```powershell
.venv/Scripts/python.exe -m src.recognition.attendance.fixed_gallery --configs configs/recognition/attendance-youtube-faces.yaml configs/recognition/attendance-vidtimit.yaml --manifest data/manifests/attendance_fixed_gallery/attempts.csv
```

Repeat the command with `--report <output.json>` and `--checkpoint <run>/attendance.pt` to compare checkpoints against cosine with frame agreement. The evaluator calibrates one threshold per method on the random dev rosters, then applies it to both random and hard test rosters. The report gives counts by source and by gallery. The same probe window can appear in several galleries.

To train the source-balanced experiment, add `--set balance_training_sources=true` to the YouTube Faces training command. Each draw chooses a source, identity, clip, and unique window uniformly at its level, then selects one of the window's gallery variants. The sampler keeps the same epoch size and training budget as the default configuration.

For checkpoint selection on fixed galleries, set `fixed_gallery_dev` to the fixed-gallery manifest and set `training.monitor` to `val_gallery_correct_rate` with mode `max`. Only dev rows enter checkpoint selection. Each epoch uses one threshold that limits unknown accepts in every source and gallery mode. The correct and wrong-enrolled rates average the four source and mode strata equally for YouTube Faces and VidTIMIT. The exported checkpoint uses the selected dev threshold. Use the fixed-gallery evaluator to report held-out results after selecting the run.

### Add the top-2 gap

The optional sixth feature is the difference between the two highest mean cosine scores across a five-frame window. The per-frame margin already in the five-feature model compares the candidate with the strongest competitor in each frame.

```powershell
.venv/Scripts/python.exe -m src.recognition.attendance.add_top2_gap --source configs/recognition/attendance-youtube-faces.yaml data/manifests/attendance_youtube_faces data/manifests/attendance_youtube_faces_top2 --source configs/recognition/attendance-vidtimit.yaml data/manifests/attendance_vidtimit_hard data/manifests/attendance_vidtimit_hard_top2 --fixed data/manifests/attendance_fixed_gallery/attempts_dev_hard.csv --fixed-output data/manifests/attendance_fixed_gallery/attempts_dev_hard_top2.csv
.venv/Scripts/python.exe -m src.recognition.attendance.train --config configs/recognition/attendance-youtube-faces.yaml --set dataset_dir=../../data/manifests/attendance_youtube_faces_top2 --set 'additional_datasets=[../../data/manifests/attendance_vidtimit_hard_top2]' --set balance_training_sources=true
```

The exporter checks the original candidates and five feature values before appending the new column. Checkpoint metadata determines which feature schema inference uses. The fixed-gallery evaluator can score both schemas against the same extended manifest.

## Train and compare

```powershell
.venv/Scripts/python.exe -m src.recognition.attendance.train --config configs/recognition/attendance.yaml
```

Train on VidTIMIT for 500 epochs with early stopping disabled:

```powershell
.venv/Scripts/python.exe -m src.recognition.attendance.train --config configs/recognition/attendance-vidtimit.yaml --set training.epochs=500 --set training.patience=null
```

The VidTIMIT configuration selects the best checkpoint by development TAR at the configured false-accept target. Higher ROC AUC and lower development loss break ties in that order.

To add difficult training attempts, export a separate dataset with `--set vidtimit.hard_mining_min_cosine=0.55` and a new `--set dataset_dir`. The preparer scores unused training windows against the nearest distractor templates. It adds high-scoring unknown attempts, their enrolled counterparts, and enrolled attempts where a distractor wins. Use the same `dataset_dir` override when training on that export. The development and test attempts follow the original sampling protocol.

Override configuration values with `--set`. The trainer fits feature scaling on train and compares the selected MLP with logistic regression and mean cosine with a consensus requirement. It chooses each threshold on dev at the configured false-accept target among incorrect candidates. Test uses the frozen dev thresholds. The test report includes correct accepts among enrolled attempts, wrong accepts among enrolled attempts, and accepts among unknown attempts. These denominators differ from pair-verification metrics.

Each run writes an `attendance.pt` checkpoint and `metrics.json` alongside the shared trainer artifacts.

## Score a window in the application

```python
from pathlib import Path

from src.recognition.attendance.model import AttendancePredictor

predictor = AttendancePredictor(
    Path("runs/attendance/<run>/attendance.pt"),
    encoder_sha256=embedder.model_version,
)
decision = predictor.predict(frame_embeddings, gallery_templates)
```

Supply a matrix of embeddings for one face track and a dictionary mapping enrolled identities to templates. Use `gallery_template` from the attendance features module to build each template. The predictor checks the gallery size, frame count, and encoder version. The app can treat `decision.accepted` as a condition for automatic attendance alongside its liveness decision. Low scores can trigger a retry or manual review.

VGGFace2 windows use separate still images. Use independent labeled video sessions to compare the frozen thresholds. The current still-image protocol has few wrong-enrolled candidates, and the sigmoid score reflects the simulated data distribution.
