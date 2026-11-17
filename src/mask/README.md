# Face mask detection (individual feature, Nam Khánh)

Two classes on a face crop: `mask` / `no_mask`. Trained on FMLD (MAFA + WIDER FACE), about 50 000 real faces.

## Data (not in git, see `data/`)

```
data/FMLD_annotations/{train,test}/*.xml        FMLD labels   github.com/borutb-fri/FMLD
data/MAFA/{train,test}-images/images/*.jpg      MAFA photos   kaggle.com/rahulmangalampalli/mafa-data
data/WIDER/WIDER_{train,val}/images/<event>/    WIDER FACE    huggingface.co/datasets/CUHK-CSE/wider_face
```

`masked_face` and `incorrectly_masked_face` become `mask`, `unmasked_face` becomes `no_mask`.
FMLD's own train/test split is kept; val is 10% of the train photos (split by photo).

| Faces (>= 40 px) | mask | no_mask |
| --- | --- | --- |
| train + val | 24 903 | 24 398 |
| test | 4 992 | 7 275 |

## Run (from the repo root)

```bash
python -m src.mask.prepare                      # crop faces -> data/mask_faces (10-20 min, CPU)
python -m src.mask.train                        # MobileNetV3-Small, 8 epochs, early stop
python -m src.mask.train --arch mobilenetv3l    # MobileNetV3-Large, second candidate
python -m src.mask.train --arch mobilenetv4m    # MobileNetV4-Conv-Medium (2024), needs `pip install timm`
python -m src.mask.evaluate --checkpoint model/Mask/mask_mobilenetv3s_<date>.pt
python -m src.mask.evaluate --checkpoint ... --extra data/mask_webcam   # + own photos
python -m pytest tests/test_mask.py             # whole pipeline on fake data, about 10 s
```

Defaults fit a 4 GB GPU: mixed precision, batch 64 (Small) / 48 (Large, V4-Medium), 4 loader workers.
If you see `CUDA out of memory`, add `--batch-size 32`. If Windows stalls in the data loader, add `--workers 2`.

`--extra` takes a folder with `mask/` and `no_mask/` photos (for example webcam snapshots of the team);
faces are found with the app's YuNet detector and cropped exactly as in the app.

Outputs: `model/Mask/mask_<arch>_<date>.pt`, `reports/figures/mask_<arch>_training.png`,
`mask_<arch>_confusion.png`, `mask_results.csv` / `.md` (accuracy, macro F1, AUC, recall per class,
per source, CPU ms per face).

## Model

ImageNet MobileNetV3 (Small / Large, torchvision) and MobileNetV4-Conv-Medium (`timm`
`mobilenetv4_conv_medium.e500_r224_in1k`), new 2-way head, all layers fine-tuned. Input: `crop_face` at
224 px and scale 1.3 so the chin and nose stay in view. AdamW, warmup then cosine LR, label smoothing
0.05, light augmentation (crop >= 80% of the area, flip, colour jitter, grayscale, blur). The checkpoint
with the best val macro F1 is kept.

## Use in the app

A MobileNetV4 checkpoint also needs `timm` installed where the app runs.

```python
from src.mask.predictor import MaskPredictor
mask = MaskPredictor("model/Mask/mask_mobilenetv3s_<date>.pt")
label, prob_mask = mask.predict_frame(frame, box)
```

Note for the app: YuNet at score 0.8 missed 3 of 16 masked faces in a small FMLD sample (1 of 16 at 0.5),
so lower the detector score to about 0.6 when this feature is on.
