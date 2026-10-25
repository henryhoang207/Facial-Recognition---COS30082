# Data

Datasets live in the shared Drive folder `FaceGuard/data/`, not in git. Thành keeps this list up to date.

| Folder on Drive | Task | Source | Notes |
| --- | --- | --- | --- |
| `face/` | Face recognition | Kaggle dataset from the project brief | `classification_data/`, `verification_data/`, `verification_pairs_val.txt` |
| `emotion/` | Emotion detection | FER2013 | 7 classes, 48×48 greyscale |
| `spoof_public/` | Anti-spoofing | CelebA-Spoof | Keep the original layout: `CelebA_Spoof/Data/{train,test}/<id>/{live,spoof}/` with the `_BB.txt` files, plus `CelebA_Spoof/metas/` |
| `spoof_recorded/` | Anti-spoofing | Our own recording sessions | Layout below; keep test people out of training |

Zip each anti-spoofing folder as `spoof_public.zip` / `spoof_recorded.zip` too: the Colab notebook unzips them to local disk, which is much faster than reading from Drive.

**Recorded spoof data layout.** One folder per person, one subfolder per class. `live` means a real face; any other name is an attack type and becomes a row in the per-attack results. Images or videos both work (every 10th video frame is kept). Faces are found automatically.

```
spoof_recorded/
  hoan/
    live/      webcam videos or photos of the real face
    print/     the camera pointed at a printed photo of hoan
    replay/    the camera pointed at a phone or laptop screen showing hoan
  thanh/
    ...
```

To work locally, copy or symlink a folder here, e.g. `data/face/`. Everything in `data/` except this file is ignored by git.
