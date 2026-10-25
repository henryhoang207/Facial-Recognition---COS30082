# Data

Datasets live in the shared Drive folder `FaceGuard/data/`, not in git. Thành keeps this list up to date.

| Folder on Drive | Task | Source | Notes |
| --- | --- | --- | --- |
| `face/` | Face recognition | Kaggle dataset from the project brief | `classification_data/`, `verification_data/`, `verification_pairs_val.txt` |
| `emotion/` | Emotion detection | FER2013 | 7 classes, 48×48 greyscale |
| `spoof_public/` | Anti-spoofing | _TBD (CelebA-Spoof subset or LCC-FASD)_ | |
| `spoof_recorded/` | Anti-spoofing | Our own recording sessions | One folder per person; keep test people out of training |

To work locally, copy or symlink a folder here, e.g. `data/face/`. Everything in `data/` except this file is ignored by git.
