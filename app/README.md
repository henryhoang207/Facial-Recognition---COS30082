# Web app

Owner: Nam Khánh. Webcam capture, registration of new face IDs, live identity / liveness / emotion display, attendance log.

Registered faces and attendance logs go in `app/instance/`, which git ignores.

## Run the backend

From the repo root:

```bash
pip install -r requirements.txt
python -m app.run                              # real models from app/config.yaml
python -m app.run --config app/config.mock.yaml  # dummy models, no checkpoints needed
```

Open <http://127.0.0.1:8000/docs> for the interactive API. The first start downloads the YuNet detector (230 KB) into `app/weights/`.

Checkpoints are read from `model/` (see `app/config.yaml`), which git ignores:

```
model/Anti-spoofing/Liveness.pt
model/Classification/Model_A.pt
model/Emotion/emotionv4.pt
model/individual/Nam Khanh/mask_mobilenetv3s.pt             Individual > Nam Khánh (face mask)
model/individual/Gia Khanh/MLP-variable.pt                   Individual > Gia Khánh (attendance MLP)
model/individual/Hoan/adaptive-liveness-distilled.pt         Individual > Hoan (adaptive liveness)
model/individual/Duc Anh/best (2).pt                         Individual > Đức Anh (ASL signs, MLP + GRU)
model/individual/Thanh/model.safetensors                     Individual > Đức Thành (head pose)
model/individual/Duc/best.pt                                 Individual > Mạnh Đức (face parsing)
artifacts/hand_gestures/assets/hand_landmarker.task          MediaPipe hand tracker for Đức Anh's tab
```

All individual checkpoints are optional: a missing or broken one only disables its own tab, which then shows the
reason. Đức Anh's tab also needs `pip install -r requirements/hand-gestures.txt` and the
[hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)
file. MediaPipe installs `opencv-contrib-python` 5, which replaces the pinned OpenCV 4; afterwards run
`pip uninstall -y opencv-contrib-python` and `pip install --force-reinstall opencv-python==4.13.0.92`.

**Individual → Gia Khánh** runs the attendance MLP with Model A. It verifies five consecutive live frames against the current gallery and records accepted check-ins in the active session. Register at least one person before testing. Use the variable gallery checkpoint from `src.recognition.attendance.train_variable_gallery` on the individual-feature branch. The original checkpoint requires exactly six people.

To swap a model, change its `type` / `path` in `app/config.yaml` and call `POST /api/models/reload` (or restart). No code changes.

### ASL Alphabet on the Live page

The ASL card beside the camera shows the current letter and classifier confidence, including when only a hand is visible. Install the gesture dependencies with `python -m pip install -r requirements/hand-gestures.txt`. Both configurations use the latest completed Alphabet MLP from `artifacts/hand_gestures/alphabet/` and the MediaPipe asset at `artifacts/hand_gestures/assets/hand_landmarker.task`. To pin a run, set `models.gesture.path` to its directory containing `metadata.json` and `best.pt`. [Alphabet training and assets](../docs/hand-gestures/implementation-plan.md#asl-alphabet-follow-up)

Mock data for the UI: `python -m app.backend.seed --config app/config.mock.yaml --reset` (6 people, 2 sessions, 3 spoof events).

Tests: `python -m pytest tests/test_app_backend.py` (dummy models, about 2 s).

## Run the frontend

The built page is in `app/frontend/dist/`, so `python -m app.run` already serves the full UI at <http://127.0.0.1:8000>.
Open it on `localhost` (browsers only allow the camera on `localhost` or HTTPS).

To change the UI (needs Node.js 20.19+):

```bash
cd app/frontend
npm install
npm run dev      # http://localhost:5173, hot reload; API calls go to the backend on :8000
npm run build    # writes app/frontend/dist, served by python -m app.run
```

Pages include **Live** (camera, ASL recognition, check-ins, and anti-spoofing, top 3 identity matches, and emotion scores
for the selected face), **Register** (10-frame capture with a quality checklist),
**Attendance** (sessions, present / absent / blocked, CSV export), **Individual** (one tab per team member for their
own feature; members and features are listed in `app/frontend/src/members.ts`), **Settings** (thresholds, pipeline,
models, registered people).

Individual > Nam Khánh, **face mask detection**: camera with a mask / no-mask label on every face, counts, mask
probability for the selected face (this frame and smoothed) and the model's FMLD test results. It uses `mode=mask`,
which runs only the detector and the mask classifier (MobileNetV3-Small, `src/mask/`), with the YuNet score lowered to
`models.mask.detector_score` (0.6) because masked faces score lower. Nothing is logged to attendance in this mode.
The mask threshold can be changed in Settings.

The other tabs use the modes in `app/backend/individual/` (one module per feature, configured under
`models.individual` in `app/config.yaml`). None of them writes attendance.

| Tab | Mode | What runs |
| --- | --- | --- |
| Hoan | `challenge` | Distilled MobileNetV3-Small liveness on every frame. A face above the checkpoint threshold after 5 frames passes; a doubtful face must turn to its own left and right (random order, 20°, 10 s). The turn is measured from YuNet's eye and nose points, so a flat photo cannot follow it. Only a verified face is recognised. |
| Đức Anh | `gesture` | Đức Anh's code from `src/hand-gestures/` (MediaPipe hand landmarks, 2 s window, sequence features, MLP + GRU). No face detection. |
| Đức Thành | `head_pose` | `src/head_pose/predictor.py`: Thành's checkpoint rebuilt as HRNet-W48 (timm) + a 1/8-feature neck and 1024-d hidden layer; its 68-point 300W head is decoded from a 256 px crop (1.4 x the face box) and fitted to a generic 3D face with `solvePnP` for yaw / pitch / roll. Only the 1/8 feature the head needs is computed; on CPU it picks bf16 when that is faster, `device: auto` uses a GPU when there is one, and the model runs on a background thread so the stream keeps its frame rate (landmarks follow the face box between results); a result whose landmarks disagree with YuNet's 5 points (hand over the face) falls back to those points. Needs `timm` (in requirements.txt). |
| Mạnh Đức | `face_parsing` | `src/face_parsing/predictor.py`: the BiSeNet-style network (ResNet-18, 11 LaPa classes, 512 px) rebuilt from Mạnh Đức's checkpoint, which loads strictly. Kept at 512 px (lower sizes lose the eyes and lips); bf16 on CPUs that support it (same masks, about 2x faster), GPU with `device: auto`, and a background thread so the stream keeps its frame rate; the mask is coloured, sent as a small PNG and moved with the face box between results. |

## Layout

```
app/
├─ run.py                 entry point (uvicorn)
├─ config.yaml            real models;  config.mock.yaml: dummy models
├─ frontend/              React + TypeScript + Vite (src/pages, src/components, src/hooks)
└─ backend/
   ├─ main.py             FastAPI app, CORS for the Vite dev server, serves frontend/dist
   ├─ context.py          shared state: config, models, gallery, repository, services
   ├─ config.py           YAML config + runtime overrides (app/instance/settings.json)
   ├─ models/             registry (config -> models), YuNet/Haar detector, emotion GRU loader, dummies
   ├─ pipeline/           tracker.py (IoU tracking + smoothing), pipeline.py (per-frame flow)
   ├─ gallery.py          registered embeddings in RAM, cosine matching
   ├─ services/           registration (10 frames -> mean embedding), attendance + spoof logging
   ├─ db/                 SQLModel tables, SQLite engine (WAL), repository (all SQL)
   ├─ api/                rest.py (/api/*), stream.py (/ws/stream)
   └─ seed.py             mock data
```

## Per-frame flow (`/ws/stream?mode=pipeline`)

```
browser JPEG ──► decode ──► detect (YuNet) ──► track (IoU) ──► for each face:
                                                   liveness.predict_frame (224 px, scale 2.0)
                                                   ├─ spoof ─► spoof_events (once per track per 5 s) ─► stop
                                                   └─ live ──► embed (112 px) ─► gallery cosine match
                                                               ─► emotion (GRU state per track)
                                                               ─► 5 stable frames? ─► attendance (once per session)
◄── JSON result per frame (+ attendance_logged / spoof_blocked events)
```

- Labels are smoothed per track over `pipeline.smooth_window` frames: mean P(live), majority vote for identity, mean emotion probabilities.
- A person is logged only after `stable_frames` consecutive frames that are all live and all the same person.
- Thresholds: `null` in the config means "use the one stored in the checkpoint" (liveness 0.985, Model A 0.416). They can be changed at runtime with `PUT /api/settings`.
- Measured on a 4-thread CPU with the three real models: about 70 ms per frame with one face (detect 10, liveness 23, embed 13, emotion 23).

Each face in a `pipeline` result carries all three model outputs, including the top 3 gallery matches. The frame-level `gesture` field is independent of face detection. The same socket supports `register_check` for registration, `liveness`, `recognition`, and `emotion` for single-model runs, and `mask` and `attendance_mlp` for the Individual page.

WebSocket messages: client sends binary `4-byte big-endian frame_id + JPEG`; server answers with JSON `{"type": "result", "frame_id", "latency_ms", "faces": [{"track_id", "box": [x1, y1, x2, y2], "liveness", "identity", "emotion", "logged"}]}`, then any events.

## Database (`app/instance/faceguard.db`, SQLite)

```
persons 1──n embeddings          (vector float32 BLOB, dim, model_version, preprocessing_version)
persons 1──n attendance n──1 sessions     UNIQUE(session_id, person_id)
sessions 1──n spoof_events       (no person_id: a spoof never reaches recognition)
```

- Matching never queries the database: embeddings for the current `model_version` are loaded into `GalleryCache` at start and updated on register / delete.
- Registration saves wide face crops in `app/instance/faces/<person_id>/`. When the embedder changes, startup re-embeds everyone from those crops, so nobody has to register again.
- `INSERT ... ON CONFLICT DO NOTHING` on `UNIQUE(session_id, person_id)` keeps one attendance row per person per session.
- WAL mode lets the pipeline write while the attendance page reads.
- Exactly one session is active (`ended_at IS NULL`); starting a new one closes the previous one. If none exists, one is created on start.

## REST API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | models, thresholds, gallery size, active session |
| GET / PUT | `/api/settings` | thresholds, pipeline, feature flags (saved to `app/instance/settings.json`) |
| POST | `/api/models/reload` | rebuild models from `app/config.yaml` |
| GET | `/api/persons` | registered people (`embedding_ok` = usable with the current embedder) |
| POST | `/api/persons/register` | multipart `name`, `student_id`, `force`, `frames[]` (JPEG); 409 if the face is already registered |
| DELETE | `/api/persons/{id}` | remove a person, their embeddings, faces and attendance |
| GET / POST | `/api/sessions` | list / start a session |
| PATCH | `/api/sessions/{id}/end` | end a session |
| GET | `/api/sessions/{id}/attendance` | present + absent people, stats |
| GET | `/api/sessions/{id}/spoof-events` | blocked spoof attempts |
| GET | `/api/sessions/{id}/export.csv` | attendance as CSV |
| GET | `/api/snapshots/{path}` | spoof snapshot image |
