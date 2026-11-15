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
model/Anti-spoofing/liveness_mobilenetv3_2026-11-02.pt
model/Classification/Model_A.pt
model/Emotion/emotion_mobilenetv3_gru_gpu_2026-11-04.pt
```

To swap a model, change its `type` / `path` in `app/config.yaml` and call `POST /api/models/reload` (or restart). No code changes.

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

Pages: **Live** (camera plus one panel with all three models for the selected face: anti-spoofing, face recognition
with the top 3 matches, and all 7 emotion scores; check-ins below), **Register** (10-frame capture with a quality checklist),
**Attendance** (sessions, present / absent / blocked, CSV export), **Settings** (thresholds, pipeline, models, registered people).

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

Each face in a `pipeline` result carries all three model outputs, including the top 3 gallery matches. Other modes on the same socket: `register_check` (face count, size and liveness per frame, used by Register), and `liveness`, `recognition`, `emotion` to run one model on its own.

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
