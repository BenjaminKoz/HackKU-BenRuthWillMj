# HackKU 2026 — ASL Translator

Real-time American Sign Language translator. Webcam → hand landmarks → letter → English sentence → spoken voice.

**Team:** Benjamin Kozlowski · Ruth Higgason · MJ McGee · William Hedges

## Architecture

```
Browser (React + Vite)                     Backend (FastAPI)
┌──────────────────────────┐               ┌──────────────────────────────┐
│  Webcam                  │               │                              │
│    ↓                     │               │                              │
│  MediaPipe Hands (WASM)  │  landmarks →  │  /classify   RandomForest    │
│    ↓                     │   letter  ←   │                              │
│  Letter buffer           │  letters  →   │  /compose    Gemini 2.5      │
│    ↓                     │  sentence ←   │                              │
│  Sentence                │  text     →   │  /speak      ElevenLabs      │
│    ↓                     │   mp3     ←   │                              │
│  🔊 Audio playback       │               │                              │
└──────────────────────────┘               └──────────────────────────────┘
```

**Hand tracking runs in the browser** via MediaPipe Tasks Vision — no GPU, no upload of raw video. Only 21 anonymized landmark points go to the backend.

**MLH tracks targeted:** Best Use of Gemini API · Best Use of ElevenLabs.

## Quick start

### 1. Backend

```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

# copy and fill in API keys
cp ../.env.example .env
# edit .env: GEMINI_API_KEY, ELEVENLABS_API_KEY

uvicorn app.main:app --reload --port 8000
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173.

### 2.1 Frontend env for production

For local dev, leave `VITE_API_BASE_URL` unset and Vite will proxy `/api/*` to
`http://127.0.0.1:8000`.

For a Vercel deployment, set `VITE_API_BASE_URL` in the Vercel project settings
before building. It must point at your deployed backend origin, for example:

```bash
VITE_API_BASE_URL=https://your-backend-host.example.com
```

If this variable is missing in production, the frontend will call relative
paths like `/api/sentence-game/generate` on the Vercel app itself, and those
requests will fail unless you also deploy the FastAPI backend behind the same
origin.

### 3. Train the classifier

The `/classify` endpoint needs a trained model at `backend/models/asl_classifier.joblib`. Two options:

**Option A — record your own (fastest, ~10 min):**

```bash
cd backend
pip install -r training/requirements-capture.txt
python training/capture.py
# Hold each ASL letter; press that letter key to record a sample. Aim for 20+ per letter.
python training/train.py
```

**Option B — Kaggle ASL Alphabet dataset (~20 min extract, best generalization):**

```bash
cd backend
pip install -r training/requirements-capture.txt  # mediapipe + opencv
pip install kaggle                                # for auto-download; skip if downloading manually

# Auto-download (needs a Kaggle API token at ~/.kaggle/kaggle.json)
python training/download_kaggle.py
# -- or -- manually grab https://www.kaggle.com/datasets/grassknoted/asl-alphabet
# and extract into data/asl_alphabet_train/<LABEL>/*.jpg

# Extract landmarks (cached) + train. 300 samples/class keeps this ~10 min.
python training/train.py --images ../data/asl_alphabet_train --skip-labels nothing,space,del
```

The first run of `train.py --images` writes `data/landmarks_images.csv`. Subsequent runs re-use that cache, so retraining after tweaks is seconds, not minutes. Pass `--rebuild-image-cache` to force re-extraction.

**Option C — hybrid (most robust demo):** do B, then have each teammate run A (capture.py) for ~5 samples per letter. `train.py` automatically merges both CSVs.

## API keys

- **Gemini** — free at https://aistudio.google.com/apikey
- **ElevenLabs** — free tier at https://elevenlabs.io/app/settings/api-keys

Default voice is Sarah (`EXAVITQu4vr4xnSDxMaL`). Override via `ELEVENLABS_VOICE_ID`.

## Project layout

```
backend/
  app/
    main.py           FastAPI app + CORS
    routes/
      classify.py     POST /api/classify
      compose.py      POST /api/compose   (Gemini)
      speak.py        POST /api/speak     (ElevenLabs)
    services/
      classifier.py   Loads joblib model, normalizes landmarks
  training/
    capture.py        Webcam landmark recorder → CSV
    train.py          Train RandomForest on landmarks
  models/             Trained classifier (git-ignored)

frontend/
  src/
    App.tsx                  Main UI
    components/Webcam.tsx    Camera + landmark overlay
    hooks/useHandLandmarker  MediaPipe Tasks Vision hook
    lib/api.ts               Fetch helpers

data/                        Training data (git-ignored)
```

## Word mode

The translator has a **Letters / Words** toggle. Word mode recognizes whole
signs (e.g. HELLO, THANKS) instead of spelling them letter-by-letter.

Starter vocabulary (10 one-handed signs): `HELLO, THANKS, YES, NO, PLEASE,
SORRY, LOVE, HELP, NAME, YOU`.

### Capture word clips

```bash
cd backend
python training/capture.py
# Press TAB to enter word mode.
# Press 1–9 or 0 to select a target word (shown on screen).
# Press SPACE to record a 90-frame (~3s) clip — sign the word the instant you press.
# Aim for 15+ clips per word per signer. Vary hand position and speed slightly.
# Press TAB again to return to letter capture.
```

Clips are appended to `data/landmarks_words.csv`.

### Train the word model

```bash
cd backend
python training/train_words.py
# Writes backend/models/asl_words_classifier.joblib
```

The word model is loaded lazily by the backend. Without it, Word mode returns
blank predictions — letter mode is unaffected.

## Notes on the ML approach

- Hand landmarks are normalized: translate to wrist, scale by middle-finger MCP distance. This makes the classifier invariant to where your hand is on screen and how far from the camera.
- The RandomForest is cheap to train (seconds) and plenty accurate for 24 static ASL letters. J and Z require motion, handled by a separate motion classifier that fires when the frame buffer contains a completed gesture.
- The frontend waits for a letter to hold steady across 4 frames above 60% confidence before committing it to the buffer — cuts down on misreads during finger-shuffle transitions.
- Word mode uses the same "clip + keyframes + wrist trajectory" recipe as the motion classifier, scaled up to 90 frames and more keyframes so multi-phase signs stay separable.
