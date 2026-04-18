# Command cheat sheet

Quick-reference for running the app day-to-day. For first-time setup (venv, npm install, API keys), see `README.md`.

Each section shows both **PowerShell (Windows)** and **bash / zsh (macOS/Linux)**. Run from the repo root.

---

## Start the backend

**PowerShell:**
```powershell
cd backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

**macOS/Linux:**
```bash
cd backend
./.venv/bin/python -m uvicorn app.main:app --reload
```

- Runs on http://localhost:8000
- `--reload` auto-restarts when `.py` files change
- Health check: open http://localhost:8000/api/health → should return `{"status":"ok"}`

To pick up a newly trained model, **Ctrl+C and restart** (the classifier caches in memory).

---

## Start the frontend

In a second terminal (same command on both platforms):

```bash
cd frontend
npm run dev
```

- Opens http://localhost:5173
- Requires the backend running for `/classify`, `/compose`, `/speak` to work

---

## Record training samples from your webcam

Adds to `data/landmarks.csv`. Your samples are weighted 50× by default during training, so your hand dominates the classifier.

**PowerShell:**
```powershell
cd backend
.\.venv\Scripts\python.exe training\capture.py
```

**macOS/Linux:**
```bash
cd backend
./.venv/bin/python training/capture.py
```

Controls (focus must be on the webcam window, not the terminal):
- `a` through `z`: record one sample labeled with that letter
- `ESC`: quit

Tip: hold the sign steady, then tap the letter key 15-20 times with slight wrist rotation, tilt, and distance changes between taps — variety matters.

---

## Retrain the classifier

After recording new samples, or to tune weighting:

**PowerShell:**
```powershell
cd backend
.\.venv\Scripts\python.exe training\train.py --skip-labels nothing,space,del
```

**macOS/Linux:**
```bash
cd backend
./.venv/bin/python training/train.py --skip-labels nothing,space,del
```

Useful flags:
- `--user-weight 100` — how much your own captures count vs Kaggle (default 50)
- `--trees 500` — more trees = slightly better accuracy, slower training (default 300)
- `--rebuild-image-cache` — re-extract landmarks from Kaggle images (default: reuse `data/landmarks_images.csv`)

Writes the new model to `backend/models/asl_classifier.joblib`. **Restart uvicorn** to load it.

---

## Download the Kaggle dataset (one-time)

Only needed if `data/asl_alphabet_train/` doesn't exist yet.

**PowerShell:**
```powershell
cd backend
.\.venv\Scripts\python.exe training\download_kaggle.py
```

**macOS/Linux:**
```bash
cd backend
./.venv/bin/python training/download_kaggle.py
```

Requires `~/.kaggle/kaggle.json` with your Kaggle API token.

---

## First-time landmark extraction from Kaggle

Only needed if `data/landmarks_images.csv` doesn't exist. Takes ~4 min with 11 workers.

**PowerShell:**
```powershell
cd backend
.\.venv\Scripts\python.exe training\train.py --images ..\data\asl_alphabet_train\asl_alphabet_train --skip-labels nothing,space,del --samples-per-class 300
```

**macOS/Linux:**
```bash
cd backend
./.venv/bin/python training/train.py --images ../data/asl_alphabet_train/asl_alphabet_train --skip-labels nothing,space,del --samples-per-class 300
```

---

## Smoke-test the backend

Health endpoint (same on all platforms):
```bash
curl http://localhost:8000/api/health
```

### Classify with fake landmarks (should return a letter)

**PowerShell:**
```powershell
$body = @{ landmarks = 1..21 | ForEach-Object { @{ x = 0.5; y = 0.5; z = 0.0 } } } | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri http://localhost:8000/api/classify -Method POST -ContentType "application/json" -Body $body
```

**macOS/Linux:**
```bash
python3 -c 'import json; print(json.dumps({"landmarks": [{"x":0.5,"y":0.5,"z":0.0}]*21}))' \
  | curl -s -X POST http://localhost:8000/api/classify \
      -H 'Content-Type: application/json' --data-binary @-
```

### Compose (needs `GEMINI_API_KEY` in `backend/.env`)

**PowerShell:**
```powershell
$body = @{ letters = "HELLOWORLD" } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/api/compose -Method POST -ContentType "application/json" -Body $body
```

**macOS/Linux:**
```bash
curl -s -X POST http://localhost:8000/api/compose \
  -H 'Content-Type: application/json' \
  -d '{"letters":"HELLOWORLD"}'
```

### Speak (needs `ELEVENLABS_API_KEY`, streams mp3 bytes)

**PowerShell:**
```powershell
$body = @{ text = "Hello world" } | ConvertTo-Json
Invoke-WebRequest -Uri http://localhost:8000/api/speak -Method POST -ContentType "application/json" -Body $body -OutFile test.mp3
```

**macOS/Linux:**
```bash
curl -s -X POST http://localhost:8000/api/speak \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello world"}' \
  --output test.mp3
```

---

## Reset training data

If your landmark captures went sideways and you want to start over:

**PowerShell:**
```powershell
Remove-Item data\landmarks.csv
# then re-run capture.py
```

**macOS/Linux:**
```bash
rm data/landmarks.csv
# then re-run capture.py
```

To re-extract Kaggle landmarks from scratch:

**PowerShell:**
```powershell
Remove-Item data\landmarks_images.csv
# then re-run train.py --images ... --rebuild-image-cache
```

**macOS/Linux:**
```bash
rm data/landmarks_images.csv
# then re-run train.py --images ... --rebuild-image-cache
```
