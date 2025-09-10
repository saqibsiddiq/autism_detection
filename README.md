# 👁️‍🗨️ AutismDetect - Gaze Detector for Early ASD Screening

AutismDetect is an AI-powered eye-tracking system designed to support early and objective diagnosis of Autism Spectrum Disorder (ASD). It leverages webcam-based gaze detection and machine learning to identify atypical gaze patterns commonly associated with ASD in children.

## 🧭 Monorepo Structure

```
frontend/   # React app (Vite + Material UI) – main website
backend/    # Node.js API (Express + better-sqlite3)
app.py      # Streamlit dashboard (camera/gaze demo & analysis)
asddb.sqlite3  # SQLite database shared by API and Streamlit
```

## 🚀 Quick Start (Local)

Prereqs: Node 18+, Python 3.11+, a working webcam

1) Backend
```bash
cd backend
npm install
npm start
# Health check → http://localhost:5001/api/health
```

2) Frontend (dev)
```bash
cd frontend
npm install
npm run dev
# Open → http://localhost:5173
```

3) Streamlit UI
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r <(python - <<'PY'
import tomllib;print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))
PY
)
streamlit run app.py
# Open → http://localhost:8501
```

Notes:
- Frontend proxy forwards `/api/*` to `http://localhost:5001`.
- DB path defaults to `data/asddb.sqlite3`. Set `DATABASE_URL` to use Postgres or a different SQLite path.

## 🧩 Features
- Real‑time webcam preview and future MediaPipe gaze analysis (React)
- REST API for sessions, assessments, gaze data, and admin stats
- Streamlit dashboard for demo and exploration

## 🐳 Docker (optional)
We provide Dockerfiles and `docker-compose.yml`. If `docker compose` isn’t installed, you can run images individually (see comments in the compose file).

## ☁️ Deploy Streamlit (Render/Heroku/Streamlit Cloud)

CI: ![CI](https://github.com/saqibsiddiq/autism_detection/actions/workflows/ci.yml/badge.svg)

### Render (recommended)
1. Create new Web Service → connect this repo
2. Runtime: Python 3.12; Build Command: `pip install -r requirements.txt`
3. Start Command: `streamlit run app.py --server.port $PORT --server.headless true`
   - If Render defaults to Python 3.13, add `runtime.txt` with `python-3.11` (in repo) or set the environment to 3.11 in service settings.
4. For persistence: add a Disk mounted at `/opt/data` and set `DATABASE_URL=sqlite:////opt/data/asddb.sqlite3`
   - Or use managed Postgres and set `DATABASE_URL=postgresql+psycopg2://USER:PASSWORD@HOST:PORT/DBNAME`

### Streamlit Community Cloud
1. New app → select repo and `app.py`
2. Advanced settings → add `requirements.txt`
3. Deploy

### Heroku
1. `heroku create` → set stack to `heroku-22`
2. `heroku buildpacks:add heroku/python`
3. `git push heroku main`
4. `Procfile` already included

## 🗂️ API Endpoints (backend)
- `GET /api/health` – service status
- `POST /api/session` – create/update user session
- `POST /api/assessments` – create assessment
- `POST /api/assessments/:id/gaze` – save gaze data (array or aggregate)
- `POST /api/assessments/:id/results` – save final results
- `GET /api/admin/stats` – admin metrics

## 🧪 Development Tips
- If port conflicts occur, stop old processes using `lsof -ti tcp:PORT | xargs kill -9`.
- Hard refresh the frontend (Ctrl+F5) after config changes.

## 📄 License & Disclaimer
Educational research tool only. Not a medical device. No clinical use.



