# LTV Software

Software-first development workspace for the Lookout Tower Vehicle. The initial increment is a local dashboard and API using simulated vehicle data; it does not connect to cameras, speakers, or face galleries.

## Development order

1. Establish backend API, roles, configuration, audit events, and offline event handling.
2. Build the operator and control-room dashboard against mock/replay data.
3. Add recorded-video replay and measurable acceptance checks.
4. Integrate and validate analytics models.
5. Connect vehicle hardware after API and safety workflows are stable.

Stage 2 now has an evaluation-only CSRNet candidate and a script for benchmarking on a public held-out crowd-count dataset. It is not connected to the dashboard or alert generator. See [the crowd-density benchmark guide](docs/crowd-density-benchmark.md).

AI outputs are recommendations for an officer. The app must not broadcast announcements or confirm identities automatically.

## Initial structure

- `backend/`: Python API service, SQLite demo data, and initial audit events.
- `frontend/`: TypeScript React dashboard shell.
- `docs/`: requirements traceability and architecture notes.

## Prerequisites

- Python 3.11+
- Node.js 20+

## Run locally

Backend:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

For this local demo, set `LTV_DEMO_PASSWORD` before the first backend start. The backend creates `operator`, `supervisor`, and `administrator` demo accounts with that same password (default: `demo-only-change-me`). These shared demo accounts are for local development only, not deployment.

Frontend (in another terminal):

```bash
cd frontend
npm install
npm run dev
```

The frontend uses simulated readings until the API integration step. Do not use this scaffold for live operations.

## Local replay and synthetic scenarios

After signing in, load a recorded video in the dashboard. Playback stays in the browser and is not uploaded. The player supports pause/play, playback speed, and repeat. Video playback is not connected to an AI model.

The camera setup panel can save drawn zones and a four-corner ground rectangle. When the rectangle's real width and depth are entered, the API applies a basic planar perspective transform and reports estimated zone areas. This is a setup prototype: it does not correct 360° lens distortion, verify the camera view, count people, or produce real people/m² values. Treat all area results as experimental until checked against measured ground areas for the actual camera.

Use the scenario controls to exercise dashboard behavior with made-up readings: Steady, Crowd rising, Surge, and Critical density. The dashboard refreshes these readings every two seconds. The synthetic alert flow uses the requirement density bands, suppresses repeat alerts of the same level in a zone for 60 seconds, supports officer acknowledgement, and marks an unacknowledged red alert escalated after 30 seconds while the simulation is running. Starting or stopping a scenario resets its current synthetic readings and alert queue. Scenario start/stop and acknowledgements are recorded in the audit log.

These scenarios are for software demonstration only. They do not analyze the video, use an AI model, establish real-world calibration, or represent actual crowd conditions. The synthetic simulator state resets if the backend restarts.

The API creates `backend/ltv.sqlite3` on first startup. Set `LTV_DATABASE_PATH` to use another local database file. Sign-in sessions expire after eight hours. The local demo cookie is configured for localhost development; production authentication, secure cookie settings, account provisioning, and stronger audit-log protections remain future work.

Supervisors and administrators can search the audit log and download filtered CSV/PDF exports. Administrators can configure retention values and record/revoke written watchlist authorizations with an expiry. The configured log retention is enforced; clip and face-image cleanup will become enforceable when those storage features are implemented. Recording a watchlist authorization does not enable matching; face/watchlist search is not implemented in this scaffold.
