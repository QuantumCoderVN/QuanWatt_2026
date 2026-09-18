# HHL Power Flow Dashboard

Run the dashboard locally to compare quantum HHL and classical power flow results.

## Requirements

- Python 3.12 with pip
- Node.js 20 or newer with npm
- A local copy of this repository

The commands below use Linux/macOS. Start each terminal in the repository root (`QuanWatt_2026`).

## 1. Start the backend

In your first terminal:

```bash
cd app
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn api:app --reload --port 8000
```

On Windows PowerShell, use `py -3.12 -m venv .venv` to create the environment and `.\.venv\Scripts\Activate.ps1` to activate it.

Keep this terminal running. Check the backend at [http://localhost:8000/api/health](http://localhost:8000/api/health); it should show `"status": "healthy"`.

## 2. Start the frontend

Open a second terminal in the repository root:

```bash
cd app/app/frontend
npm ci
npm run dev
```

Keep this terminal running too. Open [http://localhost:3000](http://localhost:3000), or the local URL printed by Vite if that port is busy.

The frontend connects to the backend on port 8000 automatically. No environment file is needed for local use.

## 3. Use the app

1. Start with the default 3-bus network and solver settings.
2. Click **Run Power Flow Analysis** and wait for the simulation to finish. Progress appears in the backend terminal.
3. Compare the HHL and classical results, convergence charts, and bus voltages.
4. Edit the buses, generators, branches, or solver settings, then run the analysis again.

## Stop and restart

Press `Ctrl+C` in each terminal to stop the app.

For later runs, activate the existing Python environment and start the backend, then run `npm run dev` in the frontend folder. Repeat dependency installation only when the requirements or package lock file changes.

## Troubleshooting

- **The dashboard cannot load:** Make sure the backend is running on port 8000, check the health link above, and refresh the dashboard.
- **Python reports a missing module:** Activate `app/.venv` and rerun `python -m pip install -r requirements.txt` from the outer `app` folder.
- **The analysis takes a while:** HHL runs a quantum simulation locally. Try the default 3-bus case first and check the backend terminal for progress or errors.
