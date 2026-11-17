# QuanWatt HHL Power Flow

Professional layout:

```text
app/
  api.py              # ASGI / Vercel entry (re-exports backend.app)
  backend/            # FastAPI proxy, FDLS, Quapp HHL client
  frontend/           # React + Vite dashboard
  scripts/            # CLI helpers (demo / Quapp smoke)
  .env                # Quapp secrets (not committed)
  requirements.txt
  vercel.json
```

Flow: **UI → proxy (FDLS) → Quapp HHL (`Ax=b`) → live progress → HHL results**.

## Requirements

- Conda env `scquan` (or Python 3.12 + `requirements.txt`)
- Node.js 20+ / npm
- `.env` with `QUAPP_BASE_URL`, `QUAPP_ACCESS_TOKEN` (or `QUAPP_API_TOKEN`),
  `QUAPP_TENANT_ID`, `QUAPP_PROJECT_ID`, `QUAPP_FUNCTION_NAME`, `QUAPP_DEVICE_ID`

## Run locally

```bash
conda activate scquan
cd app
python -m uvicorn api:app --reload --port 8000
```

```bash
cd app/frontend
npm ci
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). Vite proxies `/api` → `:8000`.

## Scripts

```bash
cd app
python scripts/quapp_smoke.py      # test Quapp invoke/poll
python scripts/hhl_demo.py         # CLI power-flow on default 3-bus case
```

## Troubleshooting

- Dashboard blank: ensure uvicorn is on 8000, then refresh.
- Quapp auth errors: refresh tokens in `.env`.
- Slow solves: each Quapp call is ~1 min; default budget is 15 minutes.
