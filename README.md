# Funpay Automation (Golden Key)

Minimal FastAPI control panel scaffold that polls Funpay chat messages for configured node IDs using a Golden Key token, stores them in SQLite, and exposes simple REST endpoints.

## What it does
- Polls `GET <FUNPAY_MESSAGES_PATH>?node=<id>&since_id=<last_id>` on `FUNPAY_BASE_URL` with your Golden Key header.
- Persists messages and last seen IDs per node in `data.db` (SQLite).
- REST API:
  - `GET /api/health`
  - `GET /api/nodes`
  - `POST /api/nodes` `{ "node": 123 }`
  - `GET /api/messages?node=123&limit=50&offset=0`

## Quick start (local)
```bash
python -m venv .venv
.venv\Scripts\activate             # on Windows
pip install -r requirements.txt
cp .env.example .env               # fill FUNPAY_GOLDEN_KEY and other values
uvicorn app:app --reload
```
Open `http://localhost:8000/docs` to poke the API.

## Configuration (`.env`)
- `FUNPAY_GOLDEN_KEY` – your Golden Key (keep secret; do not commit).
- `FUNPAY_BASE_URL` – default `https://funpay.com`.
- `FUNPAY_MESSAGES_PATH` – messages endpoint path (inspect DevTools; default `/chat/messages`).
- `FUNPAY_AUTH_HEADER` – header name to send the key (e.g., `Authorization` or `X-Access-Token`).
- `FUNPAY_AUTH_PREFIX` – optional prefix (default `Bearer`, set empty to send raw key).
- `FUNPAY_POLL_SECONDS` – poll interval seconds.
- `FUNPAY_DEFAULT_NODES` – comma-separated node IDs to start polling on boot.

## Deployment (Railway)
- Create a Railway service, set secrets from `.env.example` (especially `FUNPAY_GOLDEN_KEY`).
- Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`.
- Ensure `data.db` persistence if you need durable storage; otherwise it will reset on each deploy.

## Notes
- Replace `FUNPAY_MESSAGES_PATH`/params to match the real Funpay chat API you see in your browser’s network tab.
- Message IDs are treated as integers; if the API returns non-numeric IDs, adjust `coerce_int` and the schema.
- The poller logs rate limits (429) and unauthorized (401) and will skip updates until fixed.
