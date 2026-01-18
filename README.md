# Funpay Automation (Golden Key)

Minimal FastAPI control panel with a built-in web UI. It mirrors the `funpayapi` approach (https://github.com/controlll/funpayapi): logs in with your Golden Key cookie, polls the `runner/` endpoint for chat messages, stores them in SQLite, and exposes REST + a UI.

## What it does
- Loads Funpay app data (`userId`, `csrf-token`) with your `golden_key` cookie.
- Polls `/runner/` with `chat_node` objects for configured user IDs and parses returned message HTML.
- Persists messages and last seen IDs per chat in `data.db`.
- REST:
  - `GET /api/health`
  - `GET /api/nodes`
  - `POST /api/nodes` `{ "node": 123 }` (user IDs to watch)
  - `GET /api/messages?node=123&limit=50&offset=0`
  - `POST /api/messages/send` `{ "node": 123, "message": "hi" }`
- Web UI: served at `/` from `static/index.html` (add nodes, view messages, send replies).

## Quick start (local)
```bash
python -m venv .venv
.venv\Scripts\activate             # on Windows
pip install -r requirements.txt
cp .env.example .env               # fill FUNPAY_GOLDEN_KEY
uvicorn app:app --reload
```
Open `http://localhost:8000/docs`.

## Config (`.env`)
- `FUNPAY_GOLDEN_KEY` – your Golden Key (secret).
- `FUNPAY_BASE_URL` – default `https://funpay.com`.
- `FUNPAY_POLL_SECONDS` – poll interval seconds.
- `FUNPAY_DEFAULT_NODES` – comma-separated *user IDs* to watch on startup.

## Railway
- Set env vars above as Railway secrets; start command `uvicorn app:app --host 0.0.0.0 --port $PORT`.
- Add a Volume or external DB if you need `data.db` persistence.

## Notes
- Uses the same runner payload shape as `funpayapi`. The first time a node is added, the poller fetches the current last message ID to avoid replaying history.
- Message IDs are integers; adjust schema/parsing if Funpay changes formats.
