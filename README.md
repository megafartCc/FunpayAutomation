# Funpay Automation (React UI + FastAPI)

FastAPI backend plus a React single-page UI. It mirrors the `funpayapi` flow (https://github.com/controlll/funpayapi): logs in with a Golden Key cookie, polls the `runner/` endpoint for chats, stores messages in SQLite, and lets you view/send chats and manage lot prices from the browser.

## Features
- Golden Key is entered in the UI (not hardcoded). Optional initial key via env for auto-start.
- Chat polling via `/runner/` `chat_node` objects; messages stored in `data.db`.
- Send replies from the UI.
- Lots panel: view your offers and update prices.
- REST API surface:
  - `GET /api/session`, `POST /api/session` `{ golden_key, base_url? }`
  - `GET /api/nodes`, `POST /api/nodes` `{ node }`
  - `GET /api/messages?node=...&limit=...`, `POST /api/messages/send` `{ node, message }`
  - `GET /api/lots`, `POST /api/lots/price` `{ node, offer, price }`
  - `GET /api/accounts`, `POST /api/accounts` `{ label?, username, password, steam_id?, login_status?, twofa_otp? }`
  - `PUT /api/accounts/{id}` `{ label?, username?, password?, steam_id?, login_status?, twofa_otp? }`
  - `POST /api/accounts/{id}/login` `{ guard_code?, email_code? }`, `POST /api/accounts/{id}/logout`
  - `GET /api/accounts/{id}/status`

## Quick start (backend)
```bash
python -m venv .venv
.venv\Scripts\activate             # Windows
pip install -r requirements.txt
cp .env.example .env               # optional: set FUNPAY_BASE_URL, poll seconds; Golden Key can stay blank
uvicorn app:app --reload
```
API docs: `http://localhost:8000/docs`
UI (requires built assets in `static/`): `http://localhost:8000/`

## Frontend (React, Vite)
```bash
cd frontend
npm install
npm run dev        # proxies /api to :8000
npm run build      # outputs to ../static
```
The backend serves the built assets from `/static` at the root path.

## Env (`.env`)
- `FUNPAY_BASE_URL` – default `https://funpay.com`
- `FUNPAY_POLL_SECONDS` – polling interval for runner
- `FUNPAY_DEFAULT_NODES` – comma-separated user IDs to start watching (optional)
- `FUNPAY_GOLDEN_KEY` – optional initial key to auto-start session; otherwise enter via UI
- `MYSQLHOST`, `MYSQLPORT`, `MYSQLUSER`, `MYSQLPASSWORD`, `MYSQLDATABASE` – MySQL connection for persistent storage

## Railway
- Build frontend before deploy (`npm run build` in `frontend`) so `static/` has assets, or add a Nixpacks phase to build the frontend.
- Set env vars above as Railway secrets; start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`.
- Add a Volume or external DB if you need `data.db` persistence.

## Notes
- Messages are parsed from runner HTML; if Funpay markup changes, adjust the selectors in `parse_message_html`.
- Price updates mimic `funpayapi` by posting the offer edit form to `/lots/offerSave` with the current CSRF token.
