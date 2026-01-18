import asyncio
import json
import logging
import os
import time
import contextlib
from dataclasses import dataclass
from typing import Iterable, Optional

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query
from sqlmodel import Field, Session, SQLModel, create_engine, select

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)


# -------- Settings --------
@dataclass
class Settings:
    golden_key: Optional[str]
    base_url: str
    messages_path: str
    auth_header: str
    auth_prefix: Optional[str]
    poll_seconds: float
    default_nodes: list[int]

    @classmethod
    def load(cls) -> "Settings":
        base_url = os.getenv("FUNPAY_BASE_URL", "https://funpay.com").rstrip("/")
        messages_path = os.getenv("FUNPAY_MESSAGES_PATH", "/chat/messages")
        auth_header = os.getenv("FUNPAY_AUTH_HEADER", "Authorization")
        auth_prefix = os.getenv("FUNPAY_AUTH_PREFIX", "Bearer")
        poll_seconds = float(os.getenv("FUNPAY_POLL_SECONDS", "3"))
        default_nodes_env = os.getenv("FUNPAY_DEFAULT_NODES", "")
        default_nodes: list[int] = []
        for part in default_nodes_env.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                default_nodes.append(int(part))
            except ValueError:
                logging.warning("Could not parse node id '%s' from FUNPAY_DEFAULT_NODES", part)

        golden_key = os.getenv("FUNPAY_GOLDEN_KEY")
        if not golden_key:
            logging.warning("FUNPAY_GOLDEN_KEY not set; poller will stay idle until provided.")

        return cls(
            golden_key=golden_key,
            base_url=base_url,
            messages_path=messages_path,
            auth_header=auth_header,
            auth_prefix=auth_prefix,
            poll_seconds=poll_seconds,
            default_nodes=default_nodes,
        )

    @property
    def auth_value(self) -> Optional[str]:
        if not self.golden_key:
            return None
        if self.auth_prefix:
            return f"{self.auth_prefix} {self.golden_key}"
        return self.golden_key


settings = Settings.load()


# -------- Database --------
engine = create_engine("sqlite:///data.db", connect_args={"check_same_thread": False})


class Node(SQLModel, table=True):
    id: int = Field(primary_key=True)
    last_id: Optional[int] = None


class Message(SQLModel, table=True):
    id: int = Field(primary_key=True)
    node_id: int = Field(primary_key=True)
    sender: Optional[str] = None
    body: Optional[str] = None
    created_at: Optional[str] = None
    raw: Optional[str] = None


def init_db() -> None:
    SQLModel.metadata.create_all(engine)
    if settings.default_nodes:
        with Session(engine) as session:
            for node_id in settings.default_nodes:
                if not session.get(Node, node_id):
                    session.add(Node(id=node_id))
            session.commit()


def get_session():
    with Session(engine) as session:
        yield session


# -------- FastAPI app --------
app = FastAPI(title="Funpay Automation", version="0.1.0")


@app.on_event("startup")
async def on_startup():
    init_db()
    app.state.stop_event = asyncio.Event()
    headers = {
        "Accept": "application/json",
        "User-Agent": "funpay-automation/0.1",
    }
    if settings.auth_value:
        headers[settings.auth_header] = settings.auth_value
    app.state.http_client = httpx.AsyncClient(base_url=settings.base_url, headers=headers, timeout=15)
    if settings.golden_key:
        app.state.poller_task = asyncio.create_task(poller())
    else:
        app.state.poller_task = None


@app.on_event("shutdown")
async def on_shutdown():
    app.state.stop_event.set()
    poller_task = getattr(app.state, "poller_task", None)
    if poller_task:
        poller_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await poller_task
    client: httpx.AsyncClient = app.state.http_client
    await client.aclose()


# -------- API routes --------
@app.get("/api/health")
async def health():
    return {"status": "ok", "polling": bool(settings.golden_key)}


@app.get("/api/nodes")
def list_nodes(session: Session = Depends(get_session)):
    nodes = session.exec(select(Node)).all()
    return nodes


class NodeCreate(SQLModel):
    node: int


@app.post("/api/nodes")
def add_node(payload: NodeCreate, session: Session = Depends(get_session)):
    existing = session.get(Node, payload.node)
    if existing:
        return {"status": "exists", "node": payload.node}
    session.add(Node(id=payload.node))
    session.commit()
    return {"status": "ok", "node": payload.node}


@app.get("/api/messages")
def list_messages(
    node: int = Query(..., description="Chat node id"),
    limit: int = Query(50, le=200, ge=1),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_session),
):
    stmt = (
        select(Message)
        .where(Message.node_id == node)
        .order_by(Message.id.desc())
        .offset(offset)
        .limit(limit)
    )
    return session.exec(stmt).all()


# -------- Poller --------
async def poller():
    if not settings.golden_key:
        return

    client: httpx.AsyncClient = app.state.http_client
    stop_event: asyncio.Event = app.state.stop_event
    log = logging.getLogger("poller")

    while not stop_event.is_set():
        start = time.time()
        try:
            with Session(engine) as session:
                nodes = session.exec(select(Node)).all()
                for node in nodes:
                    await poll_node(session, client, log, node)
        except Exception as exc:
            log.exception("Poller failure: %s", exc)
        elapsed = time.time() - start
        delay = max(settings.poll_seconds - elapsed, 0.5)
        await asyncio.sleep(delay)


async def poll_node(session: Session, client: httpx.AsyncClient, log: logging.Logger, node: Node):
    params = {"node": node.id}
    if node.last_id is not None:
        params["since_id"] = node.last_id

    try:
        resp = await client.get(settings.messages_path, params=params)
    except Exception as exc:
        log.warning("Node %s request error: %s", node.id, exc)
        return

    if resp.status_code == 429:
        log.warning("Node %s rate limited; backing off", node.id)
        await asyncio.sleep(5)
        return
    if resp.status_code == 401:
        log.error("Unauthorized (401) for node %s. Check FUNPAY_GOLDEN_KEY.", node.id)
        return

    try:
        resp.raise_for_status()
    except Exception as exc:
        log.warning("Node %s HTTP error: %s", node.id, exc)
        return

    try:
        payload = resp.json()
    except Exception as exc:
        log.warning("Node %s JSON decode error: %s", node.id, exc)
        return

    messages = payload.get("messages", payload)
    if not isinstance(messages, Iterable):
        log.warning("Node %s unexpected payload shape: %s", node.id, type(messages))
        return

    new_last_id = node.last_id or 0
    inserted = 0

    for item in messages:
        mid = coerce_int(item.get("id") or item.get("message_id") or item.get("mid"))
        if mid is None:
            continue
        sender = item.get("from") or item.get("sender") or item.get("user")
        body = item.get("text") or item.get("body") or item.get("message")
        created_at = item.get("created_at") or item.get("timestamp") or item.get("time")

        msg = Message(
            id=mid,
            node_id=node.id,
            sender=str(sender) if sender is not None else None,
            body=str(body) if body is not None else None,
            created_at=str(created_at) if created_at is not None else None,
            raw=json.dumps(item, ensure_ascii=False),
        )
        session.merge(msg)
        inserted += 1
        if mid > new_last_id:
            new_last_id = mid

    if inserted:
        node.last_id = new_last_id
        session.add(node)
        session.commit()
        log.info("Node %s: stored %s messages, last_id=%s", node.id, inserted, new_last_id)


def coerce_int(value: object) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    import uvicorn

    init_db()
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
