import asyncio
import contextlib
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Iterable, Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from sqlmodel import Field, Session, SQLModel, create_engine, select

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)


# -------- Settings --------
@dataclass
class Settings:
    base_url: str
    poll_seconds: float
    default_nodes: list[str]
    initial_key: Optional[str]

    @classmethod
    def load(cls) -> "Settings":
        base_url = os.getenv("FUNPAY_BASE_URL", "https://funpay.com").rstrip("/")
        poll_seconds = float(os.getenv("FUNPAY_POLL_SECONDS", "3"))
        default_nodes_env = os.getenv("FUNPAY_DEFAULT_NODES", "")
        default_nodes: list[str] = []
        for part in default_nodes_env.split(","):
            part = part.strip()
            if part:
                default_nodes.append(part)

        initial_key = os.getenv("FUNPAY_GOLDEN_KEY")
        if initial_key:
            logging.info("Initial FUNPAY_GOLDEN_KEY found; will start session automatically.")

        return cls(
            base_url=base_url,
            poll_seconds=poll_seconds,
            default_nodes=default_nodes,
            initial_key=initial_key,
        )


settings = Settings.load()


# -------- Helpers --------
def get_users_node(id1: str, id2: str) -> str:
    a, b = sorted([str(id1), str(id2)], key=lambda x: int(x))
    return f"users-{a}-{b}"


def coerce_int(value: object) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass
class ParsedMessage:
    username: Optional[str]
    body: Optional[str]
    created_at: Optional[str]


def parse_message_html(html: str) -> ParsedMessage:
    soup = BeautifulSoup(html or "", "html.parser")
    body_el = soup.select_one(".chat-msg-text")
    date_el = soup.select_one(".chat-msg-date")
    username_el = soup.select_one(".media-user-name")
    return ParsedMessage(
        username=username_el.get_text(strip=True) if username_el else None,
        body=body_el.get_text("\n", strip=True) if body_el else None,
        created_at=(date_el.get("title") or date_el.get_text(strip=True)) if date_el else None,
    )


# -------- Database --------
engine = create_engine("sqlite:///data.db", connect_args={"check_same_thread": False})


class Node(SQLModel, table=True):
    id: str = Field(primary_key=True)
    last_id: Optional[int] = None


class Message(SQLModel, table=True):
    id: int = Field(primary_key=True)
    node_id: str = Field(primary_key=True)
    author: Optional[str] = None
    username: Optional[str] = None
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


async def start_session(golden_key: str, base_url: Optional[str] = None):
    base = (base_url or settings.base_url).rstrip("/")
    settings.base_url = base
    # stop existing poller
    poller_task = getattr(app.state, "poller_task", None)
    if poller_task:
        poller_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await poller_task
    stop_event = asyncio.Event()
    app.state.stop_event = stop_event

    # close previous client if any
    prev_client: Optional[FunpayClient] = getattr(app.state, "fp_client", None)
    if prev_client:
        with contextlib.suppress(Exception):
            await prev_client.close()

    client = FunpayClient(base, golden_key)
    app.state.fp_client = client
    await client.start()
    app.state.poller_task = asyncio.create_task(poller())
    logging.info("Session started for user %s", client.user_id)
    return client


def get_session():
    with Session(engine) as session:
        yield session


# -------- Funpay client (funpayapi-like) --------
class FunpayClient:
    def __init__(self, base_url: str, golden_key: str):
        self.base_url = base_url
        self.golden_key = golden_key
        self.client: Optional[httpx.AsyncClient] = None
        self.user_id: Optional[str] = None
        self.csrf_token: Optional[str] = None

    async def start(self):
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=20,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/129.0.0.0 Safari/537.36",
            },
        )
        if self.golden_key:
            domain = urlparse(self.base_url).hostname
            self.client.cookies.set("golden_key", self.golden_key, domain=domain)
        await self.load_app_data()

    async def close(self):
        if self.client:
            await self.client.aclose()

    async def ensure_ready(self):
        if not self.client:
            await self.start()
        if not self.user_id or not self.csrf_token:
            await self.load_app_data()

    async def load_app_data(self):
        await self._ensure_client()
        resp = await self.client.get("/")
        resp.raise_for_status()
        self._capture_cookies(resp)
        data = extract_app_data(resp.text)
        self.user_id = str(data.get("userId"))
        self.csrf_token = data.get("csrf-token")
        if not self.user_id:
            raise RuntimeError("Failed to read userId from app data.")

    async def get_user_last_message(self, other_user_id: str) -> Optional[int]:
        await self.ensure_ready()
        chat_node = get_users_node(self.user_id, other_user_id)
        resp = await self.client.get(f"/chat/?node={chat_node}")
        resp.raise_for_status()
        self._capture_cookies(resp)
        soup = BeautifulSoup(resp.text, "html.parser")
        items = soup.select(".chat-msg-item.chat-msg-with-head")
        if not items:
            return None
        last = items[-1]
        msg_id_attr = last.get("id")
        if not msg_id_attr:
            return None
        try:
            return int(msg_id_attr.split("-")[1])
        except Exception:
            return None

    async def runner(self, objects: list[dict], request: Optional[dict] = None) -> dict:
        await self.ensure_ready()
        data = {
            "objects": json.dumps(objects, ensure_ascii=False),
            "request": json.dumps(request, ensure_ascii=False) if request is not None else "false",
            "csrf_token": self.csrf_token or "",
        }
        headers = {
            "accept": "application/json, text/javascript, */*; q=0.01",
            "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
            "x-requested-with": "XMLHttpRequest",
        }
        resp = await self.client.post("/runner/", data=data, headers=headers)
        self._capture_cookies(resp)
        resp.raise_for_status()
        return resp.json()

    async def _ensure_client(self):
        if not self.client:
            await self.start()

    def _capture_cookies(self, resp: httpx.Response):
        # httpx client keeps cookies automatically; nothing to do unless golden_key missing.
        pass

    async def send_chat_message(self, other_user_id: str, content: str, last_message: Optional[int] = None):
        await self.ensure_ready()
        chat_node = get_users_node(self.user_id, other_user_id)

        if last_message is None:
            try:
                last_message = await self.get_user_last_message(other_user_id)
            except Exception:
                last_message = 0
        last_message = int(last_message or 0)

        objects = [
            {"type": "orders_counters", "id": self.user_id, "tag": "py", "data": True},
            {"type": "chat_counter", "id": self.user_id, "tag": "py", "data": True},
            {
                "type": "chat_node",
                "id": chat_node,
                "data": {"node": chat_node, "last_message": last_message, "content": ""},
            },
        ]
        request = {
            "action": "chat_message",
            "data": {"node": chat_node, "last_message": last_message + 1, "content": content},
        }
        payload = await self.runner(objects, request=request)
        return payload

    async def get_offers(self) -> list[dict]:
        await self.ensure_ready()
        resp = await self.client.get(f"/users/{self.user_id}/")
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        result: list[dict] = []
        for offer_block in soup.select(".mb20 .offer"):
            title = offer_block.select_one(".offer-list-title h3 a")
            if not title:
                continue
            group_name = title.get_text(strip=True)
            href = title.get("href", "")
            node = href.strip("/").split("/")[-1] if href else None
            if not node:
                continue
            group = {"group_name": group_name, "node": node, "offers": []}
            for item in offer_block.select(".tc-item"):
                name_el = item.select_one(".tc-desc-text")
                price_el = item.select_one(".tc-price")
                href_item = item.get("href") or ""
                offer_id = None
                if "id=" in href_item:
                    try:
                        offer_id = href_item.split("id=")[1].split("&")[0]
                    except Exception:
                        offer_id = None
                group["offers"].append(
                    {
                        "name": name_el.get_text(strip=True) if name_el else "",
                        "price": price_el.get_text(strip=True) if price_el else "",
                        "id": offer_id,
                    }
                )
            result.append(group)
        return result

    async def update_price(self, node: str, offer: str, price: float):
        await self.ensure_ready()
        route = f"/lots/offerEdit?node={node}&offer={offer}"
        resp = await self.client.get(route)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        form = soup.select_one(".form-offer-editor")
        if not form:
            raise RuntimeError("Offer form not found.")
        price = float(price)

        data: dict[str, str] = {}
        for field in form.select("input, textarea, select"):
            name = field.get("name")
            if not name:
                continue
            tag = field.name
            if tag == "select":
                selected = field.select_one("option[selected]") or field.select_one("option")
                if selected:
                    data[name] = selected.get("value", "")
            elif field.get("type") in ["checkbox", "radio"]:
                if field.has_attr("checked"):
                    data[name] = field.get("value", "on")
            else:
                data[name] = field.get("value", "")

        data["price"] = str(price)
        if self.csrf_token:
            data["csrf_token"] = self.csrf_token
        headers = {
            "accept": "application/json, text/javascript, */*; q=0.01",
            "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
            "x-requested-with": "XMLHttpRequest",
        }
        save = await self.client.post("/lots/offerSave", data=data, headers=headers)
        save.raise_for_status()
        return True


def extract_app_data(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    body = soup.find("body")
    if not body:
        raise RuntimeError("No <body> found in Funpay response.")
    raw = body.get("data-app-data")
    if not raw:
        raise RuntimeError("data-app-data not found on Funpay page.")
    return json.loads(raw)


# -------- FastAPI app --------
app = FastAPI(title="Funpay Automation", version="0.2.0")


@app.on_event("startup")
async def on_startup():
    init_db()
    app.state.stop_event = asyncio.Event()
    app.state.fp_client: Optional[FunpayClient] = None
    app.state.poller_task = None
    if settings.initial_key:
        await start_session(settings.initial_key, settings.base_url)


@app.on_event("shutdown")
async def on_shutdown():
    app.state.stop_event.set()
    poller_task = getattr(app.state, "poller_task", None)
    if poller_task:
        poller_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await poller_task
    client: FunpayClient = app.state.fp_client
    if client:
        await client.close()


# -------- API routes --------
@app.get("/api/health")
async def health():
    client: Optional[FunpayClient] = getattr(app.state, "fp_client", None)
    return {
        "status": "ok",
        "polling": bool(client),
        "userId": getattr(client, "user_id", None),
    }


@dataclass
class SessionStatus:
    polling: bool
    userId: Optional[str]
    baseUrl: str


class SessionCreate(SQLModel):
    golden_key: str
    base_url: Optional[str] = None


@app.get("/api/session")
async def session_status():
    client: Optional[FunpayClient] = getattr(app.state, "fp_client", None)
    return SessionStatus(
        polling=bool(client),
        userId=getattr(client, "user_id", None),
        baseUrl=settings.base_url,
    ).__dict__


@app.post("/api/session")
async def session_create(payload: SessionCreate):
    if not payload.golden_key.strip():
        raise HTTPException(status_code=400, detail="Golden Key is required.")
    client = await start_session(payload.golden_key.strip(), payload.base_url)
    return {"status": "ok", "userId": client.user_id, "baseUrl": client.base_url}


@app.get("/api/nodes")
def list_nodes(session: Session = Depends(get_session)):
    nodes = session.exec(select(Node)).all()
    return nodes


class NodeCreate(SQLModel):
    node: str


@app.post("/api/nodes")
def add_node(payload: NodeCreate, session: Session = Depends(get_session)):
    node_id = str(payload.node)
    existing = session.get(Node, node_id)
    if existing:
        return {"status": "exists", "node": node_id}
    session.add(Node(id=node_id))
    session.commit()
    return {"status": "ok", "node": node_id}


class SendMessage(SQLModel):
    node: str
    message: str


class UpdatePrice(SQLModel):
    node: str
    offer: str
    price: float


@app.post("/api/messages/send")
async def send_message(payload: SendMessage, session: Session = Depends(get_session)):
    node_id = str(payload.node)
    msg = payload.message.strip()
    if not msg:
        return {"status": "error", "error": "Empty message"}

    record = session.get(Node, node_id)
    if not record:
        record = Node(id=node_id, last_id=0)
        session.add(record)
        session.commit()

    client: Optional[FunpayClient] = getattr(app.state, "fp_client", None)
    if not client:
        raise HTTPException(status_code=400, detail="No active session. Set Golden Key first.")
    try:
        await client.send_chat_message(node_id, msg, last_message=record.last_id or 0)
        record.last_id = (record.last_id or 0) + 1
        session.add(record)
        session.commit()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {"status": "ok", "node": node_id}


@app.get("/api/messages")
def list_messages(
    node: str = Query(..., description="User ID of the chat partner"),
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


@app.get("/api/lots")
async def list_lots():
    client: Optional[FunpayClient] = getattr(app.state, "fp_client", None)
    if not client:
        raise HTTPException(status_code=400, detail="No active session. Set Golden Key first.")
    offers = await client.get_offers()
    return offers


@app.post("/api/lots/price")
async def update_lot_price(payload: UpdatePrice):
    client: Optional[FunpayClient] = getattr(app.state, "fp_client", None)
    if not client:
        raise HTTPException(status_code=400, detail="No active session. Set Golden Key first.")
    try:
        await client.update_price(payload.node, payload.offer, payload.price)
        return {"status": "ok"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# -------- Static --------
app.mount("/", StaticFiles(directory="static", html=True), name="static")


# -------- Poller --------
async def poller():
    client: Optional[FunpayClient] = getattr(app.state, "fp_client", None)
    if not client:
        return

    stop_event: asyncio.Event = app.state.stop_event
    log = logging.getLogger("poller")

    while not stop_event.is_set():
        start = time.time()
        try:
            with Session(engine) as session:
                current_client: Optional[FunpayClient] = getattr(app.state, "fp_client", None)
                if not current_client:
                    break
                nodes = session.exec(select(Node)).all()
                for node in nodes:
                    await poll_node(session, current_client, log, node)
        except Exception as exc:
            log.exception("Poller failure: %s", exc)
        elapsed = time.time() - start
        delay = max(settings.poll_seconds - elapsed, 0.5)
        await asyncio.sleep(delay)


async def poll_node(session: Session, client: FunpayClient, log: logging.Logger, node: Node):
    await client.ensure_ready()
    chat_node = get_users_node(client.user_id, node.id)

    if node.last_id is None:
        try:
            last_seen = await client.get_user_last_message(node.id)
            node.last_id = last_seen or 0
            session.add(node)
            session.commit()
            log.info("Node %s: initialized last_id=%s", node.id, node.last_id)
        except Exception as exc:
            log.warning("Node %s: failed to fetch last message id: %s", node.id, exc)
            node.last_id = node.last_id or 0

    objects = [
        {"type": "orders_counters", "id": client.user_id, "tag": "py", "data": True},
        {"type": "chat_counter", "id": client.user_id, "tag": "py", "data": True},
        {
            "type": "chat_node",
            "id": chat_node,
            "data": {"node": chat_node, "last_message": int(node.last_id or 0), "content": ""},
        },
    ]

    try:
        payload = await client.runner(objects)
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status == 429:
            log.warning("Node %s rate limited; backing off", node.id)
            await asyncio.sleep(5)
        elif status == 401:
            log.error("Unauthorized (401) for node %s. Check Golden Key/session.", node.id)
        else:
            log.warning("Node %s HTTP error %s: %s", node.id, status, exc)
        return
    except Exception as exc:
        log.warning("Node %s runner error: %s", node.id, exc)
        return

    objects_resp = payload.get("objects", [])
    if not isinstance(objects_resp, Iterable):
        log.warning("Node %s unexpected runner payload", node.id)
        return

    inserted = 0
    new_last_id = node.last_id or 0

    for obj in objects_resp:
        if obj.get("type") != "chat_node" or obj.get("id") != chat_node:
            continue
        data = obj.get("data") or {}
        messages = data.get("messages") or []
        for item in messages:
            mid = coerce_int(item.get("id"))
            if mid is None:
                continue
            parsed = parse_message_html(item.get("html") or "")
            msg = Message(
                id=mid,
                node_id=node.id,
                author=str(item.get("author")) if item.get("author") is not None else None,
                username=parsed.username,
                body=parsed.body,
                created_at=parsed.created_at,
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


if __name__ == "__main__":
    import uvicorn

    init_db()
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
