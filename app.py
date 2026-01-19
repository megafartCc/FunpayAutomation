import asyncio
import contextlib
import json
import logging
import os
import time
import re
from dataclasses import dataclass
from typing import Iterable, Optional
from urllib.parse import quote_plus, urlparse

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import Response
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
        base_url = "https://funpay.com"
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


def normalize_avatar_url(base_url: str, url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if url.startswith("//"):
        return f"{base_url.split(':', 1)[0]}:{url}"
    if url.startswith("/"):
        return f"{base_url.rstrip('/')}{url}"
    return url


# -------- Database --------
def build_database_url() -> str:
    explicit_url = os.getenv("DATABASE_URL") or os.getenv("FUNPAY_DATABASE_URL")
    if explicit_url:
        return explicit_url

    mysql_host = os.getenv("MYSQLHOST")
    if mysql_host:
        mysql_port = os.getenv("MYSQLPORT", "3306")
        mysql_user = quote_plus(os.getenv("MYSQLUSER", ""))
        mysql_password = quote_plus(os.getenv("MYSQLPASSWORD", ""))
        mysql_db = os.getenv("MYSQLDATABASE", "")
        return f"mysql+pymysql://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{mysql_db}"

    return "sqlite:///data.db"


database_url = build_database_url()
connect_args = {"check_same_thread": False} if database_url.startswith("sqlite:///") else {}
engine = create_engine(database_url, connect_args=connect_args)


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
    base = settings.base_url
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
        self.dialog_backoff_until = 0.0
        self.chat_backoff_until = 0.0

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
        if time.time() < self.chat_backoff_until:
            return None
        chat_node = get_users_node(self.user_id, other_user_id)
        try:
            resp = await self.client.get(f"/chat/?node={chat_node}")
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                self.chat_backoff_until = time.time() + 30
                return None
            raise
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

    async def get_dialogs(self) -> list[dict]:
        await self.ensure_ready()
        if time.time() < self.dialog_backoff_until:
            return []
        try:
            resp = await self.client.get("/chat/")
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                self.dialog_backoff_until = time.time() + 30
            raise
        soup = BeautifulSoup(resp.text, "html.parser")
        dialogs: list[dict] = []
        for item in soup.select(".contact-item"):
            node_id = item.get("data-id")
            href = item.get("href", "")
            if not node_id and "node=" in href:
                node_id = href.split("node=")[1].split("&")[0]

            name_el = item.select_one(".media-user-name")
            name = name_el.get_text(strip=True) if name_el else None

            preview_el = item.select_one(".contact-item-message")
            preview = preview_el.get_text(" ", strip=True) if preview_el else None

            time_el = item.select_one(".contact-item-time")
            time_text = time_el.get_text(strip=True) if time_el else None

            avatar_url = None
            img = item.select_one("img")
            if img and img.get("src"):
                avatar_url = img.get("src")
            else:
                avatar_el = item.select_one(".avatar-photo")
                if avatar_el:
                    style = avatar_el.get("style", "")
                    match = re.search(r"url\\(([^)]+)\\)", style)
                    if match:
                        avatar_url = match.group(1).strip("'\"")
            avatar_url = normalize_avatar_url(self.base_url, avatar_url)

            dialogs.append(
                {
                    "node_id": node_id,
                    "name": name,
                    "preview": preview,
                    "time": time_text,
                    "avatar": avatar_url,
                }
            )
        return dialogs

    async def get_partner_from_node(self, node_id: str) -> dict:
        await self.ensure_ready()
        resp = await self.client.get(f"/chat/?node={node_id}")
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        user_id = None
        name = None
        avatar_url = None

        link = soup.select_one(".chat-header .media-user-name a[href*='/users/']")
        if link and link.get("href"):
            match = re.search(r"/users/(\\d+)/", link.get("href"))
            if match:
                user_id = match.group(1)
            name = link.get_text(strip=True)
        else:
            name_el = soup.select_one(".chat-header .media-user-name")
            if name_el:
                name = name_el.get_text(strip=True)

        panel = soup.select_one(".param-item.chat-panel")
        if panel and panel.get("data-id"):
            user_id = user_id or panel.get("data-id")

        img = soup.select_one(".chat-header img")
        if img and img.get("src"):
            avatar_url = img.get("src")
        else:
            avatar_el = soup.select_one(".chat-header .avatar-photo")
            if avatar_el:
                style = avatar_el.get("style", "")
                match = re.search(r"url\\(([^)]+)\\)", style)
                if match:
                    avatar_url = match.group(1).strip("'\"")
        avatar_url = normalize_avatar_url(self.base_url, avatar_url)

        return {"user_id": user_id, "name": name, "avatar": avatar_url}

    async def get_chat_history(
        self,
        other_user_id: str,
        chat_node: Optional[str] = None,
        limit: int = 150,
    ) -> list[dict]:
        await self.ensure_ready()
        node = chat_node or get_users_node(self.user_id, other_user_id)
        limit = max(1, min(int(limit or 150), 300))
        results: list[dict] = []
        seen_ids: set[int] = set()
        last_message: Optional[int] = None
        pages = 0

        while len(results) < limit and pages < 10:
            url = f"/chat/?node={node}"
            if last_message:
                url = f"{url}&last_message={last_message}"
            resp = await self.client.get(url)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            page_items = soup.select(".chat-msg-item[id^='message-']")
            if not page_items:
                break

            page_results: list[dict] = []
            for item in page_items:
                msg_id_attr = item.get("id", "")
                match = re.search(r"message-(\d+)", msg_id_attr)
                if not match:
                    continue
                msg_id = int(match.group(1))
                if msg_id in seen_ids:
                    continue
                seen_ids.add(msg_id)
                parsed = parse_message_html(str(item))

                author = None
                author_link = item.select_one(".chat-msg-author-link[href*='/users/']")
                if author_link and author_link.get("href"):
                    author_match = re.search(r"/users/(\d+)/", author_link.get("href"))
                    if author_match:
                        author = author_match.group(1)

                page_results.append(
                    {
                        "id": msg_id,
                        "author": author,
                        "username": parsed.username,
                        "body": parsed.body,
                        "created_at": parsed.created_at,
                        "raw": str(item),
                    }
                )

            if not page_results:
                break

            results.extend(page_results)
            oldest_id = min(item["id"] for item in page_results)
            if last_message is not None and oldest_id >= last_message:
                break
            last_message = oldest_id
            pages += 1

        return results[:limit]

    async def get_partner_info(self, other_user_id: str) -> dict:
        chat_node = get_users_node(self.user_id, other_user_id)
        return await self.get_partner_from_node(chat_node)

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


def is_allowed_avatar_host(host: Optional[str]) -> bool:
    if not host:
        return False
    host = host.lower()
    return host.endswith("funpay.com")


# -------- FastAPI app --------
app = FastAPI(title="Funpay Automation", version="0.2.0")


@app.on_event("startup")
async def on_startup():
    init_db()
    app.state.stop_event = asyncio.Event()
    app.state.fp_client: Optional[FunpayClient] = None
    app.state.poller_task = None
    app.state.partner_cache = {}
    app.state.partner_cache_ttl = 600
    app.state.dialog_cache = {}
    app.state.dialog_cache_ttl = 900
    app.state.dialog_list_cache = {"ts": 0.0, "data": []}
    if settings.initial_key:
        await start_session(settings.initial_key, settings.base_url)
    else:
        logging.error("FUNPAY_GOLDEN_KEY not set in env; backend will not poll without it.")


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

@app.get("/api/session")
async def session_status():
    client: Optional[FunpayClient] = getattr(app.state, "fp_client", None)
    return SessionStatus(
        polling=bool(client),
        userId=getattr(client, "user_id", None),
        baseUrl=settings.base_url,
    ).__dict__


@app.get("/api/nodes")
async def list_nodes(session: Session = Depends(get_session)):
    nodes = session.exec(select(Node)).all()
    result = []
    client: Optional[FunpayClient] = getattr(app.state, "fp_client", None)
    cache: dict = getattr(app.state, "partner_cache", {})
    ttl = getattr(app.state, "partner_cache_ttl", 600)
    now = time.time()

    for n in nodes:
        last_msg = (
            session.exec(
                select(Message).where(Message.node_id == n.id).order_by(Message.id.desc()).limit(1)
            ).first()
        )

        partner_name = None
        partner_avatar = None
        cached = cache.get(n.id)
        if cached and now - cached.get("ts", 0) < ttl:
            partner_name = cached.get("name")
            partner_avatar = cached.get("avatar")
        elif client:
            try:
                info = await client.get_partner_info(n.id)
                partner_name = info.get("name")
                partner_avatar = info.get("avatar")
                cache[n.id] = {"name": partner_name, "avatar": partner_avatar, "ts": now}
            except Exception:
                if cached:
                    partner_name = cached.get("name")
                    partner_avatar = cached.get("avatar")

        result.append(
            {
                "id": n.id,
                "last_id": n.last_id,
                "last_username": getattr(last_msg, "username", None),
                "last_body": getattr(last_msg, "body", None),
                "last_created_at": getattr(last_msg, "created_at", None),
                "partner_name": partner_name,
                "partner_avatar": partner_avatar,
            }
        )

    app.state.partner_cache = cache
    return result


@app.get("/api/dialogs")
async def list_dialogs(
    resolve: bool = Query(False, description="Resolve missing user_id/avatar via chat page."),
    resolve_limit: int = Query(50, ge=0, le=200),
    session: Session = Depends(get_session),
):
    client: Optional[FunpayClient] = getattr(app.state, "fp_client", None)
    if not client:
        raise HTTPException(status_code=400, detail="No active session. Set Golden Key first.")

    now = time.time()
    list_cache = app.state.dialog_list_cache
    cached_dialogs = list_cache.get("data") or []
    cached_ts = list_cache.get("ts", 0.0)
    dialogs = []
    if now - cached_ts < 20 and cached_dialogs:
        dialogs = cached_dialogs
    else:
        try:
            dialogs = await client.get_dialogs()
            list_cache["data"] = dialogs
            list_cache["ts"] = now
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429 and cached_dialogs:
                dialogs = cached_dialogs
                list_cache["ts"] = now
            else:
                raise HTTPException(status_code=502, detail="FunPay rate limited. Try again soon.")
    cache: dict = getattr(app.state, "dialog_cache", {})
    ttl = getattr(app.state, "dialog_cache_ttl", 900)
    result = []

    resolved_count = 0

    for dialog in dialogs:
        node_id = dialog.get("node_id")
        if not node_id:
            continue

        cached = cache.get(node_id)
        user_id = None
        name = dialog.get("name")
        avatar = dialog.get("avatar")

        # Avoid per-dialog /chat/?node=... requests (rate limits).
        # Optionally resolve missing data on demand.
        if cached:
            user_id = cached.get("user_id")
            name = cached.get("name") or name
            avatar = cached.get("avatar") or avatar
        if resolve and resolved_count < resolve_limit and (not user_id or not avatar):
            try:
                info = await client.get_partner_from_node(node_id)
                user_id = info.get("user_id") or user_id
                name = info.get("name") or name
                avatar = info.get("avatar") or avatar
                cache[node_id] = {
                    "user_id": user_id,
                    "name": name,
                    "avatar": avatar,
                    "ts": now,
                }
                resolved_count += 1
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    break
            except Exception:
                pass

        result.append(
            {
                "node_id": node_id,
                "user_id": user_id,
                "name": name,
                "avatar": avatar,
                "preview": dialog.get("preview"),
                "time": dialog.get("time"),
            }
        )

    app.state.dialog_cache = cache
    return result


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


class SyncMessages(SQLModel):
    node: Optional[str] = None
    chat_node: Optional[str] = None
    limit: Optional[int] = 150


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


@app.post("/api/messages/sync")
async def sync_messages(payload: SyncMessages, session: Session = Depends(get_session)):
    node_id = str(payload.node) if payload.node else None
    client: Optional[FunpayClient] = getattr(app.state, "fp_client", None)
    if not client:
        raise HTTPException(status_code=400, detail="No active session. Set Golden Key first.")

    resolved_name = None
    resolved_avatar = None
    if not node_id:
        if not payload.chat_node:
            raise HTTPException(status_code=400, detail="Either node or chat_node is required.")
        info = await client.get_partner_from_node(payload.chat_node)
        node_id = info.get("user_id")
        resolved_name = info.get("name")
        resolved_avatar = info.get("avatar")
        if not node_id:
            raise HTTPException(status_code=502, detail="Failed to resolve partner user_id for dialog.")

    record = session.get(Node, node_id)
    if not record:
        record = Node(id=node_id, last_id=0)
        session.add(record)
        session.commit()

    try:
        history = await client.get_chat_history(
            node_id, chat_node=payload.chat_node, limit=payload.limit or 150
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    updated_last = record.last_id or 0
    for item in history:
        mid = coerce_int(item.get("id"))
        if mid is None:
            continue
        msg = Message(
            id=mid,
            node_id=node_id,
            author=item.get("author"),
            username=item.get("username"),
            body=item.get("body"),
            created_at=item.get("created_at"),
            raw=item.get("raw"),
        )
        session.merge(msg)
        if mid > updated_last:
            updated_last = mid

    record.last_id = updated_last
    session.add(record)
    session.commit()

    if payload.chat_node:
        cache: dict = getattr(app.state, "dialog_cache", {})
        existing = cache.get(payload.chat_node, {}) if isinstance(cache, dict) else {}
        cache[payload.chat_node] = {
            "user_id": node_id,
            "name": resolved_name or existing.get("name"),
            "avatar": resolved_avatar or existing.get("avatar"),
            "ts": time.time(),
        }
        app.state.dialog_cache = cache

    return {
        "status": "ok",
        "count": len(history),
        "last_id": updated_last,
        "user_id": node_id,
        "name": resolved_name,
        "avatar": resolved_avatar,
    }


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




@app.get("/api/avatar")
async def proxy_avatar(url: str = Query(..., description="Avatar URL to proxy")):
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not is_allowed_avatar_host(parsed.hostname):
        raise HTTPException(status_code=400, detail="Invalid avatar URL.")
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/129.0.0.0 Safari/537.36",
            },
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail="Avatar fetch failed.")
        content_type = resp.headers.get("content-type", "image/jpeg")
        return Response(
            content=resp.content,
            media_type=content_type,
            headers={"Cache-Control": "public, max-age=3600"},
        )


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
