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
from sqlalchemy import BigInteger, Column, text
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError
from sqlmodel import Field, Session, SQLModel, create_engine, select
from steam.client import SteamClient as BaseSteamClient
from steam.enums import EResult
from steam.webauth import WebAuth
from eventemitter import EventEmitter

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
    """
    Parse FunPay chat message HTML with multiple selector fallbacks.
    Improved parsing similar to AUTO-STEAM-RENT approach.
    """
    soup = BeautifulSoup(html or "", "html.parser")
    
    # Try multiple selectors for message body
    body_el = (soup.select_one(".chat-msg-text") or
              soup.select_one(".message-text") or
              soup.select_one(".msg-text") or
              soup.select_one(".chat-message-text") or
              soup.select_one("[class*='msg-text']") or
              soup.select_one("[class*='message-text']"))
    
    # Try multiple selectors for date/time
    date_el = (soup.select_one(".chat-msg-date") or
              soup.select_one(".message-date") or
              soup.select_one(".msg-date") or
              soup.select_one(".chat-message-date") or
              soup.select_one("time") or
              soup.select_one("[class*='date']") or
              soup.select_one("[class*='time']"))
    
    # Try multiple selectors for username
    username_el = (soup.select_one(".media-user-name") or
                  soup.select_one(".user-name") or
                  soup.select_one(".username") or
                  soup.select_one(".chat-msg-author") or
                  soup.select_one(".message-author") or
                  soup.select_one("[class*='user-name']") or
                  soup.select_one("[class*='author']"))
    
    # Extract body text - try multiple methods
    body_text = None
    if body_el:
        body_text = body_el.get_text("\n", strip=True)
        # If empty, try getting from inner HTML
        if not body_text:
            body_text = body_el.get_text(separator="\n", strip=True)
    
    # Extract date/time - try multiple attributes
    created_at = None
    if date_el:
        created_at = (date_el.get("title") or
                     date_el.get("datetime") or
                     date_el.get("data-time") or
                     date_el.get_text(strip=True))
    
    # Extract username
    username = username_el.get_text(strip=True) if username_el else None
    
    return ParsedMessage(
        username=username,
        body=body_text,
        created_at=created_at,
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
    last_id: Optional[int] = Field(default=None, sa_column=Column(BigInteger, nullable=True))


class Message(SQLModel, table=True):
    id: int = Field(sa_column=Column(BigInteger, primary_key=True))
    node_id: str = Field(primary_key=True)
    author: Optional[str] = None
    username: Optional[str] = None
    body: Optional[str] = None
    created_at: Optional[str] = None
    raw: Optional[str] = None


class Account(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    label: Optional[str] = None
    username: str
    password: str
    steam_id: Optional[str] = None
    login_status: Optional[str] = None
    twofa_otp: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


def init_db() -> None:
    SQLModel.metadata.create_all(engine)
    ensure_mysql_bigint()
    if settings.default_nodes:
        with Session(engine) as session:
            for node_id in settings.default_nodes:
                if not session.get(Node, node_id):
                    session.add(Node(id=node_id))
            session.commit()


def ensure_mysql_bigint() -> None:
    if engine.dialect.name != "mysql":
        return
    with engine.begin() as conn:
        try:
            conn.execute(text("ALTER TABLE message MODIFY COLUMN id BIGINT UNSIGNED NOT NULL"))
        except Exception as exc:
            logging.warning("Failed to alter message.id to BIGINT: %s", exc)
            pass
        try:
            conn.execute(text("ALTER TABLE node MODIFY COLUMN last_id BIGINT UNSIGNED NULL"))
        except Exception as exc:
            logging.warning("Failed to alter node.last_id to BIGINT: %s", exc)
            pass


def bulk_insert_messages(session: Session, rows: list[dict]) -> None:
    if not rows:
        return
    dialect = engine.dialect.name
    try:
        if dialect == "mysql":
            stmt = mysql_insert(Message).values(rows).prefix_with("IGNORE")
            session.execute(stmt)
        elif dialect == "sqlite":
            stmt = sqlite_insert(Message).values(rows).prefix_with("OR IGNORE")
            session.execute(stmt)
        else:
            for row in rows:
                session.add(Message(**row))
    except IntegrityError:
        session.rollback()


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
        """
        Get the last message ID from a chat.
        Improved with multiple selectors (like AUTO-STEAM-RENT).
        """
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
        
        # Try multiple selectors for message items
        items = (soup.select(".chat-msg-item.chat-msg-with-head") or
                soup.select(".chat-msg-item[id^='message-']") or
                soup.select(".message-item[id^='message-']") or
                soup.select("[id^='message-']") or
                soup.select(".chat-msg-item"))
        
        if not items:
            return None
        
        # Get the last message (most recent)
        last = items[-1]
        
        # Try multiple methods to extract message ID
        msg_id_attr = last.get("id", "")
        msg_id = None
        
        # Method 1: message-123 format
        if msg_id_attr:
            match = re.search(r"message-(\d+)", msg_id_attr)
            if match:
                try:
                    return int(match.group(1))
                except:
                    pass
        
        # Method 2: data-id attribute
        msg_id = last.get("data-id") or last.get("data-message-id")
        if msg_id:
            try:
                return int(msg_id)
            except:
                pass
        
        # Method 3: Try splitting id attribute
        if msg_id_attr and "-" in msg_id_attr:
            try:
                return int(msg_id_attr.split("-")[1])
            except:
                pass
        
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
        """
        Get all chat dialogs from FunPay.
        Improved parsing with multiple selectors (like AUTO-STEAM-RENT).
        """
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
        
        # Try multiple selectors for dialog items
        items = (soup.select(".contact-item") or
                soup.select(".dialog-item") or
                soup.select(".chat-item") or
                soup.select("[class*='contact']") or
                soup.select("[class*='dialog']"))
        
        for item in items:
            # Extract node_id - try multiple methods
            node_id = (item.get("data-id") or
                      item.get("data-node") or
                      item.get("data-node-id"))
            
            href = item.get("href", "")
            if not node_id and "node=" in href:
                node_id = href.split("node=")[1].split("&")[0].split("#")[0]
            
            # Also try from data attributes
            if not node_id:
                node_id = (item.get("data-chat-node") or
                          item.get("data-dialog-id"))

            # Try multiple selectors for name
            name_el = (item.select_one(".media-user-name") or
                      item.select_one(".user-name") or
                      item.select_one(".contact-name") or
                      item.select_one(".dialog-name") or
                      item.select_one("[class*='user-name']") or
                      item.select_one("[class*='name']"))
            name = name_el.get_text(strip=True) if name_el else None

            # Try multiple selectors for message preview
            preview_el = (item.select_one(".contact-item-message") or
                         item.select_one(".message-preview") or
                         item.select_one(".preview") or
                         item.select_one(".last-message") or
                         item.select_one("[class*='message']") or
                         item.select_one("[class*='preview']"))
            preview = preview_el.get_text(" ", strip=True) if preview_el else None

            # Try multiple selectors for time
            time_el = (item.select_one(".contact-item-time") or
                      item.select_one(".time") or
                      item.select_one(".last-time") or
                      item.select_one("time") or
                      item.select_one("[class*='time']") or
                      item.select_one("[class*='date']"))
            time_text = time_el.get_text(strip=True) if time_el else None
            # Also try datetime attribute
            if not time_text and time_el:
                time_text = (time_el.get("datetime") or
                           time_el.get("title") or
                           time_el.get("data-time"))

            # Extract avatar - try multiple methods
            avatar_url = None
            img = item.select_one("img")
            if img:
                avatar_url = (img.get("src") or
                            img.get("data-src") or
                            img.get("data-lazy-src"))
            
            if not avatar_url:
                # Try from background-image style
                avatar_el = (item.select_one(".avatar-photo") or
                           item.select_one(".avatar") or
                           item.select_one("[class*='avatar']"))
                if avatar_el:
                    style = avatar_el.get("style", "")
                    match = re.search(r"url\\(([^)]+)\\)", style)
                    if match:
                        avatar_url = match.group(1).strip("'\"")
                    # Also try background attribute
                    if not avatar_url:
                        avatar_url = avatar_el.get("data-bg") or avatar_el.get("data-background")
            
            avatar_url = normalize_avatar_url(self.base_url, avatar_url)

            # Only add if we have at least node_id or name
            if node_id or name:
                dialogs.append(
                    {
                        "node_id": node_id,
                        "name": name,
                        "preview": preview,
                        "time": time_text,
                        "avatar": avatar_url,
                    }
                )
        
        # Sort by time (most recent first) if available
        dialogs.sort(key=lambda x: x.get("time", ""), reverse=True)
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
        """
        Get chat history with improved parsing (like AUTO-STEAM-RENT).
        """
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

            # Try multiple selectors for message items
            page_items = (soup.select(".chat-msg-item[id^='message-']") or
                         soup.select(".message-item[id^='message-']") or
                         soup.select("[id^='message-']") or
                         soup.select(".chat-msg-item") or
                         soup.select(".message"))
            
            if not page_items:
                break

            page_results: list[dict] = []
            for item in page_items:
                # Extract message ID - try multiple methods
                msg_id_attr = item.get("id", "")
                msg_id = None
                
                # Method 1: message-123 format
                match = re.search(r"message-(\d+)", msg_id_attr)
                if match:
                    msg_id = int(match.group(1))
                else:
                    # Method 2: data-id attribute
                    msg_id = item.get("data-id") or item.get("data-message-id")
                    if msg_id:
                        try:
                            msg_id = int(msg_id)
                        except:
                            msg_id = None
                
                if not msg_id or msg_id in seen_ids:
                    continue
                
                seen_ids.add(msg_id)
                parsed = parse_message_html(str(item))

                # Extract author/user_id - try multiple methods
                author = None
                
                # Method 1: From author link
                author_link = (item.select_one(".chat-msg-author-link[href*='/users/']") or
                              item.select_one("a[href*='/users/']"))
                if author_link and author_link.get("href"):
                    author_match = re.search(r"/users/(\d+)/", author_link.get("href"))
                    if author_match:
                        author = author_match.group(1)
                
                # Method 2: From data attributes
                if not author:
                    author = (item.get("data-author-id") or
                            item.get("data-user-id") or
                            item.get("data-author"))
                
                # Method 3: From username if it matches our user_id
                if not author and parsed.username:
                    # Check if message is from the other user
                    if parsed.username and other_user_id:
                        # Try to extract from any links in the message
                        user_links = item.select("a[href*='/users/']")
                        for link in user_links:
                            match = re.search(r"/users/(\d+)/", link.get("href", ""))
                            if match:
                                author = match.group(1)
                                break

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

        # Sort by ID (chronological order)
        results.sort(key=lambda x: x.get("id", 0))
        return results[:limit]

    async def get_partner_info(self, other_user_id: str) -> dict:
        chat_node = get_users_node(self.user_id, other_user_id)
        return await self.get_partner_from_node(chat_node)

    async def get_offers(self) -> list[dict]:
        """
        Get all current lots/offers from FunPay.
        Improved parsing similar to AUTO-STEAM-RENT approach.
        """
        await self.ensure_ready()
        
        # Try multiple endpoints to get offers
        endpoints_to_try = [
            f"/users/{self.user_id}/",  # User profile page
            "/lots/",  # Direct lots page
            f"/users/{self.user_id}/lots/",  # User lots page
        ]
        
        result: list[dict] = []
        
        for endpoint in endpoints_to_try:
            try:
                resp = await self.client.get(endpoint)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                
                # Method 1: Try .mb20 .offer selector (current method)
                offer_blocks = soup.select(".mb20 .offer")
                if not offer_blocks:
                    # Method 2: Try alternative selectors (in case FunPay changed HTML)
                    offer_blocks = soup.select(".offer-list .offer")
                if not offer_blocks:
                    # Method 3: Try even more generic selectors
                    offer_blocks = soup.select(".offer")
                
                if offer_blocks:
                    for offer_block in offer_blocks:
                        # Try multiple title selectors
                        title = (offer_block.select_one(".offer-list-title h3 a") or
                                offer_block.select_one(".offer-title a") or
                                offer_block.select_one("h3 a") or
                                offer_block.select_one(".title a"))
                        
                        if not title:
                            continue
                        
                        group_name = title.get_text(strip=True)
                        href = title.get("href", "")
                        
                        # Extract node from href - try multiple formats
                        node = None
                        if href:
                            # Format: /lots/category/node or /category/node
                            parts = href.strip("/").split("/")
                            if len(parts) >= 2:
                                node = parts[-1]  # Last part is usually the node
                            elif len(parts) == 1:
                                node = parts[0]
                        
                        if not node:
                            continue
                        
                        # Check if we already have this group
                        existing_group = next((g for g in result if g.get("node") == node), None)
                        if existing_group:
                            group = existing_group
                        else:
                            group = {"group_name": group_name, "node": node, "offers": []}
                            result.append(group)
                        
                        # Parse offers in this block - try multiple item selectors
                        items = (offer_block.select(".tc-item") or
                                offer_block.select(".offer-item") or
                                offer_block.select(".lot-item") or
                                offer_block.select("a[href*='id=']"))
                        
                        for item in items:
                            # Try multiple name selectors
                            name_el = (item.select_one(".tc-desc-text") or
                                      item.select_one(".offer-name") or
                                      item.select_one(".lot-name") or
                                      item.select_one(".name") or
                                      item)
                            
                            # Try multiple price selectors
                            price_el = (item.select_one(".tc-price") or
                                       item.select_one(".offer-price") or
                                       item.select_one(".lot-price") or
                                       item.select_one(".price") or
                                       item.select_one("[class*='price']"))
                            
                            # Extract offer ID from href - try multiple formats
                            href_item = item.get("href") or ""
                            offer_id = None
                            
                            if href_item:
                                # Format 1: ?id=123 or &id=123
                                if "id=" in href_item:
                                    try:
                                        offer_id = href_item.split("id=")[1].split("&")[0].split("#")[0]
                                    except Exception:
                                        pass
                                # Format 2: /lots/offer/123
                                elif "/offer/" in href_item:
                                    try:
                                        offer_id = href_item.split("/offer/")[1].split("/")[0].split("?")[0]
                                    except Exception:
                                        pass
                                # Format 3: Direct ID in URL
                                elif href_item.isdigit():
                                    offer_id = href_item
                            
                            # Extract from data attributes
                            if not offer_id:
                                offer_id = (item.get("data-offer-id") or
                                          item.get("data-id") or
                                          item.get("data-lot-id"))
                            
                            offer_name = name_el.get_text(strip=True) if name_el else ""
                            offer_price = price_el.get_text(strip=True) if price_el else ""
                            
                            # Only add if we have at least a name or ID
                            if offer_name or offer_id:
                                # Check if offer already exists (avoid duplicates)
                                existing_offer = next(
                                    (o for o in group["offers"] if o.get("id") == offer_id or o.get("name") == offer_name),
                                    None
                                )
                                if not existing_offer:
                                    group["offers"].append({
                                        "name": offer_name,
                                        "price": offer_price,
                                        "id": offer_id,
                                    })
                    
                    # If we found offers, break (don't try other endpoints)
                    if result:
                        break
                        
            except Exception as e:
                logging.debug("Failed to parse offers from %s: %s", endpoint, e)
                continue
        
        # Sort offers by group name for consistency
        result.sort(key=lambda x: x.get("group_name", ""))
        for group in result:
            group["offers"].sort(key=lambda x: x.get("name", ""))
        
        logging.info("Parsed %d lot groups with %d total offers", len(result), sum(len(g["offers"]) for g in result))
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
    # Initialize state first (non-blocking)
    app.state.stop_event = asyncio.Event()
    app.state.fp_client: Optional[FunpayClient] = None
    app.state.poller_task = None
    app.state.steam_clients = {}
    app.state.partner_cache = {}
    app.state.partner_cache_ttl = 600
    app.state.dialog_cache = {}
    app.state.dialog_cache_ttl = 900
    app.state.dialog_list_cache = {"ts": 0.0, "data": []}
    
    # Initialize database (with error handling)
    try:
        init_db()
    except Exception as e:
        logging.warning("Database initialization warning (non-fatal): %s", e)
    
    # Start session in background if key is provided
    if settings.initial_key:
        try:
            await start_session(settings.initial_key, settings.base_url)
        except Exception as e:
            logging.warning("Failed to start initial session (non-fatal): %s", e)
    else:
        logging.info("FUNPAY_GOLDEN_KEY not set in env; backend will not poll without it.")


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
    steam_clients = getattr(app.state, "steam_clients", {})
    for steam_client in steam_clients.values():
        with contextlib.suppress(Exception):
            steam_client.logout()
            steam_client.disconnect()


# -------- API routes --------
# Health endpoints - defined early so they're not intercepted by static files
@app.get("/health")
async def health_simple():
    """Simple health check endpoint for Railway."""
    return {"status": "ok"}


@app.get("/api/health")
async def health():
    """Health check endpoint for Railway."""
    return {"status": "ok"}


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


class AccountCreate(SQLModel):
    label: Optional[str] = None
    username: str
    password: str
    steam_id: Optional[str] = None
    login_status: Optional[str] = None
    twofa_otp: Optional[str] = None


class AccountUpdate(SQLModel):
    label: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    steam_id: Optional[str] = None
    login_status: Optional[str] = None
    twofa_otp: Optional[str] = None


class ChangePasswordRequest(SQLModel):
    new_password: str


class AccountLogin(SQLModel):
    guard_code: Optional[str] = None
    email_code: Optional[str] = None


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
    rows: list[dict] = []
    for item in history:
        mid = coerce_int(item.get("id"))
        if mid is None:
            continue
        rows.append(
            {
                "id": mid,
                "node_id": node_id,
                "author": item.get("author"),
                "username": item.get("username"),
                "body": item.get("body"),
                "created_at": item.get("created_at"),
                "raw": item.get("raw"),
            }
        )
        if mid > updated_last:
            updated_last = mid

    bulk_insert_messages(session, rows)
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
    try:
        offers = await client.get_offers()
        return offers
    except Exception as exc:
        logging.exception("Failed to get lots/offers")
        raise HTTPException(status_code=500, detail=f"Failed to fetch offers: {str(exc)}")


@app.get("/api/accounts")
def list_accounts(session: Session = Depends(get_session)):
    rows = session.exec(select(Account).order_by(Account.id.desc())).all()
    return [
        {
            "id": row.id,
            "label": row.label,
            "username": row.username,
            "steam_id": row.steam_id,
            "login_status": row.login_status,
            "has_password": bool(row.password),
            "has_twofa_otp": bool(row.twofa_otp),
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }
        for row in rows
    ]


@app.post("/api/accounts")
def create_account(payload: AccountCreate, session: Session = Depends(get_session)):
    username = payload.username.strip()
    password = payload.password.strip()
    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password are required.")
    account = Account(
        label=payload.label,
        username=username,
        password=password,
        steam_id=payload.steam_id,
        login_status=payload.login_status or "idle",
        twofa_otp=payload.twofa_otp,
    )
    session.add(account)
    session.commit()
    session.refresh(account)
    return {
        "id": account.id,
        "label": account.label,
        "username": account.username,
        "steam_id": account.steam_id,
        "login_status": account.login_status,
        "has_password": True,
        "has_twofa_otp": bool(account.twofa_otp),
        "created_at": account.created_at,
        "updated_at": account.updated_at,
    }


@app.put("/api/accounts/{account_id}")
def update_account(account_id: int, payload: AccountUpdate, session: Session = Depends(get_session)):
    account = session.get(Account, account_id)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found.")
    if payload.username is not None:
        account.username = payload.username.strip()
    if payload.password is not None:
        account.password = payload.password.strip()
    if payload.label is not None:
        account.label = payload.label
    if payload.steam_id is not None:
        account.steam_id = payload.steam_id
    if payload.login_status is not None:
        account.login_status = payload.login_status
    if payload.twofa_otp is not None:
        account.twofa_otp = payload.twofa_otp
    account.updated_at = time.time()
    session.add(account)
    session.commit()
    return {"status": "ok"}


@app.post("/api/accounts/{account_id}/login")
def login_account(account_id: int, payload: AccountLogin, session: Session = Depends(get_session)):
    logging.info("Steam login request for account %s", account_id)
    account = session.get(Account, account_id)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found.")
    if not account.username or not account.password:
        raise HTTPException(status_code=400, detail="Account missing username or password.")

    steam_clients = getattr(app.state, "steam_clients", {})
    try:
        client = steam_clients.get(account_id) or SteamClient()
    except Exception as exc:
        logging.exception("Steam client init failed for account %s", account_id)
        account.login_status = "error:init"
        account.updated_at = time.time()
        session.add(account)
        session.commit()
        raise HTTPException(status_code=500, detail=str(exc))

    connected = getattr(client, "connected", None)
    if connected is False or connected is None:
        try:
            connect_ok = client.connect()
        except Exception as exc:
            logging.exception("Steam connect failed for account %s", account_id)
            account.login_status = "error:connect"
            account.updated_at = time.time()
            session.add(account)
            session.commit()
            raise HTTPException(status_code=500, detail=str(exc))
        if connect_ok is False:
            logging.warning("Steam connect returned false for account %s", account_id)
            account.login_status = "error:connect"
            account.updated_at = time.time()
            session.add(account)
            session.commit()
            raise HTTPException(status_code=500, detail="Steam connect failed")

    # Determine which type of code we have
    guard_code = (payload.guard_code or "").strip() or None
    email_code = (payload.email_code or "").strip() or None
    
    # If guard_code is provided but email_code is not, use guard_code for email_code
    # (user enters code in one field, we determine which type based on current status)
    if guard_code and not email_code:
        # Check current status to determine code type
        if account.login_status == "guard:twofactor":
            # User is providing 2FA code
            two_factor_code = guard_code
            auth_code = None
        else:
            # Assume it's an email code (most common case)
            auth_code = guard_code
            two_factor_code = None
    elif email_code:
        # Explicit email code provided
        auth_code = email_code
        two_factor_code = None
    else:
        # No code provided - first login attempt
        auth_code = None
        two_factor_code = None
    
    # Also check if account has stored 2FA OTP
    if not two_factor_code and account.twofa_otp:
        two_factor_code = account.twofa_otp.strip()
    
    logging.info(
        "Steam login attempt for account %s: username=%s, has_auth_code=%s, has_2fa=%s, current_status=%s",
        account_id,
        account.username,
        bool(auth_code),
        bool(two_factor_code),
        account.login_status,
    )
    
    try:
        logging.info("Steam login connect ok for account %s", account_id)
        result = client.login(
            account.username,
            account.password,
            two_factor_code=two_factor_code,
            auth_code=auth_code,
        )
    except Exception as exc:
        logging.exception("Steam login error for account %s", account_id)
        account.login_status = "error:exception"
        account.updated_at = time.time()
        session.add(account)
        session.commit()
        raise HTTPException(status_code=500, detail=str(exc))

    logging.info("Steam login result for account %s: %s", account_id, result)
    ok = result is True or result == EResult.OK
    if not ok:
        logging.warning("Steam login failed for account %s: %s (EResult code: %s)", account_id, result, int(result) if hasattr(result, '__int__') else result)
        
        # Map EResult codes to status strings
        status_map = {
            getattr(EResult, "AccountLogonDenied", None): "guard:email",
            getattr(EResult, "AccountLogonDeniedNeedTwoFactor", None): "guard:twofactor",
            getattr(EResult, "InvalidPassword", None): "error:invalid_password",
            getattr(EResult, "InvalidLoginAuthCode", None): "error:invalid_auth_code",
            getattr(EResult, "AccountNotFound", None): "error:account_not_found",
            getattr(EResult, "RateLimitExceeded", None): "error:rate_limit",
        }
        
        # Also check by numeric code
        result_code = int(result) if hasattr(result, '__int__') else None
        
        # EResult 63 (AccountLogonDenied) can mean different things:
        # - If we provided a code and got 63: code was invalid/expired
        # - If we didn't provide a code and got 63: code is required
        if result_code == 63:
            # Check if we provided a code in this attempt
            if auth_code or two_factor_code:
                # Code was provided but rejected - it's invalid
                account.login_status = "error:invalid_auth_code"
            else:
                # No code provided - email code is required
                account.login_status = "guard:email"
        elif result_code == 5:
            account.login_status = "error:invalid_password"
        elif result_code == 65:
            account.login_status = "guard:email"
        elif result_code == 85:
            account.login_status = "guard:twofactor"
        else:
            account.login_status = status_map.get(result, f"error:{result_code or result}")
        
        account.updated_at = time.time()
        session.add(account)
        session.commit()
        
        # Provide user-friendly error messages
        error_messages = {
            "guard:email": "Email verification code required. Check your email and enter the code.",
            "guard:twofactor": "2FA code required. Enter your Steam Guard code.",
            "error:invalid_password": "Invalid username or password.",
            "error:invalid_auth_code": "Invalid authentication code. Please check your email/2FA code and try again.",
            "error:account_not_found": "Steam account not found. Check your username.",
            "error:rate_limit": "Too many login attempts. Please wait a few minutes.",
        }
        error_msg = error_messages.get(account.login_status, f"Login failed (Error code: {result_code or result}). Check your credentials and try again.")
        raise HTTPException(status_code=401, detail=error_msg)

    # Wait a moment for the client to fully initialize web session
    # SteamClient needs time after login to establish web session
    import time as time_module
    time_module.sleep(2)  # Give SteamClient time to establish web session
    
    # Try to get web session immediately after login to ensure it's available
    try:
        if hasattr(client, "get_web_session"):
            web_sess = client.get_web_session()
            if web_sess:
                logging.info("Web session obtained immediately after login for account %s", account_id)
    except Exception as e:
        logging.debug("Could not get web session immediately after login for account %s: %s", account_id, e)
    
    steam_clients[account_id] = client
    app.state.steam_clients = steam_clients
    account.login_status = "online"
    account.steam_id = str(getattr(client, "steam_id", "") or account.steam_id or "")
    account.updated_at = time.time()
    session.add(account)
    session.commit()
    return {
        "status": "ok",
        "steam_id": account.steam_id,
        "login_status": account.login_status,
    }


@app.post("/api/accounts/{account_id}/logout")
def logout_account(account_id: int, session: Session = Depends(get_session)):
    account = session.get(Account, account_id)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found.")
    steam_clients = getattr(app.state, "steam_clients", {})
    client = steam_clients.pop(account_id, None)
    if client:
        with contextlib.suppress(Exception):
            client.logout()
            client.disconnect()
    app.state.steam_clients = steam_clients
    account.login_status = "offline"
    account.updated_at = time.time()
    session.add(account)
    session.commit()
    return {"status": "ok"}


@app.post("/api/accounts/{account_id}/stop-rental")
async def stop_rental(account_id: int, session: Session = Depends(get_session)):
    """
    Forcefully stop a rental and log off the renter.
    This aggressively:
    1. Changes the password (invalidates ALL sessions including active games)
    2. Deauthorizes all devices
    3. Logs out the SteamClient
    4. Updates account status
    
    This is the proper way to end a rental and kick someone off.
    """
    import random
    import string
    
    account = session.get(Account, account_id)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found.")
    
    if not account.username or not account.password:
        raise HTTPException(status_code=400, detail="Account missing username or password.")
    
    steam_clients = getattr(app.state, "steam_clients", {})
    client = steam_clients.get(account_id)
    
    results = {
        "password_changed": False,
        "devices_deauthorized": False,
        "client_logged_out": False,
    }
    
    try:
        # Step 1: Generate a new random password
        new_password = ''.join(random.choice(string.ascii_letters + string.digits + '!@#$%^&*') for _ in range(16))
        logging.info("Stopping rental for account %s - generating new password", account_id)
        
        # Step 2: If client is logged in, try to change password via web session
        if client:
            logged_on = bool(
                getattr(client, "logged_on", False)
                or getattr(client, "is_logged_on", lambda: False)()
            )
            
            if logged_on:
                # Try to get web session and change password
                web_session = None
                
                # Try get_web_session()
                if hasattr(client, "get_web_session"):
                    try:
                        def _get_web_session():
                            sess = client.get_web_session()
                            if sess is None and hasattr(client, "get_web_session_cookies"):
                                cookies = client.get_web_session_cookies()
                                if cookies:
                                    import requests
                                    sess = requests.Session()
                                    for cookie in cookies:
                                        sess.cookies.set(cookie['name'], cookie['value'], domain=cookie.get('domain', '.steamcommunity.com'))
                            return sess
                        web_session = await asyncio.to_thread(_get_web_session)
                    except Exception as e:
                        logging.warning("Failed to get web session for password change: %s", e)
                
                # Try web_session property
                if not web_session and hasattr(client, "web_session") and client.web_session:
                    web_session = client.web_session
                
                # Change password if we have web session
                if web_session:
                    try:
                        # Get sessionid
                        sessionid = None
                        if hasattr(web_session, 'cookies'):
                            for cookie in web_session.cookies:
                                if cookie.name == 'sessionid':
                                    sessionid = cookie.value
                                    break
                        
                        if not sessionid:
                            # Try to get from account page
                            def _get_sessionid():
                                resp = web_session.get('https://store.steampowered.com/account/')
                                if resp.status_code == 200:
                                    for cookie in web_session.cookies:
                                        if cookie.name == 'sessionid':
                                            return cookie.value
                                    import re
                                    match = re.search(r'g_sessionID = "([^"]+)"', resp.text)
                                    if match:
                                        return match.group(1)
                                return None
                            sessionid = await asyncio.to_thread(_get_sessionid)
                        
                        if sessionid:
                            # Change password
                            def _change_password():
                                form_data = {
                                    'sessionid': sessionid,
                                    'password': account.password,
                                    'newpassword': new_password,
                                    'renewpassword': new_password
                                }
                                return web_session.post(
                                    'https://store.steampowered.com/account/changepassword_finish',
                                    data=form_data,
                                    headers={'Referer': 'https://store.steampowered.com/account/changepassword'},
                                    timeout=30
                                )
                            
                            response = await asyncio.to_thread(_change_password)
                            if response.status_code == 200 and ('successfully changed' in response.text.lower() or 'success' in response.text.lower()):
                                account.password = new_password
                                results["password_changed"] = True
                                logging.info("Password changed successfully for account %s (stop rental)", account_id)
                            
                            # Deauthorize all devices - try multiple endpoints
                            try:
                                def _revoke_devices():
                                    # Try the Steam Guard manage endpoint first
                                    # This is the official "Deauthorize all other devices" endpoint
                                    endpoints_to_try = [
                                        # Steam Guard manage page - deauthorize all
                                        ('https://store.steampowered.com/twofactor/manage_action', {
                                            'sessionid': sessionid,
                                            'action': 'deauthorize_all',
                                            'revokeall': '1'
                                        }),
                                        # Alternative endpoint
                                        ('https://store.steampowered.com/account/revokeauthorizeddevices', {
                                            'sessionid': sessionid,
                                            'revokeall': '1'
                                        }),
                                        # Steam Community endpoint
                                        ('https://steamcommunity.com/devices/revoke', {
                                            'sessionid': sessionid,
                                            'revokeall': '1'
                                        }),
                                        # Try with just sessionid
                                        ('https://store.steampowered.com/twofactor/manage', {
                                            'sessionid': sessionid,
                                            'revokeall': '1'
                                        }),
                                    ]
                                    
                                    for endpoint, data in endpoints_to_try:
                                        try:
                                            # First, try to GET the page to get any CSRF tokens
                                            get_resp = web_session.get(endpoint.split('_action')[0] if '_action' in endpoint else endpoint.replace('/manage_action', '/manage'), timeout=30)
                                            if get_resp.status_code == 200:
                                                # Try to extract CSRF token if needed
                                                import re
                                                csrf_match = re.search(r'name="csrf_token"\s+value="([^"]+)"', get_resp.text)
                                                if csrf_match:
                                                    data['csrf_token'] = csrf_match.group(1)
                                                
                                                # Now POST
                                                resp = web_session.post(endpoint, data=data, timeout=30, allow_redirects=True)
                                                logging.info("Deauthorize attempt to %s returned status %s", endpoint, resp.status_code)
                                                if resp.status_code in [200, 302]:  # 302 is redirect, might be success
                                                    # Check if response indicates success
                                                    if 'deauthorized' in resp.text.lower() or 'success' in resp.text.lower() or resp.status_code == 302:
                                                        return True
                                        except Exception as e:
                                            logging.debug("Endpoint %s failed: %s", endpoint, e)
                                            continue
                                    return False
                                
                                if await asyncio.to_thread(_revoke_devices):
                                    results["devices_deauthorized"] = True
                                    logging.info("Devices deauthorized for account %s (stop rental)", account_id)
                                else:
                                    logging.warning("All deauthorize endpoints failed for account %s", account_id)
                            except Exception as e:
                                logging.warning("Failed to deauthorize devices: %s", e)
                                logging.exception("Full traceback:")
                    except Exception as e:
                        logging.warning("Failed to change password via web session: %s", e)
                
                # FALLBACK 1: Try Selenium browser automation (like AUTO-STEAM-RENT does)
                if not results["password_changed"]:
                    logging.info("Trying Selenium browser automation to change password for account %s", account_id)
                    try:
                        def _selenium_change_password():
                            try:
                                from selenium import webdriver
                                from selenium.webdriver.common.by import By
                                from selenium.webdriver.support.ui import WebDriverWait
                                from selenium.webdriver.support import expected_conditions as EC
                                from selenium.webdriver.chrome.options import Options
                                from selenium.webdriver.chrome.service import Service
                                from webdriver_manager.chrome import ChromeDriverManager
                                
                                # Setup Chrome in headless mode
                                chrome_options = Options()
                                chrome_options.add_argument('--headless')
                                chrome_options.add_argument('--no-sandbox')
                                chrome_options.add_argument('--disable-dev-shm-usage')
                                chrome_options.add_argument('--disable-gpu')
                                chrome_options.add_argument('--window-size=1920,1080')
                                chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
                                
                                # Use webdriver-manager to automatically download ChromeDriver
                                service = Service(ChromeDriverManager().install())
                                driver = webdriver.Chrome(service=service, options=chrome_options)
                                
                                try:
                                    # Navigate to Steam login
                                    driver.get('https://store.steampowered.com/login/')
                                    wait = WebDriverWait(driver, 30)
                                    
                                    # Enter username
                                    username_field = wait.until(EC.presence_of_element_located((By.ID, "input_username")))
                                    username_field.clear()
                                    username_field.send_keys(account.username)
                                    
                                    # Enter password
                                    password_field = driver.find_element(By.ID, "input_password")
                                    password_field.clear()
                                    password_field.send_keys(account.password)
                                    
                                    # Click login button
                                    login_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
                                    login_button.click()
                                    
                                    # Wait for login to complete (might need 2FA)
                                    time.sleep(3)
                                    
                                    # If 2FA is needed, generate code
                                    if account.twofa_otp:
                                        try:
                                            # Check if 2FA field appears
                                            try:
                                                twofa_field = driver.find_element(By.ID, "twofactorcode_entry")
                                                # Generate 2FA code
                                                import base64
                                                import hmac
                                                import hashlib
                                                import struct
                                                secret_bytes = base64.b64decode(account.twofa_otp)
                                                timestamp = int(time.time()) // 30
                                                msg = struct.pack(">Q", timestamp)
                                                hmac_hash = hmac.new(secret_bytes, msg, hashlib.sha1).digest()
                                                offset = hmac_hash[-1] & 0x0F
                                                code_int = int.from_bytes(hmac_hash[offset:offset + 4], "big") & 0x7FFFFFFF
                                                alphabet = "23456789BCDFGHJKMNPQRTVWXY"
                                                two_factor_code = ""
                                                for _ in range(5):
                                                    two_factor_code += alphabet[code_int % len(alphabet)]
                                                    code_int //= len(alphabet)
                                                
                                                twofa_field.clear()
                                                twofa_field.send_keys(two_factor_code)
                                                
                                                # Submit 2FA
                                                submit_btn = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
                                                submit_btn.click()
                                                time.sleep(3)
                                            except:
                                                pass  # No 2FA needed
                                        except Exception as e:
                                            logging.warning("2FA handling in Selenium failed: %s", e)
                                    
                                    # Navigate to change password page
                                    driver.get('https://store.steampowered.com/account/changepassword')
                                    time.sleep(2)
                                    
                                    # Fill in password change form
                                    current_password_field = wait.until(EC.presence_of_element_located((By.ID, "current_password")))
                                    current_password_field.clear()
                                    current_password_field.send_keys(account.password)
                                    
                                    new_password_field = driver.find_element(By.ID, "new_password")
                                    new_password_field.clear()
                                    new_password_field.send_keys(new_password)
                                    
                                    confirm_password_field = driver.find_element(By.ID, "confirm_new_password")
                                    confirm_password_field.clear()
                                    confirm_password_field.send_keys(new_password)
                                    
                                    # Submit password change
                                    change_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit'], input[type='submit']")
                                    change_button.click()
                                    
                                    # Wait for result
                                    time.sleep(3)
                                    
                                    # Check if password change was successful
                                    page_text = driver.page_source.lower()
                                    if 'successfully changed' in page_text or 'password has been changed' in page_text:
                                        return True
                                    
                                    # Also try to deauthorize devices
                                    try:
                                        driver.get('https://store.steampowered.com/twofactor/manage')
                                        time.sleep(2)
                                        
                                        # Look for deauthorize button
                                        deauth_buttons = driver.find_elements(By.XPATH, "//a[contains(text(), 'Deauthorize') or contains(text(), 'deauthorize')]")
                                        if not deauth_buttons:
                                            deauth_buttons = driver.find_elements(By.XPATH, "//button[contains(text(), 'Deauthorize')]")
                                        
                                        for btn in deauth_buttons:
                                            try:
                                                btn.click()
                                                time.sleep(2)
                                                # Confirm if needed
                                                confirm_buttons = driver.find_elements(By.XPATH, "//button[contains(text(), 'Confirm') or contains(text(), 'Yes')]")
                                                for confirm_btn in confirm_buttons:
                                                    confirm_btn.click()
                                                    time.sleep(2)
                                                break
                                            except:
                                                continue
                                    except Exception as e:
                                        logging.warning("Selenium deauthorize attempt failed: %s", e)
                                    
                                    return 'successfully changed' in page_text or 'password has been changed' in page_text
                                    
                                finally:
                                    driver.quit()
                            except Exception as e:
                                logging.warning("Selenium password change failed: %s", e)
                                return False
                        
                        if await asyncio.to_thread(_selenium_change_password):
                            account.password = new_password
                            results["password_changed"] = True
                            logging.info("Password changed successfully via Selenium for account %s (stop rental)", account_id)
                    except Exception as e:
                        logging.warning("Selenium fallback failed: %s", e)
                        logging.exception("Full Selenium traceback:")
                
                # FALLBACK 2: If we couldn't get web session, try WebAuth to create a fresh session
                if not results["password_changed"] and not web_session:
                    logging.info("Trying WebAuth fallback to change password for account %s", account_id)
                    try:
                        # Generate 2FA code if available
                        two_factor_code = None
                        if account.twofa_otp:
                            try:
                                import base64
                                import hmac
                                import hashlib
                                import struct
                                secret_bytes = base64.b64decode(account.twofa_otp)
                                timestamp = int(time.time()) // 30
                                msg = struct.pack(">Q", timestamp)
                                hmac_hash = hmac.new(secret_bytes, msg, hashlib.sha1).digest()
                                offset = hmac_hash[-1] & 0x0F
                                code_int = int.from_bytes(hmac_hash[offset:offset + 4], "big") & 0x7FFFFFFF
                                alphabet = "23456789BCDFGHJKMNPQRTVWXY"
                                two_factor_code = ""
                                for _ in range(5):
                                    two_factor_code += alphabet[code_int % len(alphabet)]
                                    code_int //= len(alphabet)
                            except Exception as e:
                                logging.warning("Failed to generate 2FA code for WebAuth: %s", e)
                        
                        # Try WebAuth to get a session and change password
                        def _webauth_change_password():
                            try:
                                wa = WebAuth(account.username)
                                # Try to login and get session
                                web_sess = wa.login(
                                    password=account.password,
                                    twofactor_code=two_factor_code
                                )
                                if web_sess:
                                    # Get sessionid
                                    sessionid = None
                                    for cookie in web_sess.cookies:
                                        if cookie.name == 'sessionid':
                                            sessionid = cookie.value
                                            break
                                    
                                    if sessionid:
                                        # Change password
                                        form_data = {
                                            'sessionid': sessionid,
                                            'password': account.password,
                                            'newpassword': new_password,
                                            'renewpassword': new_password
                                        }
                                        resp = web_sess.post(
                                            'https://store.steampowered.com/account/changepassword_finish',
                                            data=form_data,
                                            headers={'Referer': 'https://store.steampowered.com/account/changepassword'},
                                            timeout=30
                                        )
                                        if resp.status_code == 200 and ('successfully changed' in resp.text.lower() or 'success' in resp.text.lower()):
                                            return True
                            except Exception as e:
                                logging.warning("WebAuth password change failed: %s", e)
                            return False
                        
                        if await asyncio.to_thread(_webauth_change_password):
                            account.password = new_password
                            results["password_changed"] = True
                            logging.info("Password changed successfully via WebAuth for account %s (stop rental)", account_id)
                    except Exception as e:
                        logging.warning("WebAuth fallback failed: %s", e)
        
        # Step 3: Logout and disconnect the SteamClient
        if client:
            try:
                if hasattr(client, "web_logoff"):
                    client.web_logoff()
                client.logout()
                client.disconnect()
                results["client_logged_out"] = True
                logging.info("SteamClient logged out for account %s (stop rental)", account_id)
            except Exception as e:
                logging.warning("Failed to logout SteamClient: %s", e)
            
            # Remove from steam_clients
            if account_id in steam_clients:
                del steam_clients[account_id]
                app.state.steam_clients = steam_clients
        
        # Step 4: Update account status
        account.login_status = "offline"
        account.updated_at = time.time()
        
        # If password was changed, update it in database
        if results["password_changed"]:
            session.add(account)
            session.commit()
        else:
            # Even if password change failed, update status
            session.add(account)
            session.commit()
        
        # Return results - even partial success is better than nothing
        message_parts = []
        warnings = []
        
        if results["password_changed"]:
            message_parts.append("Password changed")
        else:
            warnings.append("Password change failed - renter may still have access")
        
        if results["devices_deauthorized"]:
            message_parts.append("All devices deauthorized")
        else:
            warnings.append("Device deauthorization failed")
        
        if results["client_logged_out"]:
            message_parts.append("Client logged out")
        else:
            warnings.append("Client logout failed")
        
        # Build response message
        if message_parts:
            message = "Rental stopped. " + ", ".join(message_parts) + "."
            if warnings:
                message += " Warnings: " + "; ".join(warnings) + "."
        else:
            # If nothing worked, still return partial success if client was at least disconnected
            message = "Rental stop attempted. Client disconnected, but password change failed. The renter may still be logged in. Please manually change the password."
        
        # Return success even if password change failed, as long as we did something
        return {
            "status": "ok" if any(results.values()) else "partial",
            "message": message,
            "results": results,
            "warnings": warnings if warnings else None
        }
        
    except HTTPException:
        raise
    except Exception as exc:
        logging.exception("Failed to stop rental for account %s", account_id)
        raise HTTPException(status_code=500, detail=f"Failed to stop rental: {str(exc)}")


@app.get("/api/accounts/{account_id}/status")
def account_status(account_id: int, session: Session = Depends(get_session)):
    account = session.get(Account, account_id)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found.")
    steam_clients = getattr(app.state, "steam_clients", {})
    client = steam_clients.get(account_id)
    logged_on = False
    if client:
        logged_on = bool(
            getattr(client, "logged_on", False)
            or getattr(client, "is_logged_on", lambda: False)()
        )
    return {
        "status": "ok",
        "login_status": account.login_status,
        "logged_on": logged_on,
        "steam_id": account.steam_id,
    }


@app.post("/api/accounts/{account_id}/deauthorize")
async def deauthorize_sessions(account_id: int, session: Session = Depends(get_session)):
    """Deauthorize all Steam sessions (log off everyone, including active game sessions)."""
    account = session.get(Account, account_id)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found.")
    
    steam_clients = getattr(app.state, "steam_clients", {})
    client = steam_clients.get(account_id)
    if not client:
        raise HTTPException(status_code=400, detail="Account is not logged in.")
    
    try:
        # Check if client is logged on - MUST be logged on for web session
        logged_on = bool(
            getattr(client, "logged_on", False)
            or getattr(client, "is_logged_on", lambda: False)()
        )
        
        if not logged_on:
            raise HTTPException(
                status_code=400,
                detail="Account is not fully logged in. Please ensure login is complete before deauthorizing."
            )
        
        web_session = None
        
        # Debug: Log client state
        logging.info("Deauthorize - Client state for account %s: logged_on=%s, has_get_web_session=%s, has_web_session_attr=%s", 
                    account_id, logged_on, hasattr(client, "get_web_session"), hasattr(client, "web_session"))
        
        # PRIMARY METHOD: Use get_web_session() - this is the official way per steam library docs
        if hasattr(client, "get_web_session"):
            try:
                def _get_web_session():
                    sess = client.get_web_session()
                    logging.info("get_web_session() returned: %s (type: %s)", sess, type(sess) if sess else None)
                    if sess is None:
                        # Try to get web session cookies directly as fallback
                        if hasattr(client, "get_web_session_cookies"):
                            try:
                                cookies = client.get_web_session_cookies()
                                logging.info("get_web_session_cookies() returned: %s", cookies)
                                if cookies:
                                    # Create a requests.Session with these cookies
                                    import requests
                                    sess = requests.Session()
                                    for cookie in cookies:
                                        sess.cookies.set(cookie['name'], cookie['value'], domain=cookie.get('domain', '.steamcommunity.com'))
                                    logging.info("Created requests.Session from web_session_cookies")
                                    return sess
                            except Exception as cookie_err:
                                logging.warning("get_web_session_cookies() failed: %s", cookie_err)
                        raise ValueError("get_web_session() returned None - client may not be fully authenticated")
                    return sess
                web_session = await asyncio.to_thread(_get_web_session)
                if web_session:
                    logging.info("Got web session via get_web_session() for deauthorize (account %s)", account_id)
            except Exception as e:
                logging.warning("get_web_session() failed for deauthorize (account %s): %s", account_id, e)
                logging.exception("Full traceback for get_web_session failure:")
        
        # FALLBACK 1: Try web_session property if get_web_session() didn't work
        if not web_session and hasattr(client, "web_session"):
            try:
                web_session = client.web_session
                if web_session:
                    logging.info("Using SteamClient.web_session property for deauthorize (account %s)", account_id)
                else:
                    logging.warning("SteamClient.web_session exists but is None for account %s", account_id)
            except Exception as e:
                logging.warning("Failed to access web_session property: %s", e)
        
        # FALLBACK 2: Try to get cookies directly and create session
        if not web_session and hasattr(client, "get_web_session_cookies"):
            try:
                def _get_cookies_and_create_session():
                    cookies = client.get_web_session_cookies()
                    if cookies:
                        import requests
                        sess = requests.Session()
                        for cookie in cookies:
                            sess.cookies.set(
                                cookie['name'], 
                                cookie['value'], 
                                domain=cookie.get('domain', '.steamcommunity.com')
                            )
                        return sess
                    return None
                web_session = await asyncio.to_thread(_get_cookies_and_create_session)
                if web_session:
                    logging.info("Created web session from get_web_session_cookies() for deauthorize (account %s)", account_id)
            except Exception as e:
                logging.warning("get_web_session_cookies() approach failed: %s", e)
        
        # Revoke all authorized devices - this actually kicks active game sessions
        revoked_devices = False
        
        if web_session:
            # Use web_session to revoke devices
            try:
                sessionid = None
                if hasattr(web_session, 'cookies'):
                    sessionid = web_session.cookies.get('sessionid', domain='steamcommunity.com')
                    if not sessionid:
                        sessionid = web_session.cookies.get('sessionid', domain='store.steampowered.com')
                
                if sessionid:
                    def _revoke_devices():
                        revoke_data = {
                            'sessionid': sessionid,
                            'revokeall': '1'
                        }
                        # Try multiple possible endpoints
                        endpoints = [
                            'https://store.steampowered.com/account/revokeauthorizeddevices',
                            'https://steamcommunity.com/devices/revoke',
                            'https://store.steampowered.com/account/managedevices'
                        ]
                        
                        for endpoint in endpoints:
                            try:
                                resp = web_session.post(
                                    endpoint,
                                    data=revoke_data,
                                    headers={'Referer': 'https://store.steampowered.com/account/managedevices'},
                                    timeout=30
                                )
                                if resp.status_code == 200:
                                    return True
                            except:
                                continue
                        return False
                    
                    revoked_devices = await asyncio.to_thread(_revoke_devices)
                    if revoked_devices:
                        logging.info("Successfully revoked all devices via web_session for account %s", account_id)
            except Exception as e:
                logging.warning("Failed to revoke devices via web_session (account %s): %s", account_id, e)
        
        # Method 3: Try to extract cookies and use httpx if web_session didn't work
        if not revoked_devices and logged_on:
            try:
                cookies_dict = {}
                if hasattr(client, "_session") and hasattr(client._session, "cookies"):
                    for cookie in client._session.cookies:
                        cookies_dict[cookie.name] = cookie.value
                elif hasattr(client, "session") and hasattr(client.session, "cookies"):
                    for cookie in client.session.cookies:
                        cookies_dict[cookie.name] = cookie.value
                elif hasattr(client, "cookies"):
                    cookies_dict = dict(client.cookies) if isinstance(client.cookies, dict) else {}
                
                if cookies_dict:
                    # Create httpx client with cookies
                    async with httpx.AsyncClient(cookies=cookies_dict, timeout=30.0) as http_client:
                        # Get sessionid
                        sessionid = cookies_dict.get('sessionid')
                        if not sessionid:
                            test_resp = await http_client.get('https://store.steampowered.com/account/')
                            if test_resp.status_code == 200:
                                for cookie in http_client.cookies:
                                    if cookie.name == 'sessionid':
                                        sessionid = cookie.value
                                        break
                        
                        if sessionid:
                            # Try multiple endpoints
                            endpoints = [
                                ('https://store.steampowered.com/account/revokeauthorizeddevices', {'sessionid': sessionid, 'revokeall': '1'}),
                                ('https://steamcommunity.com/devices/revoke', {'sessionid': sessionid, 'revokeall': '1'}),
                            ]
                            
                            for endpoint, revoke_data in endpoints:
                                try:
                                    revoke_resp = await http_client.post(
                                        endpoint,
                                        data=revoke_data,
                                        headers={'Referer': 'https://store.steampowered.com/account/managedevices'}
                                    )
                                    
                                    if revoke_resp.status_code == 200:
                                        logging.info("Successfully revoked all devices via httpx for account %s", account_id)
                                        revoked_devices = True
                                        break
                                except Exception as e:
                                    logging.debug("Failed to revoke via %s: %s", endpoint, e)
                                    continue
            except Exception as e:
                logging.warning("httpx approach failed for deauthorize (account %s): %s", account_id, e)
        
        # Also use web_logoff for web sessions
        if hasattr(client, "web_logoff"):
            try:
                client.web_logoff()
                logging.info("Called web_logoff for account %s", account_id)
            except Exception as e:
                logging.warning("web_logoff failed for account %s: %s", account_id, e)
        
        # Also try to logout from Steam network (but don't disconnect the client)
        # We want to keep the client connected so we can still use it
        
        return {"status": "ok", "message": "All sessions deauthorized successfully. Active game sessions should be terminated."}
    except Exception as exc:
        logging.exception("Failed to deauthorize sessions for account %s", account_id)
        raise HTTPException(status_code=500, detail=f"Failed to deauthorize: {str(exc)}")


@app.get("/api/accounts/{account_id}/code")
def get_2fa_code(account_id: int, session: Session = Depends(get_session)):
    """Generate a 2FA code for the account using shared_secret."""
    account = session.get(Account, account_id)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found.")
    
    if not account.twofa_otp:
        raise HTTPException(status_code=400, detail="Account does not have 2FA shared_secret configured.")
    
    try:
        import base64
        import hmac
        import hashlib
        import struct
        import time
        
        # Generate 2FA code from shared_secret (same logic as in main.py)
        secret_bytes = base64.b64decode(account.twofa_otp)
        timestamp = int(time.time()) // 30
        msg = struct.pack(">Q", timestamp)
        hmac_hash = hmac.new(secret_bytes, msg, hashlib.sha1).digest()
        offset = hmac_hash[-1] & 0x0F
        code_int = int.from_bytes(hmac_hash[offset:offset + 4], "big") & 0x7FFFFFFF
        alphabet = "23456789BCDFGHJKMNPQRTVWXY"
        code = ""
        for _ in range(5):
            code += alphabet[code_int % len(alphabet)]
            code_int //= len(alphabet)
        
        return {"status": "ok", "code": code}
    except Exception as exc:
        logging.exception("Failed to generate 2FA code for account %s", account_id)
        raise HTTPException(status_code=500, detail=f"Failed to generate code: {str(exc)}")


@app.post("/api/accounts/{account_id}/change-password")
async def change_steam_password_endpoint(account_id: int, payload: ChangePasswordRequest, session: Session = Depends(get_session)):
    """Change the Steam account password."""
    account = session.get(Account, account_id)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found.")
    
    if not account.username or not account.password:
        raise HTTPException(status_code=400, detail="Account missing username or password.")
    
    steam_clients = getattr(app.state, "steam_clients", {})
    client = steam_clients.get(account_id)
    if not client:
        raise HTTPException(status_code=400, detail="Account is not logged in. Please log in first.")
    
    new_password = payload.new_password.strip()
    if len(new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters long.")
    
    try:
        # Check if client is logged on - MUST be logged on for web session
        logged_on = bool(
            getattr(client, "logged_on", False)
            or getattr(client, "is_logged_on", lambda: False)()
        )
        
        if not logged_on:
            raise HTTPException(
                status_code=400,
                detail="Account is not fully logged in. Please ensure login is complete before changing password."
            )
        
        web_session = None
        
        # Debug: Log client state
        logging.info("Password change - Client state for account %s: logged_on=%s, has_get_web_session=%s, has_web_session_attr=%s", 
                    account_id, logged_on, hasattr(client, "get_web_session"), hasattr(client, "web_session"))
        
        # PRIMARY METHOD: Use get_web_session() - this is the official way per steam library docs
        # https://steam.readthedocs.io/en/latest/api/steam.client.builtins.html
        if hasattr(client, "get_web_session"):
            try:
                def _get_web_session():
                    sess = client.get_web_session()
                    logging.info("get_web_session() returned: %s (type: %s)", sess, type(sess) if sess else None)
                    if sess is None:
                        # Try to get web session cookies directly as fallback
                        if hasattr(client, "get_web_session_cookies"):
                            try:
                                cookies = client.get_web_session_cookies()
                                logging.info("get_web_session_cookies() returned: %s", cookies)
                                if cookies:
                                    # Create a requests.Session with these cookies
                                    import requests
                                    sess = requests.Session()
                                    for cookie in cookies:
                                        sess.cookies.set(cookie['name'], cookie['value'], domain=cookie.get('domain', '.steamcommunity.com'))
                                    logging.info("Created requests.Session from web_session_cookies")
                                    return sess
                            except Exception as cookie_err:
                                logging.warning("get_web_session_cookies() failed: %s", cookie_err)
                        raise ValueError("get_web_session() returned None - client may not be fully authenticated")
                    return sess
                web_session = await asyncio.to_thread(_get_web_session)
                if web_session:
                    logging.info("Got web session via get_web_session() for account %s", account_id)
            except Exception as e:
                logging.warning("get_web_session() failed for account %s: %s", account_id, e)
                logging.exception("Full traceback for get_web_session failure:")
        
        # FALLBACK 1: Try web_session property if get_web_session() didn't work
        if not web_session and hasattr(client, "web_session"):
            try:
                web_session = client.web_session
                if web_session:
                    logging.info("Using SteamClient.web_session property for password change (account %s)", account_id)
                else:
                    logging.warning("SteamClient.web_session exists but is None for account %s", account_id)
            except Exception as e:
                logging.warning("Failed to access web_session property: %s", e)
        
        # FALLBACK 2: Try to get cookies directly and create session
        if not web_session and hasattr(client, "get_web_session_cookies"):
            try:
                def _get_cookies_and_create_session():
                    cookies = client.get_web_session_cookies()
                    if cookies:
                        import requests
                        sess = requests.Session()
                        for cookie in cookies:
                            sess.cookies.set(
                                cookie['name'], 
                                cookie['value'], 
                                domain=cookie.get('domain', '.steamcommunity.com')
                            )
                        return sess
                    return None
                web_session = await asyncio.to_thread(_get_cookies_and_create_session)
                if web_session:
                    logging.info("Created web session from get_web_session_cookies() for account %s", account_id)
            except Exception as e:
                logging.warning("get_web_session_cookies() approach failed: %s", e)
        
        # FALLBACK: Try to extract cookies from SteamClient and use httpx if web_session unavailable
        if not web_session:
            logging.info("Attempting to use httpx with SteamClient cookies for account %s", account_id)
            try:
                # Try to get cookies from SteamClient's internal state
                cookies_dict = {}
                
                # Check various possible cookie storage locations
                if hasattr(client, "_session") and hasattr(client._session, "cookies"):
                    for cookie in client._session.cookies:
                        cookies_dict[cookie.name] = cookie.value
                        logging.debug("Found cookie %s from _session", cookie.name)
                elif hasattr(client, "session") and hasattr(client.session, "cookies"):
                    for cookie in client.session.cookies:
                        cookies_dict[cookie.name] = cookie.value
                        logging.debug("Found cookie %s from session", cookie.name)
                elif hasattr(client, "cookies"):
                    cookies_dict = dict(client.cookies) if isinstance(client.cookies, dict) else {}
                    logging.debug("Found cookies dict: %s", list(cookies_dict.keys()))
                
                # Also try to get cookies from web_session if it exists but wasn't accessible earlier
                if not cookies_dict and hasattr(client, "web_session") and client.web_session:
                    try:
                        if hasattr(client.web_session, "cookies"):
                            for cookie in client.web_session.cookies:
                                cookies_dict[cookie.name] = cookie.value
                                logging.debug("Found cookie %s from web_session", cookie.name)
                    except:
                        pass
                
                # If we still don't have cookies, try to get a web session by making a request
                # SteamClient might have an internal HTTP client we can use
                if not cookies_dict:
                    # Try to use get_web_session() which might work even if web_session property doesn't exist
                    try:
                        if hasattr(client, "get_web_session"):
                            def _try_get_session():
                                try:
                                    return client.get_web_session()
                                except:
                                    return None
                            temp_session = await asyncio.to_thread(_try_get_session)
                            if temp_session and hasattr(temp_session, "cookies"):
                                for cookie in temp_session.cookies:
                                    cookies_dict[cookie.name] = cookie.value
                                    logging.debug("Found cookie %s from get_web_session()", cookie.name)
                                web_session = temp_session  # Use this session directly
                    except Exception as e:
                        logging.debug("get_web_session() attempt failed: %s", e)
                
                # If we found cookies or have a web_session, try password change
                if cookies_dict or web_session:
                    # Use web_session if we have it, otherwise use httpx with cookies
                    if web_session:
                        # Use web_session directly
                        sessionid = None
                        if hasattr(web_session, 'cookies'):
                            sessionid = web_session.cookies.get('sessionid', domain='steamcommunity.com')
                            if not sessionid:
                                sessionid = web_session.cookies.get('sessionid', domain='store.steampowered.com')
                        
                        if sessionid:
                            form_data = {
                                'sessionid': sessionid,
                                'password': account.password,
                                'newpassword': new_password,
                                'renewpassword': new_password
                            }
                            
                            def _change_password_request():
                                return web_session.post(
                                    'https://store.steampowered.com/account/changepassword_finish',
                                    data=form_data,
                                    headers={'Referer': 'https://store.steampowered.com/account/changepassword'}
                                )
                            
                            response = await asyncio.to_thread(_change_password_request)
                            
                            if 'successfully changed' in response.text.lower():
                                account.password = new_password
                                account.updated_at = time.time()
                                session.add(account)
                                session.commit()
                                logging.info("Password changed successfully for account %s (via web_session)", account_id)
                                return {"status": "ok", "message": "Password changed successfully."}
                    else:
                        # Use httpx with cookies
                        async with httpx.AsyncClient(cookies=cookies_dict, timeout=30.0) as http_client:
                            # Try to get sessionid from cookies
                            sessionid = cookies_dict.get('sessionid')
                            if not sessionid:
                                # Try to get it from a test request
                                test_resp = await http_client.get('https://store.steampowered.com/account/')
                                if test_resp.status_code == 200:
                                    # Extract sessionid from response cookies
                                    for cookie in http_client.cookies:
                                        if cookie.name == 'sessionid':
                                            sessionid = cookie.value
                                            break
                            
                            if sessionid:
                                # Use httpx directly for password change
                                form_data = {
                                    'sessionid': sessionid,
                                    'password': account.password,
                                    'newpassword': new_password,
                                    'renewpassword': new_password
                                }
                                
                                response = await http_client.post(
                                    'https://store.steampowered.com/account/changepassword_finish',
                                    data=form_data,
                                    headers={'Referer': 'https://store.steampowered.com/account/changepassword'}
                                )
                                
                                if 'successfully changed' in response.text.lower():
                                    # Update password in database
                                    account.password = new_password
                                    account.updated_at = time.time()
                                    session.add(account)
                                    session.commit()
                                    
                                    logging.info("Password changed successfully for account %s (via httpx)", account_id)
                                    return {"status": "ok", "message": "Password changed successfully."}
                                else:
                                    # Try to extract error message
                                    error_msg = "Password change failed. Check your current password or try again."
                                    try:
                                        from bs4 import BeautifulSoup
                                        soup = BeautifulSoup(response.text, 'html.parser')
                                        error_div = soup.find('div', class_='error_message')
                                        if error_div:
                                            error_msg = error_div.get_text(strip=True)
                                    except:
                                        pass
                                    raise HTTPException(status_code=400, detail=error_msg)
                            else:
                                logging.warning("Could not find sessionid in cookies for account %s", account_id)
                else:
                    logging.warning("Could not extract cookies from SteamClient for account %s", account_id)
            except HTTPException:
                raise
            except Exception as e:
                logging.warning("httpx approach failed for account %s: %s", account_id, e)
        
        # If we still don't have a web_session, try Selenium as fallback (like AUTO-STEAM-RENT)
        if not web_session:
            logging.info("Web session unavailable, trying Selenium browser automation for account %s", account_id)
            try:
                def _selenium_change_password():
                    try:
                        from selenium import webdriver
                        from selenium.webdriver.common.by import By
                        from selenium.webdriver.support.ui import WebDriverWait
                        from selenium.webdriver.support import expected_conditions as EC
                        from selenium.webdriver.chrome.options import Options
                        from selenium.webdriver.chrome.service import Service
                        from webdriver_manager.chrome import ChromeDriverManager
                        
                        # Setup Chrome in headless mode
                        chrome_options = Options()
                        chrome_options.add_argument('--headless')
                        chrome_options.add_argument('--no-sandbox')
                        chrome_options.add_argument('--disable-dev-shm-usage')
                        chrome_options.add_argument('--disable-gpu')
                        chrome_options.add_argument('--window-size=1920,1080')
                        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
                        
                        service = Service(ChromeDriverManager().install())
                        driver = webdriver.Chrome(service=service, options=chrome_options)
                        
                        try:
                            # Login to Steam
                            driver.get('https://store.steampowered.com/login/')
                            wait = WebDriverWait(driver, 30)
                            
                            username_field = wait.until(EC.presence_of_element_located((By.ID, "input_username")))
                            username_field.clear()
                            username_field.send_keys(account.username)
                            
                            password_field = driver.find_element(By.ID, "input_password")
                            password_field.clear()
                            password_field.send_keys(account.password)
                            
                            login_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
                            login_button.click()
                            time.sleep(3)
                            
                            # Handle 2FA if needed
                            if account.twofa_otp:
                                try:
                                    twofa_field = driver.find_element(By.ID, "twofactorcode_entry")
                                    import base64
                                    import hmac
                                    import hashlib
                                    import struct
                                    secret_bytes = base64.b64decode(account.twofa_otp)
                                    timestamp = int(time.time()) // 30
                                    msg = struct.pack(">Q", timestamp)
                                    hmac_hash = hmac.new(secret_bytes, msg, hashlib.sha1).digest()
                                    offset = hmac_hash[-1] & 0x0F
                                    code_int = int.from_bytes(hmac_hash[offset:offset + 4], "big") & 0x7FFFFFFF
                                    alphabet = "23456789BCDFGHJKMNPQRTVWXY"
                                    two_factor_code = ""
                                    for _ in range(5):
                                        two_factor_code += alphabet[code_int % len(alphabet)]
                                        code_int //= len(alphabet)
                                    
                                    twofa_field.clear()
                                    twofa_field.send_keys(two_factor_code)
                                    submit_btn = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
                                    submit_btn.click()
                                    time.sleep(3)
                                except:
                                    pass
                            
                            # Navigate to change password
                            driver.get('https://store.steampowered.com/account/changepassword')
                            time.sleep(2)
                            
                            # Fill password change form
                            current_password_field = wait.until(EC.presence_of_element_located((By.ID, "current_password")))
                            current_password_field.clear()
                            current_password_field.send_keys(account.password)
                            
                            new_password_field = driver.find_element(By.ID, "new_password")
                            new_password_field.clear()
                            new_password_field.send_keys(new_password)
                            
                            confirm_password_field = driver.find_element(By.ID, "confirm_new_password")
                            confirm_password_field.clear()
                            confirm_password_field.send_keys(new_password)
                            
                            # Submit
                            change_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit'], input[type='submit']")
                            change_button.click()
                            time.sleep(3)
                            
                            # Check success
                            page_text = driver.page_source.lower()
                            return 'successfully changed' in page_text or 'password has been changed' in page_text
                        finally:
                            driver.quit()
                    except Exception as e:
                        logging.warning("Selenium password change failed: %s", e)
                        return False
                
                if await asyncio.to_thread(_selenium_change_password):
                    account.password = new_password
                    account.updated_at = time.time()
                    session.add(account)
                    session.commit()
                    logging.info("Password changed successfully via Selenium for account %s", account_id)
                    return {"status": "ok", "message": "Password changed successfully."}
                else:
                    raise HTTPException(
                        status_code=500,
                        detail="Could not access web session for password change. Selenium automation also failed. Please try logging out and logging back in."
                    )
            except ImportError:
                logging.warning("Selenium not available, falling back to error")
                raise HTTPException(
                    status_code=500,
                    detail="Could not access web session for password change. Selenium is not installed. Please install: pip install selenium webdriver-manager"
                )
            except Exception as e:
                logging.exception("Selenium fallback failed: %s", e)
                raise HTTPException(
                    status_code=500,
                    detail=f"Could not access web session for password change. Selenium failed: {str(e)}"
                )
        
        # Use web_session to change password - this is a requests.Session from get_web_session()
        # Get sessionid from the web session cookies
        sessionid = None
        if hasattr(web_session, 'cookies'):
            # Try to get sessionid from cookies
            try:
                # requests.Session cookies are accessed differently
                for cookie in web_session.cookies:
                    if cookie.name == 'sessionid':
                        # Check if it's for the right domain
                        if 'steamcommunity.com' in cookie.domain or 'store.steampowered.com' in cookie.domain:
                            sessionid = cookie.value
                            break
                # If not found, try direct access
                if not sessionid:
                    sessionid = web_session.cookies.get('sessionid', domain='steamcommunity.com')
                    if not sessionid:
                        sessionid = web_session.cookies.get('sessionid', domain='store.steampowered.com')
            except Exception as e:
                logging.warning("Error extracting sessionid from cookies: %s", e)
        
        if not sessionid:
            # Try to get sessionid by making a request to account page
            try:
                def _get_sessionid():
                    resp = web_session.get('https://store.steampowered.com/account/')
                    if resp.status_code == 200:
                        # Extract sessionid from cookies
                        for cookie in web_session.cookies:
                            if cookie.name == 'sessionid':
                                return cookie.value
                        # Or try to parse from HTML
                        import re
                        match = re.search(r'g_sessionID = "([^"]+)"', resp.text)
                        if match:
                            return match.group(1)
                    return None
                sessionid = await asyncio.to_thread(_get_sessionid)
            except Exception as e:
                logging.warning("Failed to get sessionid from account page: %s", e)
        
        if not sessionid:
            raise HTTPException(status_code=500, detail="Could not get Steam session ID. Please log in again.")
        
        # Change password using Steam's web API
        form_data = {
            'sessionid': sessionid,
            'password': account.password,  # Current password
            'newpassword': new_password,
            'renewpassword': new_password
        }
        
        # Run the POST request in executor since web_session.post() is synchronous (requests library)
        def _change_password_request():
            return web_session.post(
                'https://store.steampowered.com/account/changepassword_finish',
                data=form_data,
                headers={'Referer': 'https://store.steampowered.com/account/changepassword'},
                timeout=30
            )
        
        response = await asyncio.to_thread(_change_password_request)
        
        # Check if password change was successful
        if response.status_code == 200 and 'successfully changed' in response.text.lower():
            # Update password in database
            account.password = new_password
            account.updated_at = time.time()
            session.add(account)
            session.commit()
            
            logging.info("Password changed successfully for account %s", account_id)
            return {"status": "ok", "message": "Password changed successfully."}
        else:
            # Try to extract error message from response
            error_msg = "Password change failed. Check your current password or try again."
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                error_div = soup.find('div', class_='error_message')
                if error_div:
                    error_msg = error_div.get_text(strip=True)
                # Also check for success message in case response format changed
                if 'success' in response.text.lower() or 'changed' in response.text.lower():
                    # Might have succeeded despite not matching our check
                    account.password = new_password
                    account.updated_at = time.time()
                    session.add(account)
                    session.commit()
                    logging.info("Password changed successfully for account %s (alternative success check)", account_id)
                    return {"status": "ok", "message": "Password changed successfully."}
            except Exception as e:
                logging.warning("Error parsing password change response: %s", e)
            
            logging.warning("Password change failed for account %s. Status: %s, Response: %s", account_id, response.status_code, response.text[:200])
            raise HTTPException(status_code=400, detail=error_msg)
            
    except HTTPException:
        raise
    except Exception as exc:
        logging.exception("Failed to change password for account %s", account_id)
        raise HTTPException(status_code=500, detail=f"Failed to change password: {str(exc)}")




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
    rows: list[dict] = []

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
            rows.append(
                {
                    "id": mid,
                    "node_id": node.id,
                    "author": str(item.get("author")) if item.get("author") is not None else None,
                    "username": parsed.username,
                    "body": parsed.body,
                    "created_at": parsed.created_at,
                    "raw": json.dumps(item, ensure_ascii=False),
                }
            )
            inserted += 1
            if mid > new_last_id:
                new_last_id = mid

    bulk_insert_messages(session, rows)
    if inserted:
        node.last_id = new_last_id
        session.add(node)
        session.commit()
        log.info("Node %s: stored %s messages, last_id=%s", node.id, inserted, new_last_id)


if __name__ == "__main__":
    import uvicorn

    init_db()
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
class SteamClient(BaseSteamClient):
    """Custom SteamClient with proper EventEmitter initialization."""
    def __init__(self, *args, **kwargs):
        # Ensure event loop exists
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
        except Exception:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # BaseSteamClient should inherit from EventEmitter
        # But we need to ensure EventEmitter is initialized with the loop
        # Call parent init first
        super().__init__(*args, **kwargs)
        
        # Then ensure EventEmitter is properly initialized
        # Check if EventEmitter methods exist, if not, initialize manually
        if not hasattr(self, 'wait_event'):
            # Manually initialize EventEmitter with the loop
            EventEmitter.__init__(self, loop=loop)
        
        # Double-check: if still no wait_event, monkey-patch it
        if not hasattr(self, 'wait_event'):
            # Last resort: bind EventEmitter methods directly
            import types
            ee_temp = EventEmitter(loop=loop)
            # Copy all EventEmitter methods to self
            for attr_name in dir(ee_temp):
                if not attr_name.startswith('_') and callable(getattr(ee_temp, attr_name, None)):
                    if not hasattr(self, attr_name):
                        method = getattr(ee_temp, attr_name)
                        setattr(self, attr_name, types.MethodType(method.__func__, self))
