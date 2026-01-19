import asyncio
import base64
import contextlib
import hashlib
import hmac
import json
import logging
import os
import secrets
import string
import struct
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


# -------- Steam helpers --------
STEAM_GUARD_ALPHABET = "23456789BCDFGHJKMNPQRTVWXY"
_steam_client_class = None


def _ensure_event_loop() -> asyncio.AbstractEventLoop:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        pass
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


def get_steam_client_class():
    global _steam_client_class
    if _steam_client_class is not None:
        return _steam_client_class
    try:
        from steam.client import SteamClient as BaseSteamClient
        from eventemitter import EventEmitter
    except ImportError as exc:
        raise RuntimeError("Steam dependencies missing. Install steam and eventemitter.") from exc

    class SteamClient(BaseSteamClient):
        def __init__(self, *args, **kwargs):
            loop = _ensure_event_loop()
            super().__init__(*args, **kwargs)
            if not hasattr(self, "wait_event"):
                EventEmitter.__init__(self, loop=loop)
            if not hasattr(self, "wait_event"):
                import types

                ee_temp = EventEmitter(loop=loop)
                for attr_name in dir(ee_temp):
                    if attr_name.startswith("_"):
                        continue
                    value = getattr(ee_temp, attr_name, None)
                    if callable(value) and not hasattr(self, attr_name):
                        setattr(self, attr_name, types.MethodType(value.__func__, self))

    _steam_client_class = SteamClient
    return SteamClient


def generate_steam_guard_code(shared_secret: str) -> Optional[str]:
    try:
        secret_bytes = base64.b64decode(shared_secret)
    except Exception:
        return None
    timestamp = int(time.time()) // 30
    msg = struct.pack(">Q", timestamp)
    hmac_hash = hmac.new(secret_bytes, msg, hashlib.sha1).digest()
    offset = hmac_hash[-1] & 0x0F
    code_int = int.from_bytes(hmac_hash[offset:offset + 4], "big") & 0x7FFFFFFF
    code = ""
    for _ in range(5):
        code += STEAM_GUARD_ALPHABET[code_int % len(STEAM_GUARD_ALPHABET)]
        code_int //= len(STEAM_GUARD_ALPHABET)
    return code


def normalize_guard_code(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    cleaned = code.strip().upper()
    if re.fullmatch(r"[A-Z0-9]{5}", cleaned):
        return cleaned
    return None


def resolve_guard_code(stored: Optional[str]) -> Optional[str]:
    if not stored:
        return None
    generated = generate_steam_guard_code(stored)
    if generated:
        return generated
    return normalize_guard_code(stored)


def generate_password(length: int = 16) -> str:
    alphabet = string.ascii_letters + string.digits + "!@#$"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def safe_disconnect(client) -> None:
    if not client:
        return
    try:
        if getattr(client, "logged_on", False):
            client.logout()
        elif hasattr(client, "disconnect"):
            client.disconnect()
    except Exception:
        pass


def map_login_result(result) -> str:
    try:
        from steam.enums import EResult
    except ImportError:
        return "error:exception"

    if result == "connect":
        return "error:connect"
    if result == EResult.OK:
        return "online"
    if result == EResult.AccountLogonDenied:
        return "guard:email"
    if result == EResult.AccountLoginDeniedNeedTwoFactor:
        return "guard:twofactor"
    if result in {EResult.InvalidPassword}:
        return "error:invalid_password"
    if hasattr(EResult, "AccountNotFound") and result == EResult.AccountNotFound:
        return "error:account_not_found"
    if hasattr(EResult, "AccountLoginDeniedThrottle") and result == EResult.AccountLoginDeniedThrottle:
        return "error:rate_limit"
    if hasattr(EResult, "RateLimitExceeded") and result == EResult.RateLimitExceeded:
        return "error:rate_limit"
    if hasattr(EResult, "InvalidLoginAuthCode") and result == EResult.InvalidLoginAuthCode:
        return "error:invalid_auth_code"
    return "error:exception"


def login_error_message(status: str) -> str:
    if status == "guard:email":
        return "Email guard code required. Use login with the email code first."
    if status == "guard:twofactor":
        return "Two-factor code required. Use login with the 2FA code first."
    if status == "error:invalid_password":
        return "Invalid username or password."
    if status == "error:invalid_auth_code":
        return "Invalid guard code."
    if status == "error:account_not_found":
        return "Steam account not found."
    if status == "error:rate_limit":
        return "Steam rate limit. Try again later."
    if status == "error:connect":
        return "Failed to connect to Steam."
    return "Steam login failed."


def steam_login(account, guard_code: Optional[str] = None):
    SteamClient = get_steam_client_class()
    try:
        from steam.enums import EResult
    except ImportError as exc:
        raise RuntimeError("Steam dependency missing: steam") from exc

    client = SteamClient()
    if not client.connect():
        return None, "connect"

    auth_code = None
    two_factor_code = None
    cleaned_guard = normalize_guard_code(guard_code)
    if cleaned_guard:
        if account.login_status == "guard:email":
            auth_code = cleaned_guard
        else:
            two_factor_code = cleaned_guard
    else:
        fallback = resolve_guard_code(account.twofa_otp)
        if fallback:
            two_factor_code = fallback

    result = client.login(
        account.username,
        account.password,
        two_factor_code=two_factor_code,
        auth_code=auth_code,
    )
    if result is None:
        result = EResult.Fail
    return client, result


def deauthorize_all_sessions(client) -> bool:
    try:
        client.web_logoff()
        return True
    except Exception:
        return False


def change_steam_password(client, old_password: str, new_password: str) -> bool:
    try:
        sessionid = client.web_session.cookies.get("sessionid", domain="steamcommunity.com")
        if not sessionid:
            sessionid = client.web_session.cookies.get("sessionid")
        if not sessionid:
            return False
        form_data = {
            "sessionid": sessionid,
            "password": old_password,
            "newpassword": new_password,
            "renewpassword": new_password,
        }
        response = client.web_session.post(
            "https://store.steampowered.com/account/changepassword_finish",
            data=form_data,
            headers={"Referer": "https://store.steampowered.com/account/changepassword"},
        )
        return "successfully changed" in response.text.lower()
    except Exception:
        return False


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
    username: str = Field(index=True)
    password: str
    steam_id: Optional[str] = None
    login_status: str = Field(default="idle")
    twofa_otp: Optional[str] = None




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

    async def get_orders(self) -> list[dict]:
        """
        Get active orders from FunPay (AUTO-STEAM-RENT style).
        """
        await self.ensure_ready()
        
        # Try multiple endpoints for orders
        endpoints = [
            "/orders/",
            "/orders?type=purchases",  # Purchases (we're the seller)
            "/orders?type=sales",  # Sales
        ]
        
        for endpoint in endpoints:
            try:
                resp = await self.client.get(endpoint)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                
                orders = []
                
                # Try multiple selectors for order items
                order_selectors = [
                    ".order-item",
                    ".order",
                    "[data-order-id]",
                    ".tc-item.order",
                ]
                
                for selector in order_selectors:
                    order_items = soup.select(selector)
                    if order_items:
                        for item in order_items:
                            order_id = (item.get("data-order-id") or
                                       item.get("data-id") or
                                       item.get("id", "").replace("order-", ""))
                            
                            if not order_id:
                                continue
                            
                            # Extract buyer info
                            buyer_link = item.select_one("a[href*='/users/']")
                            buyer_id = None
                            if buyer_link:
                                href = buyer_link.get("href", "")
                                match = re.search(r"/users/(\d+)/", href)
                                if match:
                                    buyer_id = match.group(1)
                            
                            # Extract lot name
                            lot_name_el = (item.select_one(".order-lot-name") or
                                         item.select_one(".lot-name") or
                                         item.select_one(".order-title") or
                                         item.select_one("h3") or
                                         item.select_one("h4"))
                            lot_name = lot_name_el.get_text(strip=True) if lot_name_el else ""
                            
                            # Extract amount
                            amount_el = (item.select_one(".order-amount") or
                                       item.select_one(".amount") or
                                       item.select_one(".price") or
                                       item.select_one("[class*='amount']"))
                            amount_text = amount_el.get_text(strip=True) if amount_el else "0"
                            # Extract number from amount text
                            amount_match = re.search(r"(\d+(?:[.,]\d+)?)", amount_text.replace(",", "."))
                            amount = float(amount_match.group(1)) if amount_match else 0.0
                            
                            # Extract status
                            status_el = (item.select_one(".order-status") or
                                       item.select_one(".status") or
                                       item.select_one("[class*='status']"))
                            status = status_el.get_text(strip=True) if status_el else ""
                            
                            orders.append({
                                "order_id": order_id,
                                "buyer_id": buyer_id,
                                "lot_name": lot_name,
                                "amount": amount,
                                "status": status,
                            })
                        
                        if orders:
                            logging.info("Found %d orders from endpoint: %s", len(orders), endpoint)
                            return orders
                
            except Exception as e:
                logging.warning("Failed to fetch orders from %s: %s", endpoint, e)
                continue
        
        logging.warning("Failed to fetch orders from any known endpoint")
        return []


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
    
    rental_cleanup_task = getattr(app.state, "rental_cleanup_task", None)
    if rental_cleanup_task:
        rental_cleanup_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await rental_cleanup_task
    
    client: FunpayClient = app.state.fp_client
    if client:
        await client.close()


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
        # Try multiple times with better error handling (AUTO-STEAM-RENT style)
        last_error = None
        for attempt in range(3):
            try:
                dialogs = await client.get_dialogs()
                list_cache["data"] = dialogs
                list_cache["ts"] = now
                break  # Success
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if exc.response.status_code == 429:
                    if cached_dialogs:
                        dialogs = cached_dialogs
                        list_cache["ts"] = now
                        break
                    if attempt < 2:
                        await asyncio.sleep(5)
                        continue
                    raise HTTPException(status_code=502, detail="FunPay rate limited. Try again soon.")
                elif exc.response.status_code == 401:
                    raise HTTPException(status_code=401, detail="Unauthorized. Check Golden Key.")
                else:
                    if attempt < 2:
                        await asyncio.sleep(1)
                        continue
                    raise HTTPException(status_code=502, detail=f"FunPay error: {exc}")
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    await asyncio.sleep(1)
                    continue
                # On final failure, use cached if available
                if cached_dialogs:
                    dialogs = cached_dialogs
                    list_cache["ts"] = now
                    logging.warning("Using cached dialogs after error: %s", exc)
                else:
                    raise HTTPException(status_code=500, detail=f"Failed to load dialogs: {exc}")
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


class AccountLogin(SQLModel):
    guard_code: Optional[str] = None
    email_code: Optional[str] = None


class AccountPasswordChange(SQLModel):
    new_password: str


def account_to_dict(account: Account) -> dict:
    return {
        "id": account.id,
        "label": account.label,
        "username": account.username,
        "steam_id": account.steam_id,
        "login_status": account.login_status,
        "has_twofa_otp": bool(account.twofa_otp),
    }


def get_account_or_404(session: Session, account_id: int) -> Account:
    account = session.get(Account, account_id)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found.")
    return account


def login_or_raise(session: Session, account: Account) -> object:
    client, result = steam_login(account)
    if result is None:
        safe_disconnect(client)
        account.login_status = "error:exception"
        session.add(account)
        session.commit()
        raise HTTPException(status_code=500, detail="Steam login failed.")

    status = map_login_result(result)
    account.login_status = status
    if status != "online":
        session.add(account)
        session.commit()
        safe_disconnect(client)
        raise HTTPException(status_code=400, detail=login_error_message(status))

    if not account.steam_id and getattr(client, "steam_id", None):
        account.steam_id = str(client.steam_id)
    session.add(account)
    session.commit()
    return client


@app.get("/api/accounts")
def list_accounts(session: Session = Depends(get_session)):
    accounts = session.exec(select(Account)).all()
    return [account_to_dict(account) for account in accounts]


@app.post("/api/accounts")
def create_account(payload: AccountCreate, session: Session = Depends(get_session)):
    username = payload.username.strip()
    password = payload.password.strip()
    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password are required.")

    existing = session.exec(select(Account).where(Account.username == username)).first()
    if existing:
        raise HTTPException(status_code=409, detail="Account already exists.")

    account = Account(
        label=(payload.label or "").strip() or None,
        username=username,
        password=password,
        steam_id=(payload.steam_id or "").strip() or None,
        login_status=payload.login_status or "idle",
        twofa_otp=(payload.twofa_otp or "").strip() or None,
    )
    session.add(account)
    session.commit()
    session.refresh(account)
    return account_to_dict(account)


@app.put("/api/accounts/{account_id}")
def update_account(
    account_id: int,
    payload: AccountUpdate,
    session: Session = Depends(get_session),
):
    account = get_account_or_404(session, account_id)
    if payload.label is not None:
        account.label = payload.label.strip() or None
    if payload.username is not None:
        account.username = payload.username.strip()
    if payload.password is not None:
        account.password = payload.password.strip()
    if payload.steam_id is not None:
        account.steam_id = payload.steam_id.strip() or None
    if payload.login_status is not None:
        account.login_status = payload.login_status.strip() or "idle"
    if payload.twofa_otp is not None:
        account.twofa_otp = payload.twofa_otp.strip() or None
    session.add(account)
    session.commit()
    session.refresh(account)
    return account_to_dict(account)


@app.post("/api/accounts/{account_id}/login")
def login_account(
    account_id: int,
    payload: AccountLogin,
    session: Session = Depends(get_session),
):
    account = get_account_or_404(session, account_id)
    guard_code = payload.guard_code or payload.email_code
    try:
        client, result = steam_login(account, guard_code=guard_code)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    status = map_login_result(result)
    account.login_status = status
    if status == "online" and not account.steam_id and getattr(client, "steam_id", None):
        account.steam_id = str(client.steam_id)
    session.add(account)
    session.commit()
    safe_disconnect(client)
    return {"status": account.login_status}


@app.post("/api/accounts/{account_id}/logout")
def logout_account(account_id: int, session: Session = Depends(get_session)):
    account = get_account_or_404(session, account_id)
    account.login_status = "idle"
    session.add(account)
    session.commit()
    return {"status": "ok"}


@app.post("/api/accounts/{account_id}/deauthorize")
def deauthorize_account(account_id: int, session: Session = Depends(get_session)):
    account = get_account_or_404(session, account_id)
    client = login_or_raise(session, account)
    ok = deauthorize_all_sessions(client)
    safe_disconnect(client)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to deauthorize sessions.")
    return {"status": "ok"}


@app.post("/api/accounts/{account_id}/change-password")
def change_password(
    account_id: int,
    payload: AccountPasswordChange,
    session: Session = Depends(get_session),
):
    new_password = payload.new_password.strip()
    if len(new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")

    account = get_account_or_404(session, account_id)
    client = login_or_raise(session, account)
    ok = change_steam_password(client, account.password, new_password)
    safe_disconnect(client)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to change password.")
    account.password = new_password
    session.add(account)
    session.commit()
    return {"status": "ok"}


@app.post("/api/accounts/{account_id}/stop-rental")
def stop_rental(account_id: int, session: Session = Depends(get_session)):
    account = get_account_or_404(session, account_id)
    client = login_or_raise(session, account)
    warnings: list[str] = []

    if not deauthorize_all_sessions(client):
        warnings.append("Failed to deauthorize sessions.")

    new_password = generate_password()
    if change_steam_password(client, account.password, new_password):
        account.password = new_password
    else:
        warnings.append("Failed to change password.")

    session.add(account)
    session.commit()
    safe_disconnect(client)

    message = "Stop rental completed."
    if warnings:
        message = "Stop rental completed with warnings."
    return {"status": "ok", "message": message, "warnings": warnings, "new_password": new_password}


@app.get("/api/accounts/{account_id}/code")
def get_twofa_code(account_id: int, session: Session = Depends(get_session)):
    account = get_account_or_404(session, account_id)
    code = resolve_guard_code(account.twofa_otp)
    if not code:
        raise HTTPException(status_code=400, detail="No 2FA secret configured.")
    return {"code": code}




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
    """Get lots/offers with improved error handling (AUTO-STEAM-RENT style)."""
    client: Optional[FunpayClient] = getattr(app.state, "fp_client", None)
    if not client:
        raise HTTPException(status_code=400, detail="No active session. Set Golden Key first.")
    
    # Try multiple times with better error handling
    last_error = None
    for attempt in range(3):
        try:
            offers = await client.get_offers()
            if offers:
                return offers
            # If empty, might be temporary, try again
            if attempt < 2:
                await asyncio.sleep(1)
                continue
            return []  # Return empty array instead of error
        except httpx.HTTPStatusError as exc:
            last_error = exc
            if exc.response.status_code == 429:
                # Rate limited - wait longer
                if attempt < 2:
                    await asyncio.sleep(5)
                    continue
            elif exc.response.status_code == 401:
                raise HTTPException(status_code=401, detail="Unauthorized. Check Golden Key.")
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                await asyncio.sleep(1)
                continue
    
    # If all attempts failed, return empty array instead of error
    logging.warning("Failed to load lots after retries: %s", last_error)
    return []


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
