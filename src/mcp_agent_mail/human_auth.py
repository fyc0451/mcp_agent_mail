"""Small standalone JWT issuer for beta Human identities.

The service is intentionally separate from Agent Cockpit: Cockpit may proxy a
login request, but only this process can read the signing key or password DB.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import hmac
import ipaddress
import json
import os
import re
import secrets
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from authlib.jose import JsonWebKey, JsonWebToken
from authlib.jose.errors import JoseError
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

_USERNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_PROJECT_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_PASSWORD_MIN_BYTES = 12
_PASSWORD_MAX_BYTES = 256
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_PRIVATE_HTTP_V4 = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
)
_PRIVATE_HTTP_V6 = ipaddress.ip_network("fc00::/7")


@dataclass(frozen=True)
class HumanAuthConfig:
    data_dir: Path
    issuer: str
    audience: str
    token_ttl_seconds: int = 8 * 60 * 60


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class RegistrationRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    display_name: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=256)
    invite_code: str = Field(min_length=1, max_length=256)


class InvitationRequest(BaseModel):
    expires_in: int = Field(default=24 * 60 * 60, ge=300, le=7 * 24 * 60 * 60)
    project_slug: str | None = Field(default=None, min_length=1, max_length=128)


class UserStatusRequest(BaseModel):
    status: str = Field(min_length=1, max_length=16)


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _password_hash(password: str, *, salt: bytes | None = None) -> str:
    encoded = password.encode("utf-8")
    if not _PASSWORD_MIN_BYTES <= len(encoded) <= _PASSWORD_MAX_BYTES:
        raise ValueError("Password must be between 12 and 256 UTF-8 bytes")
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(
        encoded,
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=32,
    )
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${_b64(salt)}${_b64(digest)}"


def _password_matches(password: str, encoded_hash: str) -> bool:
    try:
        name, n, r, p, salt, expected = encoded_hash.split("$", 5)
        if name != "scrypt":
            return False
        candidate = hashlib.scrypt(
            password.encode("utf-8"),
            salt=_unb64(salt),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(_unb64(expected)),
        )
        return hmac.compare_digest(candidate, _unb64(expected))
    except (TypeError, ValueError, MemoryError):
        return False


def _write_private_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)


class HumanAuthStore:
    def __init__(self, config: HumanAuthConfig):
        self.config = config
        self.config.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.config.data_dir.chmod(0o700)
        self.db_path = self.config.data_dir / "users.sqlite3"
        self.key_path = self.config.data_dir / "signing-key.pem"
        self._ensure_schema()
        self._key = self._load_or_create_key()
        public_pem = self._key.as_pem(is_private=False)
        self.kid = hashlib.sha256(public_pem).hexdigest()[:16]
        public_jwk = self._key.as_dict(is_private=False)
        public_jwk.update({"kid": self.kid, "use": "sig", "alg": "RS256"})
        self.jwks = {"keys": [public_jwk]}
        # A real hash keeps unknown-user failures on the same expensive path.
        self._dummy_hash = _password_hash(secrets.token_urlsafe(24))

    @contextlib.contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.db_path, timeout=5)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    subject TEXT PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    display_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    roles_json TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(users)").fetchall()
            }
            if "status" not in columns:
                connection.execute(
                    "ALTER TABLE users ADD COLUMN status TEXT NOT NULL DEFAULT 'active'"
                )
                connection.execute(
                    "UPDATE users SET status = 'disabled' WHERE active = 0"
                )
            if "created_at" not in columns:
                connection.execute(
                    "ALTER TABLE users ADD COLUMN created_at INTEGER NOT NULL DEFAULT 0"
                )
                connection.execute(
                    "UPDATE users SET created_at = ? WHERE created_at = 0",
                    (int(time.time()),),
                )
            if "requested_project_slug" not in columns:
                connection.execute(
                    "ALTER TABLE users ADD COLUMN requested_project_slug TEXT"
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS invitations (
                    code_hash TEXT PRIMARY KEY,
                    created_by TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL,
                    used_by TEXT,
                    used_at INTEGER
                )
                """
            )
            invitation_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(invitations)").fetchall()
            }
            if "project_slug" not in invitation_columns:
                connection.execute(
                    "ALTER TABLE invitations ADD COLUMN project_slug TEXT"
                )
        self.db_path.chmod(0o600)

    def _load_or_create_key(self):
        if not self.key_path.exists():
            key = JsonWebKey.generate_key("RSA", 2048, is_private=True)
            with contextlib.suppress(FileExistsError):
                _write_private_file(self.key_path, key.as_pem(is_private=True))
        self.key_path.chmod(0o600)
        return JsonWebKey.import_key(self.key_path.read_bytes(), {"kty": "RSA"})

    def bootstrap_admin(
        self,
        *,
        username: str,
        display_name: str,
        credentials_path: Path,
    ) -> bool:
        username = username.strip().lower()
        display_name = display_name.strip()
        if not _USERNAME_RE.fullmatch(username):
            raise ValueError("Invalid bootstrap username")
        if not display_name or len(display_name) > 128:
            raise ValueError("Invalid bootstrap display name")
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT subject FROM users WHERE username = ?", (username,)
            ).fetchone()
            if existing is not None:
                return False
            if connection.execute("SELECT 1 FROM users LIMIT 1").fetchone() is not None:
                raise ValueError("Refusing to bootstrap an admin into a non-empty user database")
            password = secrets.token_urlsafe(24)
            connection.execute(
                """
                INSERT INTO users(
                    subject, username, display_name, password_hash, roles_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    f"human:{username}",
                    username,
                    display_name,
                    _password_hash(password),
                    json.dumps(["writer", "admin"]),
                    int(time.time()),
                ),
            )
            _write_private_file(
                credentials_path,
                (json.dumps({"username": username, "password": password}, ensure_ascii=False) + "\n").encode(),
            )
        return True

    def authenticate(self, username: str, password: str) -> sqlite3.Row | None:
        username = username.strip().lower()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT subject, username, display_name, password_hash, roles_json, status
                FROM users WHERE username = ? AND active = 1 AND status = 'active'
                """,
                (username,),
            ).fetchone()
        stored_hash = row["password_hash"] if row is not None else self._dummy_hash
        if not _password_matches(password, stored_hash):
            return None
        return row

    @staticmethod
    def profile(user: sqlite3.Row) -> dict[str, Any]:
        return {
            "username": user["username"],
            "display_name": user["display_name"],
            "roles": json.loads(user["roles_json"]),
            "status": user["status"],
        }

    def active_user(self, subject: str) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT subject, username, display_name, roles_json, status
                FROM users WHERE subject = ? AND active = 1 AND status = 'active'
                """,
                (subject,),
            ).fetchone()

    def create_invitation(
        self, *, created_by: str, expires_in: int, project_slug: str | None = None
    ) -> dict[str, Any]:
        if project_slug is not None:
            project_slug = project_slug.strip().lower()
            if not _PROJECT_SLUG_RE.fullmatch(project_slug):
                raise ValueError("Invalid project slug")
        now = int(time.time())
        code = secrets.token_urlsafe(24)
        code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
        expires_at = now + expires_in
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO invitations(
                    code_hash, created_by, created_at, expires_at, project_slug
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (code_hash, created_by, now, expires_at, project_slug),
            )
        return {
            "invite_code": code,
            "expires_at": expires_at,
            "project_slug": project_slug,
        }

    def _valid_team_code(self, invite_code: str) -> bool:
        """可重复使用的团队码(可选,存放在 data_dir/team-code)。

        团队码不过期、不限次数,按次读取文件,轮换无需重启;
        注册仍落 pending,需管理员激活后才能登录,泄露不会直接放行。
        文件缺失/不可读/非 UTF-8/超限一律视为未配置(返回 False)。
        """
        try:
            raw = (self.config.data_dir / "team-code").read_bytes()
        except OSError:
            return False
        if len(raw) > 4096:
            return False
        try:
            expected = raw.decode("utf-8").strip()
        except UnicodeDecodeError:
            return False
        if not expected:
            return False
        # 统一按 UTF-8 bytes 比较:str compare_digest 遇非 ASCII 抛 TypeError
        return hmac.compare_digest(
            invite_code.strip().encode("utf-8"), expected.encode("utf-8")
        )

    def register(
        self,
        *,
        username: str,
        display_name: str,
        password: str,
        invite_code: str,
    ) -> dict[str, Any]:
        username = username.strip().lower()
        display_name = display_name.strip()
        if not _USERNAME_RE.fullmatch(username):
            raise ValueError("Invalid username")
        if not display_name or len(display_name) > 128:
            raise ValueError("Invalid display name")
        code_hash = hashlib.sha256(invite_code.strip().encode("utf-8")).hexdigest()
        now = int(time.time())
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            invitation = connection.execute(
                """
                SELECT code_hash, project_slug FROM invitations
                WHERE code_hash = ? AND used_at IS NULL AND expires_at >= ?
                """,
                (code_hash, now),
            ).fetchone()
            # 团队码命中时不消费任何 invitation 行:团队码可能复用了
            # 已消费/过期邀请码的值,改写其 used_by/used_at 会破坏审计。
            consume_invitation = invitation is not None
            if invitation is None and not self._valid_team_code(invite_code):
                raise ValueError("Invalid or expired invitation")
            existing = connection.execute(
                "SELECT 1 FROM users WHERE username = ?", (username,)
            ).fetchone()
            if existing is not None:
                raise FileExistsError("Username is already registered")
            subject = f"human:{username}"
            connection.execute(
                """
                INSERT INTO users(
                    subject, username, display_name, password_hash, roles_json,
                    active, status, created_at, requested_project_slug
                ) VALUES (?, ?, ?, ?, ?, 0, 'pending', ?, ?)
                """,
                (
                    subject,
                    username,
                    display_name,
                    _password_hash(password),
                    json.dumps(["writer"]),
                    now,
                    invitation["project_slug"] if invitation is not None else None,
                ),
            )
            if consume_invitation:
                connection.execute(
                    "UPDATE invitations SET used_by = ?, used_at = ? WHERE code_hash = ?",
                    (subject, now, code_hash),
                )
        return {
            "username": username,
            "display_name": display_name,
            "status": "pending",
            "requested_project_slug": (
                invitation["project_slug"] if invitation is not None else None
            ),
        }

    def list_users(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT subject, username, display_name, roles_json, status,
                       created_at, requested_project_slug
                FROM users ORDER BY created_at, username
                """
            ).fetchall()
        return [
            {
                "subject": row["subject"],
                "username": row["username"],
                "display_name": row["display_name"],
                "roles": json.loads(row["roles_json"]),
                "status": row["status"],
                "requested_project_slug": row["requested_project_slug"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def set_user_status(
        self, *, actor_subject: str, username: str, status: str
    ) -> dict[str, Any]:
        if status not in {"active", "disabled"}:
            raise ValueError("Status must be active or disabled")
        username = username.strip().lower()
        with self._connect() as connection:
            target = connection.execute(
                """
                SELECT subject, username, display_name, roles_json, status,
                       created_at, requested_project_slug
                FROM users WHERE username = ?
                """,
                (username,),
            ).fetchone()
            if target is None:
                raise LookupError("User not found")
            if target["subject"] == actor_subject:
                raise ValueError("Administrators cannot change their own status")
            if "admin" in json.loads(target["roles_json"]):
                raise ValueError("Administrator accounts cannot be disabled here")
            connection.execute(
                "UPDATE users SET status = ?, active = ? WHERE subject = ?",
                (status, 1 if status == "active" else 0, target["subject"]),
            )
        return {
            "subject": target["subject"],
            "username": target["username"],
            "display_name": target["display_name"],
            "roles": json.loads(target["roles_json"]),
            "status": status,
            "requested_project_slug": target["requested_project_slug"],
            "created_at": target["created_at"],
        }

    def issue_token(self, user: sqlite3.Row) -> tuple[str, int]:
        now = int(time.time())
        expires = now + self.config.token_ttl_seconds
        claims = {
            "iss": self.config.issuer,
            "aud": self.config.audience,
            "sub": user["subject"],
            "iat": now,
            "nbf": now,
            "exp": expires,
            "jti": secrets.token_urlsafe(18),
            "preferred_username": user["username"],
            "name": user["display_name"],
            "role": json.loads(user["roles_json"]),
        }
        token = JsonWebToken(["RS256"]).encode(
            {"alg": "RS256", "typ": "JWT", "kid": self.kid},
            claims,
            self._key,
        )
        return token.decode("ascii"), self.config.token_ttl_seconds


class _RateLimiter:
    """按 key 的滑动窗口失败限流:5 次/300s,成功后清零。"""

    def __init__(self) -> None:
        self.failures: dict[str, list[float]] = {}

    def blocked(self, key: str, now: float) -> bool:
        recent = [value for value in self.failures.get(key, []) if now - value < 300]
        self.failures[key] = recent
        return len(recent) >= 5

    def failed(self, key: str, now: float) -> None:
        self.failures.setdefault(key, []).append(now)
        if len(self.failures) > 2048:
            self.failures.pop(next(iter(self.failures)))

    def succeeded(self, key: str) -> None:
        self.failures.pop(key, None)


def create_app(config: HumanAuthConfig) -> FastAPI:
    store = HumanAuthStore(config)
    limiter = _RateLimiter()
    register_limiter = _RateLimiter()
    app = FastAPI(title="Agent Hub Human Auth", docs_url=None, redoc_url=None)
    app.state.store = store

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/.well-known/openid-configuration")
    async def discovery() -> dict[str, Any]:
        return {
            "issuer": config.issuer,
            "jwks_uri": f"{config.issuer}/.well-known/jwks.json",
            "token_endpoint": f"{config.issuer}/token",
            "id_token_signing_alg_values_supported": ["RS256"],
        }

    @app.get("/.well-known/jwks.json")
    async def jwks() -> dict[str, Any]:
        return store.jwks

    def authenticated_user(request: Request, *, admin: bool = False) -> sqlite3.Row:
        authorization = request.headers.get("authorization", "")
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Authentication required")
        try:
            claims = JsonWebToken(["RS256"]).decode(
                authorization[7:].strip(), store._key
            )
            claims.validate()
        except (JoseError, ValueError, TypeError):
            raise HTTPException(status_code=401, detail="Authentication required") from None
        audience = claims.get("aud")
        if (
            claims.get("iss") != config.issuer
            or audience != config.audience
            or not isinstance(claims.get("sub"), str)
        ):
            raise HTTPException(status_code=401, detail="Authentication required")
        user = store.active_user(claims["sub"])
        if user is None:
            raise HTTPException(status_code=401, detail="Authentication required")
        if admin and "admin" not in json.loads(user["roles_json"]):
            raise HTTPException(status_code=403, detail="Administrator role required")
        return user

    @app.post("/token")
    async def token(request: Request, body: LoginRequest) -> dict[str, Any]:
        username = body.username.strip().lower()
        client_host = request.client.host if request.client else "unknown"
        limit_key = f"{client_host}:{username}"
        now = time.monotonic()
        if limiter.blocked(limit_key, now):
            raise HTTPException(status_code=429, detail="Too many login attempts")
        user = store.authenticate(username, body.password)
        if user is None:
            limiter.failed(limit_key, now)
            raise HTTPException(status_code=401, detail="Invalid username or password")
        limiter.succeeded(limit_key)
        access_token, expires_in = store.issue_token(user)
        return {
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": expires_in,
            "profile": store.profile(user),
        }

    @app.get("/me")
    async def me(request: Request) -> dict[str, Any]:
        return {"profile": store.profile(authenticated_user(request))}

    @app.post("/register", status_code=201)
    async def register(request: Request, body: RegistrationRequest) -> dict[str, Any]:
        # 按 client IP 失败限流:可复用团队码把爆破/抢占用户名的风险
        # 从一次性高熵码扩大为可持续尝试,必须兜底(成功后清零)。
        client_host = request.client.host if request.client else "unknown"
        now = time.monotonic()
        if register_limiter.blocked(client_host, now):
            raise HTTPException(status_code=429, detail="Too many registration attempts")
        try:
            account = store.register(
                username=body.username,
                display_name=body.display_name,
                password=body.password,
                invite_code=body.invite_code,
            )
        except FileExistsError as exc:
            register_limiter.failed(client_host, now)
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            register_limiter.failed(client_host, now)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        register_limiter.succeeded(client_host)
        return {"account": account}

    @app.post("/admin/invitations", status_code=201)
    async def create_invitation(
        request: Request, body: InvitationRequest
    ) -> dict[str, Any]:
        admin_user = authenticated_user(request, admin=True)
        try:
            return store.create_invitation(
                created_by=admin_user["subject"],
                expires_in=body.expires_in,
                project_slug=body.project_slug,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/admin/users")
    async def list_users(request: Request) -> dict[str, Any]:
        authenticated_user(request, admin=True)
        return {"users": store.list_users()}

    @app.patch("/admin/users/{username}")
    async def update_user(
        username: str, request: Request, body: UserStatusRequest
    ) -> dict[str, Any]:
        admin_user = authenticated_user(request, admin=True)
        try:
            user = store.set_user_status(
                actor_subject=admin_user["subject"],
                username=username,
                status=body.status,
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"user": user}

    return app


def _validated_issuer(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Issuer must be an absolute HTTP(S) URL without credentials")
    return value.rstrip("/")


def _validated_bind_host(value: str, *, allow_private_http: bool = False) -> str:
    """Keep the issuer loopback-only unless an exact private IP is opted in."""
    if value == "localhost":
        return value
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise ValueError(
            "bind host must be loopback or RFC1918/ULA private IP"
        ) from exc
    if address.is_loopback:
        return value
    private_lan = (
        address.version == 4
        and any(address in network for network in _PRIVATE_HTTP_V4)
    ) or (
        address.version == 6 and address in _PRIVATE_HTTP_V6
    )
    if not private_lan:
        raise ValueError("bind host must be loopback or RFC1918/ULA private IP")
    if not allow_private_http:
        raise ValueError("private bind requires explicit --allow-private-http")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the standalone beta Human JWT issuer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--issuer", required=True)
    parser.add_argument("--audience", default="mcp-agent-mail-human")
    parser.add_argument("--bootstrap-username")
    parser.add_argument("--bootstrap-display-name")
    parser.add_argument("--bootstrap-credentials-file", type=Path)
    parser.add_argument(
        "--allow-private-http",
        action="store_true",
        help="allow an exact RFC1918/ULA bind address on a trusted private network",
    )
    args = parser.parse_args()
    try:
        _validated_bind_host(
            args.host, allow_private_http=args.allow_private_http,
        )
    except ValueError as exc:
        parser.error(str(exc))
    config = HumanAuthConfig(
        data_dir=args.data_dir,
        issuer=_validated_issuer(args.issuer),
        audience=args.audience,
    )
    app = create_app(config)
    bootstrap_values = (
        args.bootstrap_username,
        args.bootstrap_display_name,
        args.bootstrap_credentials_file,
    )
    if any(bootstrap_values) and not all(bootstrap_values):
        parser.error("all bootstrap arguments are required together")
    if all(bootstrap_values):
        app.state.store.bootstrap_admin(
            username=args.bootstrap_username,
            display_name=args.bootstrap_display_name,
            credentials_path=args.bootstrap_credentials_file,
        )
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover
    main()
