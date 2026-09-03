"""HTTP transport helpers wrapping FastMCP with FastAPI."""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import hashlib
import hmac
import importlib
import json
import logging
import os
import re
import uuid
from collections.abc import MutableMapping
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Protocol, cast

import structlog
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.exception_handlers import http_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy import and_, func, or_, select, text, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError, NoResultFound
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import Receive, Scope, Send

from .app import (
    _agent_in_project_scope,
    _agent_referenced_as_default,
    _deliver_channel_mentions,
    _ensure_human,
    _expire_stale_file_reservations,
    _format_cross_project_agent_address,
    _human_by_subject,
    _membership_handle_taken,
    _sender_display_name,
    _set_agent_owner,
    _team_project_for_routing,
    _tool_metrics_snapshot,
    _upsert_project_human_membership,
    _validate_default_agent,
    build_mcp_server,
    get_project_sibling_data,
    refresh_project_sibling_suggestions,
    sweep_stale_agents,
    update_project_sibling_status,
)
from .config import Settings, get_settings
from .db import ensure_schema, get_session
from .models import (
    Agent,
    Channel,
    ChannelMessage,
    Human,
    HumanInboxItem,
    HumanPresence,
    Message,
    Project,
    ProjectHumanMembership,
    SessionLeadBinding,
    SessionLeadReplyDraft,
    SessionLeadReplyKey,
    TeamAttachment,
    TeamProject,
    TeamProjectAgentBinding,
)
from .storage import (
    ProjectArchive,
    archive_write_lock,
    collect_lock_status,
    ensure_archive,
    get_agent_communication_graph,
    get_archive_tree,
    get_commit_detail,
    get_fd_headroom,
    get_fd_usage,
    get_file_content,
    get_historical_inbox_snapshot,
    get_lock_telemetry,
    get_message_commit_sha,
    get_recent_commits,
    get_repo_cache_stats,
    get_timeline_commits,
    proactive_fd_cleanup,
    write_agent_profile,
    write_file_reservation_record,
)

_HUMAN_PRESENCE_TTL_SECONDS = 60
_SESSION_LEAD_RUNTIME_TTL_SECONDS = 15
_HUMAN_PRESENCE_CLIENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{15,127}$")
_TEAM_ATTACHMENT_MAX_BYTES = 10 * 1024 * 1024
_TEAM_ATTACHMENT_MAX_PER_MESSAGE = 4
_TEAM_ATTACHMENT_MEDIA_TYPES = {
    ".csv": "text/csv",
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".json": "application/json",
    ".log": "text/plain",
    ".md": "text/markdown",
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".txt": "text/plain",
    ".webp": "image/webp",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".zip": "application/zip",
}


def _team_attachment_root(settings: Settings | None = None) -> Path:
    active_settings = settings or get_settings()
    root = Path(active_settings.storage.root).expanduser().resolve() / "team-attachments"
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with contextlib.suppress(OSError):
        root.chmod(0o700)
    return root


def _team_attachment_filename(value: str | None) -> tuple[str, str]:
    raw = (value or "").replace("\\", "/")
    filename = raw.rsplit("/", 1)[-1].strip()
    if (
        not filename
        or len(filename) > 255
        or any(ord(char) < 32 for char in filename)
    ):
        raise HTTPException(status_code=400, detail="附件文件名无效")
    suffix = Path(filename).suffix.lower()
    media_type = _TEAM_ATTACHMENT_MEDIA_TYPES.get(suffix)
    if media_type is None:
        raise HTTPException(status_code=415, detail="不支持该附件类型")
    return filename, media_type


def _public_team_attachment(row: TeamAttachment) -> dict[str, Any]:
    return {
        "id": row.token,
        "filename": row.filename,
        "media_type": row.media_type,
        "size": row.size,
        "sha256": row.sha256,
    }


async def _project_slug_from_id(pid: int | None) -> str | None:
    if pid is None:
        return None
    async with get_session() as session:
        row = await session.execute(text("SELECT slug FROM projects WHERE id = :pid"), {"pid": pid})
        res = row.fetchone()
        return res[0] if res and res[0] else None


async def _revalidate_claim_write_window(
    agent: Agent,
    presented_token: str,
    project: Project,
    human: Human,
    *,
    session: AsyncSession,
) -> None:
    """Re-check claim preconditions inside the write transaction (#1013).

    While a claim waited on the per-agent lock, the token may have rotated,
    the source project may have been archived, or the caller's membership may
    have been removed. Nothing may be written unless all still hold. Token
    comparison is constant-time and failures stay opaque.
    """
    stored = agent.registration_token or ""
    if not stored or not hmac.compare_digest(stored, presented_token):
        raise HTTPException(status_code=403, detail="Invalid agent credentials")
    source = await session.get(Project, agent.project_id)
    if source is None or source.archived_at is not None:
        raise HTTPException(status_code=403, detail="Invalid agent credentials")
    membership_row = await session.execute(
        select(ProjectHumanMembership).where(
            cast(Any, ProjectHumanMembership.project_id) == project.id,
            cast(Any, ProjectHumanMembership.human_id) == human.id,
        )
    )
    membership = membership_row.scalars().first()
    if membership is None or membership.status != "active":
        raise HTTPException(status_code=403, detail="Active project membership is required")


async def _ensure_ack_escalation_holder(
    *,
    settings: Settings,
    project_id: int,
    project_slug: str | None,
    recipient_agent_id: int,
    recipient_name: str,
    claim_name: str,
    now: datetime,
    now_naive: datetime,
) -> tuple[int, str]:
    """Return the holder identity for ACK escalation, creating the ops holder if needed.

    When a synthetic holder must be created, the DB insert happens first and the
    archive profile write follows only after the session has closed. This keeps
    the ACK worker out of the DB->archive lock ordering that can deadlock mixed
    HTTP and MCP traffic.
    """
    holder_agent_id = int(recipient_agent_id)
    holder_agent_name = recipient_name
    holder_profile_payload: dict[str, Any] | None = None

    async with get_session() as s_holder:
        hid_row = await s_holder.execute(
            text("SELECT id FROM agents WHERE project_id = :pid AND name = :name"),
            {"pid": project_id, "name": claim_name},
        )
        hid = hid_row.scalar_one_or_none()
        if isinstance(hid, int):
            return hid, claim_name

        await s_holder.execute(
            text(
                "INSERT OR IGNORE INTO agents(project_id, name, program, model, task_description, inception_ts, last_active_ts, attachments_policy, contact_policy) VALUES (:pid, :name, :program, :model, :task, :ts, :ts, :attachments_policy, :contact_policy)"
            ),
            {
                "pid": project_id,
                "name": claim_name,
                "program": "ops",
                "model": "system",
                "task": "ops-escalation",
                "ts": now_naive,
                "attachments_policy": "auto",
                "contact_policy": "auto",
            },
        )
        await s_holder.commit()
        hid_row2 = await s_holder.execute(
            text("SELECT id FROM agents WHERE project_id = :pid AND name = :name"),
            {"pid": project_id, "name": claim_name},
        )
        hid2 = hid_row2.scalar_one_or_none()
        if isinstance(hid2, int):
            holder_agent_id = hid2
            holder_agent_name = claim_name
            if project_slug:
                holder_profile_payload = {
                    "id": holder_agent_id,
                    "name": holder_agent_name,
                    "program": "ops",
                    "model": "system",
                    "task_description": "ops-escalation",
                    "inception_ts": now.isoformat(),
                    "last_active_ts": now.isoformat(),
                    "project_id": project_id,
                    "attachments_policy": "auto",
                    "contact_policy": "auto",
                }

    if holder_profile_payload is not None and project_slug:
        archive = await ensure_archive(settings, project_slug)
        async with archive_write_lock(archive):
            await write_agent_profile(archive, holder_profile_payload)

    return holder_agent_id, holder_agent_name


def _http_sender_identity(
    *,
    message_project_id: int | None,
    sender_name: str | None,
    sender_project_id: int | None,
    sender_project_human_key: str | None,
    sender_project_slug: str | None,
) -> tuple[str, dict[str, str]]:
    canonical_sender = (sender_name or "").strip() or "Unknown"
    sender_display = _sender_display_name(
        message_project_id=message_project_id,
        sender_name=canonical_sender,
        sender_project_id=sender_project_id,
        sender_project_slug=sender_project_slug,
    )
    metadata: dict[str, str] = {"sender_name": canonical_sender}
    if (
        message_project_id is None
        or sender_project_id is None
        or sender_project_id == message_project_id
    ):
        return sender_display, metadata
    if sender_project_human_key:
        metadata["sender_project"] = sender_project_human_key
    if sender_project_slug:
        metadata["sender_project_slug"] = sender_project_slug
        metadata["sender_address"] = _format_cross_project_agent_address(
            sender_project_slug,
            canonical_sender,
        )
    return sender_display, metadata


_HTTP_MESSAGE_SUBJECT_SLUG_RE = re.compile(r"[^a-zA-Z0-9]+")


def _coerce_http_archive_timestamp(created_ts_raw: Any) -> datetime:
    try:
        if isinstance(created_ts_raw, str):
            text_value = (
                created_ts_raw.replace("Z", "+00:00")
                if created_ts_raw.endswith("Z")
                else created_ts_raw
            )
            dt = datetime.fromisoformat(text_value)
        else:
            dt = created_ts_raw
        if not isinstance(dt, datetime):
            raise TypeError("created timestamp must be a datetime")
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def _build_http_archive_message_filename(created_ts_raw: Any, subject_raw: str, message_id: int) -> tuple[str, str, str]:
    dt = _coerce_http_archive_timestamp(created_ts_raw)
    y_dir = dt.strftime("%Y")
    m_dir = dt.strftime("%m")
    created_iso = dt.strftime("%Y-%m-%dT%H-%M-%SZ")
    subject_slug = (
        _HTTP_MESSAGE_SUBJECT_SLUG_RE.sub("-", subject_raw).strip("-_").lower()[:80]
        or "message"
    )
    return y_dir, m_dir, f"{created_iso}__{subject_slug}__{message_id}.md"


async def _delete_messages_from_archive(
    *,
    settings: Settings,
    project_slug: str,
    messages_to_delete: list[tuple[Any, ...]],
    recip_map: dict[int, list[str]],
    commit_message: str,
) -> int:
    archive = await ensure_archive(settings, project_slug)
    git_paths_removed: list[str] = []
    seen_git_paths: set[str] = set()

    async with archive_write_lock(archive):
        for mrow in messages_to_delete:
            msg_id = int(mrow[0])
            y_dir, m_dir, filename = _build_http_archive_message_filename(
                mrow[1],
                str(mrow[2] or ""),
                msg_id,
            )
            sender_name = str(mrow[3] or "")

            candidate_dirs = [
                archive.root / "messages" / y_dir / m_dir,
                archive.root / "agents" / sender_name / "outbox" / y_dir / m_dir,
            ]
            for recip_name in recip_map.get(msg_id, []):
                candidate_dirs.append(
                    archive.root / "agents" / recip_name / "inbox" / y_dir / m_dir
                )

            for cdir in candidate_dirs:
                fpath = cdir / filename
                rel = fpath.relative_to(archive.repo_root).as_posix()
                try:
                    await asyncio.to_thread(fpath.unlink)
                except FileNotFoundError:
                    continue
                except OSError:
                    continue
                if rel not in seen_git_paths:
                    seen_git_paths.add(rel)
                    git_paths_removed.append(rel)

        if git_paths_removed:
            actor_module = importlib.import_module("git")
            actor_cls = actor_module.Actor
            git_actor = actor_cls(
                settings.storage.git_author_name,
                settings.storage.git_author_email,
            )
            await asyncio.to_thread(
                archive.repo.index.remove,
                git_paths_removed,
                working_tree=False,
            )
            await asyncio.to_thread(
                archive.repo.index.commit,
                commit_message,
                author=git_actor,
                committer=git_actor,
            )

    return len(git_paths_removed)


__all__ = ["build_http_app", "create_app", "main"]


class _FastMCPHttpApp(Protocol):
    def http_app(self, *args: Any, **kwargs: Any) -> FastAPI: ...


class _FastAPILifespan(Protocol):
    def lifespan(self, app: FastAPI) -> Any: ...


def _expanduser_resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def _path_exists(path: Path) -> bool:
    return path.exists()


def _open_git_repo(repo_root: Path):
    from git import Repo as GitRepo

    return GitRepo(str(repo_root))


async def _open_existing_project_archive(settings: Settings, slug: str) -> ProjectArchive | None:
    """Open an existing project archive for read-only routes without creating new directories."""
    repo_root = await asyncio.to_thread(_expanduser_resolve_path, Path(settings.storage.root))
    if not await asyncio.to_thread(_path_exists, repo_root / ".git"):
        return None
    project_root = repo_root / "projects" / slug
    if not await asyncio.to_thread(_path_exists, project_root):
        return None
    repo = await asyncio.to_thread(_open_git_repo, repo_root)
    return ProjectArchive(
        settings=settings,
        slug=slug,
        root=project_root,
        repo=repo,
        lock_path=project_root / ".archive.lock",
        repo_root=repo_root,
    )


def _collect_retention_quota_report_sync(settings: Settings) -> dict[str, Any]:
    import datetime as _dt
    import fnmatch as _fnmatch

    storage_root = _expanduser_resolve_path(Path(settings.storage.root))
    projects_root = storage_root / "projects"
    cutoff = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(
        days=int(settings.retention_max_age_days)
    )
    old_messages = 0
    total_attach_bytes = 0
    per_project_attach: dict[str, int] = {}
    per_project_inbox_counts: dict[str, int] = {}
    ignore_patterns = list(getattr(settings, "retention_ignore_project_patterns", []) or [])

    for proj_dir in projects_root.iterdir() if projects_root.exists() else []:
        if not proj_dir.is_dir():
            continue
        proj_name = proj_dir.name
        if any(_fnmatch.fnmatch(proj_name, pat) for pat in ignore_patterns):
            continue
        msg_root = proj_dir / "messages"
        if msg_root.exists():
            for ydir in msg_root.iterdir():
                for mdir in ydir.iterdir() if ydir.is_dir() else []:
                    for file_path in mdir.iterdir() if mdir.is_dir() else []:
                        if file_path.suffix.lower() != ".md":
                            continue
                        with contextlib.suppress(Exception):
                            ts = _dt.datetime.fromtimestamp(file_path.stat().st_mtime, _dt.timezone.utc)
                            if ts < cutoff:
                                old_messages += 1
        inbox_root = proj_dir / "agents"
        if inbox_root.exists():
            count_inbox = 0
            for inbox_file in inbox_root.rglob("inbox/*/*/*.md"):
                with contextlib.suppress(Exception):
                    if inbox_file.is_file():
                        count_inbox += 1
            per_project_inbox_counts[proj_name] = count_inbox
        att_root = proj_dir / "attachments"
        if att_root.exists():
            for attachment_file in att_root.rglob("*.webp"):
                with contextlib.suppress(Exception):
                    size_bytes = attachment_file.stat().st_size
                    total_attach_bytes += size_bytes
                    per_project_attach[proj_name] = per_project_attach.get(proj_name, 0) + size_bytes

    return {
        "old_messages": old_messages,
        "retention_max_age_days": int(settings.retention_max_age_days),
        "total_attachments_bytes": total_attach_bytes,
        "quota_limit_bytes": int(settings.quota_attachments_limit_bytes),
        "per_project_attach": per_project_attach,
        "per_project_inbox_counts": per_project_inbox_counts,
    }


async def _collect_retention_quota_report(settings: Settings) -> dict[str, Any]:
    return await asyncio.to_thread(_collect_retention_quota_report_sync, settings)


def _collect_archive_guide_stats_sync(settings: Settings) -> dict[str, Any]:
    import subprocess as _subprocess
    from itertools import islice

    storage_root = str(_expanduser_resolve_path(Path(settings.storage.root)))
    repo_root = Path(storage_root)
    total_commits = "0"
    project_count = 0
    repo_size = "0 MB"
    last_commit_time = "Never"

    if _path_exists(repo_root / ".git"):
        repo = None
        try:
            repo = _open_git_repo(repo_root)
            commit_count = sum(1 for _ in repo.iter_commits(max_count=10000))
            total_commits = "10,000+" if commit_count == 10000 else f"{commit_count:,}"
            last_commit = next(repo.iter_commits(max_count=1), None)
            last_commit_time = last_commit.authored_datetime.strftime("%b %d, %Y") if last_commit else "Never"

            projects_dir = repo_root / "projects"
            if projects_dir.exists():
                project_count = sum(1 for p in islice(projects_dir.iterdir(), 100) if p.is_dir())

            try:
                result = _subprocess.run(
                    ["du", "-sh", str(repo_root)],
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                )
                repo_size = result.stdout.split()[0] if getattr(result, "returncode", 1) == 0 else "Unknown"
            except (_subprocess.TimeoutExpired, FileNotFoundError, PermissionError, OSError):
                repo_size = "Unknown"
        except Exception:
            pass
        finally:
            if repo is not None:
                repo.close()

    return {
        "storage_root": storage_root,
        "total_commits": total_commits,
        "project_count": project_count,
        "repo_size": repo_size,
        "last_commit_time": last_commit_time,
    }


def _decode_jwt_header_segment(token: str) -> dict[str, object] | None:
    """Return decoded JWT header without verifying signature."""
    try:
        segment = token.split(".", 1)[0]
        padded = segment + "=" * (-len(segment) % 4)
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


_LOGGING_CONFIGURED = False

# Pre-compiled regex patterns for HTTP validators
_SLUG_VALIDATOR_RE = re.compile(r"^[a-z0-9_-]+$", re.IGNORECASE)
_AGENT_NAME_VALIDATOR_RE = re.compile(r"^[A-Za-z0-9]+$")
_MENTION_HANDLE_VALIDATOR_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TIMESTAMP_VALIDATOR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")
_HUB_HUMAN_RELAY_PROGRAM = "team-human-relay"

_LIKE_ESCAPE_CHAR = "!"


def _like_escape(term: str) -> str:
    """Escape LIKE wildcards for literal substring matching."""
    return term.replace("!", "!!").replace("%", "!%").replace("_", "!_")


def _configure_logging(settings: Settings) -> None:
    """Initialize structlog and stdlib logging formatting."""
    # Idempotent setup
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
    ]
    if settings.log_json_enabled:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.processors.KeyValueRenderer(key_order=["event", "path", "status"]))
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, settings.log_level.upper(), logging.INFO)),
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

    # Suppress verbose MCP library logging for stateless HTTP sessions
    # "Terminating session: None" is routine for stateless mode and just noise
    logging.getLogger("mcp.server.streamable_http").setLevel(logging.WARNING)
    logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.WARNING)

    # Suppress verbose aiosqlite DEBUG logs (functools.partial cursor/operation noise)
    logging.getLogger("aiosqlite").setLevel(logging.INFO)

    # Suppress verbose git library DEBUG logs (Popen commands, platform detection)
    logging.getLogger("git.util").setLevel(logging.INFO)
    logging.getLogger("git.cmd").setLevel(logging.INFO)

    # Suppress filelock DEBUG logs (lock acquire/release routine operations)
    logging.getLogger("filelock").setLevel(logging.INFO)

    # Suppress SSE ping keepalive debug logs (periodic noise every 15s)
    logging.getLogger("sse_starlette.sse").setLevel(logging.INFO)

    # Add filter to suppress verbose tracebacks for expected/recoverable errors
    # FastMCP's tool_manager uses logger.exception() which prints full tracebacks
    # even for expected errors like "agent not found" or "git lock contention".
    # This filter intercepts those and removes the traceback for cleaner logs.
    class ExpectedErrorFilter(logging.Filter):
        """Filter that suppresses tracebacks for expected/recoverable tool errors.

        Expected errors include:
        - ToolExecutionError with recoverable=True
        - Agent not found / project not found
        - Git index.lock contention
        - Resource busy / database lock

        These are normal operational conditions in multi-agent environments
        and don't need full stack traces cluttering the logs.
        """

        # Keywords that indicate an expected/recoverable error
        _EXPECTED_PATTERNS = (
            "not found in project",
            "index.lock",
            "git_index_lock",
            "resource_busy",
            "temporarily locked",
            "recoverable=true",
            "use register_agent",
            "available agents:",
        )

        def filter(self, record: logging.LogRecord) -> bool:
            # Only process records from FastMCP tool_manager with exception info
            if not record.exc_info or record.exc_info[1] is None:
                return True

            exc = record.exc_info[1]
            exc_str = str(exc).lower()

            # Check if this is an expected error based on message content
            is_expected = any(pattern in exc_str for pattern in self._EXPECTED_PATTERNS)

            # Also check for our ToolExecutionError with recoverable flag
            if hasattr(exc, "recoverable") and exc.recoverable:
                is_expected = True

            # Check the cause chain for ToolExecutionError
            cause = getattr(exc, "__cause__", None)
            if cause is not None:
                cause_str = str(cause).lower()
                if any(pattern in cause_str for pattern in self._EXPECTED_PATTERNS):
                    is_expected = True
                if hasattr(cause, "recoverable") and cause.recoverable:
                    is_expected = True

            if is_expected:
                # Clear exc_info to prevent traceback printing, but keep the log message
                record.exc_info = None
                record.exc_text = None
                # Downgrade from ERROR to INFO for expected errors
                if record.levelno >= logging.ERROR:
                    record.levelno = logging.INFO
                    record.levelname = "INFO"

            return True

    # Apply filter to FastMCP's tool_manager logger
    fastmcp_logger = logging.getLogger("fastmcp.tools.tool_manager")
    fastmcp_logger.addFilter(ExpectedErrorFilter())

    # mark configured
    _LOGGING_CONFIGURED = True


# In-process JWKS cache: avoid refetching the JWKS document on every request
# (#212). Keyed by JWKS URL; entries expire after _JWKS_CACHE_TTL_SECONDS.
_JWKS_CACHE_TTL_SECONDS = 300.0
_jwks_cache: dict[str, tuple[float, Any]] = {}
_jwks_cache_lock = asyncio.Lock()


async def _fetch_jwks(jwks_url: str, *, force: bool = False):
    """Return a parsed JWKS key set for ``jwks_url``, using a TTL cache.

    On a cache miss/expiry (or when ``force`` is set, e.g. after an unknown
    ``kid``), the document is refetched. On fetch/parse failure the last good
    cached key set (if any) is returned so transient outages don't break auth.
    """
    from time import monotonic

    jose_mod = importlib.import_module("authlib.jose")
    JsonWebKey = jose_mod.JsonWebKey

    now = monotonic()
    async with _jwks_cache_lock:
        cached = _jwks_cache.get(jwks_url)
        if cached is not None and not force and (now - cached[0]) < _JWKS_CACHE_TTL_SECONDS:
            return cached[1]

    try:
        httpx = importlib.import_module("httpx")
        AsyncClient = httpx.AsyncClient
        async with AsyncClient(timeout=5) as client:
            jwks = (await client.get(jwks_url)).json()
        key_set = JsonWebKey.import_key_set(jwks)
    except Exception:
        # Fall back to any cached (possibly stale) key set on fetch failure.
        async with _jwks_cache_lock:
            cached = _jwks_cache.get(jwks_url)
        return cached[1] if cached is not None else None

    async with _jwks_cache_lock:
        _jwks_cache[jwks_url] = (monotonic(), key_set)
    return key_set


async def _introspect_jwt(introspection_url: str, token: str) -> bool:
    """Ask the issuer whether a locally verified JWT is still active.

    Redirects and all network/JSON failures fail closed so a compromised or
    unavailable issuer cannot silently turn revocation checks into allow.
    """
    try:
        httpx = importlib.import_module("httpx")
        AsyncClient = httpx.AsyncClient
        async with AsyncClient(timeout=5, follow_redirects=False) as client:
            response = await client.post(
                introspection_url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                },
            )
        if response.status_code != 200:
            return False
        payload = response.json()
        return isinstance(payload, dict) and payload.get("active") is True
    except Exception:
        return False


def _select_jwks_key(key_set, header: dict, algorithms: list[str]):
    """Resolve the verification key from a JWKS key set by ``kid``.

    Never blindly picks ``keys[0]`` (#211). With a ``kid`` we look it up
    directly; an unknown ``kid`` returns ``None``. Without a ``kid`` this also
    returns ``None`` -- the caller falls back to verifying against each
    algorithm-compatible candidate (see ``_jwks_candidate_keys``) instead.
    """
    kid = header.get("kid")
    if kid:
        with contextlib.suppress(Exception):
            return key_set.find_by_kid(kid)
    return None


def _jwks_candidate_keys(key_set, header: dict, algorithms: list[str]) -> list:
    """Return JWKS keys to try when no ``kid`` is present.

    Filters by signing use and by algorithm compatibility (matching the key's
    declared ``alg`` when present, otherwise the key type implied by the
    configured algorithms). Blind ``keys[0]`` selection is never used.
    """
    alg_set = {str(a) for a in algorithms}
    # Map configured JWS algorithms to acceptable JWK key types.
    kty_for_alg = {
        "HS": "oct", "RS": "RSA", "PS": "RSA",
        "ES": "EC", "Ed": "OKP",
    }
    wanted_kty = {kty_for_alg[a[:2]] for a in alg_set if a[:2] in kty_for_alg}
    candidates = []
    for key in list(getattr(key_set, "keys", []) or []):
        with contextlib.suppress(Exception):
            use = key.tokens.get("use") if hasattr(key, "tokens") else None
            if use not in (None, "sig"):
                continue
            key_alg = key.tokens.get("alg") if hasattr(key, "tokens") else None
            if key_alg is not None and str(key_alg) not in alg_set:
                continue
            kty = getattr(key, "kty", None) or (key.tokens.get("kty") if hasattr(key, "tokens") else None)
            if wanted_kty and kty is not None and kty not in wanted_kty:
                continue
            candidates.append(key)
    return candidates


_SESSION_LEAD_CAPABILITY_RE = re.compile(
    r"^/hub/api/projects/[A-Za-z0-9_-]+/session-lead/"
    r"(?:status|reply|reply-drafts|inbox/claim|inbox/[1-9][0-9]*/complete)$"
)


def _is_session_lead_capability(path: str, method: str) -> bool:
    """Recognize endpoints authenticated by a binding-scoped capability."""
    return method.upper() == "POST" and bool(
        _SESSION_LEAD_CAPABILITY_RE.fullmatch(path)
    )


class BearerAuthMiddleware(BaseHTTPMiddleware):
    def __init__(
        self, app: FastAPI, token: str, allow_localhost: bool = False, jwt_enabled: bool = False
    ) -> None:
        super().__init__(app)
        self._token = token
        self._allow_localhost = allow_localhost
        # When JWT auth is also enabled, a static-bearer mismatch must NOT
        # short-circuit before the inner SecurityAndRateLimitMiddleware gets a
        # chance to validate a JWT (#210). In that case we accept any Bearer
        # token here and let the JWT path render the final auth decision.
        self._jwt_enabled = jwt_enabled

    @staticmethod
    def _is_localhost(host: str) -> bool:
        """Check if host is a localhost address, including IPv4-mapped IPv6."""
        if not host:
            return False
        # Standard localhost addresses
        if host in {"127.0.0.1", "::1", "localhost"}:
            return True
        # IPv4-mapped IPv6 address (::ffff:127.0.0.1)
        return bool(host.lower().startswith("::ffff:") and host[7:] == "127.0.0.1")

    @staticmethod
    def _has_forwarded_headers(request: Request) -> bool:
        """Detect proxy-forwarded headers to avoid trusting localhost behind proxies."""
        headers = request.headers
        return any(
            name in headers
            for name in ("x-forwarded-for", "x-forwarded-proto", "x-forwarded-host", "forwarded")
        )

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        if request.method == "OPTIONS":  # allow CORS preflight
            return await call_next(request)
        if request.url.path.startswith("/health/") or request.url.path == "/api/health":
            return await call_next(request)
        if _is_session_lead_capability(request.url.path, request.method):
            return await call_next(request)
        if _localhost_bypass_allowed(
            request,
            allow_localhost=self._allow_localhost,
        ):
            return await call_next(request)
        auth_header = request.headers.get("Authorization", "")
        expected_header = f"Bearer {self._token}"
        # Use constant-time comparison to prevent timing attacks
        if hmac.compare_digest(auth_header, expected_header):
            return await call_next(request)
        # Static bearer did not match. If JWT auth is enabled, defer to the inner
        # JWT-validating middleware instead of rejecting here, so EITHER a valid
        # static bearer OR a valid JWT is accepted (#210).
        if self._jwt_enabled and auth_header.startswith("Bearer "):
            return await call_next(request)
        return JSONResponse({"detail": "Unauthorized"}, status_code=status.HTTP_401_UNAUTHORIZED)


def _localhost_bypass_allowed(request: Request, *, allow_localhost: bool) -> bool:
    """Return whether this request qualifies for localhost auth bypass."""
    if not allow_localhost:
        return False
    try:
        client_host = request.client.host if request.client else ""
    except Exception:
        client_host = ""
    return BearerAuthMiddleware._is_localhost(client_host) and not BearerAuthMiddleware._has_forwarded_headers(
        request
    )


class SecurityAndRateLimitMiddleware(BaseHTTPMiddleware):
    """JWT auth (optional), RBAC, and token-bucket rate limiting.

    - If JWT is enabled, validates Authorization: Bearer <token> using either HMAC secret or JWKS URL.
    - Enforces basic RBAC when enabled: read-only roles may only call whitelisted tools and resource reads.
    - Applies per-endpoint token-bucket limits (tools vs resources) with in-memory or Redis backend.
    """

    def __init__(self, app: FastAPI, settings: Settings):
        super().__init__(app)
        self.settings = settings
        self._jwt_enabled = bool(getattr(settings.http, "jwt_enabled", False))
        self._rbac_enabled = bool(getattr(settings.http, "rbac_enabled", True))
        self._reader_roles = set(getattr(settings.http, "rbac_reader_roles", []) or [])
        self._writer_roles = set(getattr(settings.http, "rbac_writer_roles", []) or [])
        self._readonly_tools = set(getattr(settings.http, "rbac_readonly_tools", []) or [])
        self._default_role = getattr(settings.http, "rbac_default_role", "tools")
        # Token bucket state (memory)
        from time import monotonic

        self._monotonic = monotonic
        self._buckets: dict[str, tuple[float, float]] = {}
        self._last_cleanup = monotonic()
        # Redis client (optional)
        self._redis = None
        if getattr(settings.http, "rate_limit_backend", "memory") == "redis" and getattr(
            settings.http, "rate_limit_redis_url", ""
        ):
            try:
                redis_asyncio = importlib.import_module("redis.asyncio")
                Redis = redis_asyncio.Redis
                self._redis = Redis.from_url(settings.http.rate_limit_redis_url)
            except Exception:
                self._redis = None

    def _cleanup_buckets(self, now: float) -> None:
        """Remove stale buckets to prevent memory leaks."""
        # Evict buckets not accessed in the last hour
        expiration = 3600.0
        cutoff = now - expiration
        # Create list of keys to remove to avoid runtime modification errors during iteration
        to_remove = [k for k, (_, ts) in self._buckets.items() if ts < cutoff]
        for k in to_remove:
            self._buckets.pop(k, None)

    async def _decode_jwt(self, token: str) -> dict | None:
        """Validate and decode JWT, returning claims or None on failure."""
        with contextlib.suppress(Exception):
            jose_mod = importlib.import_module("authlib.jose")
            JsonWebKey = jose_mod.JsonWebKey
            JsonWebToken = jose_mod.JsonWebToken
            algs = list(getattr(self.settings.http, "jwt_algorithms", ["HS256"]))
            jwt = JsonWebToken(algs)
            audience = getattr(self.settings.http, "jwt_audience", None) or None
            issuer = getattr(self.settings.http, "jwt_issuer", None) or None
            jwks_url = getattr(self.settings.http, "jwt_jwks_url", None) or None
            secret = getattr(self.settings.http, "jwt_secret", None) or None

            header = _decode_jwt_header_segment(token)
            if header is None:
                return None
            key = None
            candidate_keys: list = []
            if jwks_url:
                with contextlib.suppress(Exception):
                    key_set = await _fetch_jwks(jwks_url)
                    if key_set is None:
                        return None
                    if header.get("kid"):
                        key = _select_jwks_key(key_set, header, algs)
                        # Unknown kid: the cached JWKS may be stale; force a
                        # refresh once before giving up (#212).
                        if key is None:
                            key_set = await _fetch_jwks(jwks_url, force=True)
                            if key_set is not None:
                                key = _select_jwks_key(key_set, header, algs)
                    else:
                        # No kid: never blind-pick keys[0]. Try every
                        # algorithm-compatible key during verification (#211).
                        candidate_keys = _jwks_candidate_keys(key_set, header, algs)
            elif secret:
                with contextlib.suppress(Exception):
                    key = JsonWebKey.import_key(secret, {"kty": "oct"})
            keys_to_try = candidate_keys if candidate_keys else ([key] if key is not None else [])
            if not keys_to_try:
                return None
            for candidate in keys_to_try:
                with contextlib.suppress(Exception):
                    claims = jwt.decode(token, candidate)
                    claims.validate()
                    if audience:
                        token_audience = claims.get("aud")
                        if isinstance(token_audience, str):
                            audience_matches = token_audience == audience
                        elif isinstance(token_audience, (list, tuple)):
                            audience_matches = audience in token_audience
                        else:
                            audience_matches = False
                        if not audience_matches:
                            continue
                    if issuer and str(claims.get("iss") or "") != issuer:
                        continue
                    introspection_url = (
                        getattr(self.settings.http, "jwt_introspection_url", None)
                        or None
                    )
                    if introspection_url and not await _introspect_jwt(
                        introspection_url, token
                    ):
                        continue
                    return dict(claims)
        return None

    @staticmethod
    def _classify_request(path: str, method: str, body_bytes: bytes) -> tuple[str, str | None]:
        """Return (kind, tool_name) where kind is 'tools'|'resources'|'other'."""
        if method.upper() != "POST":
            return "other", None
        if not body_bytes:
            return "other", None
        with contextlib.suppress(Exception):
            import json as _json

            payload = _json.loads(body_bytes)
            rpc_method = str(payload.get("method", ""))
            if rpc_method == "tools/call":
                params = payload.get("params", {}) or {}
                tool_name = params.get("name")
                return "tools", tool_name if isinstance(tool_name, str) else None
            if rpc_method.startswith("resources/"):
                return "resources", None
            return "other", None
        return "other", None

    @staticmethod
    def _coerce_rpm(value: object, default: int) -> int:
        # An explicit 0 disables the limit and must survive (#213); only a
        # missing/None value falls back to the default. Use a None check rather
        # than ``value or default`` (which would turn 0 into ``default``).
        if value is None:
            return default
        with contextlib.suppress(Exception):
            return int(value)
        return default

    def _rate_limits_for(self, kind: str) -> tuple[int, int]:
        # return (per_minute, burst)
        if kind == "tools":
            rpm = self._coerce_rpm(getattr(self.settings.http, "rate_limit_tools_per_minute", 60), 60)
            burst = int(getattr(self.settings.http, "rate_limit_tools_burst", 0) or 0)
        elif kind == "resources":
            rpm = self._coerce_rpm(getattr(self.settings.http, "rate_limit_resources_per_minute", 120), 120)
            burst = int(getattr(self.settings.http, "rate_limit_resources_burst", 0) or 0)
        else:
            rpm = self._coerce_rpm(getattr(self.settings.http, "rate_limit_per_minute", 60), 60)
            burst = 0
        # rpm <= 0 means "disabled" (handled by _consume_bucket); don't synthesize
        # a positive burst that would re-enable limiting.
        burst = int(burst) if burst > 0 else max(1, rpm)
        return rpm, burst

    async def _consume_bucket(self, key: str, per_minute: int, burst: int) -> bool:
        """Return True if token granted, False if limited."""
        if per_minute <= 0:
            return True
        rate_per_sec = per_minute / 60.0
        now = self._monotonic()

        # Redis backend
        if self._redis is not None:
            try:
                lua = (
                    "local key = KEYS[1]\n"
                    "local now = tonumber(ARGV[1])\n"
                    "local rate = tonumber(ARGV[2])\n"
                    "local burst = tonumber(ARGV[3])\n"
                    "local state = redis.call('HMGET', key, 'tokens', 'ts')\n"
                    "local tokens = tonumber(state[1]) or burst\n"
                    "local ts = tonumber(state[2]) or now\n"
                    "local delta = now - ts\n"
                    "tokens = math.min(burst, tokens + delta * rate)\n"
                    "local allowed = 0\n"
                    "if tokens >= 1 then\n"
                    "  tokens = tokens - 1\n"
                    "  allowed = 1\n"
                    "end\n"
                    "redis.call('HMSET', key, 'tokens', tokens, 'ts', now)\n"
                    "redis.call('EXPIRE', key, math.ceil(burst / math.max(rate, 0.001)))\n"
                    "return allowed\n"
                )
                allowed = await self._redis.eval(lua, 1, f"rl:{key}", now, rate_per_sec, burst)
                return bool(int(allowed or 0) == 1)
            except Exception:
                # Fallback to memory on Redis failure
                pass

        # In-memory token bucket
        tokens, ts = self._buckets.get(key, (float(burst), now))
        elapsed = max(0.0, now - ts)
        tokens = min(float(burst), tokens + elapsed * rate_per_sec)
        if tokens < 1.0:
            self._buckets[key] = (tokens, now)
            return False
        tokens -= 1.0
        self._buckets[key] = (tokens, now)
        return True

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        # Perform periodic cleanup of in-memory rate limit buckets
        if self._redis is None:
            now = self._monotonic()
            if now - self._last_cleanup > 60.0:
                self._cleanup_buckets(now)
                self._last_cleanup = now

        # Allow CORS preflight and health endpoints
        if request.method == "OPTIONS" or request.url.path.startswith("/health/") or request.url.path == "/api/health":
            return await call_next(request)

        # Only read/patch body for POST requests. GET (including SSE) must not receive http.request messages.
        body_bytes = b""
        if request.method.upper() == "POST":
            try:
                body_bytes = await request.body()
                body_sent = False

                async def _receive() -> dict:
                    nonlocal body_sent
                    if body_sent:
                        return {"type": "http.request", "body": b"", "more_body": False}
                    body_sent = True
                    return {"type": "http.request", "body": body_bytes, "more_body": False}

                cast(Any, request)._receive = _receive
            except Exception:
                body_bytes = b""

        kind, tool_name = self._classify_request(request.url.path, request.method, body_bytes)

        # Capability-auth reply endpoint carries its own token (#1093); skip
        # JWT/RBAC here but keep rate limiting below.
        session_lead_capability = _is_session_lead_capability(
            request.url.path, request.method
        )

        # JWT auth (if enabled)
        if self._jwt_enabled and not session_lead_capability:
            auth_header = request.headers.get("Authorization", "")
            # #210: when JWT is enabled, a valid *static* bearer is still accepted
            # as the OR-alternative to a JWT (the outer BearerAuthMiddleware defers
            # Bearer requests here without distinguishing the two). Check it first so
            # static-bearer clients keep working once JWT is turned on; a static
            # bearer is treated exactly as it is when JWT is disabled (default role).
            static_token = getattr(self.settings.http, "bearer_token", "") or ""
            if static_token and hmac.compare_digest(auth_header, f"Bearer {static_token}"):
                roles = {self._default_role}
            else:
                if not auth_header.startswith("Bearer "):
                    return JSONResponse({"detail": "Unauthorized"}, status_code=status.HTTP_401_UNAUTHORIZED)
                token = auth_header.split(" ", 1)[1].strip()
                claims_dict = await self._decode_jwt(token)
                if claims_dict is None:
                    return JSONResponse({"detail": "Unauthorized"}, status_code=status.HTTP_401_UNAUTHORIZED)
                claims = cast(dict[str, Any], claims_dict)
                request.state.jwt_claims = claims
                roles_raw = claims.get(self.settings.http.jwt_role_claim, [])
                if isinstance(roles_raw, str):
                    roles = {roles_raw}
                elif isinstance(roles_raw, (list, tuple)):
                    roles = {str(r) for r in roles_raw}
                else:
                    roles = set()
                if not roles:
                    roles = {self._default_role}
        else:
            roles = {self._default_role}
            # Elevate localhost to writer when unauthenticated localhost is allowed
            if _localhost_bypass_allowed(
                request,
                allow_localhost=bool(getattr(self.settings.http, "allow_localhost_unauthenticated", False)),
            ):
                roles.add("writer")

        # RBAC enforcement (skip for localhost when allowed)
        is_local_ok = _localhost_bypass_allowed(
            request,
            allow_localhost=bool(getattr(self.settings.http, "allow_localhost_unauthenticated", False)),
        )
        if self._rbac_enabled and not is_local_ok and kind in {"tools", "resources"}:
            is_reader = bool(roles & self._reader_roles)
            is_writer = bool(roles & self._writer_roles) or (not roles)
            if kind == "resources":
                pass  # readers allowed
            elif kind == "tools":
                if not tool_name:
                    # Without name, assume write-required to be safe
                    if not is_writer:
                        return JSONResponse({"detail": "Forbidden"}, status_code=status.HTTP_403_FORBIDDEN)
                else:
                    if tool_name in self._readonly_tools:
                        if not is_reader and not is_writer:
                            return JSONResponse({"detail": "Forbidden"}, status_code=status.HTTP_403_FORBIDDEN)
                    else:
                        if not is_writer:
                            return JSONResponse({"detail": "Forbidden"}, status_code=status.HTTP_403_FORBIDDEN)

        # Rate limiting
        if self.settings.http.rate_limit_enabled:
            rpm, burst = self._rate_limits_for(kind)
            identity = request.client.host if request.client else "ip-unknown"
            # Prefer stable subject from JWT if present
            with contextlib.suppress(Exception):
                maybe_claims = getattr(request.state, "jwt_claims", None)
                if isinstance(maybe_claims, dict):
                    sub = maybe_claims.get("sub")
                    if isinstance(sub, str) and sub:
                        identity = f"sub:{sub}"
            endpoint = tool_name or "*"
            key = f"{kind}:{endpoint}:{identity}"
            allowed = await self._consume_bucket(key, rpm, burst)
            if not allowed:
                return JSONResponse({"detail": "Rate limit exceeded"}, status_code=status.HTTP_429_TOO_MANY_REQUESTS)

        return await call_next(request)


async def readiness_check() -> None:
    await ensure_schema()
    async with get_session() as session:
        await session.execute(text("SELECT 1"))

    # Fail readiness if FD usage from lockfile leaks is critically high.
    # This gives orchestrators a signal to restart the process before it
    # becomes completely wedged (issue #116).
    current, limit = get_fd_usage()
    if current >= 0 and limit > 0:
        headroom_pct = (limit - current) / limit
        if headroom_pct < 0.10:
            lock_stats = get_lock_telemetry()
            raise RuntimeError(
                f"FD exhaustion imminent: {current}/{limit} FDs in use "
                f"({round(headroom_pct * 100, 1)}% headroom). "
                f"Lock telemetry: {lock_stats}"
            )


def create_app() -> FastAPI:
    """Zero-argument ASGI app factory for ``uvicorn ... --factory`` (#214).

    ``build_http_app`` requires a ``Settings`` argument, so it cannot be used
    directly as a uvicorn ``--factory`` target. This wrapper resolves settings
    from the environment and builds the app, matching the documented command.
    """
    return build_http_app(get_settings())


def build_http_app(settings: Settings, server=None) -> FastAPI:
    # Configure logging once
    _configure_logging(settings)
    if server is None:
        server = build_mcp_server()

    # Build MCP HTTP sub-app with stateless mode for ASGI test transports
    mcp_http_app = cast(_FastMCPHttpApp, server).http_app(
        path="/",
        stateless_http=True,
        json_response=True,
    )

    # Second, STATEFUL MCP sub-app (issue #250): stateless mode creates a new
    # transport per request and never issues an ``Mcp-Session-Id`` header, so
    # session-bound agent authentication (#148) could never persist across
    # HTTP tool calls — ``create_agent_identity(return_registration_token=false)``
    # followed by any protected call failed with AUTHENTICATION_REQUIRED.
    # A bare flip to ``stateless_http=False`` would break handshake-skipping
    # clients (e.g. ntm's HTTP client), so we mount BOTH: the stateful app at
    # '/mcp' for spec-compliant MCP clients that keep a session, and the
    # stateless app at '/api' (and the configured base) for one-shot clients.
    mcp_stateful_http_app = cast(_FastMCPHttpApp, server).http_app(
        path="/",
        stateless_http=False,
        json_response=True,
    )

    # no-op wrapper removed; using explicit stateless adapter below

    # Background workers lifecycle
    async def _startup() -> None:  # pragma: no cover - service lifecycle
        # Note: no early return here -- the FD health monitor always runs,
        # even when optional workers are disabled by feature flags.

        async def _worker_cleanup() -> None:
            while True:
                try:
                    await ensure_schema()
                    async with get_session() as session:
                        rows = await session.execute(text("SELECT DISTINCT project_id FROM file_reservations"))
                        pids = [r[0] for r in rows.fetchall() if r[0] is not None]
                    released_total = 0
                    for pid in pids:
                        with contextlib.suppress(Exception):
                            stale = await _expire_stale_file_reservations(pid)
                            released_total += len(stale)
                    try:
                        rich_console = importlib.import_module("rich.console")
                        rich_panel = importlib.import_module("rich.panel")
                        Console = rich_console.Console
                        Panel = rich_panel.Panel
                        Console().print(
                            Panel.fit(
                                f"projects_scanned={len(pids)} released={released_total}",
                                title="File Reservations Cleanup",
                                border_style="cyan",
                            )
                        )
                    except Exception:
                        pass
                    with contextlib.suppress(Exception):
                        structlog.get_logger("tasks").info(
                            "file_reservations_cleanup",
                            projects_scanned=len(pids),
                            stale_released=released_total,
                        )
                except Exception:
                    pass
                await asyncio.sleep(settings.file_reservations_cleanup_interval_seconds)

        async def _worker_ack_ttl() -> None:
            import datetime as _dt

            while True:
                try:
                    await ensure_schema()
                    async with get_session() as session:
                        result = await session.execute(
                            text(
                                """
                            SELECT m.id, m.project_id, m.created_ts, mr.agent_id
                            FROM messages m
                            JOIN message_recipients mr ON mr.message_id = m.id
                            WHERE m.ack_required = 1 AND mr.ack_ts IS NULL
                            """
                            )
                        )
                        rows = result.fetchall()
                    now = _dt.datetime.now(_dt.timezone.utc)
                    now_naive = now.replace(tzinfo=None)
                    for mid, project_id, created_ts, agent_id in rows:
                        # Normalize to timezone-aware UTC before arithmetic; SQLite may yield naive datetimes
                        ts = created_ts
                        if getattr(ts, "tzinfo", None) is None or ts.tzinfo.utcoffset(ts) is None:
                            ts = ts.replace(tzinfo=_dt.timezone.utc)
                        else:
                            ts = ts.astimezone(_dt.timezone.utc)
                        age = (now - ts).total_seconds()
                        if age >= settings.ack_ttl_seconds:
                            try:
                                rich_console = importlib.import_module("rich.console")
                                rich_panel = importlib.import_module("rich.panel")
                                rich_text = importlib.import_module("rich.text")
                                Console = rich_console.Console
                                Panel = rich_panel.Panel
                                Text = rich_text.Text
                                con = Console()
                                body = Text.assemble(
                                    ("message_id: ", "cyan"),
                                    (str(mid), "white"),
                                    "\n",
                                    ("agent_id: ", "cyan"),
                                    (str(agent_id), "white"),
                                    "\n",
                                    ("project_id: ", "cyan"),
                                    (str(project_id), "white"),
                                    "\n",
                                    ("age_s: ", "cyan"),
                                    (str(int(age)), "white"),
                                    "\n",
                                    ("ttl_s: ", "cyan"),
                                    (str(settings.ack_ttl_seconds), "white"),
                                )
                                con.print(Panel(body, title="ACK Overdue", border_style="red"))
                            except Exception:
                                print(
                                    f"ack-warning message_id={mid} project_id={project_id} agent_id={agent_id} age_s={int(age)} ttl_s={settings.ack_ttl_seconds}"
                                )
                            with contextlib.suppress(Exception):
                                structlog.get_logger("tasks").warning(
                                    "ack_overdue",
                                    message_id=str(mid),
                                    project_id=str(project_id),
                                    agent_id=str(agent_id),
                                    age_s=int(age),
                                    ttl_s=int(settings.ack_ttl_seconds),
                                )
                            if settings.ack_escalation_enabled:
                                mode = (settings.ack_escalation_mode or "log").lower()
                                if mode == "file_reservation":
                                    try:
                                        y_dir = created_ts.strftime("%Y")
                                        m_dir = created_ts.strftime("%m")
                                        # Resolve recipient name
                                        async with get_session() as s_lookup:
                                            name_row = await s_lookup.execute(
                                                text("SELECT name FROM agents WHERE id = :aid"), {"aid": agent_id}
                                            )
                                            name_res = name_row.fetchone()
                                        recipient_name = name_res[0] if name_res and name_res[0] else "*"
                                        pattern = (
                                            f"agents/{recipient_name}/inbox/{y_dir}/{m_dir}/*.md"
                                            if recipient_name != "*"
                                            else f"agents/*/inbox/{y_dir}/{m_dir}/*.md"
                                        )
                                        project_slug = await _project_slug_from_id(project_id)
                                        holder_agent_id = int(agent_id)
                                        holder_agent_name = recipient_name
                                        if settings.ack_escalation_claim_holder_name:
                                            claim_name = settings.ack_escalation_claim_holder_name
                                            holder_agent_id, holder_agent_name = await _ensure_ack_escalation_holder(
                                                settings=settings,
                                                project_id=int(project_id),
                                                project_slug=project_slug,
                                                recipient_agent_id=int(agent_id),
                                                recipient_name=recipient_name,
                                                claim_name=claim_name,
                                                now=now,
                                                now_naive=now_naive,
                                            )
                                        async with get_session() as s2:
                                            await s2.execute(
                                                text(
                                                    """
                                                INSERT INTO file_reservations(project_id, agent_id, path_pattern, exclusive, reason, created_ts, expires_ts)
                                                VALUES (:pid, :holder, :pattern, :exclusive, :reason, :cts, :ets)
                                                """
                                                ),
                                                {
                                                    "pid": project_id,
                                                    "holder": holder_agent_id,
                                                    "pattern": pattern,
                                                    "exclusive": 1 if settings.ack_escalation_claim_exclusive else 0,
                                                    "reason": "ack-overdue",
                                                    "cts": now_naive,
                                                    "ets": now_naive
                                                    + _dt.timedelta(seconds=settings.ack_escalation_claim_ttl_seconds),
                                                },
                                            )
                                            await s2.commit()
                                        # Also write JSON artifact to archive
                                        if not project_slug:
                                            raise ValueError(f"Project id {project_id} has no slug; cannot write archive artifacts.")
                                        archive = await ensure_archive(settings, project_slug)
                                        expires_at = now + _dt.timedelta(
                                            seconds=settings.ack_escalation_claim_ttl_seconds
                                        )
                                        async with archive_write_lock(archive):
                                            await write_file_reservation_record(
                                                archive,
                                                {
                                                    "project": project_slug,
                                                    "agent": holder_agent_name,
                                                    "path_pattern": pattern,
                                                    "exclusive": settings.ack_escalation_claim_exclusive,
                                                    "reason": "ack-overdue",
                                                    "created_ts": now.isoformat(),
                                                    "expires_ts": expires_at.isoformat(),
                                                },
                                            )
                                    except Exception:
                                        pass
                except Exception:
                    pass
                await asyncio.sleep(settings.ack_ttl_scan_interval_seconds)

        async def _worker_tool_metrics() -> None:
            log = structlog.get_logger("tool.metrics")
            while True:
                try:
                    snapshot = _tool_metrics_snapshot()
                    if snapshot:
                        log.info("tool_metrics_snapshot", tools=snapshot)
                except Exception:
                    pass
                await asyncio.sleep(max(5, settings.tool_metrics_emit_interval_seconds))

        async def _worker_retention_quota() -> None:
            while True:
                with contextlib.suppress(Exception):
                    report = await _collect_retention_quota_report(settings)
                    structlog.get_logger("maintenance").info(
                        "retention_quota_report",
                        **report,
                    )
                    # Quota alerts
                    limit_b = int(settings.quota_attachments_limit_bytes)
                    inbox_limit = int(settings.quota_inbox_limit_count)
                    if limit_b > 0:
                        for proj, used in report["per_project_attach"].items():
                            if used >= limit_b:
                                structlog.get_logger("maintenance").warning(
                                    "quota_attachments_exceeded", project=proj, used_bytes=used, limit_bytes=limit_b
                                )
                    if inbox_limit > 0:
                        for proj, cnt in report["per_project_inbox_counts"].items():
                            if cnt >= inbox_limit:
                                structlog.get_logger("maintenance").warning(
                                    "quota_inbox_exceeded", project=proj, inbox_count=cnt, limit=inbox_limit
                                )
                await asyncio.sleep(max(60, settings.retention_report_interval_seconds))

        async def _worker_fd_health() -> None:
            """Periodic file descriptor health monitor.

            Checks FD headroom every 30 seconds and proactively cleans up
            resources when headroom drops below safe thresholds. This prevents
            the EMFILE -> socket closed -> unreachable cascade that occurs
            under sustained multi-agent load.

            Also monitors lockfile FD leaks (issue #116) and cleans up
            deleted-but-open .lock file descriptors.

            Thresholds:
            - 30% headroom: warning logged
            - 20% headroom: proactive cleanup triggered (includes lockfile FDs)
            - 15% headroom: error logged, aggressive cleanup
            """
            _fd_logger = structlog.get_logger("fd_health")
            while True:
                try:
                    current, limit = get_fd_usage()
                    if current >= 0 and limit > 0:
                        headroom_pct = (limit - current) / limit
                        cache_stats = get_repo_cache_stats()
                        lock_stats = get_lock_telemetry()

                        if headroom_pct < 0.15:
                            # Critical: aggressive cleanup
                            _fd_logger.error(
                                "fd_health.critical",
                                current_fds=current,
                                fd_limit=limit,
                                headroom_pct=round(headroom_pct * 100, 1),
                                repo_cache=cache_stats,
                                lock_telemetry=lock_stats,
                            )
                            freed = proactive_fd_cleanup(threshold=limit)
                            if freed:
                                _fd_logger.warning(
                                    "fd_health.emergency_cleanup",
                                    freed=freed,
                                    new_headroom=get_fd_headroom(),
                                )
                        elif headroom_pct < 0.20:
                            # Low: proactive cleanup
                            _fd_logger.warning(
                                "fd_health.low",
                                current_fds=current,
                                fd_limit=limit,
                                headroom_pct=round(headroom_pct * 100, 1),
                                repo_cache=cache_stats,
                                lock_telemetry=lock_stats,
                            )
                            freed = proactive_fd_cleanup(threshold=int(limit * 0.25))
                            if freed:
                                _fd_logger.info(
                                    "fd_health.proactive_cleanup",
                                    freed=freed,
                                    new_headroom=get_fd_headroom(),
                                )
                        elif headroom_pct < 0.30:
                            # Warning only
                            _fd_logger.warning(
                                "fd_health.warning",
                                current_fds=current,
                                fd_limit=limit,
                                headroom_pct=round(headroom_pct * 100, 1),
                                repo_cache=cache_stats,
                                lock_telemetry=lock_stats,
                            )
                except Exception:
                    pass
                await asyncio.sleep(30)

        async def _worker_auto_retire_stale_agents() -> None:
            log = structlog.get_logger("maintenance.auto_retire")
            interval = max(60, int(settings.auto_retire_stale_agents_interval_seconds))
            threshold = max(60, int(settings.auto_retire_stale_agents_threshold_seconds))
            while True:
                with contextlib.suppress(Exception):
                    retired = await sweep_stale_agents(threshold_seconds=threshold)
                    if retired:
                        log.info(
                            "auto_retired_stale_agents",
                            count=len(retired),
                            threshold_seconds=threshold,
                            agents=[
                                {
                                    "agent": entry["agent_name"],
                                    "project": entry["project_key"],
                                    "last_active_ts": entry["last_active_ts"],
                                }
                                for entry in retired
                            ],
                        )
                await asyncio.sleep(interval)

        tasks = []
        # FD health monitor always runs - it's critical for preventing EMFILE cascades
        tasks.append(asyncio.create_task(_worker_fd_health()))
        if settings.file_reservations_cleanup_enabled:
            tasks.append(asyncio.create_task(_worker_cleanup()))
        if settings.ack_ttl_enabled:
            tasks.append(asyncio.create_task(_worker_ack_ttl()))
        if settings.tool_metrics_emit_enabled:
            tasks.append(asyncio.create_task(_worker_tool_metrics()))
        if settings.retention_report_enabled or settings.quota_enabled:
            tasks.append(asyncio.create_task(_worker_retention_quota()))
        if settings.auto_retire_stale_agents_enabled:
            tasks.append(asyncio.create_task(_worker_auto_retire_stale_agents()))
        fastapi_app.state._background_tasks = tasks

    async def _shutdown() -> None:  # pragma: no cover - service lifecycle
        tasks = getattr(fastapi_app.state, "_background_tasks", [])
        for task in tasks:
            task.cancel()
        # Await cancelled tasks with a timeout to prevent shutdown hangs
        # (aiosqlite cancellation can block indefinitely)
        if tasks:
            with contextlib.suppress(Exception):
                await asyncio.wait(tasks, timeout=5.0)

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan_context(app: FastAPI):
        # Ensure both mounted MCP apps initialize their internal task groups
        # (each http_app() call owns an independent StreamableHTTPSessionManager).
        mcp_lifespan_app = cast(_FastAPILifespan, mcp_http_app)
        mcp_stateful_lifespan_app = cast(_FastAPILifespan, mcp_stateful_http_app)
        async with (
            mcp_lifespan_app.lifespan(mcp_http_app),
            mcp_stateful_lifespan_app.lifespan(mcp_stateful_http_app),
        ):
            await _startup()
            try:
                yield
            finally:
                await _shutdown()

    # Now construct FastAPI with the composed lifespan so ASGI transports run it.
    # Give the app a real title/version so the auto-generated /openapi.json has a
    # proper `info` block (derive the version from installed package metadata,
    # mirroring cli._package_version; never hardcode a value that could drift).
    def _package_version() -> str:
        import importlib.metadata as _importlib_metadata

        try:
            return _importlib_metadata.version("mcp-agent-mail")
        except _importlib_metadata.PackageNotFoundError:  # pragma: no cover - dev installs
            return "0.0.0+local"

    fastapi_app = FastAPI(
        title="MCP Agent Mail",
        version=_package_version(),
        lifespan=lifespan_context,
    )

    # Simple request logging (configurable)
    if settings.http.request_log_enabled:
        import time as _time

        class RequestLoggingMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
                start = _time.time()
                method = request.method
                path = request.url.path
                client = request.client.host if request.client else "-"
                response = None
                exc: BaseException | None = None
                try:
                    response = await call_next(request)
                    return response
                except BaseException as err:
                    exc = err
                    raise
                finally:
                    # Always emit a log line, even when the handler raised (#215).
                    dur_ms = int((_time.time() - start) * 1000)
                    status_code = getattr(response, "status_code", 0) if response is not None else 500
                    with contextlib.suppress(Exception):
                        log = structlog.get_logger("http")
                        if exc is not None:
                            log.error(
                                "request",
                                method=method,
                                path=path,
                                status=status_code,
                                duration_ms=dur_ms,
                                client_ip=client,
                                error=repr(exc),
                            )
                        else:
                            log.info(
                                "request",
                                method=method,
                                path=path,
                                status=status_code,
                                duration_ms=dur_ms,
                                client_ip=client,
                            )
                    try:
                        rich_console = importlib.import_module("rich.console")
                        rich_panel = importlib.import_module("rich.panel")
                        rich_text = importlib.import_module("rich.text")
                        Console = rich_console.Console
                        Panel = rich_panel.Panel
                        Text = rich_text.Text
                        console = Console(width=100)
                        title = Text.assemble(
                            (method, "bold blue"),
                            ("  "),
                            (path, "bold white"),
                            ("  "),
                            (f"{status_code}", "bold green" if 200 <= status_code < 400 else "bold red"),
                            ("  "),
                            (f"{dur_ms}ms", "bold yellow"),
                        )
                        body = Text.assemble(
                            ("client: ", "cyan"),
                            (client, "white"),
                        )
                        if exc is not None:
                            body = Text.assemble(body, "\n", ("error: ", "cyan"), (repr(exc), "red"))
                        console.print(Panel(body, title=title, border_style="dim"))
                    except Exception:
                        suffix = f" error={exc!r}" if exc is not None else ""
                        print(
                            f"http method={method} path={path} status={status_code} ms={dur_ms} client={client}{suffix}"
                        )

        app_any = cast(Any, fastapi_app)
        app_any.add_middleware(RequestLoggingMiddleware)

    # Unified JWT/RBAC and robust rate limiter middleware
    if (
        settings.http.rate_limit_enabled
        or getattr(settings.http, "jwt_enabled", False)
        or getattr(settings.http, "rbac_enabled", True)
    ):
        app_any = cast(Any, fastapi_app)
        app_any.add_middleware(SecurityAndRateLimitMiddleware, settings=settings)
    # Bearer auth for non-localhost only; allow localhost unauth optionally for seamless local dev
    if settings.http.bearer_token:
        from typing import Any as _Any, cast as _cast  # local type-only import
        app_any = _cast(_Any, fastapi_app)
        app_any.add_middleware(
            BearerAuthMiddleware,
            token=settings.http.bearer_token,
            allow_localhost=bool(getattr(settings.http, "allow_localhost_unauthenticated", False)),
            jwt_enabled=bool(getattr(settings.http, "jwt_enabled", False)),
        )

    # Optional CORS
    if settings.cors.enabled:
        from typing import Any as _Any, cast as _cast  # local type-only import
        app_any2 = _cast(_Any, fastapi_app)
        app_any2.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors.origins or [],
            allow_credentials=settings.cors.allow_credentials,
            allow_methods=settings.cors.allow_methods or ["*"],
            allow_headers=settings.cors.allow_headers or ["*"],
        )

    # Health endpoints
    @fastapi_app.get("/health/liveness")
    async def liveness() -> JSONResponse:
        return JSONResponse({"status": "alive"})

    @fastapi_app.get("/health/readiness")
    async def readiness() -> JSONResponse:
        try:
            await readiness_check()
        except Exception as exc:
            try:
                rich_console = importlib.import_module("rich.console")
                rich_panel = importlib.import_module("rich.panel")
                Console = rich_console.Console
                Panel = rich_panel.Panel
                Console().print(Panel.fit(str(exc), title="Readiness Error", border_style="red"))
            except Exception:
                pass
            with contextlib.suppress(Exception):
                structlog.get_logger("health").error("readiness_error", error=str(exc))
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        return JSONResponse({"status": "ready"})

    @fastapi_app.get("/api/health")
    async def api_health_bypass() -> JSONResponse:
        """Lightweight health probe that bypasses the MCP transport layer.

        Returns immediately without touching the database or connection pool,
        so it stays responsive even when the MCP ASGI pipeline is saturated
        under heavy multi-agent load.
        """
        return JSONResponse({"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()})

    # M3a shared-Hub identity API. These routes only mutate Hub metadata. They
    # never invoke a local shell, terminal multiplexer, worktree, or task runner.
    async def _hub_json_body(request: Request) -> dict[str, Any]:
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Request body must be a JSON object") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object")
        return cast(dict[str, Any], payload)

    def _hub_subject(request: Request) -> str:
        claims = getattr(request.state, "jwt_claims", None)
        subject = claims.get("sub") if isinstance(claims, dict) else None
        if not isinstance(subject, str) or not subject.strip() or len(subject.strip()) > 255:
            raise HTTPException(status_code=401, detail="A valid JWT subject is required")
        return subject.strip()

    def _hub_require_global_admin(request: Request) -> None:
        claims = getattr(request.state, "jwt_claims", None)
        roles_raw = claims.get(settings.http.jwt_role_claim, []) if isinstance(claims, dict) else []
        if isinstance(roles_raw, str):
            roles = {roles_raw}
        elif isinstance(roles_raw, (list, tuple)):
            roles = {str(role) for role in roles_raw}
        else:
            roles = set()
        if "admin" not in roles:
            raise HTTPException(status_code=403, detail="Global administrator role is required")

    async def _hub_human(request: Request, *, session: AsyncSession) -> Human:
        human = await _human_by_subject(_hub_subject(request), session=session)
        if human is None:
            raise HTTPException(status_code=404, detail="Human identity is not registered")
        return human

    async def _hub_materialize_claimed_human(
        request: Request,
        *,
        session: AsyncSession,
    ) -> Human:
        """幂等补建已认证 Human,使旧客户端也能先列出 Topic。"""
        subject = _hub_subject(request)
        human = await _human_by_subject(subject, session=session)
        if human is not None:
            return human
        claims = getattr(request.state, "jwt_claims", None)
        display_name = None
        if isinstance(claims, dict):
            for key in ("name", "preferred_username"):
                value = claims.get(key)
                if isinstance(value, str) and value.strip():
                    display_name = value.strip()
                    break
        try:
            human = await _ensure_human(
                subject,
                display_name or subject,
                session=session,
            )
            await session.commit()
        except IntegrityError:
            await session.rollback()
            human = await _human_by_subject(subject, session=session)
            if human is None:
                raise HTTPException(status_code=409, detail="Human identity conflict") from None
        await session.refresh(human)
        return human

    async def _hub_team_project(slug: str, *, session: AsyncSession) -> TeamProject:
        if not _SLUG_VALIDATOR_RE.fullmatch(slug):
            raise HTTPException(status_code=400, detail="Invalid project slug")
        result = await session.execute(
            select(TeamProject).where(
                cast(Any, TeamProject.slug) == slug,
                cast(Any, TeamProject.archived_at).is_(None),
            )
        )
        team_project = result.scalars().first()
        if team_project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        return team_project

    async def _hub_project(slug: str, *, session: AsyncSession) -> Project:
        team_project = await _hub_team_project(slug, session=session)
        project = await session.get(Project, team_project.routing_project_id)
        if project is None or project.archived_at is not None:
            raise HTTPException(status_code=404, detail="Project not found")
        return project

    async def _hub_membership(
        project_id: int,
        human_id: int,
        *,
        session: AsyncSession,
    ) -> ProjectHumanMembership | None:
        result = await session.execute(
            select(ProjectHumanMembership).where(
                cast(Any, ProjectHumanMembership.project_id) == project_id,
                cast(Any, ProjectHumanMembership.human_id) == human_id,
            )
        )
        return result.scalars().first()

    async def _hub_active_membership(
        project: Project,
        human: Human,
        *,
        session: AsyncSession,
        admin: bool = False,
    ) -> ProjectHumanMembership:
        if project.id is None or human.id is None:
            raise HTTPException(status_code=403, detail="Active project membership is required")
        membership = await _hub_membership(project.id, human.id, session=session)
        if membership is None or membership.status != "active":
            raise HTTPException(status_code=403, detail="Active project membership is required")
        if admin and membership.role != "admin":
            raise HTTPException(status_code=403, detail="Project admin membership is required")
        return membership

    async def _hub_agent_manager(
        request: Request,
        agent: Agent,
        *,
        session: AsyncSession,
    ) -> tuple[Human, ProjectHumanMembership]:
        human = await _hub_human(request, session=session)
        project = await session.get(Project, agent.project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        membership = await _hub_active_membership(project, human, session=session)
        if membership.role != "admin" and agent.owner_id != human.id:
            raise HTTPException(status_code=403, detail="Agent owner or project admin is required")
        return human, membership

    async def _hub_agent_for_update(agent_id: int, *, session: AsyncSession) -> Agent | None:
        result = await session.execute(
            select(Agent)
            .where(cast(Any, Agent.id) == agent_id)
            .with_for_update()
        )
        return result.scalars().first()

    def _hub_human_payload(human: Human) -> dict[str, Any]:
        return {"id": human.id, "display_name": human.display_name}

    def _hub_membership_payload(membership: ProjectHumanMembership) -> dict[str, Any]:
        return {
            "id": membership.id,
            "project_id": membership.project_id,
            "human_id": membership.human_id,
            "mention_handle": membership.mention_handle,
            "role": membership.role,
            "status": membership.status,
            "default_agent_id": membership.default_agent_id,
        }

    def _hub_presence_timestamp(value: datetime | None) -> str | None:
        if value is None:
            return None
        return value.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

    async def _hub_presence_states(
        human_ids: list[int],
        *,
        session: AsyncSession,
    ) -> dict[int, dict[str, Any]]:
        if not human_ids:
            return {}
        rows = await session.execute(
            select(HumanPresence).where(
                cast(Any, HumanPresence.human_id).in_(human_ids)
            )
        )
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        cutoff = now - timedelta(seconds=_HUMAN_PRESENCE_TTL_SECONDS)
        states: dict[int, dict[str, Any]] = {}
        for row in rows.scalars().all():
            state = states.setdefault(
                row.human_id,
                {"online": False, "last_seen_at": None},
            )
            if state["last_seen_at"] is None or row.last_seen_at > state["last_seen_at"]:
                state["last_seen_at"] = row.last_seen_at
            if row.online and row.last_seen_at >= cutoff:
                state["online"] = True
        return {
            human_id: {
                "online": bool(state["online"]),
                "last_seen_at": _hub_presence_timestamp(state["last_seen_at"]),
            }
            for human_id, state in states.items()
        }

    async def _hub_session_lead_states(
        team_project_id: int,
        human_ids: list[int],
        *,
        session: AsyncSession,
    ) -> dict[int, dict[str, Any]]:
        if not human_ids:
            return {}
        rows = await session.execute(
            select(SessionLeadBinding).where(
                cast(Any, SessionLeadBinding.team_project_id) == team_project_id,
                cast(Any, SessionLeadBinding.human_id).in_(human_ids),
                cast(Any, SessionLeadBinding.status) == "active",
            )
        )
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
            seconds=_SESSION_LEAD_RUNTIME_TTL_SECONDS
        )
        states: dict[int, dict[str, Any]] = {}
        for binding in rows.scalars().all():
            runtime_status = (
                binding.runtime_status
                if binding.runtime_status in {"working", "idle", "blocked"}
                and binding.runtime_seen_at is not None
                and binding.runtime_seen_at >= cutoff
                else "stopped"
            )
            states[binding.human_id] = {
                "agent": {
                    "name": binding.lead_label or None,
                    "kind": None,
                    "status": runtime_status,
                    "managed": True,
                    "last_seen_at": _hub_presence_timestamp(binding.runtime_seen_at),
                }
            }
        return states

    def _hub_agent_payload(agent: Agent) -> dict[str, Any]:
        return {
            "id": agent.id,
            "name": agent.name,
            "program": agent.program,
            "model": agent.model,
            "owner_id": agent.owner_id,
            "retired": agent.retired_at is not None,
        }

    def _hub_mention_handle(value: Any) -> str:
        handle = value.strip() if isinstance(value, str) else ""
        if not _MENTION_HANDLE_VALIDATOR_RE.fullmatch(handle):
            raise HTTPException(
                status_code=400,
                detail="mention_handle must be 1-128 letters, numbers, '.', '_' or '-'",
            )
        return handle

    async def _hub_support_sender(
        project: Project,
        human: Human,
        membership: ProjectHumanMembership,
        *,
        session: AsyncSession,
    ) -> tuple[Agent, str]:
        """Resolve a support-message sender without requiring a user Agent.

        A configured, usable default Agent remains the preferred sender. Humans
        without one use an internal per-project relay row so the existing
        Message schema and delivery machinery stay intact. Relay rows are never
        exposed by Team Agent management APIs.
        """
        if membership.default_agent_id is not None:
            sender = await session.get(Agent, membership.default_agent_id)
            if (
                sender is not None
                and sender.owner_id == human.id
                and sender.retired_at is None
                and await _agent_in_project_scope(project, sender, session=session)
            ):
                return sender, (
                    "session_lead"
                    if sender.program == _SESSION_LEAD_PROGRAM
                    else "agent"
                )

        if project.id is None or human.id is None:
            raise HTTPException(status_code=409, detail="Human sender identity is unavailable")
        relay_name = f"TeamHumanRelay{human.id}"
        relay_insert = sqlite_insert(Agent).values(
            project_id=project.id,
            name=relay_name,
            program=_HUB_HUMAN_RELAY_PROGRAM,
            model="hub",
            task_description="Internal Team Hub relay for Human messages",
            owner_id=human.id,
            contact_policy="open",
        )
        await session.execute(
            relay_insert.on_conflict_do_nothing(index_elements=["project_id", "name"])
        )
        relay_row = await session.execute(
            select(Agent).where(
                cast(Any, Agent.project_id) == project.id,
                cast(Any, Agent.name) == relay_name,
                cast(Any, Agent.program) == _HUB_HUMAN_RELAY_PROGRAM,
                cast(Any, Agent.owner_id) == human.id,
                cast(Any, Agent.retired_at).is_(None),
            )
        )
        relay = relay_row.scalars().first()
        if relay is None:
            raise HTTPException(status_code=409, detail="Human sender relay is unavailable")
        return relay, "human"

    @fastapi_app.put("/hub/api/humans/me", response_class=JSONResponse)
    async def hub_upsert_human(request: Request) -> JSONResponse:
        await ensure_schema()
        body = await _hub_json_body(request)
        display_name = body.get("display_name")
        if not isinstance(display_name, str) or not display_name.strip():
            raise HTTPException(status_code=400, detail="display_name is required")
        if len(display_name.strip()) > 255:
            raise HTTPException(status_code=400, detail="display_name is too long")
        subject = _hub_subject(request)
        async with get_session() as session:
            try:
                human = await _ensure_human(
                    subject,
                    display_name.strip(),
                    session=session,
                )
                human.display_name = display_name.strip()
                session.add(human)
                await session.commit()
            except IntegrityError as exc:
                # Two tabs may bootstrap the same JWT subject concurrently.
                # The unique subject row is authoritative; reuse it rather
                # than leaking a database error to the second request.
                await session.rollback()
                human = await _human_by_subject(subject, session=session)
                if human is None:
                    raise HTTPException(status_code=409, detail="Human identity conflict") from exc
                human.display_name = display_name.strip()
                session.add(human)
                await session.commit()
            await session.refresh(human)
            return JSONResponse(_hub_human_payload(human))

    @fastapi_app.get("/hub/api/humans/me", response_class=JSONResponse)
    async def hub_get_human(request: Request) -> JSONResponse:
        await ensure_schema()
        async with get_session() as session:
            human = await _hub_human(request, session=session)
            return JSONResponse(_hub_human_payload(human))

    @fastapi_app.post("/hub/api/presence", response_class=JSONResponse)
    async def hub_update_presence(request: Request) -> JSONResponse:
        """Record one authenticated Cockpit client's online/offline state."""
        await ensure_schema()
        body = await _hub_json_body(request)
        if set(body) != {"client_id", "online"}:
            raise HTTPException(status_code=400, detail="client_id and online are required")
        client_id = body.get("client_id")
        online = body.get("online")
        if not isinstance(client_id, str) or not _HUMAN_PRESENCE_CLIENT_RE.fullmatch(client_id):
            raise HTTPException(status_code=400, detail="Invalid presence client_id")
        if not isinstance(online, bool):
            raise HTTPException(status_code=400, detail="online must be a boolean")
        async with get_session() as session:
            human = await _hub_materialize_claimed_human(request, session=session)
            if human.id is None:
                raise HTTPException(status_code=404, detail="Human identity is not registered")
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            statement = sqlite_insert(HumanPresence).values(
                human_id=human.id,
                client_id=client_id,
                online=online,
                last_seen_at=now,
            )
            await session.execute(
                statement.on_conflict_do_update(
                    index_elements=["human_id", "client_id"],
                    set_={"online": online, "last_seen_at": now},
                )
            )
            await session.commit()
            state = (await _hub_presence_states([human.id], session=session)).get(
                human.id,
                {"online": False, "last_seen_at": None},
            )
            return JSONResponse(state)

    @fastapi_app.get("/hub/api/agents", response_class=JSONResponse)
    async def hub_list_agent_directory(request: Request) -> JSONResponse:
        """List safe Agent candidates for explicit TeamProject binding.

        Ordinary Humans see only Agents they own. Global administrators may
        select unowned or other-owned Agents, but logical TeamProject routing
        Agents are excluded so one group's internal identity cannot be bound
        into another group. Registration credentials and project human keys
        never leave the Hub.
        """
        await ensure_schema()
        async with get_session() as session:
            human = await _hub_human(request, session=session)
            statement = (
                select(Agent, Project)
                .join(Project, cast(Any, Project.id) == Agent.project_id)
                .outerjoin(
                    TeamProject,
                    cast(Any, TeamProject.routing_project_id) == Project.id,
                )
                .where(
                    cast(Any, Agent.retired_at).is_(None),
                    cast(Any, Project.archived_at).is_(None),
                    cast(Any, TeamProject.id).is_(None),
                )
                .order_by(Agent.name, cast(Any, Agent.id))
                .limit(500)
            )
            if not _hub_is_global_admin(request):
                statement = statement.where(cast(Any, Agent.owner_id) == human.id)
            rows = await session.execute(statement)
            agents = []
            for agent, project in rows.all():
                payload = _hub_agent_payload(agent)
                payload["project_slug"] = project.slug
                agents.append(payload)
            return JSONResponse({"agents": agents})

    @fastapi_app.get("/hub/api/projects", response_class=JSONResponse)
    async def hub_list_projects(request: Request) -> JSONResponse:
        """List only user-created logical groups, never technical mail projects."""
        await ensure_schema()
        async with get_session() as session:
            human = await _hub_materialize_claimed_human(request, session=session)
            if human.id is None:
                raise HTTPException(status_code=404, detail="Human identity is not registered")
            rows = await session.execute(
                text(
                    """
                    SELECT tp.id, tp.slug, tp.name,
                           mine.role, mine.status, mine.mention_handle,
                           COUNT(active_members.id) AS active_member_count
                    FROM team_projects tp
                    JOIN projects routing ON routing.id = tp.routing_project_id
                    LEFT JOIN project_human_memberships mine
                      ON mine.project_id = tp.routing_project_id
                     AND mine.human_id = :human_id
                    LEFT JOIN project_human_memberships active_members
                      ON active_members.project_id = tp.routing_project_id
                     AND active_members.status = 'active'
                    WHERE tp.archived_at IS NULL AND routing.archived_at IS NULL
                    GROUP BY tp.id, tp.slug, tp.name,
                             mine.role, mine.status, mine.mention_handle
                    ORDER BY tp.created_at DESC
                    """
                ),
                {"human_id": human.id},
            )
            projects = [
                {
                    "id": row.id,
                    "slug": row.slug,
                    "name": row.name,
                    "active_member_count": row.active_member_count,
                    "membership": (
                        {
                            "role": row.role,
                            "status": row.status,
                            "mention_handle": row.mention_handle,
                        }
                        if row.status is not None
                        else None
                    ),
                }
                for row in rows
            ]
            return JSONResponse({"projects": projects})

    @fastapi_app.post("/hub/api/projects", response_class=JSONResponse, status_code=201)
    async def hub_create_project(request: Request) -> JSONResponse:
        await ensure_schema()
        _hub_require_global_admin(request)
        body = await _hub_json_body(request)
        name = body.get("name")
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > 255:
            raise HTTPException(status_code=400, detail="name is required and must be at most 255 characters")
        raw_slug = body.get("slug")
        slug = raw_slug.strip().lower() if isinstance(raw_slug, str) else ""
        if not slug or len(slug) > 128 or not _SLUG_VALIDATOR_RE.fullmatch(slug):
            raise HTTPException(
                status_code=400,
                detail="slug must be 1-128 letters, numbers, '_' or '-'",
            )
        mention_handle = _hub_mention_handle(body.get("mention_handle"))

        async with get_session() as session:
            human = await _hub_human(request, session=session)
            routing_id = uuid.uuid4().hex
            project = Project(
                slug=f"hub-group-{routing_id}",
                human_key=f"team:{routing_id}",
            )
            session.add(project)
            try:
                await session.flush()
                if project.id is None:
                    raise HTTPException(status_code=500, detail="Failed to create routing project")
                team_project = TeamProject(
                    slug=slug,
                    name=name.strip(),
                    routing_project_id=project.id,
                )
                session.add(team_project)
                await session.flush()
                membership = await _upsert_project_human_membership(
                    project=project,
                    subject=human.subject,
                    display_name=human.display_name,
                    mention_handle=mention_handle,
                    role="admin",
                    status="active",
                    session=session,
                )
                await session.commit()
            except IntegrityError as exc:
                await session.rollback()
                raise HTTPException(status_code=409, detail="Project already exists") from exc
            await session.refresh(project)
            await session.refresh(team_project)
            await session.refresh(membership)
            return JSONResponse(
                {
                    "id": team_project.id,
                    "slug": team_project.slug,
                    "name": team_project.name,
                    "membership": _hub_membership_payload(membership),
                },
                status_code=201,
            )

    @fastapi_app.post(
        "/hub/api/projects/{project_slug}/join-requests",
        response_class=JSONResponse,
        status_code=201,
    )
    async def hub_request_project_join(project_slug: str, request: Request) -> JSONResponse:
        await ensure_schema()
        body = await _hub_json_body(request)
        mention_handle = _hub_mention_handle(body.get("mention_handle"))
        async with get_session() as session:
            human = await _hub_human(request, session=session)
            project = await _hub_project(project_slug, session=session)
            if project.id is None or human.id is None:
                raise HTTPException(status_code=404, detail="Project or Human not found")
            project_id = project.id
            human_id = human.id
            existing = await _hub_membership(project_id, human_id, session=session)
            if existing is not None and existing.status == "active":
                raise HTTPException(status_code=409, detail="Human is already an active member")
            try:
                membership = await _upsert_project_human_membership(
                    project=project,
                    subject=human.subject,
                    display_name=human.display_name,
                    mention_handle=mention_handle,
                    role="member",
                    status="invited",
                    session=session,
                )
                await session.commit()
            except ValueError as exc:
                await session.rollback()
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except IntegrityError as exc:
                # Idempotent concurrent self-join: if this Human's row won in
                # another transaction, return it. A handle collision belonging
                # to someone else remains a conflict.
                await session.rollback()
                membership = await _hub_membership(project_id, human_id, session=session)
                if membership is None:
                    raise HTTPException(status_code=409, detail="Membership conflict") from exc
            await session.refresh(membership)
            return JSONResponse(
                _hub_membership_payload(membership),
                status_code=200 if existing is not None else 201,
            )

    @fastapi_app.get(
        "/hub/api/projects/{project_slug}/membership",
        response_class=JSONResponse,
    )
    async def hub_get_membership(project_slug: str, request: Request) -> JSONResponse:
        await ensure_schema()
        async with get_session() as session:
            human = await _hub_human(request, session=session)
            project = await _hub_project(project_slug, session=session)
            if project.id is None or human.id is None:
                raise HTTPException(status_code=404, detail="Membership not found")
            membership = await _hub_membership(project.id, human.id, session=session)
            if membership is None:
                raise HTTPException(status_code=404, detail="Membership not found")
            return JSONResponse(_hub_membership_payload(membership))

    @fastapi_app.patch(
        "/hub/api/projects/{project_slug}/membership",
        response_class=JSONResponse,
    )
    async def hub_update_membership(project_slug: str, request: Request) -> JSONResponse:
        await ensure_schema()
        body = await _hub_json_body(request)
        allowed = {"mention_handle", "default_agent_id"}
        if not body or not set(body).issubset(allowed):
            raise HTTPException(
                status_code=400,
                detail="Only mention_handle and default_agent_id may be changed",
            )
        async with get_session() as session:
            human = await _hub_human(request, session=session)
            project = await _hub_project(project_slug, session=session)
            membership = await _hub_active_membership(project, human, session=session)
            mention_handle = (
                _hub_mention_handle(body["mention_handle"])
                if "mention_handle" in body
                else membership.mention_handle
            )
            default_agent_id = body.get(
                "default_agent_id", membership.default_agent_id
            )
            if default_agent_id is not None and (
                isinstance(default_agent_id, bool) or not isinstance(default_agent_id, int)
            ):
                raise HTTPException(status_code=400, detail="default_agent_id must be an integer or null")
            try:
                membership = await _upsert_project_human_membership(
                    project=project,
                    subject=human.subject,
                    display_name=human.display_name,
                    mention_handle=mention_handle,
                    role=membership.role,
                    status=membership.status,
                    default_agent_id=default_agent_id,
                    session=session,
                )
                await session.commit()
            except ValueError as exc:
                await session.rollback()
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            await session.refresh(membership)
            return JSONResponse(_hub_membership_payload(membership))

    @fastapi_app.get(
        "/hub/api/projects/{project_slug}/members",
        response_class=JSONResponse,
    )
    async def hub_list_members(project_slug: str, request: Request) -> JSONResponse:
        await ensure_schema()
        async with get_session() as session:
            human = await _hub_human(request, session=session)
            team_project = await _hub_team_project(project_slug, session=session)
            project = await session.get(Project, team_project.routing_project_id)
            if project is None or project.archived_at is not None:
                raise HTTPException(status_code=404, detail="Project not found")
            caller = await _hub_active_membership(project, human, session=session)
            is_admin = caller.role == "admin"
            # Ordinary members get a minimal roster of ACTIVE members only —
            # the @mention UI needs human_id/display_name/mention_handle, but
            # invited/removed rows, opaque subjects and other members'
            # default_agent_id stay hidden. Admins keep the full view for
            # approval and member management.
            conditions = [cast(Any, ProjectHumanMembership.project_id) == project.id]
            if not is_admin:
                conditions.append(cast(Any, ProjectHumanMembership.status) == "active")
            rows = await session.execute(
                select(ProjectHumanMembership, Human)
                .join(Human, cast(Any, Human.id) == ProjectHumanMembership.human_id)
                .where(*conditions)
                .order_by(cast(Any, ProjectHumanMembership.created_at))
            )
            members = []
            member_rows = rows.all()
            presence_states = await _hub_presence_states(
                [membership.human_id for membership, _member_human in member_rows],
                session=session,
            )
            lead_states = await _hub_session_lead_states(
                cast(int, team_project.id),
                [membership.human_id for membership, _member_human in member_rows],
                session=session,
            )
            for membership, member_human in member_rows:
                if is_admin:
                    payload = _hub_membership_payload(membership)
                else:
                    payload = {
                        "human_id": membership.human_id,
                        "mention_handle": membership.mention_handle,
                        "role": membership.role,
                        "status": membership.status,
                    }
                payload["display_name"] = member_human.display_name
                payload.update(presence_states.get(
                    membership.human_id,
                    {"online": False, "last_seen_at": None},
                ))
                payload.update(lead_states.get(membership.human_id, {"agent": None}))
                members.append(payload)
            return JSONResponse({"members": members})

    @fastapi_app.post(
        "/hub/api/projects/{project_slug}/members",
        response_class=JSONResponse,
        status_code=201,
    )
    async def hub_provision_member(project_slug: str, request: Request) -> JSONResponse:
        """Project admin accepts an issuer-approved invitation target.

        Provisioning is idempotent so Cockpit can safely retry before it
        activates the Human Auth account.
        """
        await ensure_schema()
        body = await _hub_json_body(request)
        if set(body) != {"subject", "display_name", "mention_handle"}:
            raise HTTPException(
                status_code=400,
                detail="subject, display_name and mention_handle are required",
            )
        subject = body.get("subject")
        display_name = body.get("display_name")
        if not isinstance(subject, str) or not subject.strip() or len(subject.strip()) > 255:
            raise HTTPException(status_code=400, detail="Invalid Human subject")
        if (
            not isinstance(display_name, str)
            or not display_name.strip()
            or len(display_name.strip()) > 255
        ):
            raise HTTPException(status_code=400, detail="Invalid Human display name")
        mention_handle = _hub_mention_handle(body.get("mention_handle"))
        async with get_session() as session:
            actor = await _hub_human(request, session=session)
            project = await _hub_project(project_slug, session=session)
            await _hub_active_membership(project, actor, session=session, admin=True)
            existing_human = await _human_by_subject(subject.strip(), session=session)
            existing = None
            if existing_human is not None and existing_human.id is not None and project.id is not None:
                existing = await _hub_membership(
                    project.id, existing_human.id, session=session,
                )
            try:
                membership = await _upsert_project_human_membership(
                    project=project,
                    subject=subject.strip(),
                    display_name=display_name.strip(),
                    mention_handle=mention_handle,
                    role="admin" if existing is not None and existing.role == "admin" else "member",
                    status="active",
                    session=session,
                )
                await session.commit()
            except ValueError as exc:
                await session.rollback()
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except IntegrityError as exc:
                await session.rollback()
                raise HTTPException(status_code=409, detail="Membership conflict") from exc
            await session.refresh(membership)
            return JSONResponse(
                _hub_membership_payload(membership),
                status_code=200 if existing is not None else 201,
            )

    @fastapi_app.patch(
        "/hub/api/projects/{project_slug}/members/{human_id}",
        response_class=JSONResponse,
    )
    async def hub_manage_member(
        project_slug: str,
        human_id: int,
        request: Request,
    ) -> JSONResponse:
        await ensure_schema()
        body = await _hub_json_body(request)
        allowed = {"mention_handle", "role", "status"}
        if not body or not set(body).issubset(allowed):
            raise HTTPException(
                status_code=400,
                detail="Only mention_handle, role and status may be changed",
            )
        async with get_session() as session:
            actor = await _hub_human(request, session=session)
            project = await _hub_project(project_slug, session=session)
            await _hub_active_membership(project, actor, session=session, admin=True)
            if actor.id == human_id:
                raise HTTPException(status_code=400, detail="Use the self membership endpoint")
            if project.id is None:
                raise HTTPException(status_code=404, detail="Project not found")
            membership = await _hub_membership(project.id, human_id, session=session)
            target = await session.get(Human, human_id)
            if membership is None or target is None:
                raise HTTPException(status_code=404, detail="Membership not found")
            mention_handle = (
                _hub_mention_handle(body["mention_handle"])
                if "mention_handle" in body
                else membership.mention_handle
            )
            role = body.get("role", membership.role)
            member_status = body.get("status", membership.status)
            default_agent_id = (
                None if member_status == "removed" else membership.default_agent_id
            )
            try:
                membership = await _upsert_project_human_membership(
                    project=project,
                    subject=target.subject,
                    display_name=target.display_name,
                    mention_handle=mention_handle,
                    role=role,
                    status=member_status,
                    default_agent_id=default_agent_id,
                    session=session,
                )
                if member_status != "active":
                    # 成员失活: 其受管 lead 同事务解绑且 reply capability 失效(#1096)
                    team_project = await _team_project_for_routing(project.id, session=session)
                    if team_project is not None and team_project.id is not None:
                        await session.execute(
                            update(SessionLeadBinding)
                            .where(
                                cast(Any, SessionLeadBinding.team_project_id) == team_project.id,
                                cast(Any, SessionLeadBinding.human_id) == target.id,
                                cast(Any, SessionLeadBinding.status) == "active",
                            )
                            .values(
                                status="unbound",
                                reply_token_hash=None,
                                updated_at=datetime.now(timezone.utc).replace(tzinfo=None),
                            )
                        )
                await session.commit()
            except ValueError as exc:
                await session.rollback()
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            await session.refresh(membership)
            return JSONResponse(_hub_membership_payload(membership))

    @fastapi_app.get(
        "/hub/api/projects/{project_slug}/agents",
        response_class=JSONResponse,
    )
    async def hub_list_agents(project_slug: str, request: Request) -> JSONResponse:
        await ensure_schema()
        async with get_session() as session:
            human = await _hub_human(request, session=session)
            team_project = await _hub_team_project(project_slug, session=session)
            project = await session.get(Project, team_project.routing_project_id)
            if project is None or project.archived_at is not None:
                raise HTTPException(status_code=404, detail="Project not found")
            await _hub_active_membership(project, human, session=session)
            rows = await session.execute(
                select(Agent)
                .where(
                    cast(Any, Agent.project_id) == project.id,
                    cast(Any, Agent.program) != _HUB_HUMAN_RELAY_PROGRAM,
                    cast(Any, Agent.program) != _SESSION_LEAD_PROGRAM,
                )
                .order_by(Agent.name)
            )
            agents = [_hub_agent_payload(agent) for agent in rows.scalars().all()]
            # M3b-1: actively bound external agents are first-class members of
            # the group namespace — listed (so they can be picked as a default
            # agent) but never exposing credentials. Payload stays token-free.
            bound_rows = await session.execute(
                select(Agent)
                .join(
                    TeamProjectAgentBinding,
                    cast(Any, TeamProjectAgentBinding.agent_id) == Agent.id,
                )
                .where(
                    cast(Any, TeamProjectAgentBinding.team_project_id) == team_project.id,
                    cast(Any, TeamProjectAgentBinding.status) == "active",
                    cast(Any, Agent.project_id) != project.id,
                    cast(Any, Agent.program) != _HUB_HUMAN_RELAY_PROGRAM,
                )
                .order_by(Agent.name)
            )
            for agent in bound_rows.scalars().all():
                payload = _hub_agent_payload(agent)
                payload["bound_external"] = True
                agents.append(payload)
            return JSONResponse({"agents": agents})

    @fastapi_app.patch(
        "/hub/api/projects/{project_slug}/agents/{agent_id}",
        response_class=JSONResponse,
    )
    async def hub_manage_agent(
        project_slug: str,
        agent_id: int,
        request: Request,
    ) -> JSONResponse:
        await ensure_schema()
        body = await _hub_json_body(request)
        allowed = {"owner_id", "retired"}
        if not body or not set(body).issubset(allowed):
            raise HTTPException(status_code=400, detail="Only owner_id and retired may be changed")
        async with get_session() as session:
            human = await _hub_human(request, session=session)
            project = await _hub_project(project_slug, session=session)
            if project.id is None:
                raise HTTPException(status_code=404, detail="Project not found")
            membership = await _hub_active_membership(project, human, session=session)
            agent = await _hub_agent_for_update(agent_id, session=session)
            if agent is None or not await _agent_in_project_scope(project, agent, session=session):
                raise HTTPException(status_code=404, detail="Agent not found")
            # M3b-1 review: an externally bound Agent row is GLOBAL state shared
            # with its home project and other groups. A group admin may bind it,
            # but must not gain lifecycle control over it:
            #   * owner_id change → global admin only
            #   * retired change  → global admin or the agent's CURRENT owner
            # Local routing-project agents keep the existing group admin/owner rules.
            is_local_agent = agent.project_id == project.id
            is_global_admin = _hub_is_global_admin(request)

            if "owner_id" in body:
                if is_local_agent:
                    if membership.role != "admin":
                        raise HTTPException(status_code=403, detail="Project admin membership is required")
                elif not is_global_admin:
                    raise HTTPException(
                        status_code=403,
                        detail="Global administrator role is required to change a bound external agent owner",
                    )
                owner_id = body["owner_id"]
                if owner_id is not None and (
                    isinstance(owner_id, bool) or not isinstance(owner_id, int)
                ):
                    raise HTTPException(status_code=400, detail="owner_id must be an integer or null")
                if owner_id is not None:
                    owner_membership = await _hub_membership(
                        project.id,
                        owner_id,
                        session=session,
                    )
                    if owner_membership is None or owner_membership.status != "active":
                        raise HTTPException(
                            status_code=400,
                            detail="Agent owner must be an active project member",
                        )
                try:
                    agent = await _set_agent_owner(agent_id, owner_id, session=session)
                except (NoResultFound, ValueError) as exc:
                    raise HTTPException(status_code=409, detail=str(exc)) from exc

            if "retired" in body:
                retired = body["retired"]
                if not isinstance(retired, bool):
                    raise HTTPException(status_code=400, detail="retired must be a boolean")
                if is_local_agent:
                    if membership.role != "admin" and agent.owner_id != human.id:
                        raise HTTPException(
                            status_code=403,
                            detail="Agent owner or project admin is required",
                        )
                elif not is_global_admin and agent.owner_id != human.id:
                    raise HTTPException(
                        status_code=403,
                        detail="Global admin or the current agent owner is required for a bound external agent",
                    )
                if retired and agent.retired_at is None:
                    referenced = await _agent_referenced_as_default(agent_id, session=session)
                    if referenced:
                        raise HTTPException(
                            status_code=409,
                            detail="Clear membership default_agent_id before retiring this agent",
                        )
                agent.retired_at = (
                    datetime.now(timezone.utc).replace(tzinfo=None) if retired else None
                )
                session.add(agent)

            await session.commit()
            await session.refresh(agent)
            return JSONResponse(_hub_agent_payload(agent))

    class _HubHTTPDeliveryContext:
        async def info(self, message: str) -> None:
            structlog.get_logger("hub-support").info("delivery", message=message)

    @fastapi_app.post(
        "/hub/api/projects/{project_slug}/attachments",
        response_class=JSONResponse,
        status_code=201,
    )
    async def hub_upload_team_attachment(
        project_slug: str,
        request: Request,
        file: Annotated[UploadFile, File()],
    ) -> JSONResponse:
        """Store one opaque Team attachment; contents are never Agent input."""
        await ensure_schema()
        filename, media_type = _team_attachment_filename(file.filename)
        async with get_session() as session:
            human = await _hub_human(request, session=session)
            team_project = await _hub_team_project(project_slug, session=session)
            project = await session.get(Project, team_project.routing_project_id)
            if project is None or project.archived_at is not None:
                raise HTTPException(status_code=404, detail="Project not found")
            await _hub_active_membership(project, human, session=session)
            stale_before = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=1)
            stale_rows = (await session.execute(
                select(TeamAttachment).where(
                    cast(Any, TeamAttachment.project_id) == project.id,
                    cast(Any, TeamAttachment.owner_human_id) == human.id,
                    cast(Any, TeamAttachment.message_id).is_(None),
                    cast(Any, TeamAttachment.created_ts) < stale_before,
                )
            )).scalars().all()
            stale_names = [row.storage_name for row in stale_rows]
            for row in stale_rows:
                await session.delete(row)
            if stale_rows:
                await session.commit()
                for stale_name in stale_names:
                    with contextlib.suppress(OSError):
                        (_team_attachment_root(settings) / stale_name).unlink()
            pending_count = int((await session.execute(
                select(func.count(TeamAttachment.id)).where(
                    cast(Any, TeamAttachment.project_id) == project.id,
                    cast(Any, TeamAttachment.owner_human_id) == human.id,
                    cast(Any, TeamAttachment.message_id).is_(None),
                )
            )).scalar_one())
            if pending_count >= 8:
                raise HTTPException(status_code=429, detail="待发送附件过多: 请先发送或移除")
            content = await file.read(_TEAM_ATTACHMENT_MAX_BYTES + 1)
            await file.close()
            if not content:
                raise HTTPException(status_code=400, detail="附件不能为空")
            if len(content) > _TEAM_ATTACHMENT_MAX_BYTES:
                raise HTTPException(status_code=413, detail="附件不能超过 10 MiB")
            quota = int(settings.quota_attachments_limit_bytes)
            if quota > 0:
                used = int((await session.execute(
                    select(func.coalesce(func.sum(TeamAttachment.size), 0)).where(
                        cast(Any, TeamAttachment.project_id) == project.id,
                    )
                )).scalar_one())
                if used + len(content) > quota:
                    raise HTTPException(status_code=413, detail="团队附件空间已达到配额")
            token = uuid.uuid4().hex
            storage_name = token
            path = _team_attachment_root(settings) / storage_name
            try:
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(content)
                attachment = TeamAttachment(
                    token=token,
                    project_id=cast(int, project.id),
                    owner_human_id=cast(int, human.id),
                    filename=filename,
                    media_type=media_type,
                    size=len(content),
                    sha256=hashlib.sha256(content).hexdigest(),
                    storage_name=storage_name,
                )
                session.add(attachment)
                await session.commit()
                await session.refresh(attachment)
            except Exception:
                with contextlib.suppress(OSError):
                    path.unlink()
                raise
        return JSONResponse(_public_team_attachment(attachment), status_code=201)

    @fastapi_app.delete(
        "/hub/api/projects/{project_slug}/attachments/{attachment_token}",
        response_class=JSONResponse,
    )
    async def hub_delete_team_attachment(
        project_slug: str,
        attachment_token: str,
        request: Request,
    ) -> JSONResponse:
        """Delete only the caller's still-unattached upload."""
        await ensure_schema()
        if re.fullmatch(r"[0-9a-f]{32}", attachment_token) is None:
            raise HTTPException(status_code=404, detail="附件不存在")
        async with get_session() as session:
            human = await _hub_human(request, session=session)
            team_project = await _hub_team_project(project_slug, session=session)
            project = await session.get(Project, team_project.routing_project_id)
            if project is None or project.archived_at is not None:
                raise HTTPException(status_code=404, detail="Project not found")
            await _hub_active_membership(project, human, session=session)
            row = (await session.execute(
                select(TeamAttachment).where(
                    cast(Any, TeamAttachment.token) == attachment_token,
                    cast(Any, TeamAttachment.project_id) == project.id,
                    cast(Any, TeamAttachment.owner_human_id) == human.id,
                )
            )).scalars().first()
            if row is None:
                raise HTTPException(status_code=404, detail="附件不存在")
            if row.message_id is not None:
                raise HTTPException(status_code=409, detail="已发送附件不能删除")
            storage_name = row.storage_name
            await session.delete(row)
            await session.commit()
        with contextlib.suppress(OSError):
            (_team_attachment_root(settings) / storage_name).unlink()
        return JSONResponse({"deleted": True})

    @fastapi_app.get(
        "/hub/api/projects/{project_slug}/attachments/{attachment_token}",
        response_class=FileResponse,
    )
    async def hub_download_team_attachment(
        project_slug: str,
        attachment_token: str,
        request: Request,
    ) -> FileResponse:
        """Download through membership authorization; never render inline."""
        await ensure_schema()
        if re.fullmatch(r"[0-9a-f]{32}", attachment_token) is None:
            raise HTTPException(status_code=404, detail="附件不存在")
        async with get_session() as session:
            human = await _hub_human(request, session=session)
            team_project = await _hub_team_project(project_slug, session=session)
            project = await session.get(Project, team_project.routing_project_id)
            if project is None or project.archived_at is not None:
                raise HTTPException(status_code=404, detail="Project not found")
            await _hub_active_membership(project, human, session=session)
            row = (await session.execute(
                select(TeamAttachment).where(
                    cast(Any, TeamAttachment.token) == attachment_token,
                    cast(Any, TeamAttachment.project_id) == project.id,
                )
            )).scalars().first()
            if row is None or (
                row.message_id is None and row.owner_human_id != human.id
            ):
                raise HTTPException(status_code=404, detail="附件不存在")
            filename = row.filename
            media_type = row.media_type
            storage_name = row.storage_name
        path = _team_attachment_root(settings) / storage_name
        if not path.is_file():
            raise HTTPException(status_code=404, detail="附件文件不存在")
        return FileResponse(
            path,
            filename=filename,
            media_type=media_type,
            content_disposition_type="attachment",
            headers={
                "Cache-Control": "private, no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @fastapi_app.get(
        "/hub/api/projects/{project_slug}/chat/messages",
        response_class=JSONResponse,
    )
    async def hub_list_chat_messages(
        project_slug: str,
        request: Request,
        limit: int = Query(default=80, ge=1, le=200),
        before_id: int | None = Query(default=None, ge=1),
    ) -> JSONResponse:
        """Return one newest-first cursor page, rendered oldest-first."""
        await ensure_schema()
        async with get_session() as session:
            human = await _hub_human(request, session=session)
            team_project = await _hub_team_project(project_slug, session=session)
            project = await session.get(Project, team_project.routing_project_id)
            if project is None or project.archived_at is not None:
                raise HTTPException(status_code=404, detail="Project not found")
            await _hub_active_membership(project, human, session=session)
            channel_row = await session.execute(
                select(Channel).where(
                    cast(Any, Channel.project_id) == project.id,
                    cast(Any, Channel.name) == "support",
                )
            )
            channel = channel_row.scalars().first()
            if channel is None:
                return JSONResponse({
                    "channel": "support",
                    "messages": [],
                    "count": 0,
                    "has_more": False,
                    "next_before_id": None,
                })
            conditions = [cast(Any, ChannelMessage.channel_id) == channel.id]
            if before_id is not None:
                conditions.append(cast(Any, ChannelMessage.id) < before_id)
            rows = await session.execute(
                select(ChannelMessage, Agent, Human, SessionLeadBinding)
                .join(Agent, cast(Any, Agent.id) == ChannelMessage.sender_id)
                .outerjoin(Human, cast(Any, Human.id) == Agent.owner_id)
                .outerjoin(
                    SessionLeadBinding,
                    cast(Any, SessionLeadBinding.agent_id) == Agent.id,
                )
                .where(*conditions)
                .order_by(cast(Any, ChannelMessage.id).desc())
                .limit(limit + 1)
            )
            page_rows = rows.all()
            has_more = len(page_rows) > limit
            page_rows = page_rows[:limit]
            items = []
            for message, sender, sender_human, lead_binding in page_rows:
                body_md = message.body_md
                mention_handles: list[str] = []
                prefix, separator, content = body_md.partition("\n\n")
                tokens = prefix.split()
                if separator and tokens and all(
                    token.startswith("@")
                    and _MENTION_HANDLE_VALIDATOR_RE.fullmatch(token[1:])
                    for token in tokens
                ):
                    mention_handles = [token[1:] for token in tokens]
                    body_md = content
                is_relay = sender.program == _HUB_HUMAN_RELAY_PROGRAM
                is_session_lead = sender.program == _SESSION_LEAD_PROGRAM
                # M3: 受管 lead 是 Human 的代理,不是独立团队成员——sender 归
                # Human(display_name/human_id),内部 Agent 名永不透出;sender_agent
                # 只带客户端 lead_label,前端据此显示 "付彦超 · via codex-main"
                # 并识别自己的消息。
                items.append({
                    "id": message.id,
                    "subject": message.subject,
                    "body_md": body_md,
                    "attachments": [
                        item for item in message.attachments
                        if isinstance(item, dict)
                    ],
                    "mention_handles": mention_handles,
                    "importance": message.importance,
                    "created_ts": str(message.created_ts),
                    "sender_name": (
                        sender_human.display_name
                        if sender_human is not None
                        else ("Team member" if is_relay or is_session_lead else sender.name)
                    ),
                    "sender_human_id": sender_human.id if sender_human else None,
                    "sender_kind": (
                        "human"
                        if is_relay
                        else ("session_lead" if is_session_lead else "agent")
                    ),
                    "sender_agent": (
                        None
                        if is_relay
                        else (
                            lead_binding.lead_label or None
                            if is_session_lead and lead_binding is not None
                            else (None if is_session_lead else sender.name)
                        )
                    ),
                })
            items.reverse()
            return JSONResponse(
                {
                    "channel": "support",
                    "messages": items,
                    "count": len(items),
                    "has_more": has_more,
                    "next_before_id": items[0]["id"] if has_more and items else None,
                }
            )

    @fastapi_app.post(
        "/hub/api/projects/{project_slug}/support-requests",
        response_class=JSONResponse,
        status_code=201,
    )
    async def hub_post_support_request(
        project_slug: str,
        request: Request,
    ) -> JSONResponse:
        """Post a real team support message as the Human or their default Agent."""
        await ensure_schema()
        body = await _hub_json_body(request)
        allowed = {
            "subject", "body_md", "importance", "mention_handles", "attachment_ids",
        }
        if not set(body).issubset(allowed):
            raise HTTPException(status_code=400, detail="Unsupported support request field")
        subject = body.get("subject")
        body_md = body.get("body_md")
        importance = body.get("importance", "normal")
        attachment_ids = body.get("attachment_ids", [])
        if (
            not isinstance(attachment_ids, list)
            or len(attachment_ids) > _TEAM_ATTACHMENT_MAX_PER_MESSAGE
            or any(
                not isinstance(item, str)
                or re.fullmatch(r"[0-9a-f]{32}", item) is None
                for item in attachment_ids
            )
            or len(set(attachment_ids)) != len(attachment_ids)
        ):
            raise HTTPException(status_code=400, detail="附件列表无效")
        if not isinstance(subject, str) or not subject.strip() or len(subject.strip()) > 512:
            raise HTTPException(status_code=400, detail="subject is required and must be at most 512 characters")
        if not isinstance(body_md, str) or not body_md.strip() or len(body_md) > 50_000:
            raise HTTPException(status_code=400, detail="body_md is required and must be at most 50000 characters")
        if importance not in {"low", "normal", "high", "urgent"}:
            raise HTTPException(status_code=400, detail="Invalid importance")
        raw_handles = body.get("mention_handles")
        requested_handles: list[str] | None = None
        if raw_handles is not None:
            if not isinstance(raw_handles, list) or not raw_handles or len(raw_handles) > 50:
                raise HTTPException(status_code=400, detail="mention_handles must be a non-empty array")
            requested_handles = []
            for raw_handle in raw_handles:
                if not isinstance(raw_handle, str):
                    raise HTTPException(status_code=400, detail="mention_handles must contain strings")
                handle = _hub_mention_handle(raw_handle)
                if handle.lower() not in {item.lower() for item in requested_handles}:
                    requested_handles.append(handle)

        async with get_session() as session:
            human = await _hub_human(request, session=session)
            team_project = await _hub_team_project(project_slug, session=session)
            project = await session.get(Project, team_project.routing_project_id)
            if project is None or project.archived_at is not None:
                raise HTTPException(status_code=404, detail="Project not found")
            membership = await _hub_active_membership(project, human, session=session)
            if requested_handles is None and membership.role != "admin":
                raise HTTPException(
                    status_code=403,
                    detail="只有话题管理员可以使用 @all",
                )
            sender, sender_kind = await _hub_support_sender(
                project,
                human,
                membership,
                session=session,
            )
            if sender.program == _SESSION_LEAD_PROGRAM:
                lead_row = await session.execute(
                    select(SessionLeadBinding).where(
                        cast(Any, SessionLeadBinding.agent_id) == sender.id,
                    )
                )
                lead_binding = lead_row.scalars().first()
                sender_agent_label = (
                    lead_binding.lead_label or None
                    if lead_binding is not None
                    else None
                )
            elif sender.program == _HUB_HUMAN_RELAY_PROGRAM:
                sender_agent_label = None
            else:
                sender_agent_label = sender.name
            rows = await session.execute(
                select(ProjectHumanMembership).where(
                    cast(Any, ProjectHumanMembership.project_id) == project.id,
                    cast(Any, ProjectHumanMembership.status) == "active",
                    cast(Any, ProjectHumanMembership.human_id) != human.id,
                )
            )
            candidates = rows.scalars().all()
            by_handle = {item.mention_handle.lower(): item for item in candidates}
            if requested_handles is None:
                target_memberships = candidates
            else:
                missing = [handle for handle in requested_handles if handle.lower() not in by_handle]
                if missing:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Active team member not found: {missing[0]}",
                    )
                target_memberships = [by_handle[handle.lower()] for handle in requested_handles]
            if not target_memberships:
                raise HTTPException(status_code=409, detail="No other active team members")
            mention_handles = [item.mention_handle for item in target_memberships]

            attachment_rows: list[TeamAttachment] = []
            if attachment_ids:
                attachment_rows = list((await session.execute(
                    select(TeamAttachment).where(
                        cast(Any, TeamAttachment.token).in_(attachment_ids),
                        cast(Any, TeamAttachment.project_id) == project.id,
                        cast(Any, TeamAttachment.owner_human_id) == human.id,
                        cast(Any, TeamAttachment.message_id).is_(None),
                    )
                )).scalars().all())
                by_token = {item.token: item for item in attachment_rows}
                if set(by_token) != set(attachment_ids):
                    raise HTTPException(status_code=409, detail="附件不存在、已发送或不属于当前用户")
                attachment_rows = [by_token[token] for token in attachment_ids]

            channel_insert = sqlite_insert(Channel).values(
                project_id=project.id,
                name="support",
                created_ts=datetime.now(timezone.utc).replace(tzinfo=None),
            )
            await session.execute(
                channel_insert.on_conflict_do_nothing(
                    index_elements=["project_id", "name"],
                )
            )
            channel_row = await session.execute(
                select(Channel).where(
                    cast(Any, Channel.project_id) == project.id,
                    cast(Any, Channel.name) == "support",
                )
            )
            channel = channel_row.scalars().one()
            message = ChannelMessage(
                channel_id=cast(int, channel.id),
                sender_id=cast(int, sender.id),
                subject=subject.strip(),
                body_md=(
                    " ".join(f"@{handle}" for handle in mention_handles)
                    + "\n\n"
                    + body_md.strip()
                ),
                importance=importance,
                attachments=[_public_team_attachment(row) for row in attachment_rows],
            )
            sender.last_active_ts = datetime.now(timezone.utc).replace(tzinfo=None)
            session.add(sender)
            session.add(message)
            await session.flush()
            for attachment in attachment_rows:
                attachment.message_id = cast(int, message.id)
                session.add(attachment)
            await session.commit()
            await session.refresh(message)

        deliveries = await _deliver_channel_mentions(
            cast(Any, _HubHTTPDeliveryContext()),
            project,
            sender,
            message,
            mention_handles,
        )
        return JSONResponse(
            {
                "channel": "support",
                "message_id": message.id,
                "sender_agent": sender_agent_label,
                "sender_kind": sender_kind,
                "sender_human": membership.mention_handle,
                "mention_handles": mention_handles,
                "attachments": message.attachments,
                "deliveries": deliveries,
            },
            status_code=201,
        )

    # M3a human inbox (人工收件箱): read-only + mark-read over durable items
    # created by @human channel mentions without a usable default agent. Pure
    # DB access — Hub data never triggers local execution.
    @fastapi_app.get("/hub/api/inbox", response_class=JSONResponse)
    async def hub_list_inbox(
        request: Request,
        unread_only: bool = False,
        limit: int = 50,
    ) -> JSONResponse:
        await ensure_schema()
        if not 1 <= limit <= 200:
            raise HTTPException(status_code=400, detail="limit must be 1-200")
        async with get_session() as session:
            human = await _hub_human(request, session=session)
            conditions = [cast(Any, HumanInboxItem.human_id) == human.id]
            if unread_only:
                conditions.append(cast(Any, HumanInboxItem.read_ts).is_(None))
            rows = await session.execute(
                select(
                    HumanInboxItem,
                    Message,
                    Agent,
                    TeamProject,
                    Human,
                    ProjectHumanMembership,
                    SessionLeadBinding,
                )
                .join(Message, cast(Any, Message.id) == HumanInboxItem.message_id)
                .join(Agent, cast(Any, Agent.id) == Message.sender_id)
                .join(
                    TeamProject,
                    cast(Any, TeamProject.routing_project_id) == HumanInboxItem.project_id,
                )
                .outerjoin(Human, cast(Any, Human.id) == Agent.owner_id)
                .outerjoin(
                    ProjectHumanMembership,
                    and_(
                        cast(Any, ProjectHumanMembership.project_id)
                        == HumanInboxItem.project_id,
                        cast(Any, ProjectHumanMembership.human_id)
                        == Agent.owner_id,
                        cast(Any, ProjectHumanMembership.status) == "active",
                    ),
                )
                .outerjoin(
                    SessionLeadBinding,
                    cast(Any, SessionLeadBinding.agent_id) == Agent.id,
                )
                .where(*conditions)
                .order_by(cast(Any, HumanInboxItem.created_ts).desc())
                .limit(limit)
            )
            items = [
                {
                    "id": item.id,
                    "project_slug": project.slug,
                    "message_id": message.id,
                    "subject": message.subject,
                    "body_md": message.body_md,
                    "importance": message.importance,
                    "kind": item.kind,
                    # relay/受管 lead 的 sender 归 Human;内部 Agent 名(hash)永不透出,
                    # lead 只经 sender_agent 透出客户端 lead_label。
                    "sender_name": (
                        sender_human.display_name
                        if sender_human is not None
                        and sender.program in (_HUB_HUMAN_RELAY_PROGRAM, _SESSION_LEAD_PROGRAM)
                        else (
                            "Team member"
                            if sender.program in (_HUB_HUMAN_RELAY_PROGRAM, _SESSION_LEAD_PROGRAM)
                            else sender.name
                        )
                    ),
                    "sender_kind": (
                        "human"
                        if sender.program == _HUB_HUMAN_RELAY_PROGRAM
                        else ("session_lead" if sender.program == _SESSION_LEAD_PROGRAM else "agent")
                    ),
                    "sender_agent": (
                        None
                        if sender.program == _HUB_HUMAN_RELAY_PROGRAM
                        else (
                            lead_binding.lead_label or None
                            if sender.program == _SESSION_LEAD_PROGRAM
                            and lead_binding is not None
                            else (None if sender.program == _SESSION_LEAD_PROGRAM else sender.name)
                        )
                    ),
                    # Project-local handle is the only safe reply address. The
                    # display name is not unique and must never be guessed as
                    # a mention handle by Cockpit.
                    "sender_handle": (
                        sender_membership.mention_handle
                        if sender_membership is not None
                        else None
                    ),
                    "read_ts": str(item.read_ts) if item.read_ts else None,
                    "created_ts": str(item.created_ts),
                }
                for (
                    item,
                    message,
                    sender,
                    project,
                    sender_human,
                    sender_membership,
                    lead_binding,
                ) in rows.all()
            ]
            return JSONResponse({"items": items})

    @fastapi_app.post("/hub/api/inbox/mark-read", response_class=JSONResponse)
    async def hub_mark_inbox_read(request: Request) -> JSONResponse:
        await ensure_schema()
        body = await _hub_json_body(request)
        raw_ids = body.get("ids")
        if raw_ids is None:
            ids: list[int] | None = None
        elif (
            isinstance(raw_ids, list)
            and all(isinstance(value, int) and not isinstance(value, bool) for value in raw_ids)
        ):
            ids = raw_ids
        else:
            raise HTTPException(status_code=400, detail="ids must be an array of integers")
        async with get_session() as session:
            human = await _hub_human(request, session=session)
            conditions = [
                cast(Any, HumanInboxItem.human_id) == human.id,
                cast(Any, HumanInboxItem.read_ts).is_(None),
                cast(Any, HumanInboxItem.project_id).in_(
                    select(TeamProject.routing_project_id).where(
                        cast(Any, TeamProject.archived_at).is_(None)
                    )
                ),
            ]
            if ids is not None:
                conditions.append(cast(Any, HumanInboxItem.id).in_(ids))
            result = await session.execute(
                update(HumanInboxItem)
                .where(*conditions)
                .values(read_ts=datetime.now(timezone.utc).replace(tzinfo=None))
            )
            await session.commit()
            return JSONResponse({"updated": cast(Any, result).rowcount or 0})

    # M3a: explicit Agent ↔ TeamProject binding. Bind/unbind only reference an
    # existing agent id + team project slug — never a local path — and never
    # trigger Herdr/shell/worktree/task execution (pure Hub metadata).
    def _hub_is_global_admin(request: Request) -> bool:
        """Global admin = JWT whose role claim carries 'admin' (issuer convention)."""
        claims = getattr(request.state, "jwt_claims", None)
        if not isinstance(claims, dict):
            return False
        roles_raw = claims.get(settings.http.jwt_role_claim, [])
        if isinstance(roles_raw, str):
            roles = {roles_raw}
        elif isinstance(roles_raw, (list, tuple)):
            roles = {str(role) for role in roles_raw}
        else:
            roles = set()
        return "admin" in roles

    async def _hub_binding_admin(
        request: Request,
        team_project: TeamProject,
        *,
        session: AsyncSession,
    ) -> Human:
        """Only a global admin or the group's active admin may bind/unbind."""
        human = await _hub_human(request, session=session)
        if _hub_is_global_admin(request):
            return human
        project = await session.get(Project, team_project.routing_project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        await _hub_active_membership(project, human, session=session, admin=True)
        return human

    def _hub_binding_payload(binding: TeamProjectAgentBinding) -> dict[str, Any]:
        return {
            "id": binding.id,
            "team_project_id": binding.team_project_id,
            "agent_id": binding.agent_id,
            "status": binding.status,
            "bound_by_human_id": binding.bound_by_human_id,
            "created_at": str(binding.created_at),
            "updated_at": str(binding.updated_at),
        }

    @fastapi_app.post(
        "/hub/api/projects/{project_slug}/agent-bindings",
        response_class=JSONResponse,
    )
    async def hub_bind_agent(project_slug: str, request: Request) -> JSONResponse:
        await ensure_schema()
        body = await _hub_json_body(request)
        agent_id = body.get("agent_id")
        if isinstance(agent_id, bool) or not isinstance(agent_id, int):
            raise HTTPException(status_code=400, detail="agent_id must be an integer")
        async with get_session() as session:
            team_project = await _hub_team_project(project_slug, session=session)
            human = await _hub_binding_admin(request, team_project, session=session)
            agent = await session.get(Agent, agent_id)
            if agent is None:
                raise HTTPException(status_code=404, detail="Agent not found")
            if agent.retired_at is not None:
                raise HTTPException(status_code=409, detail="Cannot bind a retired agent")
            project = await session.get(Project, team_project.routing_project_id)
            if project is None or project.archived_at is not None:
                raise HTTPException(status_code=404, detail="Project not found")
            if agent.project_id != project.id and await _membership_handle_taken(
                project, agent.name, session=session
            ):
                # A bound external agent becomes addressable by name in the
                # group namespace; it must not shadow an active human handle.
                raise HTTPException(
                    status_code=409,
                    detail="Agent name collides with an active member mention_handle",
                )
            existing_row = await session.execute(
                select(TeamProjectAgentBinding).where(
                    cast(Any, TeamProjectAgentBinding.team_project_id) == team_project.id,
                    cast(Any, TeamProjectAgentBinding.agent_id) == agent_id,
                )
            )
            binding = existing_row.scalars().first()
            if binding is not None:
                if binding.status != "active":
                    # Re-binding revives the kept history row (idempotent).
                    binding.status = "active"
                    binding.bound_by_human_id = human.id
                    binding.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                    session.add(binding)
                    await session.commit()
                await session.refresh(binding)
                return JSONResponse(_hub_binding_payload(binding), status_code=200)
            # Atomic upsert: a concurrent bind of the same pair no-ops on the
            # unique constraint instead of raising, so no poisoned session /
            # IntegrityError retry path is needed. rowcount 1 = we inserted.
            insert_stmt = sqlite_insert(TeamProjectAgentBinding).values(
                team_project_id=team_project.id,
                agent_id=agent_id,
                status="active",
                bound_by_human_id=human.id,
            )
            result = await session.execute(
                insert_stmt.on_conflict_do_nothing(
                    index_elements=["team_project_id", "agent_id"]
                )
            )
            created = bool(cast(Any, result).rowcount)
            await session.commit()
            # Return the authoritative row (ours or the concurrent winner's).
            existing_row = await session.execute(
                select(TeamProjectAgentBinding).where(
                    cast(Any, TeamProjectAgentBinding.team_project_id) == team_project.id,
                    cast(Any, TeamProjectAgentBinding.agent_id) == agent_id,
                )
            )
            binding = existing_row.scalars().one()
            return JSONResponse(
                _hub_binding_payload(binding), status_code=201 if created else 200
            )

    @fastapi_app.delete(
        "/hub/api/projects/{project_slug}/agent-bindings/{agent_id}",
        response_class=JSONResponse,
    )
    async def hub_unbind_agent(
        project_slug: str,
        agent_id: int,
        request: Request,
    ) -> JSONResponse:
        await ensure_schema()
        async with get_session() as session:
            team_project = await _hub_team_project(project_slug, session=session)
            await _hub_binding_admin(request, team_project, session=session)
            existing_row = await session.execute(
                select(TeamProjectAgentBinding).where(
                    cast(Any, TeamProjectAgentBinding.team_project_id) == team_project.id,
                    cast(Any, TeamProjectAgentBinding.agent_id) == agent_id,
                )
            )
            binding = existing_row.scalars().first()
            if binding is None:
                raise HTTPException(status_code=404, detail="Agent binding not found")
            if binding.status != "unbound":
                # Unbind keeps the row as history; only status/timestamp move.
                binding.status = "unbound"
                binding.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                session.add(binding)
                await session.commit()
            await session.refresh(binding)
            return JSONResponse(_hub_binding_payload(binding))

    @fastapi_app.get(
        "/hub/api/projects/{project_slug}/agent-bindings",
        response_class=JSONResponse,
    )
    async def hub_list_agent_bindings(project_slug: str, request: Request) -> JSONResponse:
        await ensure_schema()
        async with get_session() as session:
            team_project = await _hub_team_project(project_slug, session=session)
            human = await _hub_human(request, session=session)
            project = await session.get(Project, team_project.routing_project_id)
            if project is None:
                raise HTTPException(status_code=404, detail="Project not found")
            caller = await _hub_active_membership(project, human, session=session)
            is_admin = caller.role == "admin" or _hub_is_global_admin(request)
            conditions = [
                cast(Any, TeamProjectAgentBinding.team_project_id) == team_project.id
            ]
            if not is_admin:
                # Ordinary members see active bindings only; unbind history is
                # an admin view.
                conditions.append(cast(Any, TeamProjectAgentBinding.status) == "active")
            rows = await session.execute(
                select(TeamProjectAgentBinding, Agent)
                .join(Agent, cast(Any, Agent.id) == TeamProjectAgentBinding.agent_id)
                .where(*conditions)
                .order_by(cast(Any, TeamProjectAgentBinding.created_at))
            )
            bindings = []
            for binding, agent in rows.all():
                payload = _hub_binding_payload(binding)
                payload["agent_name"] = agent.name
                bindings.append(payload)
            return JSONResponse({"bindings": bindings})

    # M3e: self-service claim of an unowned Agent by a team member who controls
    # it (proven by the agent's registration_token). The selector is what a
    # local registry can actually know: source_project_slug + agent_name
    # (never a local path, never a remote numeric id). Sets owner + creates or
    # revives the binding atomically. The token is compared in constant time
    # and never echoed to responses, exceptions, or logs; unknown selectors and
    # bad tokens get the same opaque refusal.
    _claim_locks: dict[int, asyncio.Lock] = {}
    _session_lead_locks: dict[tuple[int, int], asyncio.Lock] = {}
    _session_lead_claim_locks: dict[int, asyncio.Lock] = {}
    _session_lead_draft_decision_locks: dict[int, asyncio.Lock] = {}
    _session_lead_reply_request_locks: dict[int, asyncio.Lock] = {}
    # Exposed for race-controlled tests (pre-acquire an agent's lock).
    fastapi_app.state.hub_claim_locks = _claim_locks

    @fastapi_app.post(
        "/hub/api/projects/{project_slug}/agent-claims",
        response_class=JSONResponse,
    )
    async def hub_claim_agent(project_slug: str, request: Request) -> JSONResponse:
        await ensure_schema()
        body = await _hub_json_body(request)
        source_slug = body.get("source_project_slug")
        if not isinstance(source_slug, str) or not _SLUG_VALIDATOR_RE.fullmatch(source_slug.strip()):
            raise HTTPException(status_code=400, detail="source_project_slug must be a valid slug")
        agent_name = body.get("agent_name")
        if not isinstance(agent_name, str) or not agent_name.strip() or len(agent_name.strip()) > 128:
            raise HTTPException(status_code=400, detail="agent_name is required")
        presented = body.get("registration_token")
        if not isinstance(presented, str) or not presented.strip() or len(presented) > 255:
            raise HTTPException(status_code=400, detail="registration_token is required")
        presented_token = presented.strip()

        async with get_session() as session:
            team_project = await _hub_team_project(project_slug, session=session)
            human = await _hub_human(request, session=session)
            project = await session.get(Project, team_project.routing_project_id)
            if project is None or project.archived_at is not None:
                raise HTTPException(status_code=404, detail="Project not found")
            # Any active member may claim an agent they control; no admin power
            # is granted or required here.
            await _hub_active_membership(project, human, session=session)

            # Resolve the candidate by globally unique source slug + per-project
            # unique (case-insensitive) agent name. Unknown source slug, unknown
            # name, archived source project, and a wrong token all yield the same
            # opaque refusal so the directory cannot be probed.
            source_row = await session.execute(
                select(Project).where(
                    cast(Any, Project.slug) == source_slug.strip(),
                    cast(Any, Project.archived_at).is_(None),
                )
            )
            source_project = source_row.scalars().first()
            agent = None
            if source_project is not None:
                agent_row = await session.execute(
                    select(Agent).where(
                        cast(Any, Agent.project_id) == source_project.id,
                        # Exact match (#1013): the registry stores the precise
                        # name; a case-insensitive first() could pick the wrong
                        # case-variant twin.
                        cast(Any, Agent.name) == agent_name.strip(),
                    )
                )
                agent = agent_row.scalars().first()
            stored = agent.registration_token if agent is not None else None
            if (
                agent is None
                or not stored
                or not hmac.compare_digest(stored, presented_token)
            ):
                raise HTTPException(status_code=403, detail="Invalid agent credentials")
            agent_id = agent.id

            # Serialize the mutation critical section per agent (#996): within
            # one hub process, two concurrent claims for the same agent run
            # sequentially, so no two connections ever write the same row at
            # the same time (avoids cross-greenlet connection contention). A
            # fresh inner session starts its snapshot AFTER acquiring the
            # lock, so it always sees the previous claim's committed state.
            claim_lock = _claim_locks.setdefault(agent_id, asyncio.Lock())
            async with claim_lock, get_session() as claim_session:
                agent = await claim_session.get(Agent, agent_id)
                if agent is None:
                    raise HTTPException(status_code=403, detail="Invalid agent credentials")
                # Re-validate inside the write transaction (#1013): while this
                # request waited on the lock, the token may have rotated, the
                # source project may have been archived, or the caller's
                # membership may have been removed. Nothing may be written
                # unless all of them still hold.
                await _revalidate_claim_write_window(
                    agent, presented_token, project, human, session=claim_session
                )
                if agent.retired_at is not None:
                    raise HTTPException(status_code=409, detail="Cannot claim a retired agent")
                # Team routing agents are managed, not claimable.
                routing_row = await claim_session.execute(
                    select(TeamProject).where(
                        cast(Any, TeamProject.routing_project_id) == agent.project_id,
                        cast(Any, TeamProject.archived_at).is_(None),
                    )
                )
                if routing_row.scalars().first() is not None:
                    raise HTTPException(
                        status_code=409, detail="Team routing agents cannot be claimed"
                    )
                # Owner may only be unset or the caller; never another human's.
                if agent.owner_id is not None and agent.owner_id != human.id:
                    raise HTTPException(status_code=409, detail="Agent is already owned by another human")
                if agent.project_id != project.id and await _membership_handle_taken(
                    project, agent.name, session=claim_session
                ):
                    raise HTTPException(
                        status_code=409,
                        detail="Agent name collides with an active member mention_handle",
                    )

                if agent.owner_id is None:
                    # CAS safety net (cross-process): only claims the agent
                    # while its owner is still unset, atomically.
                    cas = await claim_session.execute(
                        update(Agent)
                        .where(
                            cast(Any, Agent.id) == agent_id,
                            cast(Any, Agent.owner_id).is_(None),
                        )
                        .values(owner_id=human.id)
                    )
                    if not cast(Any, cas).rowcount:
                        await claim_session.rollback()
                        owner_now = await claim_session.scalar(
                            select(Agent.owner_id).where(cast(Any, Agent.id) == agent_id)
                        )
                        if owner_now != human.id:
                            raise HTTPException(
                                status_code=409, detail="Agent is already owned by another human"
                            )
                    agent = await claim_session.get(Agent, agent_id)

                # Create or revive the binding (same upsert semantics as bind).
                existing_row = await claim_session.execute(
                    select(TeamProjectAgentBinding).where(
                        cast(Any, TeamProjectAgentBinding.team_project_id) == team_project.id,
                        cast(Any, TeamProjectAgentBinding.agent_id) == agent_id,
                    )
                )
                binding = existing_row.scalars().first()
                if binding is not None:
                    if binding.status != "active":
                        binding.status = "active"
                        binding.bound_by_human_id = human.id
                        binding.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                        claim_session.add(binding)
                    await claim_session.commit()
                    await claim_session.refresh(binding)
                    await claim_session.refresh(agent)
                    return JSONResponse(
                        {
                            "binding": _hub_binding_payload(binding),
                            "agent": _hub_agent_payload(agent),
                        },
                        status_code=200,
                    )
                insert_stmt = sqlite_insert(TeamProjectAgentBinding).values(
                    team_project_id=team_project.id,
                    agent_id=agent_id,
                    status="active",
                    bound_by_human_id=human.id,
                )
                result = await claim_session.execute(
                    insert_stmt.on_conflict_do_nothing(
                        index_elements=["team_project_id", "agent_id"]
                    )
                )
                created = bool(cast(Any, result).rowcount)
                await claim_session.commit()
                existing_row = await claim_session.execute(
                    select(TeamProjectAgentBinding).where(
                        cast(Any, TeamProjectAgentBinding.team_project_id) == team_project.id,
                        cast(Any, TeamProjectAgentBinding.agent_id) == agent_id,
                    )
                )
                binding = existing_row.scalars().one()
                await claim_session.refresh(agent)
                return JSONResponse(
                    {
                        "binding": _hub_binding_payload(binding),
                        "agent": _hub_agent_payload(agent),
                    },
                    status_code=201 if created else 200,
                )

    # M3 Session-Team: managed session-lead Agents. A lead Agent is created and
    # owned by the Hub inside the routing project — the client never needs an
    # Agent Mail token, and no registration_token is ever issued or returned.
    _CLIENT_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    _SESSION_LEAD_PROGRAM = "team-session-lead"

    def _new_reply_token() -> str:
        import secrets as _secrets

        return _secrets.token_urlsafe(32)

    def _hash_reply_token(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def _hub_reply_credentials_invalid() -> HTTPException:
        # 统一不透明: 不区分 未知 session / 已撤销 / 错 token
        return HTTPException(status_code=403, detail="Invalid reply credentials")

    _TEAM_PROGRESS_PHASES = {"working", "waiting", "blocked"}
    _TEAM_PROGRESS_UNSAFE_RE = re.compile(
        r"(?:https?://|`|(?:^|\s)(?:ssh|sudo|curl|wget)\s|"
        r"/(?:home|Users|root|etc|var|opt|mnt)/|"
        r"\b(?:password|passwd|secret|token|api[_ -]?key|authorization)\b|"
        r"(?:密码|密钥|令牌|凭据)|"
        r"\b[A-Za-z_][A-Za-z0-9_]{2,}=\S+|"
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b|"
        r"[A-Fa-f0-9]{32,}|[A-Za-z0-9_-]{48,})",
        re.IGNORECASE,
    )

    def _safe_team_progress_summary(value: Any) -> str | None:
        """Keep only a short user-facing sentence; fail closed on secrets/paths."""
        if value is None:
            return None
        if not isinstance(value, str):
            raise HTTPException(status_code=400, detail="Invalid progress summary")
        summary = re.sub(r"\s+", " ", value).strip()
        if not summary:
            return None
        if (
            len(summary) > 160
            or any(ord(char) < 32 for char in summary)
            or _TEAM_PROGRESS_UNSAFE_RE.search(summary)
        ):
            return None
        return summary

    def _clear_team_progress(binding: SessionLeadBinding) -> None:
        binding.progress_message_id = None
        binding.progress_phase = None
        binding.progress_summary = None
        binding.progress_sequence = 0
        binding.progress_started_at = None
        binding.progress_seen_at = None

    def _session_lead_credentials(body: dict[str, Any]) -> tuple[str, str]:
        raw_session_id = body.get("client_session_id")
        client_session_id = (
            raw_session_id.strip() if isinstance(raw_session_id, str) else ""
        )
        if not _CLIENT_SESSION_ID_RE.fullmatch(client_session_id):
            raise HTTPException(
                status_code=400, detail="client_session_id must be a valid id"
            )
        raw_token = body.get("reply_token")
        reply_token = raw_token.strip() if isinstance(raw_token, str) else ""
        if not reply_token or len(reply_token) > 128:
            raise HTTPException(status_code=400, detail="reply_token is required")
        return client_session_id, reply_token

    async def _session_lead_capability_context(
        project_slug: str,
        client_session_id: str,
        reply_token: str,
        *,
        session: AsyncSession,
    ) -> tuple[TeamProject, Project, SessionLeadBinding, Agent]:
        team_project = await _hub_team_project(project_slug, session=session)
        project = await session.get(Project, team_project.routing_project_id)
        if project is None or project.archived_at is not None:
            raise HTTPException(status_code=404, detail="Project not found")
        binding_row = await session.execute(
            select(SessionLeadBinding).where(
                cast(Any, SessionLeadBinding.team_project_id) == team_project.id,
                cast(Any, SessionLeadBinding.client_session_id) == client_session_id,
                cast(Any, SessionLeadBinding.status) == "active",
            )
        )
        binding = binding_row.scalars().first()
        if binding is None or not binding.reply_token_hash:
            raise _hub_reply_credentials_invalid()
        if not hmac.compare_digest(
            binding.reply_token_hash, _hash_reply_token(reply_token)
        ):
            raise _hub_reply_credentials_invalid()
        membership_row = await session.execute(
            select(ProjectHumanMembership).where(
                cast(Any, ProjectHumanMembership.project_id) == project.id,
                cast(Any, ProjectHumanMembership.human_id) == binding.human_id,
                cast(Any, ProjectHumanMembership.status) == "active",
                cast(Any, ProjectHumanMembership.default_agent_id) == binding.agent_id,
            )
        )
        if membership_row.scalars().first() is None:
            raise _hub_reply_credentials_invalid()
        sender = await session.get(Agent, binding.agent_id)
        if sender is None:
            raise HTTPException(
                status_code=409, detail="Managed lead agent is unavailable"
            )
        if sender.retired_at is not None:
            sender.retired_at = None
            sender.last_active_ts = datetime.now(timezone.utc).replace(tzinfo=None)
            session.add(sender)
        return team_project, project, binding, sender

    @fastapi_app.post(
        "/hub/api/projects/{project_slug}/session-lead/status",
        response_class=JSONResponse,
    )
    async def hub_session_lead_status(project_slug: str, request: Request) -> JSONResponse:
        """Record a binding-scoped runtime heartbeat for the safe team roster."""
        await ensure_schema()
        body = await _hub_json_body(request)
        required = {"client_session_id", "reply_token", "status"}
        if not required.issubset(body) or not set(body).issubset(required | {"progress"}):
            raise HTTPException(
                status_code=400,
                detail="client_session_id, reply_token and status are required; progress is optional",
            )
        client_session_id, reply_token = _session_lead_credentials(body)
        runtime_status = body.get("status")
        if runtime_status not in {"working", "idle", "blocked"}:
            raise HTTPException(status_code=400, detail="Invalid runtime status")
        async with get_session() as session:
            _team_project, _project, binding, sender = (
                await _session_lead_capability_context(
                    project_slug,
                    client_session_id,
                    reply_token,
                    session=session,
                )
            )
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            binding.runtime_status = cast(str, runtime_status)
            binding.runtime_seen_at = now
            binding.updated_at = now
            if "progress" in body:
                progress = body.get("progress")
                if progress is None:
                    _clear_team_progress(binding)
                else:
                    if not isinstance(progress, dict) or set(progress) != {
                        "inbox_item_id", "phase", "summary", "sequence", "started_at"
                    }:
                        raise HTTPException(status_code=400, detail="Invalid progress payload")
                    inbox_item_id = progress.get("inbox_item_id")
                    sequence = progress.get("sequence")
                    phase = progress.get("phase")
                    started_at = progress.get("started_at")
                    if (
                        isinstance(inbox_item_id, bool)
                        or not isinstance(inbox_item_id, int)
                        or inbox_item_id < 1
                        or isinstance(sequence, bool)
                        or not isinstance(sequence, int)
                        or not 1 <= sequence <= 2_147_483_647
                        or phase not in _TEAM_PROGRESS_PHASES
                        or isinstance(started_at, bool)
                        or not isinstance(started_at, (int, float))
                        or not now.replace(tzinfo=timezone.utc).timestamp() - 86_400
                        <= float(started_at)
                        <= now.replace(tzinfo=timezone.utc).timestamp() + 5
                    ):
                        raise HTTPException(status_code=400, detail="Invalid progress payload")
                    item = await session.get(HumanInboxItem, inbox_item_id)
                    if (
                        item is None
                        or item.project_id != _project.id
                        or item.human_id != binding.human_id
                        or item.kind != "session_lead"
                        or item.claim_binding_id != binding.id
                        or item.completed_at is not None
                        or item.claim_expires_at is None
                        or item.claim_expires_at <= now
                        or item.source_channel_message_id is None
                    ):
                        raise HTTPException(status_code=409, detail="Progress claim is unavailable")
                    if (
                        binding.progress_message_id == item.source_channel_message_id
                        and sequence < binding.progress_sequence
                    ):
                        raise HTTPException(status_code=409, detail="Progress sequence is stale")
                    binding.progress_message_id = item.source_channel_message_id
                    binding.progress_phase = cast(str, phase)
                    binding.progress_summary = _safe_team_progress_summary(
                        progress.get("summary")
                    )
                    binding.progress_sequence = sequence
                    binding.progress_started_at = datetime.fromtimestamp(
                        float(started_at), tz=timezone.utc
                    ).replace(tzinfo=None)
                    binding.progress_seen_at = now
            sender.last_active_ts = now
            session.add(binding)
            session.add(sender)
            await session.commit()
            return JSONResponse({
                "status": binding.runtime_status,
                "last_seen_at": _hub_presence_timestamp(binding.runtime_seen_at),
            })

    @fastapi_app.get(
        "/hub/api/projects/{project_slug}/progress",
        response_class=JSONResponse,
    )
    async def hub_team_progress(project_slug: str, request: Request) -> JSONResponse:
        """Return fresh, non-durable progress overlays to active team members."""
        await ensure_schema()
        async with get_session() as session:
            team_project = await _hub_team_project(project_slug, session=session)
            human = await _hub_human(request, session=session)
            project = await session.get(Project, team_project.routing_project_id)
            if project is None or project.archived_at is not None:
                raise HTTPException(status_code=404, detail="Project not found")
            await _hub_active_membership(project, human, session=session)
            cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
                seconds=_SESSION_LEAD_RUNTIME_TTL_SECONDS
            )
            rows = (
                await session.execute(
                    select(SessionLeadBinding).where(
                        cast(Any, SessionLeadBinding.team_project_id) == team_project.id,
                        cast(Any, SessionLeadBinding.status) == "active",
                        cast(Any, SessionLeadBinding.progress_message_id).is_not(None),
                        cast(Any, SessionLeadBinding.progress_seen_at) >= cutoff,
                    )
                )
            ).scalars().all()
            progress_rows: list[dict[str, Any]] = []
            for binding in rows:
                membership = await _hub_membership(
                    project.id, binding.human_id, session=session
                )
                if membership is None or membership.status != "active":
                    continue
                progress_rows.append({
                    "message_id": binding.progress_message_id,
                    "agent_name": binding.lead_label or None,
                    "phase": binding.progress_phase,
                    "summary": binding.progress_summary,
                    "sequence": binding.progress_sequence,
                    "started_at": _hub_presence_timestamp(binding.progress_started_at),
                    "updated_at": _hub_presence_timestamp(binding.progress_seen_at),
                })
            return JSONResponse({"progress": progress_rows})

    @fastapi_app.post(
        "/hub/api/projects/{project_slug}/session-lead/reply",
        response_class=JSONResponse,
    )
    async def hub_session_lead_reply(project_slug: str, request: Request) -> JSONResponse:
        """Capability-auth reply as the managed lead (no Human JWT required).

        Auth = the binding-scoped reply token (hash stored only). The managed
        lead posts to the support channel; Human-via-lead attribution and the
        Human Inbox fallback stay intact. The idempotency key occupies its
        slot in the SAME transaction as the message, and a replay re-runs the
        (idempotent) mention delivery with the ORIGINAL handles, so a crashed
        first attempt never silently loses the notification (#1101).
        """
        await ensure_schema()
        body = await _hub_json_body(request)
        client_session_id, reply_token = _session_lead_credentials(body)
        subject = body.get("subject")
        body_md = body.get("body_md")
        importance = body.get("importance", "normal")
        if not isinstance(subject, str) or not subject.strip() or len(subject.strip()) > 512:
            raise HTTPException(status_code=400, detail="subject is required and must be at most 512 characters")
        if not isinstance(body_md, str) or not body_md.strip() or len(body_md) > 50_000:
            raise HTTPException(status_code=400, detail="body_md is required and must be at most 50000 characters")
        if importance not in {"low", "normal", "high", "urgent"}:
            raise HTTPException(status_code=400, detail="Invalid importance")
        raw_handles = body.get("mention_handles")
        mention_handles: list[str] = []
        if raw_handles is not None:
            if not isinstance(raw_handles, list) or len(raw_handles) > 50:
                raise HTTPException(status_code=400, detail="mention_handles must be an array")
            for raw_handle in raw_handles:
                if not isinstance(raw_handle, str):
                    raise HTTPException(status_code=400, detail="mention_handles must contain strings")
                handle = _hub_mention_handle(raw_handle)
                if handle.lower() not in {item.lower() for item in mention_handles}:
                    mention_handles.append(handle)
        raw_key = body.get("idempotency_key")
        idem_key = raw_key.strip() if isinstance(raw_key, str) else ""
        if raw_key is not None and (not idem_key or len(idem_key) > 128):
            raise HTTPException(status_code=400, detail="idempotency_key must be 1-128 characters")
        raw_inbox_item_id = body.get("inbox_item_id")
        if raw_inbox_item_id is not None and (
            isinstance(raw_inbox_item_id, bool)
            or not isinstance(raw_inbox_item_id, int)
        ):
            raise HTTPException(status_code=400, detail="inbox_item_id must be an integer")
        raw_claim_token = body.get("claim_token")
        claim_token = (
            raw_claim_token.strip() if isinstance(raw_claim_token, str) else ""
        )
        if raw_inbox_item_id is not None and (
            not claim_token or len(claim_token) > 128
        ):
            raise HTTPException(status_code=400, detail="claim_token is required")

        replay_message: ChannelMessage | None = None
        replay_handles: list[str] = []
        replay_sender: Agent | None = None

        async with get_session() as session:
            _, project, binding, sender = await _session_lead_capability_context(
                project_slug,
                client_session_id,
                reply_token,
                session=session,
            )
            if raw_inbox_item_id is None:
                if binding.reply_mode != "auto":
                    raise HTTPException(
                        status_code=409, detail="Human confirmation is required"
                    )
            else:
                item = await session.get(HumanInboxItem, raw_inbox_item_id)
                now = datetime.now(timezone.utc).replace(tzinfo=None)
                if (
                    item is None
                    or item.project_id != project.id
                    or item.human_id != binding.human_id
                    or item.kind != "session_lead"
                    or item.claim_binding_id != binding.id
                    or not item.claim_token_hash
                    or not hmac.compare_digest(
                        item.claim_token_hash, _hash_reply_token(claim_token)
                    )
                ):
                    raise HTTPException(
                        status_code=403, detail="Invalid claim credentials"
                    )
                if item.completed_at is not None:
                    raise HTTPException(status_code=409, detail="Inbox item is completed")
                if item.claim_expires_at is None or item.claim_expires_at <= now:
                    raise HTTPException(status_code=409, detail="Inbox claim has expired")
                if item.reply_decision not in {"approved", "auto"}:
                    raise HTTPException(
                        status_code=409, detail="Human confirmation is required"
                    )

            if mention_handles:
                member_rows = await session.execute(
                    select(ProjectHumanMembership).where(
                        cast(Any, ProjectHumanMembership.project_id) == project.id,
                        cast(Any, ProjectHumanMembership.status) == "active",
                    )
                )
                known = {
                    item.mention_handle.lower()
                    for item in member_rows.scalars().all()
                }
                missing = [h for h in mention_handles if h.lower() not in known]
                if missing:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Active team member not found: {missing[0]}",
                    )

            key_row = None
            fresh_insert = True
            if idem_key:
                key_insert = sqlite_insert(SessionLeadReplyKey).values(
                    binding_id=binding.id,
                    idem_key=idem_key,
                    mention_handles=json.dumps(mention_handles),
                )
                result = await session.execute(
                    key_insert.on_conflict_do_nothing(
                        index_elements=["binding_id", "idem_key"]
                    )
                )
                fresh_insert = bool(cast(Any, result).rowcount)
                if not fresh_insert:
                    # 重放: 加载原消息与原始 handles, 恢复幂等投递(#1101)
                    key_row = (
                        await session.execute(
                            select(SessionLeadReplyKey).where(
                                cast(Any, SessionLeadReplyKey.binding_id) == binding.id,
                                cast(Any, SessionLeadReplyKey.idem_key) == idem_key,
                            )
                        )
                    ).scalars().one()
                    replay_message = await session.get(ChannelMessage, key_row.message_id)
                    if replay_message is None:
                        raise HTTPException(status_code=409, detail="Reply replay state is inconsistent")
                    replay_handles = json.loads(key_row.mention_handles or "[]")
                    replay_sender = sender
                    await session.refresh(replay_message)
                    await session.refresh(replay_sender)
                    await session.refresh(project)
                else:
                    key_row = (
                        await session.execute(
                            select(SessionLeadReplyKey).where(
                                cast(Any, SessionLeadReplyKey.binding_id) == binding.id,
                                cast(Any, SessionLeadReplyKey.idem_key) == idem_key,
                            )
                        )
                    ).scalars().first()

            if replay_message is None:
                channel_insert = sqlite_insert(Channel).values(
                    project_id=project.id,
                    name="support",
                    created_ts=datetime.now(timezone.utc).replace(tzinfo=None),
                )
                await session.execute(
                    channel_insert.on_conflict_do_nothing(index_elements=["project_id", "name"])
                )
                channel_row = await session.execute(
                    select(Channel).where(
                        cast(Any, Channel.project_id) == project.id,
                        cast(Any, Channel.name) == "support",
                    )
                )
                channel = channel_row.scalars().one()
                message = ChannelMessage(
                    channel_id=cast(int, channel.id),
                    sender_id=cast(int, sender.id),
                    subject=subject.strip(),
                    body_md=(
                        (" ".join(f"@{handle}" for handle in mention_handles) + "\n\n")
                        if mention_handles
                        else ""
                    )
                    + body_md.strip(),
                    importance=importance,
                    attachments=[],
                )
                sender.last_active_ts = datetime.now(timezone.utc).replace(tzinfo=None)
                session.add(sender)
                session.add(message)
                await session.flush()
                assert message.id is not None
                if key_row is not None:
                    key_row.message_id = message.id
                    session.add(key_row)
                await session.commit()
                await session.refresh(message)
                await session.refresh(sender)
                await session.refresh(project)
                replay_sender = sender
                replay_message = message
                replay_handles = mention_handles

        assert replay_sender is not None
        assert replay_message is not None
        deliveries = await _deliver_channel_mentions(
            cast(Any, _HubHTTPDeliveryContext()),
            project,
            cast(Agent, replay_sender),
            cast(ChannelMessage, replay_message),
            replay_handles,
            # A managed lead response is the terminal answer to one authorized
            # Human message. Keep it in the Team timeline and normal mention
            # delivery, but never turn it into another managed-lead work item:
            # two auto bindings would otherwise reply to each other forever.
            create_session_lead_inbox=False,
        )
        if fresh_insert or not idem_key:
            return JSONResponse(
                {
                    "status": "delivered",
                    "message_id": replay_message.id,
                    "client_session_id": client_session_id,
                    "deliveries": deliveries,
                },
                status_code=201,
            )
        return JSONResponse(
            {
                "status": "already_delivered",
                "message_id": replay_message.id,
                "client_session_id": client_session_id,
                "deliveries": deliveries,
            }
        )

    def _reply_draft_payload(draft: SessionLeadReplyDraft) -> dict[str, Any]:
        return {
            "id": draft.id,
            "binding_id": draft.binding_id,
            "inbox_item_id": draft.inbox_item_id,
            "subject": draft.subject,
            "body_md": draft.body_md,
            "importance": draft.importance,
            "mention_handles": json.loads(draft.mention_handles or "[]"),
            "status": draft.status,
            "message_id": draft.sent_message_id,
            "created_at": str(draft.created_at),
            "updated_at": str(draft.updated_at),
            "decided_at": str(draft.decided_at) if draft.decided_at else None,
        }

    def _reply_request_payload(
        item: HumanInboxItem, *, reply_mode: str = "confirm"
    ) -> dict[str, Any]:
        if item.reply_decision == "ignored":
            request_status = "ignored"
        elif item.completed_at is not None:
            request_status = "replied"
        elif item.reply_decision in {"approved", "auto"} or reply_mode == "auto":
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            request_status = (
                "processing"
                if item.claim_expires_at is not None
                and item.claim_expires_at > now
                and item.claim_token_hash is not None
                else "queued"
            )
        else:
            request_status = "awaiting_confirmation"
        return {
            "inbox_item_id": item.id,
            "message_id": item.source_channel_message_id,
            "status": request_status,
            "decision": (
                item.reply_decision
                if item.reply_decision is not None
                else ("auto" if reply_mode == "auto" else None)
            ),
            "decided_at": (
                str(item.reply_decided_at) if item.reply_decided_at else None
            ),
        }

    async def _human_reply_request_context(
        project_slug: str,
        inbox_item_id: int,
        request: Request,
        *,
        session: AsyncSession,
    ) -> tuple[Project, SessionLeadBinding, HumanInboxItem]:
        team_project = await _hub_team_project(project_slug, session=session)
        human = await _hub_human(request, session=session)
        project = await session.get(Project, team_project.routing_project_id)
        if project is None or project.archived_at is not None:
            raise HTTPException(status_code=404, detail="Project not found")
        membership = await _hub_active_membership(project, human, session=session)
        binding = (
            await session.execute(
                select(SessionLeadBinding).where(
                    cast(Any, SessionLeadBinding.team_project_id) == team_project.id,
                    cast(Any, SessionLeadBinding.human_id) == human.id,
                    cast(Any, SessionLeadBinding.status) == "active",
                )
            )
        ).scalars().first()
        if binding is None or membership.default_agent_id != binding.agent_id:
            raise HTTPException(status_code=409, detail="Session lead is unavailable")
        item = await session.get(HumanInboxItem, inbox_item_id)
        if (
            item is None
            or item.project_id != project.id
            or item.human_id != human.id
            or item.kind != "session_lead"
        ):
            raise HTTPException(status_code=404, detail="Reply request not found")
        return project, binding, item

    @fastapi_app.get(
        "/hub/api/projects/{project_slug}/reply-requests",
        response_class=JSONResponse,
    )
    async def hub_list_reply_requests(
        project_slug: str, request: Request
    ) -> JSONResponse:
        """List message-level reply decisions without exposing message bodies."""
        await ensure_schema()
        async with get_session() as session:
            team_project = await _hub_team_project(project_slug, session=session)
            human = await _hub_human(request, session=session)
            project = await session.get(Project, team_project.routing_project_id)
            if project is None or project.archived_at is not None:
                raise HTTPException(status_code=404, detail="Project not found")
            membership = await _hub_active_membership(project, human, session=session)
            binding = (
                await session.execute(
                    select(SessionLeadBinding).where(
                        cast(Any, SessionLeadBinding.team_project_id)
                        == team_project.id,
                        cast(Any, SessionLeadBinding.human_id) == human.id,
                        cast(Any, SessionLeadBinding.status) == "active",
                    )
                )
            ).scalars().first()
            if binding is None or membership.default_agent_id != binding.agent_id:
                raise HTTPException(
                    status_code=409, detail="Session lead is unavailable"
                )
            rows = await session.execute(
                select(HumanInboxItem)
                .where(
                    cast(Any, HumanInboxItem.project_id) == project.id,
                    cast(Any, HumanInboxItem.human_id) == human.id,
                    cast(Any, HumanInboxItem.kind) == "session_lead",
                )
                .order_by(cast(Any, HumanInboxItem.created_ts).desc())
                .limit(500)
            )
            return JSONResponse(
                {
                    "requests": [
                        _reply_request_payload(
                            item, reply_mode=binding.reply_mode
                        )
                        for item in rows.scalars().all()
                    ]
                }
            )

    @fastapi_app.post(
        "/hub/api/projects/{project_slug}/reply-requests/{inbox_item_id}/approve",
        response_class=JSONResponse,
    )
    async def hub_approve_reply_request(
        project_slug: str, inbox_item_id: int, request: Request
    ) -> JSONResponse:
        await ensure_schema()
        item_lock = _session_lead_reply_request_locks.setdefault(
            inbox_item_id, asyncio.Lock()
        )
        async with item_lock, get_session() as session:
            _, binding, item = await _human_reply_request_context(
                project_slug, inbox_item_id, request, session=session
            )
            if item.reply_decision == "ignored":
                raise HTTPException(status_code=409, detail="Reply request was ignored")
            if item.completed_at is not None:
                raise HTTPException(status_code=409, detail="Reply request is completed")
            replay = item.reply_decision in {"approved", "auto"}
            if not replay:
                now = datetime.now(timezone.utc).replace(tzinfo=None)
                item.reply_decision = (
                    "auto" if binding.reply_mode == "auto" else "approved"
                )
                item.reply_decided_at = now
                item.claim_binding_id = None
                item.claim_token_hash = None
                item.claim_expires_at = None
                session.add(item)
                await session.commit()
                await session.refresh(item)
            return JSONResponse(
                {
                    "status": "already_approved" if replay else "approved",
                    "request": _reply_request_payload(
                        item, reply_mode=binding.reply_mode
                    ),
                },
                status_code=200 if replay else 201,
            )

    @fastapi_app.post(
        "/hub/api/projects/{project_slug}/reply-requests/{inbox_item_id}/reject",
        response_class=JSONResponse,
    )
    async def hub_reject_reply_request(
        project_slug: str, inbox_item_id: int, request: Request
    ) -> JSONResponse:
        await ensure_schema()
        item_lock = _session_lead_reply_request_locks.setdefault(
            inbox_item_id, asyncio.Lock()
        )
        async with item_lock, get_session() as session:
            _, binding, item = await _human_reply_request_context(
                project_slug, inbox_item_id, request, session=session
            )
            if item.reply_decision == "ignored":
                return JSONResponse(
                    {
                        "status": "already_ignored",
                        "request": _reply_request_payload(
                            item, reply_mode=binding.reply_mode
                        ),
                    }
                )
            if item.completed_at is not None:
                raise HTTPException(status_code=409, detail="Reply request is completed")
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            if (
                item.claim_token_hash is not None
                and item.claim_expires_at is not None
                and item.claim_expires_at > now
            ):
                raise HTTPException(status_code=409, detail="Reply is already processing")
            if item.reply_decision in {"approved", "auto"}:
                raise HTTPException(status_code=409, detail="Reply request was approved")
            item.reply_decision = "ignored"
            item.reply_decided_at = now
            item.completed_at = now
            item.claim_binding_id = None
            item.claim_token_hash = None
            item.claim_expires_at = None
            session.add(item)
            await session.commit()
            await session.refresh(item)
            return JSONResponse(
                {
                    "status": "ignored",
                    "request": _reply_request_payload(
                        item, reply_mode=binding.reply_mode
                    ),
                },
                status_code=201,
            )

    async def _validate_reply_handles(
        project: Project,
        mention_handles: list[str],
        *,
        session: AsyncSession,
    ) -> None:
        if not mention_handles:
            return
        rows = await session.execute(
            select(cast(Any, ProjectHumanMembership.mention_handle)).where(
                cast(Any, ProjectHumanMembership.project_id) == project.id,
                cast(Any, ProjectHumanMembership.status) == "active",
            )
        )
        known = {str(value).lower() for value in rows.scalars().all()}
        missing = [handle for handle in mention_handles if handle.lower() not in known]
        if missing:
            raise HTTPException(
                status_code=404,
                detail=f"Active team member not found: {missing[0]}",
            )

    async def _create_session_lead_channel_message(
        project: Project,
        sender: Agent,
        *,
        subject: str,
        body_md: str,
        importance: str,
        mention_handles: list[str],
        session: AsyncSession,
    ) -> ChannelMessage:
        channel_insert = sqlite_insert(Channel).values(
            project_id=project.id,
            name="support",
            created_ts=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        await session.execute(
            channel_insert.on_conflict_do_nothing(
                index_elements=["project_id", "name"]
            )
        )
        channel = (
            await session.execute(
                select(Channel).where(
                    cast(Any, Channel.project_id) == project.id,
                    cast(Any, Channel.name) == "support",
                )
            )
        ).scalars().one()
        message = ChannelMessage(
            channel_id=cast(int, channel.id),
            sender_id=cast(int, sender.id),
            subject=subject,
            body_md=(
                (
                    " ".join(f"@{handle}" for handle in mention_handles)
                    + "\n\n"
                )
                if mention_handles
                else ""
            )
            + body_md,
            importance=importance,
            attachments=[],
        )
        sender.last_active_ts = datetime.now(timezone.utc).replace(tzinfo=None)
        session.add(sender)
        session.add(message)
        await session.flush()
        return message

    @fastapi_app.post(
        "/hub/api/projects/{project_slug}/session-lead/inbox/claim",
        response_class=JSONResponse,
    )
    async def hub_session_lead_claim_inbox(
        project_slug: str, request: Request
    ) -> JSONResponse:
        """Claim one pending message for the current active bound lead."""
        await ensure_schema()
        body = await _hub_json_body(request)
        client_session_id, reply_token = _session_lead_credentials(body)
        async with get_session() as session:
            _, project, binding, _ = await _session_lead_capability_context(
                project_slug,
                client_session_id,
                reply_token,
                session=session,
            )
            assert binding.id is not None
            claim_lock = _session_lead_claim_locks.setdefault(
                binding.id, asyncio.Lock()
            )
            async with claim_lock:
                now = datetime.now(timezone.utc).replace(tzinfo=None)
                reply_condition = (
                    cast(Any, HumanInboxItem.reply_decision).in_(
                        ["approved", "auto"]
                    )
                    if binding.reply_mode == "confirm"
                    else or_(
                        cast(Any, HumanInboxItem.reply_decision).is_(None),
                        cast(Any, HumanInboxItem.reply_decision).in_(
                            ["approved", "auto"]
                        ),
                    )
                )
                item = (
                    await session.execute(
                        select(HumanInboxItem)
                        .where(
                            cast(Any, HumanInboxItem.project_id) == project.id,
                            cast(Any, HumanInboxItem.human_id) == binding.human_id,
                            cast(Any, HumanInboxItem.kind) == "session_lead",
                            cast(Any, HumanInboxItem.completed_at).is_(None),
                            reply_condition,
                            or_(
                                cast(Any, HumanInboxItem.claim_expires_at).is_(None),
                                cast(Any, HumanInboxItem.claim_expires_at) <= now,
                            ),
                        )
                        .order_by(cast(Any, HumanInboxItem.created_ts))
                        .limit(1)
                    )
                ).scalars().first()
                if item is None:
                    return JSONResponse({"status": "empty", "message": None})
                if binding.reply_mode == "auto" and item.reply_decision is None:
                    item.reply_decision = "auto"
                    item.reply_decided_at = now
                claim_token = _new_reply_token()
                item.claim_binding_id = binding.id
                item.claim_token_hash = _hash_reply_token(claim_token)
                item.claim_expires_at = now + timedelta(minutes=15)
                if item.read_ts is None:
                    item.read_ts = now
                session.add(item)
                message = await session.get(Message, item.message_id)
                if message is None:
                    raise HTTPException(
                        status_code=409, detail="Inbox message is unavailable"
                    )
                sender = await session.get(Agent, message.sender_id)
                if sender is None:
                    raise HTTPException(
                        status_code=409, detail="Inbox sender is unavailable"
                    )
                sender_human = (
                    await session.get(Human, sender.owner_id)
                    if sender.owner_id is not None
                    else None
                )
                sender_membership = None
                if sender.owner_id is not None:
                    sender_membership = (
                        await session.execute(
                            select(ProjectHumanMembership).where(
                                cast(Any, ProjectHumanMembership.project_id)
                                == project.id,
                                cast(Any, ProjectHumanMembership.human_id)
                                == sender.owner_id,
                                cast(Any, ProjectHumanMembership.status) == "active",
                            )
                        )
                    ).scalars().first()
                await session.commit()
                return JSONResponse(
                    {
                        "status": "claimed",
                        "claim_token": claim_token,
                        "claim_expires_at": str(item.claim_expires_at),
                        "reply_mode": binding.reply_mode,
                        "message": {
                            "inbox_item_id": item.id,
                            "message_id": message.id,
                            "subject": message.subject,
                            "body_md": message.body_md,
                            "importance": message.importance,
                            "attachments": [
                                item for item in message.attachments
                                if isinstance(item, dict)
                            ],
                            "sender_name": (
                                sender_human.display_name
                                if sender_human is not None
                                else "Team member"
                            ),
                            "sender_handle": (
                                sender_membership.mention_handle
                                if sender_membership is not None
                                else None
                            ),
                            "created_ts": str(message.created_ts),
                        },
                    },
                    status_code=201,
                )

    @fastapi_app.post(
        "/hub/api/projects/{project_slug}/session-lead/inbox/{inbox_item_id}/complete",
        response_class=JSONResponse,
    )
    async def hub_session_lead_complete_inbox(
        project_slug: str, inbox_item_id: int, request: Request
    ) -> JSONResponse:
        await ensure_schema()
        body = await _hub_json_body(request)
        client_session_id, reply_token = _session_lead_credentials(body)
        raw_claim_token = body.get("claim_token")
        claim_token = (
            raw_claim_token.strip() if isinstance(raw_claim_token, str) else ""
        )
        if not claim_token or len(claim_token) > 128:
            raise HTTPException(status_code=400, detail="claim_token is required")
        async with get_session() as session:
            _, project, binding, _ = await _session_lead_capability_context(
                project_slug,
                client_session_id,
                reply_token,
                session=session,
            )
            item = await session.get(HumanInboxItem, inbox_item_id)
            if (
                item is None
                or item.project_id != project.id
                or item.human_id != binding.human_id
                or item.claim_binding_id != binding.id
                or not item.claim_token_hash
                or not hmac.compare_digest(
                    item.claim_token_hash, _hash_reply_token(claim_token)
                )
            ):
                raise HTTPException(status_code=403, detail="Invalid claim credentials")
            if item.completed_at is not None:
                return JSONResponse(
                    {"status": "already_completed", "inbox_item_id": item.id}
                )
            if item.reply_decision not in {"approved", "auto"}:
                raise HTTPException(
                    status_code=409, detail="Reply was not authorized"
                )
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            if item.claim_expires_at is None or item.claim_expires_at <= now:
                raise HTTPException(status_code=409, detail="Inbox claim has expired")
            item.completed_at = now
            if binding.progress_message_id == item.source_channel_message_id:
                _clear_team_progress(binding)
            session.add(item)
            session.add(binding)
            await session.commit()
            return JSONResponse(
                {"status": "completed", "inbox_item_id": item.id}
            )

    @fastapi_app.post(
        "/hub/api/projects/{project_slug}/session-lead/reply-drafts",
        response_class=JSONResponse,
    )
    async def hub_session_lead_create_reply_draft(
        project_slug: str, request: Request
    ) -> JSONResponse:
        """Reject the retired generate-before-approval protocol."""
        await ensure_schema()
        body = await _hub_json_body(request)
        client_session_id, reply_token = _session_lead_credentials(body)
        async with get_session() as session:
            await _session_lead_capability_context(
                project_slug,
                client_session_id,
                reply_token,
                session=session,
            )
        raise HTTPException(
            status_code=410,
            detail="Reply drafts are retired; authorize the message before generation",
        )
    @fastapi_app.get(
        "/hub/api/projects/{project_slug}/reply-drafts",
        response_class=JSONResponse,
    )
    async def hub_list_reply_drafts(
        project_slug: str, request: Request, status_filter: str = "pending"
    ) -> JSONResponse:
        await ensure_schema()
        if status_filter not in {"pending", "approved", "rejected", "all"}:
            raise HTTPException(status_code=400, detail="Invalid status_filter")
        async with get_session() as session:
            team_project = await _hub_team_project(project_slug, session=session)
            human = await _hub_human(request, session=session)
            project = await session.get(Project, team_project.routing_project_id)
            if project is None or project.archived_at is not None:
                raise HTTPException(status_code=404, detail="Project not found")
            await _hub_active_membership(project, human, session=session)
            conditions = [
                cast(Any, SessionLeadBinding.team_project_id) == team_project.id,
                cast(Any, SessionLeadBinding.human_id) == human.id,
            ]
            if status_filter != "all":
                conditions.append(
                    cast(Any, SessionLeadReplyDraft.status) == status_filter
                )
            rows = await session.execute(
                select(SessionLeadReplyDraft)
                .join(
                    SessionLeadBinding,
                    cast(Any, SessionLeadBinding.id)
                    == SessionLeadReplyDraft.binding_id,
                )
                .where(*conditions)
                .order_by(cast(Any, SessionLeadReplyDraft.created_at).desc())
            )
            return JSONResponse(
                {"drafts": [_reply_draft_payload(row) for row in rows.scalars().all()]}
            )

    async def _human_reply_draft_context(
        project_slug: str,
        draft_id: int,
        request: Request,
        *,
        session: AsyncSession,
    ) -> tuple[Project, SessionLeadBinding, SessionLeadReplyDraft, Agent]:
        team_project = await _hub_team_project(project_slug, session=session)
        human = await _hub_human(request, session=session)
        project = await session.get(Project, team_project.routing_project_id)
        if project is None or project.archived_at is not None:
            raise HTTPException(status_code=404, detail="Project not found")
        membership = await _hub_active_membership(project, human, session=session)
        row = await session.execute(
            select(SessionLeadReplyDraft, SessionLeadBinding)
            .join(
                SessionLeadBinding,
                cast(Any, SessionLeadBinding.id) == SessionLeadReplyDraft.binding_id,
            )
            .where(
                cast(Any, SessionLeadReplyDraft.id) == draft_id,
                cast(Any, SessionLeadBinding.team_project_id) == team_project.id,
                cast(Any, SessionLeadBinding.human_id) == human.id,
            )
        )
        result = row.first()
        if result is None:
            raise HTTPException(status_code=404, detail="Reply draft not found")
        draft, binding = result
        if (
            binding.status != "active"
            or membership.default_agent_id != binding.agent_id
        ):
            raise HTTPException(status_code=409, detail="Session lead is unavailable")
        sender = await session.get(Agent, binding.agent_id)
        if sender is None or sender.retired_at is not None:
            raise HTTPException(status_code=409, detail="Managed lead agent is unavailable")
        return project, binding, draft, sender

    @fastapi_app.post(
        "/hub/api/projects/{project_slug}/reply-drafts/{draft_id}/approve",
        response_class=JSONResponse,
    )
    async def hub_approve_reply_draft(
        project_slug: str, draft_id: int, request: Request
    ) -> JSONResponse:
        await ensure_schema()
        draft_lock = _session_lead_draft_decision_locks.setdefault(
            draft_id, asyncio.Lock()
        )
        async with draft_lock:
            async with get_session() as session:
                project, _, draft, sender = await _human_reply_draft_context(
                    project_slug, draft_id, request, session=session
                )
                if draft.status == "rejected":
                    raise HTTPException(status_code=409, detail="Reply draft was rejected")
                mention_handles = json.loads(draft.mention_handles or "[]")
                await _validate_reply_handles(project, mention_handles, session=session)
                replay = draft.status == "approved"
                if replay:
                    message = await session.get(ChannelMessage, draft.sent_message_id)
                    if message is None:
                        raise HTTPException(
                            status_code=409, detail="Approved draft state is inconsistent"
                        )
                else:
                    message = await _create_session_lead_channel_message(
                        project,
                        sender,
                        subject=draft.subject,
                        body_md=draft.body_md,
                        importance=draft.importance,
                        mention_handles=mention_handles,
                        session=session,
                    )
                    draft.status = "approved"
                    draft.sent_message_id = message.id
                    draft.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                    draft.decided_at = draft.updated_at
                    session.add(draft)
                    await session.commit()
                    await session.refresh(message)
                    await session.refresh(sender)
                    await session.refresh(project)
            deliveries = await _deliver_channel_mentions(
                cast(Any, _HubHTTPDeliveryContext()),
                project,
                sender,
                message,
                mention_handles,
            )
            return JSONResponse(
                {
                    "status": "already_approved" if replay else "approved",
                    "draft": _reply_draft_payload(draft),
                    "deliveries": deliveries,
                },
                status_code=200 if replay else 201,
            )

    @fastapi_app.post(
        "/hub/api/projects/{project_slug}/reply-drafts/{draft_id}/reject",
        response_class=JSONResponse,
    )
    async def hub_reject_reply_draft(
        project_slug: str, draft_id: int, request: Request
    ) -> JSONResponse:
        await ensure_schema()
        draft_lock = _session_lead_draft_decision_locks.setdefault(
            draft_id, asyncio.Lock()
        )
        async with draft_lock, get_session() as session:
            _, _, draft, _ = await _human_reply_draft_context(
                project_slug, draft_id, request, session=session
            )
            if draft.status == "approved":
                raise HTTPException(status_code=409, detail="Reply draft was approved")
            if draft.status == "rejected":
                return JSONResponse(
                    {
                        "status": "already_rejected",
                        "draft": _reply_draft_payload(draft),
                    }
                )
            draft.status = "rejected"
            draft.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            draft.decided_at = draft.updated_at
            session.add(draft)
            await session.commit()
            await session.refresh(draft)
            return JSONResponse(
                {"status": "rejected", "draft": _reply_draft_payload(draft)}
            )

    def _hub_session_lead_payload(binding: SessionLeadBinding) -> dict[str, Any]:
        return {
            "id": binding.id,
            "team_project_id": binding.team_project_id,
            "human_id": binding.human_id,
            "client_session_id": binding.client_session_id,
            "agent_id": binding.agent_id,
            "lead_label": binding.lead_label,
            "reply_mode": binding.reply_mode,
            "runtime_status": binding.runtime_status,
            "runtime_seen_at": _hub_presence_timestamp(binding.runtime_seen_at),
            "status": binding.status,
            "created_at": str(binding.created_at),
            "updated_at": str(binding.updated_at),
        }

    def _hub_session_lead_name(
        human_id: int,
        client_session_id: str,
        lead_label: str,
    ) -> str:
        fragment = re.sub(r"[^A-Za-z0-9._-]+", "-", lead_label)[:32].strip("-.") or "lead"
        session_hash = hashlib.sha256(client_session_id.encode()).hexdigest()[:12]
        return f"SessionLead{human_id}-{fragment}-{session_hash}"

    @fastapi_app.put(
        "/hub/api/projects/{project_slug}/session-lead",
        response_class=JSONResponse,
    )
    async def hub_upsert_session_lead(project_slug: str, request: Request) -> JSONResponse:
        """Create or reuse the caller's managed lead Agent for a client session,
        and atomically make it the caller's membership default.

        client_session_id is an opaque client-computed hash (session+generation
        SHA-256) used only as the idempotency key — the Hub never receives
        session names, paths, or panes. lead_label names the managed Agent.
        """
        await ensure_schema()
        body = await _hub_json_body(request)
        raw_session_id = body.get("client_session_id")
        client_session_id = raw_session_id.strip() if isinstance(raw_session_id, str) else ""
        if not _CLIENT_SESSION_ID_RE.fullmatch(client_session_id):
            raise HTTPException(
                status_code=400,
                detail="client_session_id must be 1-128 letters, numbers, '.', '_' or '-'",
            )
        raw_label = body.get("lead_label")
        lead_label = raw_label.strip() if isinstance(raw_label, str) else ""
        if (
            not lead_label
            or len(lead_label) > 128
            or any(ord(char) < 32 for char in lead_label)
        ):
            raise HTTPException(
                status_code=400,
                detail="lead_label is required, must be at most 128 characters and contain no control characters",
            )
        rotate_raw = body.get("rotate_reply_token")
        if rotate_raw is not None and not isinstance(rotate_raw, bool):
            raise HTTPException(status_code=400, detail="rotate_reply_token must be a boolean")
        rotate_token = bool(rotate_raw)
        raw_reply_mode = body.get("reply_mode")
        if raw_reply_mode is not None and raw_reply_mode not in {"confirm", "auto"}:
            raise HTTPException(
                status_code=400, detail="reply_mode must be 'confirm' or 'auto'"
            )
        async with get_session() as session:
            team_project = await _hub_team_project(project_slug, session=session)
            human = await _hub_human(request, session=session)
            if human.id is None:
                raise HTTPException(status_code=404, detail="Human identity is not registered")
            project = await session.get(Project, team_project.routing_project_id)
            if project is None or project.archived_at is not None:
                raise HTTPException(status_code=404, detail="Project not found")
            membership = await _hub_active_membership(project, human, session=session)
            # 同一 human+project 任意时刻仅一个 active lead(#1064):
            # 关键段按 (team_project, human) 串行,先保证单 active 再 upsert。
            lead_lock = _session_lead_locks.setdefault(
                (cast(int, team_project.id), human.id),
                asyncio.Lock(),
            )
            async with lead_lock:
                # 单 active 切换: 旧 binding 的 reply capability 同事务失效
                await session.execute(
                    update(SessionLeadBinding)
                    .where(
                        cast(Any, SessionLeadBinding.team_project_id) == team_project.id,
                        cast(Any, SessionLeadBinding.human_id) == human.id,
                        cast(Any, SessionLeadBinding.status) == "active",
                        cast(Any, SessionLeadBinding.client_session_id) != client_session_id,
                    )
                    .values(
                        status="unbound",
                        reply_token_hash=None,
                        runtime_status="unknown",
                        runtime_seen_at=None,
                        updated_at=datetime.now(timezone.utc).replace(tzinfo=None),
                    )
                )

                existing_row = await session.execute(
                    select(SessionLeadBinding).where(
                        cast(Any, SessionLeadBinding.team_project_id) == team_project.id,
                        cast(Any, SessionLeadBinding.human_id) == human.id,
                        cast(Any, SessionLeadBinding.client_session_id) == client_session_id,
                    )
                )
                binding = existing_row.scalars().first()
                created = False
                reply_token: str | None = None
                if binding is not None:
                    agent = await session.get(Agent, binding.agent_id)
                    if agent is None:
                        raise HTTPException(status_code=409, detail="Managed lead agent is unavailable")
                    if agent.retired_at is not None:
                        # The Hub owns the managed lifecycle: reuse reactivates.
                        agent.retired_at = None
                        session.add(agent)
                    if binding.status != "active":
                        binding.status = "active"
                        binding.runtime_status = "unknown"
                        binding.runtime_seen_at = None
                        binding.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                        session.add(binding)
                    if binding.lead_label != lead_label:
                        # Label follows the latest PUT (display name, not a key).
                        binding.lead_label = lead_label
                        binding.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                        session.add(binding)
                    mode_changed = (
                        raw_reply_mode is not None
                        and binding.reply_mode != raw_reply_mode
                    )
                    if mode_changed:
                        binding.reply_mode = cast(str, raw_reply_mode)
                        binding.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                        session.add(binding)
                    if rotate_token or mode_changed or not binding.reply_token_hash:
                        reply_token = _new_reply_token()
                        binding.reply_token_hash = _hash_reply_token(reply_token)
                        binding.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                        session.add(binding)
                else:
                    lead_name = _hub_session_lead_name(
                        human.id,
                        client_session_id,
                        lead_label,
                    )
                    if await _membership_handle_taken(project, lead_name, session=session):
                        raise HTTPException(
                            status_code=409,
                            detail="Managed lead name collides with an active member mention_handle",
                        )
                    agent_insert = sqlite_insert(Agent).values(
                        project_id=project.id,
                        name=lead_name,
                        program="team-session-lead",
                        model="hub",
                        task_description="Hub-managed session lead agent",
                        owner_id=human.id,
                        contact_policy="open",
                    )
                    await session.execute(
                        agent_insert.on_conflict_do_nothing(index_elements=["project_id", "name"])
                    )
                    agent_row = await session.execute(
                        select(Agent).where(
                            cast(Any, Agent.project_id) == project.id,
                            cast(Any, Agent.name) == lead_name,
                        )
                    )
                    agent = agent_row.scalars().first()
                    if agent is None or agent.owner_id != human.id:
                        raise HTTPException(status_code=409, detail="Managed lead agent is unavailable")
                    reply_token = _new_reply_token()
                    binding_insert = sqlite_insert(SessionLeadBinding).values(
                        team_project_id=team_project.id,
                        human_id=human.id,
                        client_session_id=client_session_id,
                        agent_id=agent.id,
                        lead_label=lead_label,
                        reply_token_hash=_hash_reply_token(reply_token),
                        reply_mode=(
                            cast(str, raw_reply_mode)
                            if raw_reply_mode is not None
                            else "confirm"
                        ),
                        status="active",
                    )
                    await session.execute(
                        binding_insert.on_conflict_do_nothing(
                            index_elements=["team_project_id", "human_id", "client_session_id"]
                        )
                    )
                    binding_row = await session.execute(
                        select(SessionLeadBinding).where(
                            cast(Any, SessionLeadBinding.team_project_id) == team_project.id,
                            cast(Any, SessionLeadBinding.human_id) == human.id,
                            cast(Any, SessionLeadBinding.client_session_id) == client_session_id,
                        )
                    )
                    binding = binding_row.scalars().first()
                    if binding is None:
                        raise HTTPException(status_code=409, detail="Session lead binding conflict")
                    created = True

                # Atomically make the lead the caller's default (same transaction).
                await _validate_default_agent(project, human.id, agent.id, session=session)
                membership.default_agent_id = agent.id
                session.add(membership)
                await session.commit()
            await session.refresh(binding)
            await session.refresh(agent)
            payload = {
                "agent": _hub_agent_payload(agent),
                "client_session_id": client_session_id,
                "active": True,
                "binding": _hub_session_lead_payload(binding),
                "membership_default_agent_id": membership.default_agent_id,
            }
            if reply_token is not None:
                # Plaintext only on creation/rotation — never stored, never logged.
                payload["reply_token"] = reply_token
            return JSONResponse(payload, status_code=201 if created else 200)

    @fastapi_app.get(
        "/hub/api/projects/{project_slug}/session-lead",
        response_class=JSONResponse,
    )
    async def hub_list_session_leads(project_slug: str, request: Request) -> JSONResponse:
        """List the caller's own session-lead bindings (active + history)."""
        await ensure_schema()
        async with get_session() as session:
            team_project = await _hub_team_project(project_slug, session=session)
            human = await _hub_human(request, session=session)
            project = await session.get(Project, team_project.routing_project_id)
            if project is None:
                raise HTTPException(status_code=404, detail="Project not found")
            await _hub_active_membership(project, human, session=session)
            rows = await session.execute(
                select(SessionLeadBinding).where(
                    cast(Any, SessionLeadBinding.team_project_id) == team_project.id,
                    cast(Any, SessionLeadBinding.human_id) == human.id,
                )
                .order_by(cast(Any, SessionLeadBinding.created_at))
            )
            bindings = [
                _hub_session_lead_payload(binding)
                for binding in rows.scalars().all()
            ]
            return JSONResponse({"bindings": bindings})

    @fastapi_app.delete(
        "/hub/api/projects/{project_slug}/session-lead",
        response_class=JSONResponse,
    )
    async def hub_unbind_session_lead(
        project_slug: str,
        request: Request,
    ) -> JSONResponse:
        """Unbind a session lead: stops routing (inbox fallback off) but
        preserves the binding row, Agent and all messages. The caller's
        default is cleared only while the record is still active AND the
        default points at this lead — anything else is an idempotent no-op."""
        await ensure_schema()
        body = await _hub_json_body(request)
        raw_session_id = body.get("client_session_id")
        client_session_id = raw_session_id.strip() if isinstance(raw_session_id, str) else ""
        if not _CLIENT_SESSION_ID_RE.fullmatch(client_session_id):
            raise HTTPException(
                status_code=400,
                detail="client_session_id must be 1-128 letters, numbers, '.', '_' or '-'",
            )
        async with get_session() as session:
            team_project = await _hub_team_project(project_slug, session=session)
            human = await _hub_human(request, session=session)
            project = await session.get(Project, team_project.routing_project_id)
            if project is None:
                raise HTTPException(status_code=404, detail="Project not found")
            membership = await _hub_active_membership(project, human, session=session)
            existing_row = await session.execute(
                select(SessionLeadBinding).where(
                    cast(Any, SessionLeadBinding.team_project_id) == team_project.id,
                    cast(Any, SessionLeadBinding.human_id) == human.id,
                    cast(Any, SessionLeadBinding.client_session_id) == client_session_id,
                )
            )
            binding = existing_row.scalars().first()
            if binding is None:
                raise HTTPException(status_code=404, detail="Session lead binding not found")
            was_active = binding.status == "active"
            if was_active:
                binding.status = "unbound"
                binding.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                session.add(binding)
            if binding.reply_token_hash is not None:
                # 解绑即失效,同一事务
                binding.reply_token_hash = None
                session.add(binding)
            # Default cleared only while the record was still active AND the
            # caller's default points at this lead (contract #1062).
            if was_active and membership.default_agent_id == binding.agent_id:
                membership.default_agent_id = None
                session.add(membership)
            await session.commit()
            await session.refresh(binding)
            return JSONResponse(_hub_session_lead_payload(binding))

    def _oauth_metadata_disabled_response() -> JSONResponse:
        return JSONResponse({"mcp_oauth": False}, status_code=404)

    def _register_oauth_metadata_disabled(path: str) -> None:
        async def _oauth_metadata_disabled() -> JSONResponse:
            return _oauth_metadata_disabled_response()

        fastapi_app.add_api_route(path, _oauth_metadata_disabled, methods=["GET"], include_in_schema=False)

    # Thin ASGI wrapper that normalizes Accept / Content-Type headers for
    # MCP clients (some omit Accept entirely) and then delegates to the
    # SDK's native mcp_http_app which properly coordinates server lifecycle,
    # request handling, and session management via StreamableHTTPSessionManager.
    #
    # In production the parent FastAPI lifespan initializes the session manager
    # task group before any requests arrive.  In test environments (httpx
    # ASGITransport) no lifespan events are sent, so the wrapper lazily enters
    # the MCP app's lifespan on first request to avoid "Task group not
    # initialized" errors.
    class _HeaderFixupMCPApp:
        """Normalize headers then delegate to the native MCP HTTP app."""

        def __init__(self, native_app: FastAPI) -> None:
            self._app = native_app
            self._lifespan_entered = False
            self._lifespan_cm: Any = None
            self._lifespan_lock: asyncio.Lock | None = None

        async def _ensure_lifespan(self) -> None:
            """Lazily enter the MCP app's lifespan if not already running.

            This handles test environments where ASGI lifespan events are never
            sent (e.g. httpx ASGITransport).  In production the parent app's
            lifespan context already calls mcp_http_app.lifespan, so the
            session manager's task group will already be initialized and this
            method is a fast no-op.

            Uses double-check locking to prevent concurrent requests from
            entering the lifespan context manager twice.
            """
            if self._lifespan_entered:
                return
            # Lazily create the lock (must be in async context for the
            # correct event loop).
            if self._lifespan_lock is None:
                self._lifespan_lock = asyncio.Lock()
            async with self._lifespan_lock:
                if self._lifespan_entered:
                    return
                # Check if the session manager is already running (production path)
                session_mgr = getattr(self._app.state, "session_manager", None)
                if session_mgr is None:
                    # Try to find it via route endpoint
                    for route in getattr(self._app, "routes", []):
                        endpoint = getattr(route, "endpoint", None)
                        sm = getattr(endpoint, "session_manager", None)
                        if sm is not None:
                            session_mgr = sm
                            break
                if session_mgr is not None and getattr(session_mgr, "_task_group", None) is not None:
                    self._lifespan_entered = True
                    return
                # Enter the MCP app's lifespan (test path)
                mcp_lifespan_app = cast(_FastAPILifespan, self._app)
                self._lifespan_cm = mcp_lifespan_app.lifespan(self._app)
                await self._lifespan_cm.__aenter__()
                self._lifespan_entered = True

        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            if scope.get("type") != "http":
                # Delegate non-HTTP scopes (e.g. lifespan) directly
                await self._app(scope, receive, send)
                return

            await self._ensure_lifespan()

            headers = list(scope.get("headers") or [])

            def _has_header(key: bytes) -> bool:
                lk = key.lower()
                return any(h[0].lower() == lk for h in headers)

            # Ensure both JSON and SSE are accepted; httpx defaults no Accept header
            headers = [(k, v) for (k, v) in headers if k.lower() != b"accept"]
            headers.append((b"accept", b"application/json, text/event-stream"))
            if scope.get("method") == "POST" and not _has_header(b"content-type"):
                headers.append((b"content-type", b"application/json"))
            new_scope = dict(scope)
            new_scope["headers"] = headers

            await self._app(new_scope, receive, send)

    # Mount at both '/base' and '/base/' to tolerate either form from clients/tests.
    # Also mount compatibility aliases for both '/api' and '/mcp' regardless of configured base.
    mount_base = settings.http.path or "/api"
    if not mount_base.startswith("/"):
        mount_base = "/" + mount_base
    base_no_slash = mount_base.rstrip("/") or "/"
    base_with_slash = base_no_slash if base_no_slash == "/" else base_no_slash + "/"
    stateless_app = _HeaderFixupMCPApp(mcp_http_app)
    stateful_app = _HeaderFixupMCPApp(mcp_stateful_http_app)

    # Path -> app mapping (issue #250): the '/mcp' compat alias is the
    # stateful, Mcp-Session-Id-issuing endpoint; '/api' and the configured
    # base stay stateless for handshake-skipping one-shot clients (e.g. ntm).
    # The CONFIGURED base always keeps the legacy stateless behavior, even if
    # an operator points it at '/mcp' — an explicit HTTP_PATH is a promise to
    # existing clients of that deployment, so we never change its semantics.
    def _app_for_mount(path: str) -> _HeaderFixupMCPApp:
        normalized = path.rstrip("/") or "/"
        if normalized == "/mcp" and base_no_slash != "/mcp":
            return stateful_app
        return stateless_app

    mount_paths = [base_no_slash, base_with_slash]
    for compat_base in ("/api", "/mcp"):
        compat_no_slash = compat_base.rstrip("/") or "/"
        compat_with_slash = compat_no_slash if compat_no_slash == "/" else compat_no_slash + "/"
        if compat_no_slash not in mount_paths:
            mount_paths.append(compat_no_slash)
        if compat_with_slash not in mount_paths:
            mount_paths.append(compat_with_slash)

    oauth_metadata_paths: set[str] = set()

    def _add_oauth_metadata_path(path: str) -> None:
        normalized = path.rstrip("/") or "/"
        oauth_metadata_paths.add(normalized)
        if normalized != "/":
            oauth_metadata_paths.add(f"{normalized}/")

    _add_oauth_metadata_path("/.well-known/oauth-authorization-server")
    _add_oauth_metadata_path("/.well-known/oauth-authorization-server/mcp")
    for mount_path in mount_paths:
        normalized = mount_path.rstrip("/") or "/"
        if normalized == "/":
            continue
        _add_oauth_metadata_path(f"{normalized}/.well-known/oauth-authorization-server")
        _add_oauth_metadata_path(f"{normalized}/.well-known/oauth-authorization-server/mcp")
        _add_oauth_metadata_path(f"/.well-known/oauth-authorization-server{normalized}")
    for path in sorted(oauth_metadata_paths):
        _register_oauth_metadata_disabled(path)

    for mount_path in mount_paths:
        with contextlib.suppress(Exception):
            fastapi_app.mount(mount_path, _app_for_mount(mount_path))

    # Expose composed lifespan via router
    fastapi_app.router.lifespan_context = lifespan_context

    # Add direct routes at no-slash base paths to tolerate clients omitting trailing slashes.
    def _register_base_passthrough(base_path_no_slash: str, base_path_with_slash: str) -> None:
        # Dispatch to the same app that is mounted at this base (issue #250:
        # '/mcp' is stateful, everything else stateless).
        target_app = _app_for_mount(base_path_no_slash)

        @fastapi_app.post(base_path_no_slash)
        async def _base_passthrough(request: Request) -> JSONResponse:
            # Re-dispatch to the mounted MCP app by calling it directly
            response_body: dict[str, Any] = {}
            status_code = 200
            headers: dict[str, str] = {}

            async def _send(message: MutableMapping[str, Any]) -> None:
                nonlocal response_body, status_code, headers
                if message.get("type") == "http.response.start":
                    status_code = int(message.get("status", 200))
                    hdrs = message.get("headers") or []
                    for k, v in hdrs:
                        headers[k.decode("latin1")] = v.decode("latin1")
                elif message.get("type") == "http.response.body":
                    body = message.get("body") or b""
                    try:
                        response_body = json.loads(body.decode("utf-8")) if body else {}
                    except Exception:
                        response_body = {}

            # If localhost and allow_localhost_unauthenticated, synthesize Authorization header automatically
            scope = dict(request.scope)
            if _localhost_bypass_allowed(
                request,
                allow_localhost=bool(settings.http.allow_localhost_unauthenticated),
            ):
                scope_headers = list(scope.get("headers") or [])
                has_auth = any(k.lower() == b"authorization" for k, _ in scope_headers)
                if not has_auth and settings.http.bearer_token:
                    scope_headers.append((b"authorization", f"Bearer {settings.http.bearer_token}".encode("latin1")))
                scope["headers"] = scope_headers
            await target_app(
                {**scope, "path": "/"},  # MCP app expects requests at its root
                request.receive,
                _send,
            )
            return JSONResponse(response_body, status_code=status_code, headers=headers)

    passthrough_pairs: list[tuple[str, str]] = [(base_no_slash, base_with_slash)]
    for compat_base in ("/api", "/mcp"):
        compat_no_slash = compat_base.rstrip("/") or "/"
        compat_with_slash = compat_no_slash if compat_no_slash == "/" else compat_no_slash + "/"
        if (compat_no_slash, compat_with_slash) not in passthrough_pairs:
            passthrough_pairs.append((compat_no_slash, compat_with_slash))
    for no_slash, with_slash in passthrough_pairs:
        _register_base_passthrough(no_slash, with_slash)

    # ----- Simple SSR Mail UI -----
    def _register_mail_ui() -> None:
        import bleach
        import markdown2

        try:
            from bleach.css_sanitizer import CSSSanitizer as _CSSSanitizerImport
        except Exception:  # tinycss2 may be missing; degrade gracefully
            _CSSSanitizer = None
        else:
            _CSSSanitizer = _CSSSanitizerImport
        CSSSanitizer = cast(Any, _CSSSanitizer)
        from jinja2 import Environment, FileSystemLoader, select_autoescape

        templates_root = Path(__file__).resolve().parent / "templates"
        env = Environment(
            loader=FileSystemLoader(str(templates_root)),
            autoescape=select_autoescape(["html", "xml"]),
            enable_async=True,
        )
        # HTML sanitizer (allow safe images and limited CSS)
        _css_sanitizer = (
            CSSSanitizer(
                allowed_css_properties=["color", "background-color", "text-align", "text-decoration", "font-weight"]
            )
            if CSSSanitizer
            else None
        )
        _html_cleaner = bleach.Cleaner(
            tags=[
                "a",
                "abbr",
                "acronym",
                "b",
                "blockquote",
                "code",
                "em",
                "i",
                "li",
                "ol",
                "ul",
                "p",
                "pre",
                "strong",
                "table",
                "thead",
                "tbody",
                "tr",
                "th",
                "td",
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
                "hr",
                "br",
                "span",
                "img",
            ],
            attributes={
                "*": ["class"],
                "a": ["href", "title", "rel"],
                "abbr": ["title"],
                "acronym": ["title"],
                "code": ["class"],
                "pre": ["class"],
                "span": ["class", "style"],
                "p": ["class", "style"],
                "table": ["class", "style"],
                "td": ["class", "style"],
                "th": ["class", "style"],
                "img": ["src", "alt", "title", "width", "height", "loading", "decoding", "class"],
            },
            protocols=["http", "https", "mailto", "data"],
            strip=True,
            css_sanitizer=_css_sanitizer,
        )

        async def _render(name: str, **ctx: Any) -> HTMLResponse:
            tpl = env.get_template(name)
            html = await tpl.render_async(**ctx)
            return HTMLResponse(html)

        def _parse_fts_query(
            raw: str, scope_preference: str | None = None
        ) -> tuple[str, str, str, list[dict[str, str]]]:
            """Return (fts_expression, like_pattern) from a user query.
            Supports subject:foo and body:"multi word" tokens; otherwise defaults to subject/body OR.
            """
            raw = (raw or "").strip()
            if not raw:
                return "", "", "both", []
            scope_pref = scope_preference if scope_preference in {"subject", "body"} else "both"
            # tokens: key:"phrase" | "phrase" | key:word | word
            parts = re.findall(r"\w+:\"[^\"]+\"|\"[^\"]+\"|\w+:[^\s]+|[^\s]+", raw)
            exprs: list[str] = []
            like_terms: list[str] = []
            like_scope = scope_pref
            tokens: list[dict[str, str]] = []

            def _quote(s: str) -> str:
                return '"' + s.replace('"', '""') + '"'

            def _like_escape(term: str) -> str:
                return term.replace("!", "!!").replace("%", "!%").replace("_", "!_")

            for p in parts:
                key = None
                val = p
                if ":" in p and not p.startswith('"'):
                    maybe_key, maybe_val = p.split(":", 1)
                    if maybe_key in {"subject", "body"}:
                        key = maybe_key
                        val = maybe_val
                val = val.strip()
                val_inner = val[1:-1] if val.startswith('"') and val.endswith('"') and len(val) >= 2 else val

                # For LIKE pattern, we want literal matching of the user's term
                like_terms.append(_like_escape(val_inner))

                if key in {"subject", "body"}:
                    exprs.append(f"{key}:{_quote(val_inner)}")
                    tokens.append({"field": key, "value": val_inner})
                else:
                    if scope_pref == "subject":
                        exprs.append(f"subject:{_quote(val_inner)}")
                        tokens.append({"field": "subject", "value": val_inner})
                    elif scope_pref == "body":
                        exprs.append(f"body:{_quote(val_inner)}")
                        tokens.append({"field": "body", "value": val_inner})
                    else:
                        exprs.append(f"(subject:{_quote(val_inner)} OR body:{_quote(val_inner)})")
                        tokens.append({"field": "both", "value": val_inner})
            fts = " AND ".join(exprs) if exprs else ""
            like_pat = "%" + "%".join(like_terms) + "%" if like_terms else ""
            return fts, like_pat, like_scope, tokens

        @fastapi_app.get("/mail/api/locks", response_class=JSONResponse)
        async def mail_lock_status() -> JSONResponse:
            """Return metadata about active archive locks for observability."""

            settings_local = get_settings()
            payload = collect_lock_status(settings_local)
            return JSONResponse(payload)

        async def _build_unified_inbox_payload(
            *, limit: int = 500, include_projects: bool = True
        ) -> dict[str, Any]:
            """Fetch unified inbox data for HTML and JSON consumers."""

            safe_limit = max(1, min(int(limit), 1000))
            messages: list[dict[str, Any]] = []
            projects: list[dict[str, Any]] = []

            try:
                await ensure_schema()

                sibling_map: dict[int, dict[str, Any]] = {}
                if include_projects:
                    await refresh_project_sibling_suggestions()
                    sibling_map = await get_project_sibling_data()

                async with get_session() as session:
                    # Fetch recent messages with sender/project and computed recipient list
                    query = text(
                        """
                        SELECT
                            m.id,
                            m.subject,
                            m.body_md,
                            LENGTH(COALESCE(m.body_md, '')) AS body_length,
                            m.created_ts,
                            m.importance,
                            m.thread_id,
                            m.project_id AS message_project_id,
                            sender.name AS sender_name,
                            sender.project_id AS sender_project_id,
                            sp.human_key AS sender_project_name,
                            sp.slug AS sender_project_slug,
                            p.slug AS project_slug,
                            p.human_key AS project_name,
                            COALESCE(
                                (
                                    SELECT GROUP_CONCAT(name, ', ')
                                    FROM (
                                        SELECT DISTINCT recip2.name AS name
                                        FROM message_recipients mr2
                                        JOIN agents recip2 ON recip2.id = mr2.agent_id
                                        WHERE mr2.message_id = m.id
                                        ORDER BY name
                                    )
                                ),
                                ''
                            ) AS recipients
                        FROM messages m
                        JOIN agents sender ON m.sender_id = sender.id
                        LEFT JOIN projects sp ON sp.id = sender.project_id
                        JOIN projects p ON m.project_id = p.id
                        ORDER BY m.created_ts DESC
                        LIMIT :limit
                        """
                    )

                    rows = await session.execute(query, {"limit": safe_limit})

                    for r in rows.mappings().all():
                        body = r["body_md"] or ""
                        raw_body_length = r["body_length"]
                        body_length = int(raw_body_length) if raw_body_length is not None else len(body)
                        excerpt = body[:150].replace('#', '').replace('*', '').replace('`', '').strip()
                        if body_length > 150:
                            excerpt += "..."

                        created_ts = r["created_ts"]
                        if isinstance(created_ts, str):
                            created_dt = datetime.fromisoformat(created_ts.replace('Z', '+00:00'))
                        else:
                            created_dt = created_ts

                        if created_dt.tzinfo is None:
                            created_dt = created_dt.replace(tzinfo=timezone.utc)
                        else:
                            created_dt = created_dt.astimezone(timezone.utc)

                        now = datetime.now(timezone.utc)
                        delta = now - created_dt

                        if delta.days < 0 or (delta.days == 0 and delta.seconds < 0):
                            created_relative = "Just now"
                        elif delta.days > 365:
                            created_relative = f"{delta.days // 365}y ago"
                        elif delta.days > 30:
                            created_relative = f"{delta.days // 30}mo ago"
                        elif delta.days > 0:
                            created_relative = f"{delta.days}d ago"
                        elif delta.seconds > 3600:
                            created_relative = f"{delta.seconds // 3600}h ago"
                        elif delta.seconds > 60:
                            created_relative = f"{delta.seconds // 60}m ago"
                        else:
                            created_relative = "Just now"

                        sender_display, sender_meta = _http_sender_identity(
                            message_project_id=r["message_project_id"],
                            sender_name=r["sender_name"],
                            sender_project_id=r["sender_project_id"],
                            sender_project_human_key=r["sender_project_name"],
                            sender_project_slug=r["sender_project_slug"],
                        )
                        message_payload = {
                            "id": r["id"],
                            "subject": r["subject"] or "(No subject)",
                            "body_md": body,
                            "body_length": body_length,
                            "excerpt": excerpt,
                            "created_ts": str(r["created_ts"]),
                            "created_full": created_dt.strftime("%B %d, %Y at %I:%M %p"),
                            "created_relative": created_relative,
                            "importance": r["importance"] or "normal",
                            "thread_id": r["thread_id"],
                            "sender": sender_display,
                            "project_slug": r["project_slug"],
                            "project_name": r["project_name"],
                            "recipients": ", ".join(
                                part.strip() for part in (r["recipients"] or "").split(",") if part.strip()
                            ),
                            "read": False,
                        }
                        message_payload.update(sender_meta)
                        messages.append(message_payload)

                    if include_projects:
                        rows = await session.execute(
                            text("SELECT id, slug, human_key, created_at, archived_at FROM projects ORDER BY created_at DESC")
                        )
                        for r in rows.fetchall():
                            project_id = int(r[0])
                            siblings = sibling_map.get(project_id, {"confirmed": [], "suggested": []})
                            projects.append(
                                {
                                    "id": project_id,
                                    "slug": r[1],
                                    "human_key": r[2],
                                    "created_at": str(r[3]),
                                    "archived_at": str(r[4]) if r[4] else None,
                                    "confirmed_siblings": siblings.get("confirmed", []),
                                    "suggested_siblings": siblings.get("suggested", []),
                                }
                            )

            except Exception as exc:  # pragma: no cover - defensive logging
                logging.error("Error fetching unified inbox data", exc_info=True, extra={"error": str(exc)})

            return {"messages": messages, "projects": projects}

        @fastapi_app.get("/mail", response_class=HTMLResponse)
        async def mail_unified_inbox() -> HTMLResponse:
            """Unified inbox showing ALL messages across ALL projects (Gmail-style) + Projects below"""

            payload = await _build_unified_inbox_payload()
            return await _render(
                "mail_unified_inbox.html",
                messages=payload.get("messages", []),
                projects=payload.get("projects", []),
            )

        @fastapi_app.get("/mail/api/unified-inbox", response_class=JSONResponse)
        async def mail_unified_inbox_api(
            limit: int = 50000,
            include_projects: bool = False,
        ) -> JSONResponse:
            """JSON feed for the unified inbox view (used for background refresh)."""

            payload = await _build_unified_inbox_payload(limit=limit, include_projects=include_projects)
            if not include_projects:
                # Reduce payload size when polling for message updates only
                payload["projects"] = []
            return JSONResponse(payload)

        @fastapi_app.post("/mail/api/delete-messages", response_class=JSONResponse)
        async def delete_messages_api(request: Request) -> JSONResponse:
            """Permanently delete messages by ID (cross-project).

            Removes messages from the SQLite database AND deletes the
            corresponding markdown files from the Git archive.
            """
            await ensure_schema()

            try:
                request_body = await request.json()
                message_ids: list[int] = request_body.get("message_ids", [])

                if not message_ids:
                    raise HTTPException(status_code=400, detail="No message IDs provided")

                if len(message_ids) > 500:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Too many messages ({len(message_ids)}). Maximum is 500."
                    )

                deleted_count = 0
                messages_by_project: dict[str, list[tuple[Any, ...]]] = {}
                recip_map: dict[int, list[str]] = {}
                async with get_session() as session:
                    placeholders = ','.join([f':mid{i}' for i in range(len(message_ids))])
                    id_params: dict[str, Any] = {f"mid{i}": mid for i, mid in enumerate(message_ids)}

                    # Fetch message metadata for Git cleanup
                    rows = await session.execute(
                        text(
                            f"""
                            SELECT m.id, m.created_ts, m.subject, s.name AS sender_name,
                                   p.slug AS project_slug
                            FROM messages m
                            JOIN agents s ON s.id = m.sender_id
                            JOIN projects p ON p.id = m.project_id
                            WHERE m.id IN ({placeholders})
                            """
                        ),
                        id_params,
                    )
                    messages_to_delete = [tuple(row) for row in rows.fetchall()]

                    if not messages_to_delete:
                        return JSONResponse({"success": True, "deleted_count": 0})

                    # Collect recipients per message
                    recip_rows = await session.execute(
                        text(
                            f"""
                            SELECT mr.message_id, a.name
                            FROM message_recipients mr
                            JOIN agents a ON a.id = mr.agent_id
                            WHERE mr.message_id IN ({placeholders})
                            """
                        ),
                        id_params,
                    )
                    for rr in recip_rows.fetchall():
                        recip_map.setdefault(int(rr[0]), []).append(rr[1])

                    for mrow in messages_to_delete:
                        slug = str(mrow[4])
                        messages_by_project.setdefault(slug, []).append(mrow)

                    # Delete from SQLite
                    await session.execute(
                        text(f"DELETE FROM message_recipients WHERE message_id IN ({placeholders})"),
                        id_params,
                    )
                    del_result = await session.execute(
                        text(f"DELETE FROM messages WHERE id IN ({placeholders})"),
                        id_params,
                    )
                    deleted_count = int(getattr(del_result, "rowcount", 0) or 0)
                    await session.commit()

                settings = get_settings()
                total_git_files_removed = 0
                for project_slug, proj_msgs in messages_by_project.items():
                    try:
                        total_git_files_removed += await _delete_messages_from_archive(
                            settings=settings,
                            project_slug=project_slug,
                            messages_to_delete=proj_msgs,
                            recip_map=recip_map,
                            commit_message=f"delete: {len(proj_msgs)} message(s) via web UI\n",
                        )
                    except Exception as archive_exc:
                        logging.getLogger(__name__).warning(
                            "Git archive cleanup failed for project %s: %s",
                            project_slug,
                            archive_exc,
                        )

                return JSONResponse({
                    "success": True,
                    "deleted_count": deleted_count,
                    "git_files_removed": total_git_files_removed,
                })

            except HTTPException:
                raise
            except Exception as exc:
                import traceback
                traceback.print_exc()
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to delete messages: {exc!s}"
                ) from exc

        # ---- Agent Retire/Unretire API ----

        @fastapi_app.post("/mail/api/retire-agent", response_class=JSONResponse)
        async def retire_agent_api(request: Request) -> JSONResponse:
            """Retire an agent (soft-delete). Preserves message history but stops new messages."""
            await ensure_schema()
            try:
                body = await request.json()
                agent_id: int | None = body.get("agent_id")
                if agent_id is None:
                    raise HTTPException(status_code=400, detail="agent_id is required")

                async with get_session() as session:
                    agent = await _hub_agent_for_update(agent_id, session=session)
                    if not agent:
                        raise HTTPException(status_code=404, detail="Agent not found")
                    if settings.http.jwt_enabled:
                        await _hub_agent_manager(request, agent, session=session)
                    referenced = await _agent_referenced_as_default(agent_id, session=session)
                    if referenced:
                        raise HTTPException(
                            status_code=409,
                            detail="Clear membership default_agent_id before retiring this agent",
                        )
                    agent.retired_at = datetime.now(timezone.utc).replace(tzinfo=None)
                    session.add(agent)
                    await session.commit()

                return JSONResponse({"success": True, "agent_id": agent_id, "status": "retired"})
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Failed to retire agent: {exc!s}") from exc

        @fastapi_app.post("/mail/api/unretire-agent", response_class=JSONResponse)
        async def unretire_agent_api(request: Request) -> JSONResponse:
            """Restore a retired agent back to active status."""
            await ensure_schema()
            try:
                body = await request.json()
                agent_id: int | None = body.get("agent_id")
                if agent_id is None:
                    raise HTTPException(status_code=400, detail="agent_id is required")

                async with get_session() as session:
                    agent = await _hub_agent_for_update(agent_id, session=session)
                    if not agent:
                        raise HTTPException(status_code=404, detail="Agent not found")
                    if settings.http.jwt_enabled:
                        await _hub_agent_manager(request, agent, session=session)
                    agent.retired_at = None
                    session.add(agent)
                    await session.commit()

                return JSONResponse({"success": True, "agent_id": agent_id, "status": "active"})
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Failed to unretire agent: {exc!s}") from exc

        # ---- Project Archive/Unarchive API ----

        @fastapi_app.post("/mail/api/archive-project", response_class=JSONResponse)
        async def archive_project_api(request: Request) -> JSONResponse:
            """Archive a project (soft-delete). Preserves all messages but hides from active lists."""
            await ensure_schema()
            try:
                body = await request.json()
                project_id: int | None = body.get("project_id")
                if project_id is None:
                    raise HTTPException(status_code=400, detail="project_id is required")

                async with get_session() as session:
                    project = await session.get(Project, project_id)
                    if not project:
                        raise HTTPException(status_code=404, detail="Project not found")
                    if settings.http.jwt_enabled:
                        human = await _hub_human(request, session=session)
                        await _hub_active_membership(
                            project,
                            human,
                            session=session,
                            admin=True,
                        )
                    project.archived_at = datetime.now(timezone.utc).replace(tzinfo=None)
                    session.add(project)
                    await session.commit()

                return JSONResponse({"success": True, "project_id": project_id, "status": "archived"})
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Failed to archive project: {exc!s}") from exc

        @fastapi_app.post("/mail/api/unarchive-project", response_class=JSONResponse)
        async def unarchive_project_api(request: Request) -> JSONResponse:
            """Restore an archived project back to active status."""
            await ensure_schema()
            try:
                body = await request.json()
                project_id: int | None = body.get("project_id")
                if project_id is None:
                    raise HTTPException(status_code=400, detail="project_id is required")

                async with get_session() as session:
                    project = await session.get(Project, project_id)
                    if not project:
                        raise HTTPException(status_code=404, detail="Project not found")
                    if settings.http.jwt_enabled:
                        human = await _hub_human(request, session=session)
                        await _hub_active_membership(
                            project,
                            human,
                            session=session,
                            admin=True,
                        )
                    project.archived_at = None
                    session.add(project)
                    await session.commit()

                return JSONResponse({"success": True, "project_id": project_id, "status": "active"})
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Failed to unarchive project: {exc!s}") from exc

        @fastapi_app.get("/mail/projects", response_class=HTMLResponse)
        async def mail_projects_list() -> HTMLResponse:
            """Projects list view (moved from /mail)"""
            await ensure_schema()
            await refresh_project_sibling_suggestions()
            sibling_map = await get_project_sibling_data()
            async with get_session() as session:
                rows = await session.execute(
                    text("SELECT id, slug, human_key, created_at, archived_at FROM projects ORDER BY created_at DESC")
                )
                projects = []
                for r in rows.fetchall():
                    project_id = int(r[0])
                    siblings = sibling_map.get(project_id, {"confirmed": [], "suggested": []})
                    projects.append(
                        {
                            "id": project_id,
                            "slug": r[1],
                            "human_key": r[2],
                            "created_at": str(r[3]),
                            "archived_at": str(r[4]) if r[4] else None,
                            "confirmed_siblings": siblings.get("confirmed", []),
                            "suggested_siblings": siblings.get("suggested", []),
                        }
                    )
            return await _render("mail_index.html", projects=projects)

        @fastapi_app.get("/mail/{project}", response_class=HTMLResponse)
        async def mail_project(
            project: str,
            q: str | None = None,
            scope: str | None = None,
            order: str | None = None,
            boost: int | None = None,
        ) -> HTMLResponse:
            if order not in ("relevance", "time", None):
                order = "relevance"
            await ensure_schema()
            async with get_session() as session:
                proj = await session.execute(
                    text("SELECT id, slug, human_key, archived_at FROM projects WHERE slug = :k OR human_key = :k"), {"k": project}
                )
                prow = proj.fetchone()
                if not prow:
                    return await _render("error.html", message="Project not found")
                pid = int(prow[0])
                project_archived_at = str(prow[3]) if prow[3] else None
                agents_q = await session.execute(
                    text("SELECT id, name, program, model, retired_at FROM agents WHERE project_id = :pid ORDER BY name"),
                    {"pid": pid},
                )
                agents = [{"id": r[0], "name": r[1], "program": r[2], "model": r[3], "retired_at": str(r[4]) if r[4] else None} for r in agents_q.fetchall()]
                matched_messages: list[dict] = []
                if q and q.strip():
                    # Prefer FTS5 when available (fts_messages maintained by triggers)
                    fts_expr, like_pat, like_scope, tokens = _parse_fts_query(q, scope)
                    weights = (0.0, 3.0, 1.0) if (boost or 0) else (0.0, 1.0, 1.0)
                    fts_sql = (
                        "SELECT m.id, m.subject, s.name AS sender_name, s.project_id AS sender_project_id, "
                        "sp.human_key AS sender_project_name, sp.slug AS sender_project_slug, "
                        "m.created_ts, m.importance, m.thread_id, "
                        "snippet(fts_messages, 2, '<mark>', '</mark>', '…', 18) AS body_snippet "
                        "FROM fts_messages "
                        "JOIN messages m ON m.id = fts_messages.rowid "
                        "JOIN agents s ON s.id = m.sender_id "
                        "LEFT JOIN projects sp ON sp.id = s.project_id "
                        "WHERE m.project_id = :pid AND fts_messages MATCH :q "
                        + (
                            "ORDER BY m.created_ts DESC "
                            if (order or "relevance") == "time"
                            else f"ORDER BY bm25(fts_messages, {weights[0]}, {weights[1]}, {weights[2]}) "
                        )
                        + "LIMIT 10000"
                    )
                    try:
                        search = await session.execute(text(fts_sql), {"pid": pid, "q": fts_expr or q})
                        matched_messages = []
                        for r in search.mappings().all():
                            sender_display, sender_meta = _http_sender_identity(
                                message_project_id=pid,
                                sender_name=r["sender_name"],
                                sender_project_id=r["sender_project_id"],
                                sender_project_human_key=r["sender_project_name"],
                                sender_project_slug=r["sender_project_slug"],
                            )
                            item = {
                                "id": r["id"],
                                "subject": r["subject"],
                                "sender": sender_display,
                                "created": str(r["created_ts"]),
                                "importance": r["importance"],
                                "thread_id": r["thread_id"],
                                "snippet": r["body_snippet"],
                                "hits": (r["body_snippet"] or "").count("<mark>"),
                            }
                            item.update(sender_meta)
                            matched_messages.append(item)
                    except Exception:
                        # Fallback to LIKE if FTS not available
                        if like_scope == "subject":
                            like_sql = (
                                "SELECT m.id, m.subject, s.name AS sender_name, s.project_id AS sender_project_id, "
                                "sp.human_key AS sender_project_name, sp.slug AS sender_project_slug, "
                                "m.created_ts, m.importance, m.thread_id "
                                "FROM messages m JOIN agents s ON s.id = m.sender_id "
                                "LEFT JOIN projects sp ON sp.id = s.project_id "
                                f"WHERE m.project_id = :pid AND m.subject LIKE :pat ESCAPE '{_LIKE_ESCAPE_CHAR}' "
                                "ORDER BY m.created_ts DESC LIMIT 10000"
                            )
                        elif like_scope == "body":
                            like_sql = (
                                "SELECT m.id, m.subject, s.name AS sender_name, s.project_id AS sender_project_id, "
                                "sp.human_key AS sender_project_name, sp.slug AS sender_project_slug, "
                                "m.created_ts, m.importance, m.thread_id "
                                "FROM messages m JOIN agents s ON s.id = m.sender_id "
                                "LEFT JOIN projects sp ON sp.id = s.project_id "
                                f"WHERE m.project_id = :pid AND m.body_md LIKE :pat ESCAPE '{_LIKE_ESCAPE_CHAR}' "
                                "ORDER BY m.created_ts DESC LIMIT 10000"
                            )
                        else:
                            like_sql = (
                                "SELECT m.id, m.subject, s.name AS sender_name, s.project_id AS sender_project_id, "
                                "sp.human_key AS sender_project_name, sp.slug AS sender_project_slug, "
                                "m.created_ts, m.importance, m.thread_id "
                                "FROM messages m JOIN agents s ON s.id = m.sender_id "
                                "LEFT JOIN projects sp ON sp.id = s.project_id "
                                f"WHERE m.project_id = :pid AND (m.subject LIKE :pat ESCAPE '{_LIKE_ESCAPE_CHAR}' "
                                f"OR m.body_md LIKE :pat ESCAPE '{_LIKE_ESCAPE_CHAR}') "
                                "ORDER BY m.created_ts DESC LIMIT 10000"
                            )
                        search = await session.execute(text(like_sql), {"pid": pid, "pat": like_pat or f"%{_like_escape(q)}%"})
                        matched_messages = []
                        for r in search.mappings().all():
                            sender_display, sender_meta = _http_sender_identity(
                                message_project_id=pid,
                                sender_name=r["sender_name"],
                                sender_project_id=r["sender_project_id"],
                                sender_project_human_key=r["sender_project_name"],
                                sender_project_slug=r["sender_project_slug"],
                            )
                            item = {
                                "id": r["id"],
                                "subject": r["subject"],
                                "sender": sender_display,
                                "created": str(r["created_ts"]),
                                "importance": r["importance"],
                                "thread_id": r["thread_id"],
                                "snippet": "",
                                "hits": 0,
                            }
                            item.update(sender_meta)
                            matched_messages.append(item)
            return await _render(
                "mail_project.html",
                project={"id": pid, "slug": prow[1], "human_key": prow[2], "archived_at": project_archived_at},
                agents=agents,
                q=q or "",
                scope=scope or "",
                order=order or "relevance",
                boost=bool(boost),
                tokens=tokens if q and q.strip() else [],
                results=matched_messages,
            )

        @fastapi_app.post("/mail/api/projects/{project_id}/siblings/{other_id}", response_class=JSONResponse)
        async def update_project_sibling(project_id: int, other_id: int, request: Request) -> JSONResponse:
            try:
                payload = await request.json()
            except Exception:
                payload = {}
            action = str(payload.get("action", "")).lower()
            if action not in {"confirm", "dismiss", "reset"}:
                return JSONResponse({"error": "Invalid action"}, status_code=status.HTTP_400_BAD_REQUEST)

            target_status = {
                "confirm": "confirmed",
                "dismiss": "dismissed",
                "reset": "suggested",
            }[action]

            try:
                suggestion = await update_project_sibling_status(project_id, other_id, target_status)
            except ValueError as exc:
                return JSONResponse({"error": str(exc)}, status_code=status.HTTP_400_BAD_REQUEST)
            except NoResultFound:
                return JSONResponse({"error": "Project pair not found"}, status_code=status.HTTP_404_NOT_FOUND)
            except Exception as exc:
                structlog.get_logger("sibling").exception(
                    "project_sibling.update_failed",
                    project_id=project_id,
                    other_id=other_id,
                    action=action,
                    error=str(exc),
                )
                return JSONResponse(
                    {"error": "Unable to update sibling status"}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            return JSONResponse({"status": suggestion["status"], "suggestion": suggestion})

        @fastapi_app.get("/mail/unified-inbox", response_class=HTMLResponse)
        async def unified_inbox(limit: int = 10000, filter_importance: str | None = None) -> HTMLResponse:
            """Unified inbox showing messages from all active agents across all projects."""
            limit = min(max(1, limit), 10000)
            await ensure_schema()
            async with get_session() as session:
                # Get all projects with their agents
                projects_query = await session.execute(
                    text(
                        """
                    SELECT p.id, p.slug, p.human_key,
                           COUNT(DISTINCT a.id) as agent_count,
                           MAX(a.last_active_ts) as last_activity
                    FROM projects p
                    LEFT JOIN agents a ON a.project_id = p.id
                    GROUP BY p.id, p.slug, p.human_key
                    ORDER BY (last_activity IS NULL) ASC, last_activity DESC, p.created_at DESC
                    """
                    )
                )
                projects_data = []
                for r in projects_query.fetchall():
                    proj_id = int(r[0])
                    # Get agents for this project
                    agents_query = await session.execute(
                        text(
                            """
                        SELECT a.id, a.name, a.program, a.model, a.last_active_ts
                        FROM agents a
                        WHERE a.project_id = :pid
                        ORDER BY a.last_active_ts DESC, a.name ASC
                        """
                        ),
                        {"pid": proj_id},
                    )

                    agents_list = []
                    for ar in agents_query.fetchall():
                        agents_list.append(
                            {
                                "id": int(ar[0]),
                                "name": ar[1],
                                "program": ar[2],
                                "model": ar[3],
                                "last_active": str(ar[4]) if ar[4] else None,
                            }
                        )

                    if agents_list:  # Only include projects with agents
                        projects_data.append(
                            {
                                "id": proj_id,
                                "slug": r[1],
                                "human_key": r[2],
                                "agent_count": int(r[3] or 0),
                                "agents": agents_list,
                            }
                        )

                # Get recent messages across all projects with thread information
                # Build WHERE clause safely using parameterized queries
                importance_conditions = []
                query_params = {"lim": limit}

                if filter_importance and filter_importance.lower() in ["urgent", "high"]:
                    importance_conditions.append("m.importance IN ('urgent', 'high')")

                where_clause = "WHERE " + " AND ".join(importance_conditions) if importance_conditions else "WHERE 1=1"

                messages_query = await session.execute(
                    text(
                        f"""
                    SELECT
                        m.id, m.subject, m.body_md, m.created_ts, m.importance, m.thread_id,
                        m.project_id AS message_project_id,
                        p.slug, p.human_key,
                        sender.name as sender_name,
                        sender.project_id AS sender_project_id,
                        sp.human_key AS sender_project_name,
                        sp.slug AS sender_project_slug,
                        COALESCE(
                            (
                                SELECT GROUP_CONCAT(name, ', ')
                                FROM (
                                    SELECT DISTINCT recip2.name AS name
                                    FROM message_recipients mr2
                                    JOIN agents recip2 ON recip2.id = mr2.agent_id
                                    WHERE mr2.message_id = m.id
                                    ORDER BY name
                                )
                            ),
                            ''
                        ) as recipient_names,
                        COUNT(DISTINCT CASE WHEN m2.id IS NOT NULL THEN m2.id END) as thread_count
                    FROM messages m
                    JOIN projects p ON p.id = m.project_id
                    JOIN agents sender ON sender.id = m.sender_id
                    LEFT JOIN projects sp ON sp.id = sender.project_id
                    LEFT JOIN message_recipients mr ON mr.message_id = m.id
                    LEFT JOIN agents recip ON recip.id = mr.agent_id
                    LEFT JOIN messages m2 ON (
                        m.thread_id IS NOT NULL
                        AND m2.thread_id = m.thread_id
                        AND m2.project_id = m.project_id
                        AND m2.id != m.id
                    )
                    {where_clause}
                    GROUP BY m.id, m.subject, m.body_md, m.created_ts, m.importance, m.thread_id,
                             m.project_id, p.slug, p.human_key, sender.name, sender.project_id, sp.human_key, sp.slug
                    ORDER BY m.created_ts DESC
                    LIMIT :lim
                    """
                    ),
                    query_params,
                )

                messages = []
                for r in messages_query.mappings().all():
                    sender_display, sender_meta = _http_sender_identity(
                        message_project_id=r["message_project_id"],
                        sender_name=r["sender_name"],
                        sender_project_id=r["sender_project_id"],
                        sender_project_human_key=r["sender_project_name"],
                        sender_project_slug=r["sender_project_slug"],
                    )
                    item = {
                        "id": int(r["id"]),
                        "subject": r["subject"],
                        "body_md": r["body_md"] or "",
                        "created": str(r["created_ts"]),
                        "importance": r["importance"] or "normal",
                        "thread_id": r["thread_id"],
                        "project_slug": r["slug"],
                        "project_name": r["human_key"],
                        "sender": sender_display,
                        "recipients": r["recipient_names"] or "",
                        "thread_count": int(r["thread_count"] or 0),
                    }
                    item.update(sender_meta)
                    messages.append(item)

            return await _render(
                "mail_unified_inbox.html",
                projects=projects_data,
                messages=messages,
                total_agents=sum(p["agent_count"] for p in projects_data),
                total_messages=len(messages),
                filter_importance=filter_importance or "",
            )

        @fastapi_app.get("/mail/{project}/inbox/{agent}", response_class=HTMLResponse)
        async def mail_inbox(project: str, agent: str, limit: int = 10000, page: int = 1) -> HTMLResponse:
            limit = min(max(1, limit), 10000)
            page = min(max(1, page), 10000)
            await ensure_schema()
            async with get_session() as session:
                prow = (
                    await session.execute(
                        text("SELECT id, slug, human_key FROM projects WHERE slug = :k OR human_key = :k"),
                        {"k": project},
                    )
                ).fetchone()
                if not prow:
                    return await _render("error.html", message="Project not found")
                pid = int(prow[0])
                arow = (
                    await session.execute(
                        text("SELECT id, name FROM agents WHERE project_id = :pid AND lower(name) = lower(:name)"),
                        {"pid": pid, "name": agent},
                    )
                ).fetchone()
                if not arow:
                    return await _render("error.html", message="Agent not found")
                offset = max(0, (max(1, page) - 1) * max(1, limit))
                inbox_rows = await session.execute(
                    text(
                        """
                    SELECT
                        m.id,
                        m.subject,
                        s.name AS sender_name,
                        s.project_id AS sender_project_id,
                        sp.human_key AS sender_project_name,
                        sp.slug AS sender_project_slug,
                        m.created_ts,
                        m.importance,
                        m.thread_id,
                        m.ack_required,
                        mr.read_ts,
                        mr.ack_ts
                    FROM messages m
                    JOIN message_recipients mr ON mr.message_id = m.id
                    JOIN agents a ON a.id = mr.agent_id
                    JOIN agents s ON s.id = m.sender_id
                    LEFT JOIN projects sp ON sp.id = s.project_id
                    WHERE m.project_id = :pid AND a.name = :name
                    ORDER BY m.created_ts DESC
                    LIMIT :lim OFFSET :off
                    """
                    ),
                    {"pid": pid, "name": agent, "lim": limit, "off": offset},
                )
                items = []
                for r in inbox_rows.mappings().all():
                    sender_display, sender_meta = _http_sender_identity(
                        message_project_id=pid,
                        sender_name=r["sender_name"],
                        sender_project_id=r["sender_project_id"],
                        sender_project_human_key=r["sender_project_name"],
                        sender_project_slug=r["sender_project_slug"],
                    )
                    read_ts = r["read_ts"]
                    ack_ts = r["ack_ts"]
                    ack_required = bool(r["ack_required"])
                    item = {
                        "id": r["id"],
                        "subject": r["subject"],
                        "sender": sender_display,
                        "created": str(r["created_ts"]),
                        "importance": r["importance"],
                        "thread_id": r["thread_id"],
                        "ack_required": ack_required,
                        "read_ts": str(read_ts) if read_ts else None,
                        "ack_ts": str(ack_ts) if ack_ts else None,
                        "unread": read_ts is None,
                        "needs_ack": ack_required and ack_ts is None,
                        "acked": ack_ts is not None,
                    }
                    item.update(sender_meta)
                    items.append(item)
            return await _render(
                "mail_inbox.html",
                project={"slug": prow[1], "human_key": prow[2]},
                agent=agent,
                items=items,
                page=page,
                limit=limit,
                next_page=page + 1,
                prev_page=page - 1 if page > 1 else None,
            )

        @fastapi_app.get("/mail/{project}/message/{mid}", response_class=HTMLResponse)
        async def mail_message(project: str, mid: int) -> HTMLResponse:
            await ensure_schema()
            async with get_session() as session:
                prow = (
                    await session.execute(
                        text("SELECT id, slug, human_key FROM projects WHERE slug = :k OR human_key = :k"),
                        {"k": project},
                    )
                ).fetchone()
                if not prow:
                    return await _render("error.html", message="Project not found")
                pid = int(prow[0])
                mrow = (
                    await session.execute(
                        text(
                            """
                            SELECT
                                m.id,
                                m.subject,
                                m.body_md,
                                s.name AS sender_name,
                                s.project_id AS sender_project_id,
                                sp.human_key AS sender_project_name,
                                sp.slug AS sender_project_slug,
                                m.created_ts,
                                m.importance,
                                m.thread_id,
                                m.ack_required,
                                m.attachments
                            FROM messages m
                            JOIN agents s ON s.id = m.sender_id
                            LEFT JOIN projects sp ON sp.id = s.project_id
                            WHERE m.project_id = :pid AND m.id = :mid
                            """
                        ),
                        {"pid": pid, "mid": mid},
                    )
                ).mappings().fetchone()
                if not mrow:
                    return await _render("error.html", message="Message not found")
                recs = await session.execute(
                    text(
                        "SELECT a.name, mr.kind, mr.read_ts, mr.ack_ts "
                        "FROM message_recipients mr JOIN agents a ON a.id = mr.agent_id "
                        "WHERE mr.message_id = :mid"
                    ),
                    {"mid": mid},
                )
                recipients = [
                    {
                        "name": r[0],
                        "kind": r[1],
                        "read_ts": str(r[2]) if r[2] else None,
                        "ack_ts": str(r[3]) if r[3] else None,
                    }
                    for r in recs.fetchall()
                ]
                ack_required_msg = bool(mrow["ack_required"])
                ack_count = sum(1 for r in recipients if r["ack_ts"])
                read_count = sum(1 for r in recipients if r["read_ts"])
                ack_summary = {
                    "ack_required": ack_required_msg,
                    "total": len(recipients),
                    "read": read_count,
                    "acked": ack_count,
                }
                # Find thread messages if thread_id is set
                thread_items: list[dict] = []
                th = mrow["thread_id"]
                if isinstance(th, str) and th.strip():
                    th_rows = await session.execute(
                        text(
                            """
                            SELECT
                                m.id,
                                m.subject,
                                s.name AS sender_name,
                                s.project_id AS sender_project_id,
                                sp.human_key AS sender_project_name,
                                sp.slug AS sender_project_slug,
                                m.created_ts
                            FROM messages m
                            JOIN agents s ON s.id = m.sender_id
                            LEFT JOIN projects sp ON sp.id = s.project_id
                            WHERE m.project_id = :pid AND (m.thread_id = :th OR m.id = :id)
                            ORDER BY m.created_ts ASC
                            """
                        ),
                        {"pid": pid, "th": th, "id": mid},
                    )
                    thread_items = []
                    for rr in th_rows.mappings().all():
                        sender_display, sender_meta = _http_sender_identity(
                            message_project_id=pid,
                            sender_name=rr["sender_name"],
                            sender_project_id=rr["sender_project_id"],
                            sender_project_human_key=rr["sender_project_name"],
                            sender_project_slug=rr["sender_project_slug"],
                        )
                        item = {
                            "id": rr["id"],
                            "subject": rr["subject"],
                            "from": sender_display,
                            "created": str(rr["created_ts"]),
                        }
                        item.update(sender_meta)
                        thread_items.append(item)
            # Convert markdown body to HTML for display (server-side render)
            body_html = (
                markdown2.markdown(mrow["body_md"] or "", extras=["fenced-code-blocks", "tables", "strike", "cuddled-lists"])
                if mrow["body_md"]
                else ""
            )
            if body_html:
                body_html = _html_cleaner.clean(body_html)

            # Get commit SHA for provenance badge
            commit_sha = None
            try:
                settings = get_settings()
                archive = await ensure_archive(settings, prow[1])
                commit_sha = await get_message_commit_sha(archive, mid)
            except Exception:
                pass  # Commit SHA is optional

            sender_display, sender_meta = _http_sender_identity(
                message_project_id=pid,
                sender_name=mrow["sender_name"],
                sender_project_id=mrow["sender_project_id"],
                sender_project_human_key=mrow["sender_project_name"],
                sender_project_slug=mrow["sender_project_slug"],
            )
            # Parse persisted attachments so the message view can render/link
            # them (#220). Stored as a JSON array column.
            message_attachments: list[dict[str, Any]] = []
            try:
                raw_attachments = mrow["attachments"]
                if isinstance(raw_attachments, str):
                    try:
                        parsed_attachments = json.loads(raw_attachments)
                    except json.JSONDecodeError:
                        parsed_attachments = []
                else:
                    parsed_attachments = raw_attachments
                if isinstance(parsed_attachments, list):
                    message_attachments = [a for a in parsed_attachments if isinstance(a, dict)]
            except Exception:
                message_attachments = []

            message_payload = {
                "id": mrow["id"],
                "subject": mrow["subject"],
                "body_md": mrow["body_md"],
                "body_html": body_html,
                "sender": sender_display,
                "created": str(mrow["created_ts"]),
                "importance": mrow["importance"],
                "thread_id": mrow["thread_id"],
                "attachments": message_attachments,
            }
            message_payload.update(sender_meta)

            return await _render(
                "mail_message.html",
                project={"slug": prow[1], "human_key": prow[2]},
                message=message_payload,
                recipients=recipients,
                ack_summary=ack_summary,
                thread_items=thread_items,
                commit_sha=commit_sha,
            )

        @fastapi_app.post("/mail/{project}/inbox/{agent}/mark-read")
        async def mark_selected_messages_read(project: str, agent: str, request: Request) -> JSONResponse:
            """Mark specific messages as read for an agent."""
            await ensure_schema()

            try:
                # Parse request body
                request_body = await request.json()
                message_ids: list[int] = request_body.get("message_ids", [])

                if not message_ids:
                    raise HTTPException(status_code=400, detail="No message IDs provided")

                # Limit to prevent SQL parameter overflow (SQLite default limit is 999)
                # Also prevents abuse - if someone wants to mark 1000+ messages, use "mark all"
                if len(message_ids) > 500:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Too many messages selected ({len(message_ids)}). Maximum is 500. Use 'Mark All Read' instead."
                    )

                async with get_session() as session:
                    # Get project
                    prow = (
                        await session.execute(
                            text("SELECT id, slug FROM projects WHERE slug = :k OR human_key = :k"),
                            {"k": project},
                        )
                    ).fetchone()
                    if not prow:
                        raise HTTPException(status_code=404, detail="Project not found")

                    pid = int(prow[0])

                    # Get agent
                    arow = (
                        await session.execute(
                            text("SELECT id FROM agents WHERE project_id = :pid AND name = :name"),
                            {"pid": pid, "name": agent},
                        )
                    ).fetchone()
                    if not arow:
                        raise HTTPException(status_code=404, detail="Agent not found")

                    aid = int(arow[0])

                    # Mark specific messages as read
                    # Use naive UTC datetime for SQLite compatibility
                    now = datetime.now(timezone.utc).replace(tzinfo=None)

                    # Use IN clause with parameter binding
                    placeholders = ','.join([f':mid{i}' for i in range(len(message_ids))])
                    params = {"aid": aid, "now": now}
                    params.update({f"mid{i}": mid for i, mid in enumerate(message_ids)})

                    result = await session.execute(
                        text(
                            f"""
                            UPDATE message_recipients
                            SET read_ts = :now
                            WHERE agent_id = :aid
                            AND message_id IN ({placeholders})
                            AND read_ts IS NULL
                            """
                        ),
                        params,
                    )
                    await session.commit()

                    count = int(getattr(result, "rowcount", 0) or 0)

                    return JSONResponse({
                        "success": True,
                        "marked_count": count,
                        "requested_count": len(message_ids),
                        "agent": agent,
                        "project": prow[1],
                    })

            except HTTPException:
                raise
            except Exception as exc:
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Failed to mark messages as read: {exc!s}") from exc

        @fastapi_app.post("/mail/{project}/inbox/{agent}/mark-all-read")
        async def mark_all_messages_read(project: str, agent: str) -> JSONResponse:
            """Mark all messages for an agent as read."""
            await ensure_schema()

            try:
                async with get_session() as session:
                    # Get project
                    prow = (
                        await session.execute(
                            text("SELECT id, slug FROM projects WHERE slug = :k OR human_key = :k"),
                            {"k": project},
                        )
                    ).fetchone()
                    if not prow:
                        raise HTTPException(status_code=404, detail="Project not found")

                    pid = int(prow[0])

                    # Get agent
                    arow = (
                        await session.execute(
                            text("SELECT id FROM agents WHERE project_id = :pid AND name = :name"),
                            {"pid": pid, "name": agent},
                        )
                    ).fetchone()
                    if not arow:
                        raise HTTPException(status_code=404, detail="Agent not found")

                    aid = int(arow[0])

                    # Mark all unread messages as read
                    # Use naive UTC datetime for SQLite compatibility
                    now = datetime.now(timezone.utc).replace(tzinfo=None)
                    result = await session.execute(
                        text(
                            """
                            UPDATE message_recipients
                            SET read_ts = :now
                            WHERE agent_id = :aid
                            AND read_ts IS NULL
                            """
                        ),
                        {"aid": aid, "now": now},
                    )
                    await session.commit()

                    count = int(getattr(result, "rowcount", 0) or 0)

                    return JSONResponse({
                        "success": True,
                        "marked_count": count,
                        "agent": agent,
                        "project": prow[1],
                    })

            except HTTPException:
                raise
            except Exception as exc:
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Failed to mark messages as read: {exc!s}") from exc

        @fastapi_app.post("/mail/{project}/inbox/{agent}/delete-messages")
        async def delete_selected_messages(project: str, agent: str, request: Request) -> JSONResponse:
            """Permanently delete specific messages for an agent.

            Removes messages from the SQLite database AND deletes the
            corresponding markdown files from the Git archive so that
            messages do not reappear after a refresh or server restart.
            """
            await ensure_schema()

            try:
                request_body = await request.json()
                message_ids: list[int] = request_body.get("message_ids", [])

                if not message_ids:
                    raise HTTPException(status_code=400, detail="No message IDs provided")

                if len(message_ids) > 500:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Too many messages selected ({len(message_ids)}). Maximum is 500."
                    )

                deleted_count = 0
                messages_to_delete: list[tuple[Any, ...]] = []
                recip_map: dict[int, list[str]] = {}
                async with get_session() as session:
                    # Resolve project
                    prow = (
                        await session.execute(
                            text("SELECT id, slug, human_key FROM projects WHERE slug = :k OR human_key = :k"),
                            {"k": project},
                        )
                    ).fetchone()
                    if not prow:
                        raise HTTPException(status_code=404, detail="Project not found")

                    pid = int(prow[0])
                    project_slug = prow[1]

                    # Resolve agent
                    arow = (
                        await session.execute(
                            text("SELECT id FROM agents WHERE project_id = :pid AND name = :name"),
                            {"pid": pid, "name": agent},
                        )
                    ).fetchone()
                    if not arow:
                        raise HTTPException(status_code=404, detail="Agent not found")

                    # Fetch message metadata before deleting so we can locate Git files
                    placeholders = ','.join([f':mid{i}' for i in range(len(message_ids))])
                    id_params: dict[str, Any] = {"pid": pid}
                    id_params.update({f"mid{i}": mid for i, mid in enumerate(message_ids)})

                    rows = await session.execute(
                        text(
                            f"""
                            SELECT m.id, m.created_ts, m.subject, s.name AS sender_name
                            FROM messages m
                            JOIN agents s ON s.id = m.sender_id
                            WHERE m.project_id = :pid
                            AND m.id IN ({placeholders})
                            """
                        ),
                        id_params,
                    )
                    messages_to_delete = [tuple(row) for row in rows.fetchall()]

                    if not messages_to_delete:
                        return JSONResponse({"success": True, "deleted_count": 0})

                    # Collect recipient names per message for inbox path removal
                    recip_rows = await session.execute(
                        text(
                            f"""
                            SELECT mr.message_id, a.name
                            FROM message_recipients mr
                            JOIN agents a ON a.id = mr.agent_id
                            WHERE mr.message_id IN ({placeholders})
                            """
                        ),
                        {f"mid{i}": mid for i, mid in enumerate(message_ids)},
                    )
                    for rr in recip_rows.fetchall():
                        recip_map.setdefault(int(rr[0]), []).append(rr[1])

                    # Delete from SQLite: recipients first, then messages
                    await session.execute(
                        text(
                            f"DELETE FROM message_recipients WHERE message_id IN ({placeholders})"
                        ),
                        {f"mid{i}": mid for i, mid in enumerate(message_ids)},
                    )
                    del_result = await session.execute(
                        text(
                            f"DELETE FROM messages WHERE project_id = :pid AND id IN ({placeholders})"
                        ),
                        id_params,
                    )
                    deleted_count = int(getattr(del_result, "rowcount", 0) or 0)
                    await session.commit()

                settings = get_settings()
                git_files_removed = 0
                try:
                    git_files_removed = await _delete_messages_from_archive(
                        settings=settings,
                        project_slug=project_slug,
                        messages_to_delete=messages_to_delete,
                        recip_map=recip_map,
                        commit_message=f"delete: {deleted_count} message(s) via web UI\n",
                    )
                except Exception as archive_exc:
                    # Archive operations are best-effort; DB deletion already happened.
                    logging.getLogger(__name__).warning(
                        "Git archive cleanup failed: %s", archive_exc
                    )

                return JSONResponse({
                    "success": True,
                    "deleted_count": deleted_count,
                    "git_files_removed": git_files_removed,
                    "agent": agent,
                    "project": project_slug,
                })

            except HTTPException:
                raise
            except Exception as exc:
                import traceback
                traceback.print_exc()
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to delete messages: {exc!s}"
                ) from exc

        @fastapi_app.get("/mail/{project}/thread/{thread_id}", response_class=HTMLResponse)
        async def mail_thread(project: str, thread_id: str) -> HTMLResponse:
            """Display all messages in a thread chronologically (Gmail-style conversation view).

            NOTE: Currently loads ALL messages in thread without pagination.
            For threads with 1000+ messages, consider adding LIMIT/OFFSET pagination.
            """
            await ensure_schema()
            async with get_session() as session:
                # Get project
                prow = (
                    await session.execute(
                        text("SELECT id, slug, human_key FROM projects WHERE slug = :k OR human_key = :k"),
                        {"k": project},
                    )
                ).fetchone()
                if not prow:
                    return await _render("error.html", message="Project not found")

                pid = int(prow[0])

                # Get all messages in this thread, ordered chronologically
                # Include messages where thread_id matches OR message id matches (for thread starter)
                try:
                    thread_id_int = int(thread_id)
                    rows = await session.execute(
                        text(
                            """
                            SELECT
                                m.id,
                                m.subject,
                                m.body_md,
                                s.name AS sender_name,
                                s.project_id AS sender_project_id,
                                sp.human_key AS sender_project_name,
                                sp.slug AS sender_project_slug,
                                m.created_ts,
                                m.importance,
                                m.thread_id
                            FROM messages m
                            JOIN agents s ON s.id = m.sender_id
                            LEFT JOIN projects sp ON sp.id = s.project_id
                            WHERE m.project_id = :pid
                            AND (m.thread_id = :tid OR m.id = :tid_int)
                            ORDER BY m.created_ts ASC
                            """
                        ),
                        {"pid": pid, "tid": thread_id, "tid_int": thread_id_int},
                    )
                except ValueError:
                    # Not an integer, just use string thread_id
                    rows = await session.execute(
                        text(
                            """
                            SELECT
                                m.id,
                                m.subject,
                                m.body_md,
                                s.name AS sender_name,
                                s.project_id AS sender_project_id,
                                sp.human_key AS sender_project_name,
                                sp.slug AS sender_project_slug,
                                m.created_ts,
                                m.importance,
                                m.thread_id
                            FROM messages m
                            JOIN agents s ON s.id = m.sender_id
                            LEFT JOIN projects sp ON sp.id = s.project_id
                            WHERE m.project_id = :pid
                            AND m.thread_id = :tid
                            ORDER BY m.created_ts ASC
                            """
                        ),
                        {"pid": pid, "tid": thread_id},
                    )

                messages = []
                for r in rows.mappings().all():
                    # Convert markdown to HTML for each message
                    body_html = ""
                    if r["body_md"]:
                        body_html = markdown2.markdown(
                            r["body_md"],
                            extras=["fenced-code-blocks", "tables", "strike", "cuddled-lists"]
                        )
                        body_html = _html_cleaner.clean(body_html)

                    sender_display, sender_meta = _http_sender_identity(
                        message_project_id=pid,
                        sender_name=r["sender_name"],
                        sender_project_id=r["sender_project_id"],
                        sender_project_human_key=r["sender_project_name"],
                        sender_project_slug=r["sender_project_slug"],
                    )
                    message = {
                        "id": r["id"],
                        "subject": r["subject"],
                        "body_md": r["body_md"],
                        "body_html": body_html,
                        "sender": sender_display,
                        "created": str(r["created_ts"]),
                        "importance": r["importance"],
                        "thread_id": r["thread_id"],
                    }
                    message.update(sender_meta)
                    messages.append(message)

                if not messages:
                    return await _render(
                        "error.html",
                        message=f"No messages found in thread '{thread_id}'. The thread may not exist or all messages may have been deleted."
                    )

                # Get unique subject (use first message's subject, with fallback)
                thread_subject = messages[0]["subject"] if messages and messages[0]["subject"] else f"Thread {thread_id}"

                return await _render(
                    "mail_thread.html",
                    project={"slug": prow[1], "human_key": prow[2]},
                    thread_id=thread_id,
                    thread_subject=thread_subject,
                    messages=messages,
                    message_count=len(messages),
                )

        # Full-text search UI across subject/body using LIKE fallback (SQLite FTS handled elsewhere)
        @fastapi_app.get("/mail/{project}/search", response_class=HTMLResponse)
        async def mail_search(
            project: str,
            q: str,
            limit: int = 10000,
            scope: str | None = None,
            order: str | None = None,
            boost: int | None = None,
        ) -> HTMLResponse:
            limit = min(max(1, limit), 10000)
            if order not in ("relevance", "time", None):
                order = "relevance"
            await ensure_schema()
            async with get_session() as session:
                prow = (
                    await session.execute(
                        text("SELECT id, slug, human_key FROM projects WHERE slug = :k OR human_key = :k"),
                        {"k": project},
                    )
                ).fetchone()
                if not prow:
                    return await _render("error.html", message="Project not found")
                pid = int(prow[0])
                fts_expr, like_pat, like_scope, tokens = _parse_fts_query(q, scope)
                weights = (0.0, 3.0, 1.0) if (boost or 0) else (0.0, 1.0, 1.0)
                fts_sql = (
                    "SELECT m.id, m.subject, s.name AS sender_name, s.project_id AS sender_project_id, "
                    "sp.human_key AS sender_project_name, sp.slug AS sender_project_slug, "
                    "m.created_ts, m.importance, m.thread_id, "
                    "snippet(fts_messages, 2, '<mark>', '</mark>', '…', 22) AS body_snippet "
                    "FROM fts_messages "
                    "JOIN messages m ON m.id = fts_messages.rowid "
                    "JOIN agents s ON s.id = m.sender_id "
                    "LEFT JOIN projects sp ON sp.id = s.project_id "
                    "WHERE m.project_id = :pid AND fts_messages MATCH :q "
                    + (
                        "ORDER BY m.created_ts DESC "
                        if (order or "relevance") == "time"
                        else f"ORDER BY bm25(fts_messages, {weights[0]}, {weights[1]}, {weights[2]}) "
                    )
                    + "LIMIT :lim"
                )
                try:
                    rows = await session.execute(text(fts_sql), {"pid": pid, "q": fts_expr or q, "lim": limit})
                    results = []
                    for r in rows.mappings().all():
                        sender_display, sender_meta = _http_sender_identity(
                            message_project_id=pid,
                            sender_name=r["sender_name"],
                            sender_project_id=r["sender_project_id"],
                            sender_project_human_key=r["sender_project_name"],
                            sender_project_slug=r["sender_project_slug"],
                        )
                        item = {
                            "id": r["id"],
                            "subject": r["subject"],
                            "from": sender_display,
                            "created": str(r["created_ts"]),
                            "importance": r["importance"],
                            "thread_id": r["thread_id"],
                            "snippet": r["body_snippet"],
                            "hits": (r["body_snippet"] or "").count("<mark>"),
                        }
                        item.update(sender_meta)
                        results.append(item)
                except Exception:
                    if like_scope == "subject":
                        like_sql = (
                            "SELECT m.id, m.subject, s.name AS sender_name, s.project_id AS sender_project_id, "
                            "sp.human_key AS sender_project_name, sp.slug AS sender_project_slug, "
                            "m.created_ts, m.importance, m.thread_id "
                            "FROM messages m JOIN agents s ON s.id = m.sender_id "
                            "LEFT JOIN projects sp ON sp.id = s.project_id "
                            f"WHERE m.project_id = :pid AND m.subject LIKE :pat ESCAPE '{_LIKE_ESCAPE_CHAR}' "
                            "ORDER BY m.created_ts DESC LIMIT :lim"
                        )
                    elif like_scope == "body":
                        like_sql = (
                            "SELECT m.id, m.subject, s.name AS sender_name, s.project_id AS sender_project_id, "
                            "sp.human_key AS sender_project_name, sp.slug AS sender_project_slug, "
                            "m.created_ts, m.importance, m.thread_id "
                            "FROM messages m JOIN agents s ON s.id = m.sender_id "
                            "LEFT JOIN projects sp ON sp.id = s.project_id "
                            f"WHERE m.project_id = :pid AND m.body_md LIKE :pat ESCAPE '{_LIKE_ESCAPE_CHAR}' "
                            "ORDER BY m.created_ts DESC LIMIT :lim"
                        )
                    else:
                        like_sql = (
                            "SELECT m.id, m.subject, s.name AS sender_name, s.project_id AS sender_project_id, "
                            "sp.human_key AS sender_project_name, sp.slug AS sender_project_slug, "
                            "m.created_ts, m.importance, m.thread_id "
                            "FROM messages m JOIN agents s ON s.id = m.sender_id "
                            "LEFT JOIN projects sp ON sp.id = s.project_id "
                            f"WHERE m.project_id = :pid AND (m.subject LIKE :pat ESCAPE '{_LIKE_ESCAPE_CHAR}' "
                            f"OR m.body_md LIKE :pat ESCAPE '{_LIKE_ESCAPE_CHAR}') "
                            "ORDER BY m.created_ts DESC LIMIT :lim"
                        )
                    rows = await session.execute(
                        text(like_sql), {"pid": pid, "pat": like_pat or f"%{_like_escape(q)}%", "lim": limit}
                    )
                    results = []
                    for r in rows.mappings().all():
                        sender_display, sender_meta = _http_sender_identity(
                            message_project_id=pid,
                            sender_name=r["sender_name"],
                            sender_project_id=r["sender_project_id"],
                            sender_project_human_key=r["sender_project_name"],
                            sender_project_slug=r["sender_project_slug"],
                        )
                        item = {
                            "id": r["id"],
                            "subject": r["subject"],
                            "from": sender_display,
                            "created": str(r["created_ts"]),
                            "importance": r["importance"],
                            "thread_id": r["thread_id"],
                            "snippet": "",
                            "hits": 0,
                        }
                        item.update(sender_meta)
                        results.append(item)
            return await _render(
                "mail_search.html",
                project={"slug": prow[1], "human_key": prow[2]},
                q=q,
                scope=scope or "",
                order=order or "relevance",
                tokens=tokens,
                results=results,
                boost=bool(boost),
            )

        # File reservations and attachments views
        @fastapi_app.get("/mail/{project}/file_reservations", response_class=HTMLResponse)
        async def mail_file_reservations(project: str) -> HTMLResponse:
            await ensure_schema()
            async with get_session() as session:
                prow = (
                    await session.execute(
                        text("SELECT id, slug, human_key FROM projects WHERE slug = :k OR human_key = :k"),
                        {"k": project},
                    )
                ).fetchone()
                if not prow:
                    return await _render("error.html", message="Project not found")
                pid = int(prow[0])
                # LEFT JOIN so orphaned reservations whose owning agent row
                # has been deleted still surface in the web UI (`a.name` will
                # be NULL — render as "<orphaned>" so operators see them and
                # can act). Matches the model-side LEFT JOIN in
                # _collect_file_reservation_statuses. (#161)
                rows = await session.execute(
                    text(
                        "SELECT c.id, a.name, c.path_pattern, c.exclusive, c.created_ts, c.expires_ts, c.released_ts, c.agent_id FROM file_reservations c LEFT JOIN agents a ON a.id = c.agent_id WHERE c.project_id = :pid ORDER BY c.created_ts DESC"
                    ),
                    {"pid": pid},
                )
                file_reservations = [
                    {
                        "id": r[0],
                        "agent": r[1] if r[1] is not None else "<orphaned>",
                        "agent_id": r[7],
                        "path_pattern": r[2],
                        "exclusive": bool(r[3]),
                        "created": str(r[4]),
                        "expires": str(r[5]) if r[5] else "",
                        "released": str(r[6]) if r[6] else "",
                    }
                    for r in rows.fetchall()
                ]
            return await _render("mail_file_reservations.html", project={"slug": prow[1], "human_key": prow[2]}, file_reservations=file_reservations)

        @fastapi_app.get("/mail/{project}/attachments", response_class=HTMLResponse)
        async def mail_attachments(project: str) -> HTMLResponse:
            await ensure_schema()
            async with get_session() as session:
                prow = (
                    await session.execute(
                        text("SELECT id, slug, human_key FROM projects WHERE slug = :k OR human_key = :k"),
                        {"k": project},
                    )
                ).fetchone()
                if not prow:
                    return await _render("error.html", message="Project not found")
                pid = int(prow[0])
                rows = await session.execute(
                    text(
                        "SELECT id, subject, created_ts, attachments FROM messages WHERE project_id = :pid AND json_array_length(attachments) > 0 ORDER BY created_ts DESC LIMIT 10000"
                    ),
                    {"pid": pid},
                )
                items = []
                for r in rows.fetchall():
                    attachments: list[dict[str, Any]] = []
                    try:
                        raw = r[3]
                        if isinstance(raw, str):
                            try:
                                parsed = json.loads(raw)
                            except json.JSONDecodeError:
                                parsed = []
                        else:
                            parsed = raw
                        if isinstance(parsed, list):
                            attachments = [a for a in parsed if isinstance(a, dict)]
                    except Exception:
                        attachments = []
                    items.append({"id": r[0], "subject": r[1], "created": str(r[2]), "attachments": attachments})
            return await _render("mail_attachments.html", project={"slug": prow[1], "human_key": prow[2]}, items=items)

        # ========== Human Overseer Routes ==========

        @fastapi_app.get("/mail/{project}/overseer/compose", response_class=HTMLResponse)
        async def overseer_compose(project: str) -> HTMLResponse:
            """Display Human Overseer message composer."""
            await ensure_schema()
            async with get_session() as session:
                # Get project
                prow = (
                    await session.execute(
                        text("SELECT id, slug, human_key FROM projects WHERE slug = :k OR human_key = :k"),
                        {"k": project},
                    )
                ).fetchone()
                if not prow:
                    return await _render("error.html", message="Project not found")

                # Get all agents for this project
                pid = int(prow[0])
                agent_rows = await session.execute(
                    text("SELECT name FROM agents WHERE project_id = :pid ORDER BY name"),
                    {"pid": pid}
                )
                agents = [{"name": r[0]} for r in agent_rows.fetchall()]

            return await _render(
                "overseer_compose.html",
                project={"slug": prow[1], "human_key": prow[2]},
                agents=agents
            )

        @fastapi_app.post("/mail/{project}/overseer/send")
        async def overseer_send(project: str, request: Request) -> JSONResponse:
            """Send message from Human Overseer to selected agents."""
            await ensure_schema()

            try:
                # Parse request body
                request_body = await request.json()
                recipients: list[str] = request_body.get("recipients", [])
                subject: str = request_body.get("subject", "").strip()
                body_md: str = request_body.get("body_md", "").strip()
                thread_id: str | None = request_body.get("thread_id")

                # Comprehensive validation
                if not recipients:
                    raise HTTPException(status_code=400, detail="At least one recipient is required")
                if len(recipients) > 100:
                    raise HTTPException(status_code=400, detail="Too many recipients (maximum 100 agents)")
                if not subject:
                    raise HTTPException(status_code=400, detail="Subject is required")
                if len(subject) > 200:
                    raise HTTPException(status_code=400, detail="Subject too long (maximum 200 characters)")
                if not body_md:
                    raise HTTPException(status_code=400, detail="Message body is required")
                if len(body_md) > 50000:
                    raise HTTPException(status_code=400, detail="Message body too long (maximum 50,000 characters)")

                # Remove duplicate recipients while preserving order
                recipients = list(dict.fromkeys(recipients))

                # Add Human Overseer preamble (pure markdown for cross-renderer compatibility)
                preamble = """---

        🚨 MESSAGE FROM HUMAN OVERSEER 🚨

        This message is from a human operator overseeing this project. Please prioritize the instructions below over your current tasks.

        You should:
        1. Temporarily pause your current work
        2. Complete the request described below
        3. Resume your original plans afterward (unless modified by these instructions)

        The human's guidance supersedes all other priorities.

        ---

        """
                full_body = preamble + body_md

                # Validate combined length (preamble + user message)
                if len(full_body) > 50000:
                    preamble_length = len(preamble)
                    max_user_length = 50000 - preamble_length
                    raise HTTPException(
                        status_code=400,
                        detail=f"Message body too long ({len(body_md)} characters). Maximum is {max_user_length} characters to accommodate the overseer preamble ({preamble_length} characters)."
                    )

                # Keep database work and archive work in separate phases so
                # the request never holds a live DB transaction while doing
                # archive/Git I/O.
                from datetime import datetime, timezone
                message_id: int | None = None
                valid_recipients: list[str] = []
                project_slug = ""
                project_human_key = ""
                overseer_name = "HumanOverseer"
                now = datetime.now(timezone.utc).replace(tzinfo=None)
                async with get_session() as session:
                    # Get project
                    prow = (
                        await session.execute(
                            text("SELECT id, slug, human_key FROM projects WHERE slug = :k OR human_key = :k"),
                            {"k": project},
                        )
                    ).fetchone()
                    if not prow:
                        raise HTTPException(status_code=404, detail="Project not found")

                    # Extract project info consistently
                    project_id = int(prow[0])
                    project_slug = prow[1]
                    project_human_key = prow[2]

                    # Get or create "HumanOverseer" agent (with race condition protection)
                    overseer_row = (
                        await session.execute(
                            text("SELECT id, name FROM agents WHERE project_id = :pid AND name = :name"),
                            {"pid": project_id, "name": overseer_name}
                        )
                    ).fetchone()

                    if not overseer_row:
                        # Create HumanOverseer agent (use INSERT OR IGNORE to handle race conditions)
                        await session.execute(
                            text("""
                                INSERT OR IGNORE INTO agents (
                                    project_id,
                                    name,
                                    program,
                                    model,
                                    task_description,
                                    contact_policy,
                                    attachments_policy,
                                    inception_ts,
                                    last_active_ts
                                )
                                VALUES (
                                    :pid,
                                    :name,
                                    :program,
                                    :model,
                                    :task,
                                    :policy,
                                    :attachments_policy,
                                    :ts,
                                    :ts
                                )
                            """),
                            {
                                "pid": project_id,
                                "name": overseer_name,
                                "program": "WebUI",
                                "model": "Human",
                                "task": "Human operator providing guidance and oversight to agents",
                                "policy": "open",
                                "attachments_policy": "auto",
                                # Use naive UTC datetime for SQLite compatibility
                                "ts": datetime.now(timezone.utc).replace(tzinfo=None),
                            },
                        )
                        # Fetch the agent (whether we just created it or another request did)
                        overseer_row = (
                            await session.execute(
                                text("SELECT id, name FROM agents WHERE project_id = :pid AND name = :name"),
                                {"pid": project_id, "name": overseer_name}
                            )
                        ).fetchone()

                        if not overseer_row:
                            raise HTTPException(status_code=500, detail="Failed to create HumanOverseer agent")

                    # Extract overseer_id for later use
                    overseer_id = overseer_row[0]

                    result = await session.execute(
                        text("""
                            INSERT INTO messages (project_id, sender_id, subject, body_md, importance, thread_id, created_ts, ack_required)
                            VALUES (:pid, :sid, :subj, :body, :imp, :tid, :ts, :ack)
                            RETURNING id
                        """),
                        {
                            "pid": project_id,
                            "sid": overseer_id,
                            "subj": subject,
                            "body": full_body,
                            "imp": "high",  # Always high importance for overseer
                            "tid": thread_id,
                            "ts": now,
                            "ack": False
                        }
                    )
                    message_row = result.fetchone()
                    if not message_row:
                        raise HTTPException(status_code=500, detail="Failed to create message")
                    message_id = message_row[0]

                    # Insert recipients (optimized: bulk SELECT + bulk INSERT instead of N+1 queries)
                    # Build SQL with proper parameter expansion for IN clause
                    placeholders = ", ".join([f":name_{i}" for i in range(len(recipients))])
                    params: dict[str, Any] = {"pid": project_id}
                    params.update({f"name_{i}": name for i, name in enumerate(recipients)})

                    # Single query to get all valid recipient IDs
                    recipient_rows = await session.execute(
                        text(f"SELECT id, name FROM agents WHERE project_id = :pid AND name IN ({placeholders})"),
                        params
                    )
                    recipient_map = {row[1]: row[0] for row in recipient_rows.fetchall()}  # name -> id mapping

                    # Build valid recipients list (only those that exist)
                    valid_recipients = [name for name in recipients if name in recipient_map]

                    # Bulk insert all message_recipients (single executemany call)
                    if valid_recipients:
                        # Prepare bulk insert params
                        insert_params = [
                            {"mid": message_id, "aid": recipient_map[name], "kind": "to"}
                            for name in valid_recipients
                        ]
                        # Use executemany for bulk insert
                        await session.execute(
                            text("""
                                INSERT INTO message_recipients (message_id, agent_id, kind)
                                VALUES (:mid, :aid, :kind)
                            """),
                            insert_params
                        )

                    # If no valid recipients found, rollback and error
                    if not valid_recipients:
                        await session.rollback()
                        raise HTTPException(
                            status_code=400,
                            detail=f"None of the specified recipients exist in this project. Available agents can be seen at /mail/{project_slug}"
                        )

                    # Update HumanOverseer activity timestamp before commit.
                    await session.execute(
                        text("UPDATE agents SET last_active_ts = :ts WHERE id = :id"),
                        {"ts": now, "id": overseer_id}
                    )

                    await session.commit()

                from .storage import ensure_archive, write_message_bundle

                settings = get_settings()
                archive = await ensure_archive(settings, project_slug)
                message_dict = {
                    "id": message_id,
                    "thread_id": thread_id,
                    "project": project_human_key,
                    "project_slug": project_slug,
                    "from": overseer_name,
                    "to": valid_recipients,
                    "cc": [],
                    "bcc": [],
                    "subject": subject,
                    "importance": "high",
                    "ack_required": False,
                    "created": now.isoformat(),
                    "attachments": [],
                }

                try:
                    async with archive_write_lock(archive):
                        await write_message_bundle(
                            archive,
                            message_dict,
                            full_body,
                            overseer_name,
                            valid_recipients,
                            extra_paths=None,
                            commit_text=f"Human Overseer message: {subject}",
                            sender_outbox_name=overseer_name,
                        )
                except Exception as git_error:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to write message to Git archive: {git_error!s}"
                    ) from git_error

                return JSONResponse({
                    "success": True,
                    "message_id": message_id,
                    "recipients": valid_recipients,
                    "sent_at": now.isoformat()
                })

            except HTTPException:
                raise
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Failed to send message: {e!s}") from e

        # ========== Archive Visualization Routes ==========

        def _validate_project_slug(slug: str) -> bool:
            """Validate project slug format to prevent path traversal."""

            # Slugs should only contain lowercase letters, numbers, hyphens, underscores
            # No path separators or relative path components
            if not slug:
                return False
            if slug in (".", "..", "/", "\\"):
                return False
            if "/" in slug or "\\" in slug or ".." in slug:
                return False
            # Should match safe slug pattern
            return bool(_SLUG_VALIDATOR_RE.match(slug))

        @fastapi_app.get("/mail/archive/guide", response_class=HTMLResponse)
        async def archive_guide() -> HTMLResponse:
            """Display the archive access guide and overview."""
            settings = get_settings()
            guide_stats = await asyncio.to_thread(_collect_archive_guide_stats_sync, settings)

            # Get list of projects for picker
            async with get_session() as session:
                rows = await session.execute(text("SELECT slug, human_key FROM projects ORDER BY human_key"))
                projects = [{"slug": r[0], "human_key": r[1]} for r in rows.fetchall()]

            return await _render(
                "archive_guide.html",
                storage_root=guide_stats["storage_root"],
                total_commits=guide_stats["total_commits"],
                project_count=guide_stats["project_count"],
                repo_size=guide_stats["repo_size"],
                last_commit_time=guide_stats["last_commit_time"],
                projects=projects,
            )

        @fastapi_app.get("/mail/archive/activity", response_class=HTMLResponse)
        async def archive_activity(limit: int = 50) -> HTMLResponse:
            """Display recent commits across all projects."""
            # Validate and cap limit to prevent DoS
            limit = max(1, min(limit, 500))  # Between 1 and 500

            settings = get_settings()
            repo_root = await asyncio.to_thread(_expanduser_resolve_path, Path(settings.storage.root))
            if not await asyncio.to_thread(_path_exists, repo_root / ".git"):
                return await _render("archive_activity.html", commits=[])

            repo = None
            try:
                repo = await asyncio.to_thread(_open_git_repo, repo_root)
                commits = await get_recent_commits(repo, limit=limit)
                return await _render("archive_activity.html", commits=commits)
            finally:
                if repo is not None:
                    await asyncio.to_thread(repo.close)

        @fastapi_app.get("/mail/archive/commit/{sha}", response_class=HTMLResponse)
        async def archive_commit(sha: str) -> HTMLResponse:
            """Display detailed commit information with diffs."""
            settings = get_settings()
            repo_root = await asyncio.to_thread(_expanduser_resolve_path, Path(settings.storage.root))
            if not await asyncio.to_thread(_path_exists, repo_root / ".git"):
                return await _render("error.html", message="Archive repository not found")

            repo = None
            try:
                repo = await asyncio.to_thread(_open_git_repo, repo_root)
                commit = await get_commit_detail(repo, sha)
                return await _render("archive_commit.html", commit=commit)
            except ValueError:
                # Validation errors (bad SHA, etc.)
                return await _render("error.html", message="Invalid commit identifier")
            except Exception:
                # Don't leak error details
                return await _render("error.html", message="Commit not found")
            finally:
                if repo is not None:
                    await asyncio.to_thread(repo.close)

        @fastapi_app.get("/mail/archive/timeline", response_class=HTMLResponse)
        async def archive_timeline(project: str | None = None) -> HTMLResponse:
            """Display communication timeline with Mermaid.js visualization."""
            # Validate project slug if provided
            if project and not _validate_project_slug(project):
                return await _render("error.html", message="Invalid project identifier")

            settings = get_settings()
            repo_root = await asyncio.to_thread(_expanduser_resolve_path, Path(settings.storage.root))
            if not await asyncio.to_thread(_path_exists, repo_root / ".git"):
                return await _render("error.html", message="Archive repository not found")

            # Default to first project if not specified
            if not project:
                async with get_session() as session:
                    row = (
                        await session.execute(text("SELECT slug, human_key FROM projects ORDER BY id LIMIT 1"))
                    ).fetchone()
                    if row:
                        project = row[0]
                    else:
                        return await _render("error.html", message="No projects found")

            # Get project name
            project_name = project
            async with get_session() as session:
                row = (
                    await session.execute(text("SELECT human_key FROM projects WHERE slug = :s"), {"s": project})
                ).fetchone()
                if row:
                    project_name = row[0]

            repo = None
            try:
                repo = await asyncio.to_thread(_open_git_repo, repo_root)
                commits = await get_timeline_commits(repo, project, limit=100)
                return await _render("archive_timeline.html", commits=commits, project=project, project_name=project_name)
            finally:
                if repo is not None:
                    await asyncio.to_thread(repo.close)

        @fastapi_app.get("/mail/archive/browser", response_class=HTMLResponse)
        async def archive_browser(project: str | None = None, path: str = "") -> HTMLResponse:
            """Browse archive files and directories."""
            if not project:
                # Show project selector - requires project parameter
                return await _render("error.html", message="Please select a project to browse")

            # Validate project slug
            if not _validate_project_slug(project):
                return await _render("error.html", message="Invalid project identifier")

            settings = get_settings()
            archive = await _open_existing_project_archive(settings, project)
            if archive is None:
                return await _render("error.html", message="Project archive not found")
            try:
                tree = await get_archive_tree(archive, path)
                return await _render("archive_browser.html", tree=tree, project=project, path=path)
            except ValueError:
                return await _render("error.html", message="Invalid archive path")
            finally:
                await asyncio.to_thread(archive.repo.close)

        @fastapi_app.get("/mail/archive/browser/{project}/file")
        async def archive_browser_file(project: str, path: str) -> JSONResponse:
            """Get file content from archive."""
            # Validate project slug
            if not _validate_project_slug(project):
                raise HTTPException(status_code=400, detail="Invalid project identifier")

            try:
                settings = get_settings()
                archive = await _open_existing_project_archive(settings, project)
                if archive is None:
                    raise HTTPException(status_code=404, detail="Project archive not found")
                try:
                    content = await get_file_content(archive, path)
                finally:
                    await asyncio.to_thread(archive.repo.close)

                if content is None:
                    raise HTTPException(status_code=404, detail="File not found")

                return JSONResponse(content=content)
            except ValueError as err:
                # Path validation errors
                raise HTTPException(status_code=400, detail="Invalid file path") from err
            except HTTPException:
                raise
            except Exception as err:
                raise HTTPException(status_code=404, detail="File not found") from err

        @fastapi_app.get("/mail/archive/browser/{project}/download")
        async def archive_browser_download(project: str, path: str) -> Response:
            """Download a file from the archive as an attachment (#221)."""
            # Validate project slug
            if not _validate_project_slug(project):
                raise HTTPException(status_code=400, detail="Invalid project identifier")

            try:
                settings = get_settings()
                archive = await _open_existing_project_archive(settings, project)
                if archive is None:
                    raise HTTPException(status_code=404, detail="Project archive not found")
                try:
                    content = await get_file_content(archive, path)
                finally:
                    await asyncio.to_thread(archive.repo.close)

                if content is None:
                    raise HTTPException(status_code=404, detail="File not found")

                # Derive a safe download filename from the (already validated)
                # path's basename; strip any directory components and quotes.
                filename = PurePosixPath(path.replace("\\", "/")).name or "download"
                filename = filename.replace('"', "").replace("\r", "").replace("\n", "")
                return Response(
                    content=content,
                    media_type="application/octet-stream",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'},
                )
            except ValueError as err:
                # Path validation errors
                raise HTTPException(status_code=400, detail="Invalid file path") from err
            except HTTPException:
                raise
            except Exception as err:
                raise HTTPException(status_code=404, detail="File not found") from err

        @fastapi_app.get("/mail/archive/network", response_class=HTMLResponse)
        async def archive_network(project: str | None = None) -> HTMLResponse:
            """Display agent communication network graph."""
            # Validate project slug if provided
            if project and not _validate_project_slug(project):
                return await _render("error.html", message="Invalid project identifier")

            settings = get_settings()
            repo_root = await asyncio.to_thread(_expanduser_resolve_path, Path(settings.storage.root))
            if not await asyncio.to_thread(_path_exists, repo_root / ".git"):
                return await _render("error.html", message="Archive repository not found")

            # Default to first project
            if not project:
                async with get_session() as session:
                    row = (
                        await session.execute(text("SELECT slug, human_key FROM projects ORDER BY id LIMIT 1"))
                    ).fetchone()
                    if row:
                        project = row[0]
                    else:
                        return await _render("error.html", message="No projects found")

            # Get project name
            project_name = project
            async with get_session() as session:
                row = (
                    await session.execute(text("SELECT human_key FROM projects WHERE slug = :s"), {"s": project})
                ).fetchone()
                if row:
                    project_name = row[0]

            repo = None
            try:
                repo = await asyncio.to_thread(_open_git_repo, repo_root)
                graph = await get_agent_communication_graph(repo, project, limit=200)
                return await _render("archive_network.html", graph=graph, project=project, project_name=project_name)
            finally:
                if repo is not None:
                    await asyncio.to_thread(repo.close)

        @fastapi_app.get("/mail/api/projects/{project}/agents")
        async def api_project_agents(project: str) -> JSONResponse:
            """Get list of agents for a project."""
            # Validate project slug
            if not _validate_project_slug(project):
                raise HTTPException(status_code=400, detail="Invalid project identifier")

            async with get_session() as session:
                # Get project ID
                proj_result = await session.execute(
                    text("SELECT id FROM projects WHERE slug = :k OR human_key = :k"),
                    {"k": project}
                )
                prow = proj_result.fetchone()
                if not prow:
                    raise HTTPException(status_code=404, detail="Project not found")

                # Get agents for this project
                agents_result = await session.execute(
                    text("SELECT name FROM agents WHERE project_id = :pid ORDER BY name"),
                    {"pid": prow[0]}
                )
                agents = [r[0] for r in agents_result.fetchall()]

            return JSONResponse({"agents": agents})

        @fastapi_app.get("/mail/archive/time-travel", response_class=HTMLResponse)
        async def archive_time_travel() -> HTMLResponse:
            """Display time-travel interface."""
            # Get all projects
            async with get_session() as session:
                rows = await session.execute(text("SELECT slug FROM projects ORDER BY human_key"))
                projects = [r[0] for r in rows.fetchall()]

            return await _render("archive_time_travel.html", projects=projects)

        @fastapi_app.get("/mail/archive/time-travel/snapshot")
        async def archive_time_travel_snapshot(project: str, agent: str, timestamp: str) -> JSONResponse:
            """Get historical inbox snapshot."""
            # Validate project slug
            if not _validate_project_slug(project):
                raise HTTPException(status_code=400, detail="Invalid project identifier")

            # Validate agent name (alphanumeric only)
            if not agent or not _AGENT_NAME_VALIDATOR_RE.match(agent):
                raise HTTPException(status_code=400, detail="Invalid agent name format")

            # Validate timestamp format (basic ISO 8601 check)
            if not timestamp or not _TIMESTAMP_VALIDATOR_RE.match(timestamp):
                raise HTTPException(status_code=400, detail="Invalid timestamp format. Use ISO 8601 format (YYYY-MM-DDTHH:MM)")

            try:
                # Get project archive
                settings = get_settings()
                repo = await _open_existing_project_archive(settings, project)
                if repo is None:
                    return JSONResponse({
                        "messages": [],
                        "snapshot_time": None,
                        "commit_sha": None,
                        "requested_time": timestamp,
                        "error": "Project archive not found",
                    })

                try:
                    # Get historical snapshot
                    snapshot = await get_historical_inbox_snapshot(repo, agent, timestamp, limit=200)
                    return JSONResponse(snapshot)
                finally:
                    await asyncio.to_thread(repo.repo.close)

            except Exception as e:
                # Log error but return empty result rather than failing
                structlog.get_logger("archive").warning(
                    "time_travel_failed",
                    project=project,
                    agent=agent,
                    timestamp=timestamp,
                    error=str(e)
                )
                return JSONResponse({
                    "messages": [],
                    "snapshot_time": None,
                    "commit_sha": None,
                    "requested_time": timestamp,
                    "error": f"Unable to retrieve historical snapshot: {e!s}"
                })


    try:
        _register_mail_ui()
    except Exception as exc:
        # templates/Jinja may be missing in some environments; UI remains optional
        with contextlib.suppress(Exception):
            structlog.get_logger("ui").error("ui_init_failed", error=str(exc))
        pass

    # Keep the auto-generated /openapi.json focused on the real API contract.
    # The browser-facing SSR mail UI (and its UI-backing JSON helpers) all live
    # under the `/mail` prefix; they are registered for humans, not as part of
    # the documented API surface, so we filter them out of the schema. The
    # routes stay fully registered and functional — they are only omitted from
    # the OpenAPI document. Using a custom app.openapi() (rather than
    # include_in_schema=False on ~33 decorators) keeps this in one place and
    # automatically covers any future /mail/* routes.
    from fastapi.openapi.utils import get_openapi as _get_openapi

    def _custom_openapi() -> dict[str, Any]:
        if fastapi_app.openapi_schema:
            return fastapi_app.openapi_schema
        schema = _get_openapi(
            title=fastapi_app.title,
            version=fastapi_app.version,
            openapi_version=fastapi_app.openapi_version,
            description=fastapi_app.description,
            routes=fastapi_app.routes,
        )
        paths = schema.get("paths")
        if isinstance(paths, dict):
            schema["paths"] = {
                path: item
                for path, item in paths.items()
                if not (path == "/mail" or path.startswith("/mail/"))
            }
        fastapi_app.openapi_schema = schema
        return schema

    # Install the custom generator (FastAPI's documented extension point for
    # overriding the OpenAPI document); cast keeps the bound-method override
    # explicit for the type checker.
    cast(Any, fastapi_app).openapi = _custom_openapi

    # Static web UI (SPA) routing support
    def _resolve_web_root() -> Path | None:
        candidates: list[Path] = []
        with contextlib.suppress(Exception):
            candidates.append(Path(__file__).resolve().parents[3] / "web")
        candidates.append(Path.cwd() / "web")
        for candidate in candidates:
            try:
                if candidate.exists() and (candidate / "index.html").exists():
                    return candidate
            except Exception:
                continue
        return None

    web_root = _resolve_web_root()
    if web_root is not None:
        fastapi_app.mount("/", StaticFiles(directory=str(web_root), html=True), name="web")

        def _is_api_path(path: str) -> bool:
            if base_no_slash == "/":
                return True
            return path == base_no_slash or path.startswith(base_no_slash + "/")

        def _should_spa_fallback(path: str) -> bool:
            if _is_api_path(path):
                return False
            # JSON API areas must keep real 404s for API clients; the SPA
            # fallback is only for browser page loads.
            if path == "/hub/api" or path.startswith("/hub/api/"):
                return False
            return not (path == "/mail" or path.startswith("/mail/"))

        @fastapi_app.exception_handler(HTTPException)
        async def spa_fallback(request: Request, exc: HTTPException):
            if exc.status_code == status.HTTP_404_NOT_FOUND and _should_spa_fallback(request.url.path):
                return FileResponse(web_root / "index.html")
            return await http_exception_handler(request, exc)

    return fastapi_app


def main() -> None:
    """Run the HTTP transport using settings-specified host/port."""

    parser = argparse.ArgumentParser(description="Run the MCP Agent Mail HTTP transport")
    parser.add_argument("--host", help="Override HTTP host", default=None)
    parser.add_argument("--port", help="Override HTTP port", type=int, default=None)
    parser.add_argument("--log-level", help="Uvicorn log level", default="info")
    # Be tolerant of extraneous argv when invoked under test runners
    args, _unknown = parser.parse_known_args()

    settings = get_settings()
    host = args.host or settings.http.host
    port = args.port or settings.http.port

    app = build_http_app(settings)
    # Disable WebSockets when running the service directly; HTTP-only transport
    import inspect as _inspect

    _sig = _inspect.signature(uvicorn.run)
    _kwargs: dict[str, Any] = {"host": host, "port": port, "log_level": args.log_level}
    if "ws" in _sig.parameters:
        _kwargs["ws"] = "none"
    uvicorn.run(app, **_kwargs)


if __name__ == "__main__":  # pragma: no cover - manual execution path
    main()
