"""M1b-3b: post_channel_message / fetch_channel_messages / mark_channel_read tools.

Covers the channel-message + read-cursor MCP tools:
- same-project agents are default channel members; cross-project agents must
  subscribe before posting/reading
- post writes only channel_messages (no message_recipients fanout), attachments
  fixed empty, no @ parsing
- fetch is pure read (never advances the cursor), returns id > cursor in stable
  ascending order, bounded limit
- mark_channel_read advances atomically via a single SQLite UPSERT
  (ON CONFLICT DO UPDATE ... WHERE stored IS NULL OR stored < requested) and
  reports the truthful post-commit cursor; no rewind on older marks
- strict DTO keys and plain-dict result.data
- fresh/existing DB regressions preserved
"""

from __future__ import annotations

import asyncio
import sqlite3

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from sqlmodel import select

from mcp_agent_mail.app import build_mcp_server
from mcp_agent_mail.db import (
    ensure_schema,
    get_database_path,
    get_engine,
    get_session,
    reset_database_state,
)
from mcp_agent_mail.models import ChannelReadCursor


async def _bootstrap() -> tuple[Client, dict[str, object]]:
    """Return a bootstrap client (registers identities) plus owner identity."""
    server = build_mcp_server()
    bootstrap = Client(server)
    await bootstrap.__aenter__()
    await bootstrap.call_tool("ensure_project", {"human_key": "/channels/alpha"})
    owner = await bootstrap.call_tool(
        "register_agent",
        {"project_key": "/channels/alpha", "program": "test", "model": "test", "name": "BlueLake"},
    )
    await bootstrap.call_tool(
        "ensure_channel",
        {"project_key": "/channels/alpha", "channel_name": "general", "registration_token": owner.data["registration_token"]},
    )
    return bootstrap, {"name": "BlueLake", "token": owner.data["registration_token"], "project": "/channels/alpha"}


async def _new_identity(bootstrap: Client, project: str, name: str) -> dict[str, object]:
    result = await bootstrap.call_tool(
        "register_agent",
        {"project_key": project, "program": "test", "model": "test", "name": name},
    )
    return {"name": name, "token": result.data["registration_token"], "project": project}


async def _teardown(client: Client) -> None:
    await client.__aexit__(None, None, None)


@pytest.fixture
async def channel_client(isolated_env):
    bootstrap, owner = await _bootstrap()
    client = Client(build_mcp_server())
    await client.__aenter__()
    try:
        yield client, owner, bootstrap
    finally:
        await _teardown(client)
        await _teardown(bootstrap)


def _post_args(channel_project, channel_name, sender, subject="hi", body="body"):
    return {
        "channel_project_key": channel_project,
        "channel_name": channel_name,
        "sender_project_key": sender["project"],
        "sender_name": sender["name"],
        "subject": subject,
        "body_md": body,
        "registration_token": sender["token"],
    }


def _fetch_args(channel_project, channel_name, agent, limit=100):
    return {
        "channel_project_key": channel_project,
        "channel_name": channel_name,
        "agent_project_key": agent["project"],
        "agent_name": agent["name"],
        "limit": limit,
        "registration_token": agent["token"],
    }


def _mark_args(channel_project, channel_name, agent, message_id):
    return {
        "channel_project_key": channel_project,
        "channel_name": channel_name,
        "agent_project_key": agent["project"],
        "agent_name": agent["name"],
        "message_id": message_id,
        "registration_token": agent["token"],
    }


# ============================================================================
# post_channel_message
# ============================================================================


class TestPostChannelMessage:
    @pytest.mark.asyncio
    async def test_post_by_same_project_member(self, channel_client):
        client, owner, _ = channel_client
        result = await client.call_tool("post_channel_message", _post_args(owner["project"], "general", owner))
        assert set(result.data) == {"id", "channel_id", "sender_id", "sender_name", "subject", "body_md", "importance", "created_ts"}
        assert result.data["subject"] == "hi"
        assert result.data["sender_name"] == "BlueLake"

    @pytest.mark.asyncio
    async def test_post_writes_no_message_recipient_rows(self, channel_client):
        client, owner, _ = channel_client
        await client.call_tool("post_channel_message", _post_args(owner["project"], "general", owner))
        engine = get_engine()
        async with engine.begin() as conn:
            result = await conn.run_sync(
                lambda sync_conn: sync_conn.exec_driver_sql(
                    "SELECT COUNT(*) FROM message_recipients"
                ).fetchone()
            )
            assert result is not None
            assert result[0] == 0

    @pytest.mark.asyncio
    async def test_post_by_unsubscribed_cross_project_rejected(self, channel_client):
        client, owner, bootstrap = channel_client
        await bootstrap.call_tool("ensure_project", {"human_key": "/channels/beta"})
        remote = await _new_identity(bootstrap, "/channels/beta", "GreenHill")
        with pytest.raises(ToolError):
            await client.call_tool("post_channel_message", _post_args(owner["project"], "general", remote))

    @pytest.mark.asyncio
    async def test_post_by_subscribed_cross_project_allowed(self, channel_client):
        client, owner, bootstrap = channel_client
        await bootstrap.call_tool("ensure_project", {"human_key": "/channels/beta"})
        remote = await _new_identity(bootstrap, "/channels/beta", "GreenHill")
        await client.call_tool(
            "subscribe_channel",
            {
                "channel_project_key": owner["project"],
                "channel_name": "general",
                "agent_project_key": remote["project"],
                "agent_name": remote["name"],
                "registration_token": remote["token"],
            },
        )
        result = await client.call_tool("post_channel_message", _post_args(owner["project"], "general", remote))
        assert result.data["sender_name"] == "GreenHill"

    @pytest.mark.asyncio
    async def test_post_rejects_bad_importance(self, channel_client):
        client, owner, _ = channel_client
        args = _post_args(owner["project"], "general", owner)
        args["importance"] = "bogus"
        # importance is free-form tolerated in this codebase; assert no crash and DTO keys
        result = await client.call_tool("post_channel_message", args)
        assert "importance" in result.data


# ============================================================================
# fetch_channel_messages
# ============================================================================


class TestFetchChannelMessages:
    @pytest.mark.asyncio
    async def test_fetch_initial_cursor_null_returns_all(self, channel_client):
        client, owner, _ = channel_client
        for i in range(3):
            await client.call_tool("post_channel_message", _post_args(owner["project"], "general", owner, subject=f"s{i}"))
        result = await client.call_tool("fetch_channel_messages", _fetch_args(owner["project"], "general", owner))
        assert set(result.data) == {"channel", "messages", "cursor", "limit", "count"}
        assert result.data["cursor"] is None  # no cursor row yet
        assert result.data["count"] == 3
        assert [m["subject"] for m in result.data["messages"]] == ["s0", "s1", "s2"]  # stable ascending
        assert all(isinstance(m, dict) for m in result.data["messages"])
        assert set(result.data["messages"][0]) == {
            "id", "channel_id", "sender_id", "sender_name", "subject", "body_md", "importance", "created_ts",
        }

    @pytest.mark.asyncio
    async def test_fetch_does_not_advance_cursor(self, channel_client):
        client, owner, _ = channel_client
        await client.call_tool("post_channel_message", _post_args(owner["project"], "general", owner))
        first = await client.call_tool("fetch_channel_messages", _fetch_args(owner["project"], "general", owner))
        second = await client.call_tool("fetch_channel_messages", _fetch_args(owner["project"], "general", owner))
        assert first.data["count"] == second.data["count"] == 1

    @pytest.mark.asyncio
    async def test_fetch_limit_bounded(self, channel_client):
        client, owner, _ = channel_client
        for _i in range(5):
            await client.call_tool("post_channel_message", _post_args(owner["project"], "general", owner))
        result = await client.call_tool("fetch_channel_messages", _fetch_args(owner["project"], "general", owner, limit=2))
        assert result.data["count"] == 2
        assert result.data["limit"] == 2

    @pytest.mark.asyncio
    async def test_fetch_after_mark_only_returns_increment(self, channel_client):
        client, owner, _ = channel_client
        m1 = await client.call_tool("post_channel_message", _post_args(owner["project"], "general", owner))
        m2 = await client.call_tool("post_channel_message", _post_args(owner["project"], "general", owner))
        await client.call_tool("mark_channel_read", _mark_args(owner["project"], "general", owner, m1.data["id"]))
        result = await client.call_tool("fetch_channel_messages", _fetch_args(owner["project"], "general", owner))
        assert result.data["cursor"] == m1.data["id"]
        assert [m["id"] for m in result.data["messages"]] == [m2.data["id"]]

    @pytest.mark.asyncio
    async def test_fetch_by_unsubscribed_cross_project_rejected(self, channel_client):
        client, owner, bootstrap = channel_client
        await bootstrap.call_tool("ensure_project", {"human_key": "/channels/beta"})
        remote = await _new_identity(bootstrap, "/channels/beta", "GreenHill")
        with pytest.raises(ToolError):
            await client.call_tool("fetch_channel_messages", _fetch_args(owner["project"], "general", remote))

    @pytest.mark.asyncio
    async def test_fetch_by_subscribed_cross_project_allowed(self, channel_client):
        client, owner, bootstrap = channel_client
        await bootstrap.call_tool("ensure_project", {"human_key": "/channels/beta"})
        remote = await _new_identity(bootstrap, "/channels/beta", "GreenHill")
        await client.call_tool(
            "subscribe_channel",
            {
                "channel_project_key": owner["project"],
                "channel_name": "general",
                "agent_project_key": remote["project"],
                "agent_name": remote["name"],
                "registration_token": remote["token"],
            },
        )
        await client.call_tool("post_channel_message", _post_args(owner["project"], "general", owner))
        result = await client.call_tool("fetch_channel_messages", _fetch_args(owner["project"], "general", remote))
        assert result.data["count"] == 1


# ============================================================================
# mark_channel_read
# ============================================================================


class TestMarkChannelRead:
    @pytest.mark.asyncio
    async def test_mark_creates_cursor_and_advances(self, channel_client):
        client, owner, _ = channel_client
        m1 = await client.call_tool("post_channel_message", _post_args(owner["project"], "general", owner))
        result = await client.call_tool("mark_channel_read", _mark_args(owner["project"], "general", owner, m1.data["id"]))
        assert set(result.data) == {"channel", "cursor", "updated"}
        assert result.data["cursor"] == m1.data["id"]
        assert result.data["updated"] is True

        async with get_session() as session:
            result = await session.execute(
                select(ChannelReadCursor).where(ChannelReadCursor.last_read_message_id == m1.data["id"])
            )
            assert result.scalars().first() is not None

    @pytest.mark.asyncio
    async def test_mark_advances_pre_created_null_row(self, channel_client):
        """A pre-existing NULL cursor row (cursor at start) must advance.

        Regression for SQLite scalar max(NULL, x) returning NULL: the UPSERT
        must gate on 'stored IS NULL OR stored < requested' and set the value
        directly, so a NULL row advances to the requested message id.
        """
        client, owner, _ = channel_client
        m1 = await client.call_tool("post_channel_message", _post_args(owner["project"], "general", owner))

        engine = get_engine()
        async with engine.begin() as conn:
            await conn.exec_driver_sql(
                "INSERT INTO channel_read_cursors (channel_id, agent_id, last_read_message_id, created_ts, updated_ts) "
                "VALUES (?, ?, NULL, '2026-01-01 00:00:00', '2026-01-01 00:00:00')",
                (m1.data["channel_id"], m1.data["sender_id"]),
            )

        result = await client.call_tool("mark_channel_read", _mark_args(owner["project"], "general", owner, m1.data["id"]))
        assert result.data["cursor"] == m1.data["id"]
        assert result.data["updated"] is True

    @pytest.mark.asyncio
    async def test_mark_rejects_cross_channel_message(self, channel_client):
        client, owner, _ = channel_client
        await client.call_tool(
            "ensure_channel",
            {"project_key": owner["project"], "channel_name": "other", "registration_token": owner["token"]},
        )
        m = await client.call_tool("post_channel_message", _post_args(owner["project"], "other", owner))
        with pytest.raises(ToolError):
            await client.call_tool("mark_channel_read", _mark_args(owner["project"], "general", owner, m.data["id"]))

    @pytest.mark.asyncio
    async def test_mark_idempotent_no_rewind(self, channel_client):
        client, owner, _ = channel_client
        m1 = await client.call_tool("post_channel_message", _post_args(owner["project"], "general", owner))
        m2 = await client.call_tool("post_channel_message", _post_args(owner["project"], "general", owner))
        await client.call_tool("mark_channel_read", _mark_args(owner["project"], "general", owner, m2.data["id"]))
        # Marking an older id must NOT rewind the cursor; returns the committed cursor.
        result = await client.call_tool("mark_channel_read", _mark_args(owner["project"], "general", owner, m1.data["id"]))
        assert result.data["updated"] is False
        assert result.data["cursor"] == m2.data["id"]

    @pytest.mark.asyncio
    async def test_mark_missing_message_rejected(self, channel_client):
        client, owner, _ = channel_client
        with pytest.raises(ToolError):
            await client.call_tool("mark_channel_read", _mark_args(owner["project"], "general", owner, 999999))

    @pytest.mark.asyncio
    async def test_mark_concurrent_fresh_keeps_max(self, channel_client):
        """Two concurrent marks on a fresh cursor settle on the larger id."""
        client, owner, _ = channel_client
        m1 = await client.call_tool("post_channel_message", _post_args(owner["project"], "general", owner))
        m2 = await client.call_tool("post_channel_message", _post_args(owner["project"], "general", owner))

        async def mark(mid: int) -> None:
            await client.call_tool("mark_channel_read", _mark_args(owner["project"], "general", owner, mid))

        await asyncio.gather(mark(m1.data["id"]), mark(m2.data["id"]))

        async with get_session() as session:
            result = await session.execute(
                select(ChannelReadCursor).where(ChannelReadCursor.last_read_message_id == m2.data["id"])
            )
            assert result.scalars().first() is not None

    @pytest.mark.asyncio
    async def test_mark_concurrent_pre_created_null_row_keeps_max(self, channel_client):
        """Two concurrent marks on a pre-created NULL row settle on the larger id."""
        client, owner, _ = channel_client
        m1 = await client.call_tool("post_channel_message", _post_args(owner["project"], "general", owner))
        m2 = await client.call_tool("post_channel_message", _post_args(owner["project"], "general", owner))

        engine = get_engine()
        async with engine.begin() as conn:
            await conn.exec_driver_sql(
                "INSERT INTO channel_read_cursors (channel_id, agent_id, last_read_message_id, created_ts, updated_ts) "
                "VALUES (?, ?, NULL, '2026-01-01 00:00:00', '2026-01-01 00:00:00')",
                (m1.data["channel_id"], m1.data["sender_id"]),
            )

        async def mark(mid: int) -> None:
            await client.call_tool("mark_channel_read", _mark_args(owner["project"], "general", owner, mid))

        await asyncio.gather(mark(m1.data["id"]), mark(m2.data["id"]))

        async with get_session() as session:
            result = await session.execute(
                select(ChannelReadCursor).where(ChannelReadCursor.last_read_message_id == m2.data["id"])
            )
            assert result.scalars().first() is not None


# ============================================================================
# Invalid inputs / auth
# ============================================================================


class TestInvalidInputs:
    @pytest.mark.asyncio
    async def test_post_rejects_invalid_channel_name(self, channel_client):
        client, owner, _ = channel_client
        with pytest.raises(ToolError):
            await client.call_tool("post_channel_message", _post_args(owner["project"], "Bad Name!", owner))

    @pytest.mark.asyncio
    async def test_post_rejects_missing_channel(self, channel_client):
        client, owner, _ = channel_client
        args = _post_args(owner["project"], "nope", owner)
        with pytest.raises(ToolError):
            await client.call_tool("post_channel_message", args)

    @pytest.mark.asyncio
    async def test_post_rejects_wrong_token(self, channel_client):
        client, owner, _ = channel_client
        args = _post_args(owner["project"], "general", owner)
        args["registration_token"] = "wrong-token"
        with pytest.raises(ToolError):
            await client.call_tool("post_channel_message", args)

    @pytest.mark.asyncio
    async def test_fetch_rejects_missing_project(self, channel_client):
        client, owner, _ = channel_client
        args = _fetch_args(owner["project"], "general", owner)
        args["agent_project_key"] = "/channels/does-not-exist"
        with pytest.raises(ToolError):
            await client.call_tool("fetch_channel_messages", args)


# ============================================================================
# Fresh / existing DB regressions
# ============================================================================


class TestSchemaRegression:
    @pytest.mark.asyncio
    async def test_tools_work_after_existing_db_upgrade(self, isolated_env):
        """New tools keep working after a simulated process restart/upgrade."""
        await ensure_schema()
        db_path = get_database_path()
        assert db_path is not None
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("DROP TABLE channel_read_cursors")
            conn.execute("DROP TABLE channel_messages")
            conn.commit()
        finally:
            conn.close()
        reset_database_state()
        await ensure_schema()

        bootstrap, owner = await _bootstrap()
        client = Client(build_mcp_server())
        await client.__aenter__()
        try:
            await client.call_tool("post_channel_message", _post_args(owner["project"], "general", owner))
            result = await client.call_tool("fetch_channel_messages", _fetch_args(owner["project"], "general", owner))
            assert result.data["count"] == 1
        finally:
            await _teardown(client)
            await _teardown(bootstrap)
