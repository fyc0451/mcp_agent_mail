"""M1b channel data layer and server tool tests.

Covers the project-scoped channel and cross-project subscription data layer:
- channel name unique per project (same name allowed across projects)
- subscriptions allow same-project and cross-project agents
- (channel_id, agent_id) unique
- DDL FKs reject orphans once enforcement is enabled
- idempotent schema init and non-regressing upgrade on an existing DB
- strict channel/subscription DTOs, authentication, and invalid inputs
"""

from __future__ import annotations

import asyncio
import sqlite3

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from sqlalchemy.exc import IntegrityError
from sqlmodel import select

from mcp_agent_mail.app import build_mcp_server
from mcp_agent_mail.db import (
    ensure_schema,
    get_database_path,
    get_engine,
    get_session,
    reset_database_state,
)
from mcp_agent_mail.models import Agent, Channel, ChannelSubscription, Project


async def _new_project(name: str) -> Project:
    async with get_session() as session:
        project = Project(slug=f"slug-{name}", human_key=f"/tmp/{name}")
        session.add(project)
        await session.commit()
        await session.refresh(project)
        return project


async def _new_agent(project: Project, name: str) -> Agent:
    async with get_session() as session:
        agent = Agent(project_id=project.id, name=name, program="test", model="test")
        session.add(agent)
        await session.commit()
        await session.refresh(agent)
        return agent


async def _new_channel(project: Project, name: str) -> Channel:
    async with get_session() as session:
        channel = Channel(project_id=project.id, name=name)
        session.add(channel)
        await session.commit()
        await session.refresh(channel)
        return channel


async def _subscribe(channel: Channel, agent: Agent) -> None:
    async with get_session() as session:
        session.add(ChannelSubscription(channel_id=channel.id, agent_id=agent.id))
        await session.commit()


async def _table_names() -> list[str]:
    engine = get_engine()
    async with engine.begin() as conn:
        result = await conn.run_sync(
            lambda sync_conn: sync_conn.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        )
        return [row[0] for row in result]


# ============================================================================
# Schema / FK metadata + idempotency
# ============================================================================


class TestSchema:
    @pytest.mark.asyncio
    async def test_ensure_schema_creates_channel_tables(self, isolated_env):
        await ensure_schema()
        names = await _table_names()
        assert "channels" in names
        assert "channel_subscriptions" in names

    @pytest.mark.asyncio
    async def test_fk_columns_target_existing_tables(self, isolated_env):
        """DDL FKs point at the real projects/channels/agents tables.

        The application engine leaves SQLite's foreign_keys pragma off by
        default (matching the existing schema); these are DDL metadata checks.
        """
        await ensure_schema()
        engine = get_engine()
        async with engine.begin() as conn:
            result = await conn.run_sync(
                lambda sync_conn: sync_conn.exec_driver_sql(
                    "SELECT \"from\", \"table\" FROM pragma_foreign_key_list('channels')"
                ).fetchall()
            )
            channel_fk = {row[1] for row in result}
            result = await conn.run_sync(
                lambda sync_conn: sync_conn.exec_driver_sql(
                    "SELECT \"from\", \"table\" FROM pragma_foreign_key_list('channel_subscriptions') "
                    "ORDER BY id"
                ).fetchall()
            )
            sub_fk = {(row[0], row[1]) for row in result}
        assert channel_fk == {"projects"}
        assert ("channel_id", "channels") in sub_fk
        assert ("agent_id", "agents") in sub_fk

    @pytest.mark.asyncio
    async def test_ensure_schema_idempotent(self, isolated_env):
        await ensure_schema()
        p1 = await _new_project("a")
        await _new_channel(p1, "general")
        await ensure_schema()
        await ensure_schema()
        names = await _table_names()
        assert "channels" in names
        assert "channel_subscriptions" in names


# ============================================================================
# Channel uniqueness (project-scoped) + subscription semantics
# ============================================================================


class TestChannelAndSubscriptions:
    @pytest.mark.asyncio
    async def test_channel_name_unique_per_project(self, isolated_env):
        await ensure_schema()
        p1 = await _new_project("a")
        p2 = await _new_project("b")
        await _new_channel(p1, "general")
        await _new_channel(p2, "general")  # same name, different project: OK
        with pytest.raises(IntegrityError):
            await _new_channel(p1, "general")  # duplicate within project: fails

    @pytest.mark.asyncio
    async def test_subscribe_same_or_other_project(self, isolated_env):
        await ensure_schema()
        p1 = await _new_project("a")
        p2 = await _new_project("b")
        local = await _new_agent(p1, "alice")
        remote = await _new_agent(p2, "bob")
        channel = await _new_channel(p1, "general")

        await _subscribe(channel, local)
        await _subscribe(channel, remote)  # cross-project subscription: OK

    @pytest.mark.asyncio
    async def test_duplicate_subscription_fails(self, isolated_env):
        await ensure_schema()
        p1 = await _new_project("a")
        agent = await _new_agent(p1, "alice")
        channel = await _new_channel(p1, "general")
        await _subscribe(channel, agent)
        with pytest.raises(IntegrityError):
            await _subscribe(channel, agent)

    @pytest.mark.asyncio
    async def test_fk_rejects_orphans_when_enforced(self, isolated_env):
        """DDL FKs reject a missing channel/agent once enforcement is enabled.

        The application engine leaves SQLite's foreign_keys pragma off by
        default (matching the existing schema); this asserts the DDL constraint
        behaves, not that production runtime rejects orphans automatically.
        """
        await ensure_schema()
        p1 = await _new_project("a")
        agent = await _new_agent(p1, "alice")
        channel = await _new_channel(p1, "general")

        engine = get_engine()
        async with engine.connect() as conn:
            await conn.exec_driver_sql("PRAGMA foreign_keys=ON")
            for channel_id, agent_id in ((999999, agent.id), (channel.id, 999999)):
                with pytest.raises(IntegrityError):
                    await conn.exec_driver_sql(
                        "INSERT INTO channel_subscriptions (channel_id, agent_id, created_ts) "
                        "VALUES (?, ?, ?)",
                        (channel_id, agent_id, "2026-01-01 00:00:00"),
                    )


# ============================================================================
# Existing DB upgrade path
# ============================================================================


class TestExistingDBUpgrade:
    @pytest.mark.asyncio
    async def test_upgrade_keeps_data_and_adds_channel_tables(self, isolated_env):
        """A legacy DB without channel tables gains them without data loss."""
        await ensure_schema()
        p1 = await _new_project("a")
        agent = await _new_agent(p1, "alice")

        db_path = get_database_path()
        assert db_path is not None
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("DROP TABLE channel_subscriptions")
            conn.execute("DROP TABLE channels")
            conn.commit()
        finally:
            conn.close()

        # Simulate a process restart so the create_all/upgrade path re-runs.
        reset_database_state()
        await ensure_schema()

        names = await _table_names()
        assert "channels" in names
        assert "channel_subscriptions" in names

        async with get_session() as session:
            result = await session.execute(select(Project).where(Project.slug == "slug-a"))
            assert result.scalars().first() is not None
            agents = await session.execute(select(Agent).where(Agent.name == "alice"))
            assert agents.scalars().first() is not None

        channel = await _new_channel(p1, "general")
        await _subscribe(channel, agent)


# ============================================================================
# M1b-2 server tools + strict DTOs
# ============================================================================


class TestChannelTools:
    @pytest.mark.asyncio
    async def test_ensure_and_list_channels_are_idempotent_and_project_scoped(self, isolated_env):
        server = build_mcp_server()
        async with Client(server) as bootstrap:
            await bootstrap.call_tool("ensure_project", {"human_key": "/channels/alpha"})
            owner = await bootstrap.call_tool(
                "register_agent",
                {
                    "project_key": "/channels/alpha",
                    "program": "test",
                    "model": "test",
                    "name": "BlueLake",
                },
            )
            token = owner.data["registration_token"]

        async with Client(server) as client:
            first = await client.call_tool(
                "ensure_channel",
                {
                    "project_key": "/channels/alpha",
                    "channel_name": "general",
                    "registration_token": token,
                },
            )
            second = await client.call_tool(
                "ensure_channel",
                {
                    "project_key": "/channels/alpha",
                    "channel_name": "general",
                    "registration_token": token,
                },
            )
            await client.call_tool(
                "ensure_channel",
                {
                    "project_key": "/channels/alpha",
                    "channel_name": "announcements",
                    "registration_token": token,
                },
            )
            for valid_name in ("a", "a.b_c-d", "a" * 128):
                await client.call_tool(
                    "ensure_channel",
                    {
                        "project_key": "/channels/alpha",
                        "channel_name": valid_name,
                        "registration_token": token,
                    },
                )
            listed = await client.call_tool("list_channels", {"project_key": "/channels/alpha"})

        assert set(first.data) == {"channel", "created"}
        assert first.data["created"] is True
        assert second.data["created"] is False
        assert first.data["channel"] == second.data["channel"]
        assert set(first.data["channel"]) == {
            "id",
            "project_id",
            "project_slug",
            "project_key",
            "name",
            "created_ts",
        }
        assert first.data["channel"]["project_key"] == "/channels/alpha"
        assert set(listed.data) == {"project_slug", "project_key", "channels", "count"}
        expected_names = sorted(["general", "announcements", "a", "a.b_c-d", "a" * 128])
        assert listed.data["count"] == len(expected_names)
        assert [channel["name"] for channel in listed.data["channels"]] == expected_names

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "channel_name",
        ["", " general", "general ", "General", "channel:general", "general/chat", "x" * 129],
    )
    async def test_ensure_channel_rejects_noncanonical_names(self, isolated_env, channel_name):
        server = build_mcp_server()
        async with Client(server) as bootstrap:
            await bootstrap.call_tool("ensure_project", {"human_key": "/channels/invalid"})
            owner = await bootstrap.call_tool(
                "register_agent",
                {
                    "project_key": "/channels/invalid",
                    "program": "test",
                    "model": "test",
                    "name": "GreenHill",
                },
            )

            with pytest.raises(ToolError) as exc_info:
                await bootstrap.call_tool(
                    "ensure_channel",
                    {
                        "project_key": "/channels/invalid",
                        "channel_name": channel_name,
                        "registration_token": owner.data["registration_token"],
                    },
                )

        assert "channel_name" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_cross_project_subscription_lifecycle_and_dto_whitelist(self, isolated_env):
        server = build_mcp_server()
        async with Client(server) as bootstrap:
            await bootstrap.call_tool("ensure_project", {"human_key": "/channels/owners"})
            owner = await bootstrap.call_tool(
                "register_agent",
                {
                    "project_key": "/channels/owners",
                    "program": "test",
                    "model": "test",
                    "name": "BlueLake",
                },
            )
            await bootstrap.call_tool("ensure_project", {"human_key": "/channels/subscribers"})
            subscriber = await bootstrap.call_tool(
                "register_agent",
                {
                    "project_key": "/channels/subscribers",
                    "program": "test",
                    "model": "test",
                    "name": "GreenHill",
                },
            )

        async with Client(server) as client:
            await client.call_tool(
                "ensure_channel",
                {
                    "project_key": "/channels/owners",
                    "channel_name": "general",
                    "registration_token": owner.data["registration_token"],
                },
            )
            with pytest.raises(ToolError) as exc_info:
                await client.call_tool(
                    "subscribe_channel",
                    {
                        "channel_project_key": "/channels/owners",
                        "channel_name": "general",
                        "agent_project_key": "/channels/subscribers",
                        "agent_name": "GreenHill",
                        "registration_token": "wrong-token",
                    },
                )
            first = await client.call_tool(
                "subscribe_channel",
                {
                    "channel_project_key": "/channels/owners",
                    "channel_name": "general",
                    "agent_project_key": "/channels/subscribers",
                    "agent_name": "GreenHill",
                    "registration_token": subscriber.data["registration_token"],
                },
            )
            second = await client.call_tool(
                "subscribe_channel",
                {
                    "channel_project_key": "/channels/owners",
                    "channel_name": "general",
                    "agent_project_key": "/channels/subscribers",
                    "agent_name": "GreenHill",
                    "registration_token": subscriber.data["registration_token"],
                },
            )
            listed = await client.call_tool(
                "list_channel_subscriptions",
                {
                    "agent_project_key": "/channels/subscribers",
                    "agent_name": "GreenHill",
                    "registration_token": subscriber.data["registration_token"],
                },
            )

        async def unsubscribe_once():
            async with Client(server) as client:
                return await client.call_tool(
                    "unsubscribe_channel",
                    {
                        "channel_project_key": "/channels/owners",
                        "channel_name": "general",
                        "agent_project_key": "/channels/subscribers",
                        "agent_name": "GreenHill",
                        "registration_token": subscriber.data["registration_token"],
                    },
                )

        removed, removed_concurrently = await asyncio.gather(unsubscribe_once(), unsubscribe_once())
        async with Client(server) as client:
            removed_again = await client.call_tool(
                "unsubscribe_channel",
                {
                    "channel_project_key": "/channels/owners",
                    "channel_name": "general",
                    "agent_project_key": "/channels/subscribers",
                    "agent_name": "GreenHill",
                    "registration_token": subscriber.data["registration_token"],
                },
            )
            empty = await client.call_tool(
                "list_channel_subscriptions",
                {
                    "agent_project_key": "/channels/subscribers",
                    "agent_name": "GreenHill",
                    "registration_token": subscriber.data["registration_token"],
                },
            )

        assert "invalid registration_token" in str(exc_info.value).lower()
        assert first.data["created"] is True
        assert second.data["created"] is False
        assert first.data["subscription"] == second.data["subscription"]
        assert set(first.data["subscription"]) == {"id", "channel", "subscriber", "created_ts"}
        assert set(first.data["subscription"]["subscriber"]) == {
            "id",
            "name",
            "project_id",
            "project_slug",
            "project_key",
        }
        assert first.data["subscription"]["subscriber"]["project_key"] == "/channels/subscribers"
        assert first.data["subscription"]["channel"]["project_key"] == "/channels/owners"
        assert listed.data["count"] == 1
        assert listed.data["subscriptions"] == [first.data["subscription"]]
        assert sorted([removed.data["removed"], removed_concurrently.data["removed"]]) == [False, True]
        assert removed_again.data["removed"] is False
        assert empty.data["count"] == 0

    @pytest.mark.asyncio
    async def test_channel_tools_reject_missing_projects(self, isolated_env):
        server = build_mcp_server()
        async with Client(server) as client:
            with pytest.raises(ToolError) as exc_info:
                await client.call_tool("list_channels", {"project_key": "/channels/not-created"})

        assert "use ensure_project" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_subscribe_channel_rejects_missing_channel(self, isolated_env):
        server = build_mcp_server()
        async with Client(server) as client:
            await client.call_tool("ensure_project", {"human_key": "/channels/missing"})
            agent = await client.call_tool(
                "register_agent",
                {
                    "project_key": "/channels/missing",
                    "program": "test",
                    "model": "test",
                    "name": "BlueLake",
                },
            )
            with pytest.raises(ToolError) as exc_info:
                await client.call_tool(
                    "subscribe_channel",
                    {
                        "channel_project_key": "/channels/missing",
                        "channel_name": "unknown",
                        "agent_project_key": "/channels/missing",
                        "agent_name": "BlueLake",
                        "registration_token": agent.data["registration_token"],
                    },
                )

        assert "use ensure_channel" in str(exc_info.value).lower()
