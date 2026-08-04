"""M1b-1: Channel + ChannelSubscription data layer tests.

Covers the project-scoped channel and cross-project subscription data layer:
- channel name unique per project (same name allowed across projects)
- subscriptions allow same-project and cross-project agents
- (channel_id, agent_id) unique
- DDL FKs reject orphans once enforcement is enabled
- idempotent schema init and non-regressing upgrade on an existing DB
"""

from __future__ import annotations

import sqlite3

import pytest
from sqlalchemy.exc import IntegrityError
from sqlmodel import select

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
