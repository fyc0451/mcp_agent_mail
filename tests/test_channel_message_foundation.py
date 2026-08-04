"""M1b-3a: ChannelMessage + ChannelReadCursor data layer tests.

Covers the channel history ("blackboard") and per-agent read cursor data layer:
- channel_messages rows are independent of message_recipients (no fanout rows)
- sender may be a cross-project agent (subscriber posting to a channel)
- channel_read_cursors keyed by (channel_id, agent_id), unique
- cursor starts at last_read_message_id=NULL (not 0) so FK enforcement never
  references a non-existent message row
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
from mcp_agent_mail.models import (
    Agent,
    Channel,
    ChannelMessage,
    ChannelReadCursor,
    ChannelSubscription,
    Project,
)


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


async def _post_message(channel: Channel, sender: Agent, subject: str = "hi") -> ChannelMessage:
    async with get_session() as session:
        msg = ChannelMessage(channel_id=channel.id, sender_id=sender.id, subject=subject, body_md="body")
        session.add(msg)
        await session.commit()
        await session.refresh(msg)
        return msg


async def _new_cursor(channel: Channel, agent: Agent) -> ChannelReadCursor:
    async with get_session() as session:
        cursor = ChannelReadCursor(channel_id=channel.id, agent_id=agent.id)
        session.add(cursor)
        await session.commit()
        await session.refresh(cursor)
        return cursor


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
    async def test_ensure_schema_creates_channel_message_tables(self, isolated_env):
        await ensure_schema()
        names = await _table_names()
        assert "channel_messages" in names
        assert "channel_read_cursors" in names

    @pytest.mark.asyncio
    async def test_fk_columns_target_existing_tables(self, isolated_env):
        """DDL FKs point at the real channels/agents/messages tables."""
        await ensure_schema()
        engine = get_engine()
        async with engine.begin() as conn:
            result = await conn.run_sync(
                lambda sync_conn: sync_conn.exec_driver_sql(
                    "SELECT \"from\", \"table\" FROM pragma_foreign_key_list('channel_messages') "
                    "ORDER BY id"
                ).fetchall()
            )
            msg_fk = {(row[0], row[1]) for row in result}
            result = await conn.run_sync(
                lambda sync_conn: sync_conn.exec_driver_sql(
                    "SELECT \"from\", \"table\" FROM pragma_foreign_key_list('channel_read_cursors') "
                    "ORDER BY id"
                ).fetchall()
            )
            cursor_fk = {(row[0], row[1]) for row in result}
        assert ("channel_id", "channels") in msg_fk
        assert ("sender_id", "agents") in msg_fk
        assert ("channel_id", "channels") in cursor_fk
        assert ("agent_id", "agents") in cursor_fk
        assert ("last_read_message_id", "channel_messages") in cursor_fk

    @pytest.mark.asyncio
    async def test_ensure_schema_idempotent(self, isolated_env):
        await ensure_schema()
        p1 = await _new_project("a")
        ch = await _new_channel(p1, "general")
        agent = await _new_agent(p1, "alice")
        await _new_cursor(ch, agent)
        await ensure_schema()
        await ensure_schema()
        names = await _table_names()
        assert "channel_messages" in names
        assert "channel_read_cursors" in names


# ============================================================================
# ChannelMessage semantics
# ============================================================================


class TestChannelMessage:
    @pytest.mark.asyncio
    async def test_post_message_roundtrip(self, isolated_env):
        await ensure_schema()
        p1 = await _new_project("a")
        agent = await _new_agent(p1, "alice")
        channel = await _new_channel(p1, "general")

        msg = await _post_message(channel, agent, subject="hello")
        assert msg.id is not None

        async with get_session() as session:
            result = await session.execute(
                select(ChannelMessage).where(ChannelMessage.id == msg.id)
            )
            stored = result.scalars().first()
            assert stored is not None
            assert stored.subject == "hello"
            assert stored.sender_id == agent.id
            assert stored.channel_id == channel.id

    @pytest.mark.asyncio
    async def test_no_message_recipient_fanout_rows(self, isolated_env):
        """Channel messages do NOT write message_recipients fanout rows."""
        await ensure_schema()
        p1 = await _new_project("a")
        agent = await _new_agent(p1, "alice")
        channel = await _new_channel(p1, "general")
        await _post_message(channel, agent)

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
    async def test_cross_project_sender_can_post(self, isolated_env):
        """A cross-project agent (subscriber) can post to the channel."""
        await ensure_schema()
        p1 = await _new_project("a")
        p2 = await _new_project("b")
        channel = await _new_channel(p1, "general")
        remote_agent = await _new_agent(p2, "bob")  # agent from another project

        async with get_session() as session:
            session.add(ChannelSubscription(channel_id=channel.id, agent_id=remote_agent.id))
            await session.commit()

        msg = await _post_message(channel, remote_agent, subject="cross")
        assert msg.id is not None

    @pytest.mark.asyncio
    async def test_fk_rejects_orphans_when_enforced(self, isolated_env):
        """DDL FKs reject a missing channel/sender once enforcement is enabled.

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
            for channel_id, sender_id in ((999999, agent.id), (channel.id, 999999)):
                with pytest.raises(IntegrityError):
                    await conn.exec_driver_sql(
                        "INSERT INTO channel_messages (channel_id, sender_id, subject, body_md, importance, created_ts) "
                        "VALUES (?, ?, 'x', 'y', 'normal', '2026-01-01 00:00:00')",
                        (channel_id, sender_id),
                    )


# ============================================================================
# ChannelReadCursor semantics
# ============================================================================


class TestChannelReadCursor:
    @pytest.mark.asyncio
    async def test_cursor_initial_value_is_null(self, isolated_env):
        """Cursor starts at NULL (not 0) so FK enforcement never orphans."""
        await ensure_schema()
        p1 = await _new_project("a")
        agent = await _new_agent(p1, "alice")
        channel = await _new_channel(p1, "general")

        cursor = await _new_cursor(channel, agent)
        assert cursor.last_read_message_id is None

        async with get_session() as session:
            result = await session.execute(
                select(ChannelReadCursor).where(ChannelReadCursor.id == cursor.id)
            )
            stored = result.scalars().first()
            assert stored is not None
            assert stored.last_read_message_id is None

    @pytest.mark.asyncio
    async def test_cursor_unique_per_channel_and_agent(self, isolated_env):
        await ensure_schema()
        p1 = await _new_project("a")
        p2 = await _new_project("b")
        alice = await _new_agent(p1, "alice")
        bob = await _new_agent(p2, "bob")
        channel = await _new_channel(p1, "general")
        other = await _new_channel(p1, "other")

        await _new_cursor(channel, alice)
        await _new_cursor(channel, bob)  # different agent on same channel: OK
        await _new_cursor(other, alice)  # same agent on different channel: OK
        with pytest.raises(IntegrityError):
            await _new_cursor(channel, alice)  # duplicate (channel, agent): fails

    @pytest.mark.asyncio
    async def test_resubscribe_preserves_cursor(self, isolated_env):
        """A cursor survives unsubscribe/resubscribe: no DELETE on the cursor row.

        Re-subscribing the same (channel, agent) after an unsubscribe must
        restore the prior cursor rather than starting over, so history is not
        re-delivered. The data layer keeps the cursor row; the unsubscribe
        tool must delete only the subscription row.
        """
        await ensure_schema()
        p1 = await _new_project("a")
        agent = await _new_agent(p1, "alice")
        channel = await _new_channel(p1, "general")
        msg = await _post_message(channel, agent)

        # Establish a real subscription lifecycle: subscribe first.
        async with get_session() as session:
            session.add(ChannelSubscription(channel_id=channel.id, agent_id=agent.id))
            await session.commit()

        cursor = await _new_cursor(channel, agent)
        async with get_session() as session:
            cursor = await session.get(ChannelReadCursor, cursor.id)
            assert cursor is not None
            cursor.last_read_message_id = msg.id
            session.add(cursor)
            await session.commit()

        # Unsubscribe: delete only the subscription row, keep the cursor row,
        # and prove the subscription is really gone.
        async with get_session() as session:
            result = await session.execute(
                select(ChannelSubscription).where(
                    ChannelSubscription.channel_id == channel.id,
                    ChannelSubscription.agent_id == agent.id,
                )
            )
            sub = result.scalars().first()
            assert sub is not None
            await session.delete(sub)
            await session.commit()

        async with get_session() as session:
            result = await session.execute(
                select(ChannelSubscription).where(
                    ChannelSubscription.channel_id == channel.id,
                    ChannelSubscription.agent_id == agent.id,
                )
            )
            assert result.scalars().first() is None

        # Resubscribe: the cursor row must still be present with the old value.
        async with get_session() as session:
            session.add(ChannelSubscription(channel_id=channel.id, agent_id=agent.id))
            await session.commit()

        async with get_session() as session:
            result = await session.execute(
                select(ChannelReadCursor).where(
                    ChannelReadCursor.channel_id == channel.id,
                    ChannelReadCursor.agent_id == agent.id,
                )
            )
            restored = result.scalars().first()
            assert restored is not None
            assert restored.last_read_message_id == msg.id

    @pytest.mark.asyncio
    async def test_cursor_advance_to_existing_message(self, isolated_env):
        """Advancing to a real channel_messages.id is allowed."""
        await ensure_schema()
        p1 = await _new_project("a")
        agent = await _new_agent(p1, "alice")
        channel = await _new_channel(p1, "general")
        msg = await _post_message(channel, agent)
        cursor = await _new_cursor(channel, agent)

        async with get_session() as session:
            cursor = await session.get(ChannelReadCursor, cursor.id)
            assert cursor is not None
            cursor.last_read_message_id = msg.id
            session.add(cursor)
            await session.commit()

        async with get_session() as session:
            result = await session.execute(
                select(ChannelReadCursor).where(ChannelReadCursor.id == cursor.id)
            )
            stored = result.scalars().first()
            assert stored is not None
            assert stored.last_read_message_id == msg.id

    @pytest.mark.asyncio
    async def test_cursor_fk_rejects_orphans_when_enforced(self, isolated_env):
        """DDL FKs on channel_read_cursors reject orphans once enforced.

        Covers all three FK columns: last_read_message_id (orphan message),
        channel_id (orphan channel), and agent_id (orphan agent, with the
        cursor's last_read_message_id left NULL as the initial value).
        """
        await ensure_schema()
        p1 = await _new_project("a")
        agent = await _new_agent(p1, "alice")
        channel = await _new_channel(p1, "general")

        engine = get_engine()
        async with engine.connect() as conn:
            await conn.exec_driver_sql("PRAGMA foreign_keys=ON")
            with pytest.raises(IntegrityError):
                await conn.exec_driver_sql(
                    "INSERT INTO channel_read_cursors (channel_id, agent_id, last_read_message_id, created_ts, updated_ts) "
                    "VALUES (?, ?, 999999, '2026-01-01 00:00:00', '2026-01-01 00:00:00')",
                    (channel.id, agent.id),
                )
            # Orphan channel with a NULL cursor initial value.
            with pytest.raises(IntegrityError):
                await conn.exec_driver_sql(
                    "INSERT INTO channel_read_cursors (channel_id, agent_id, last_read_message_id, created_ts, updated_ts) "
                    "VALUES (?, ?, NULL, '2026-01-01 00:00:00', '2026-01-01 00:00:00')",
                    (999999, agent.id),
                )
            # Orphan agent with a NULL cursor initial value.
            with pytest.raises(IntegrityError):
                await conn.exec_driver_sql(
                    "INSERT INTO channel_read_cursors (channel_id, agent_id, last_read_message_id, created_ts, updated_ts) "
                    "VALUES (?, ?, NULL, '2026-01-01 00:00:00', '2026-01-01 00:00:00')",
                    (channel.id, 999999),
                )


# ============================================================================
# Existing DB upgrade path
# ============================================================================


class TestExistingDBUpgrade:
    @pytest.mark.asyncio
    async def test_upgrade_keeps_data_and_adds_channel_message_tables(self, isolated_env):
        """A legacy DB without channel_message tables gains them without data loss."""
        await ensure_schema()
        p1 = await _new_project("a")
        agent = await _new_agent(p1, "alice")
        channel = await _new_channel(p1, "general")
        await _post_message(channel, agent)
        await _new_cursor(channel, agent)

        db_path = get_database_path()
        assert db_path is not None
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("DROP TABLE channel_read_cursors")
            conn.execute("DROP TABLE channel_messages")
            conn.commit()
        finally:
            conn.close()

        # Simulate a process restart so the create_all/upgrade path re-runs.
        reset_database_state()
        await ensure_schema()

        names = await _table_names()
        assert "channel_messages" in names
        assert "channel_read_cursors" in names

        async with get_session() as session:
            result = await session.execute(select(Project).where(Project.slug == "slug-a"))
            assert result.scalars().first() is not None
            agents = await session.execute(select(Agent).where(Agent.name == "alice"))
            assert agents.scalars().first() is not None
            channels = await session.execute(
                select(Channel).where(Channel.name == "general")
            )
            assert channels.scalars().first() is not None

        channel2 = await _new_channel(p1, "rebuilt")
        msg2 = await _post_message(channel2, agent)
        cursor = await _new_cursor(channel2, agent)
        async with get_session() as session:
            cursor = await session.get(ChannelReadCursor, cursor.id)
            assert cursor is not None
            cursor.last_read_message_id = msg2.id
            session.add(cursor)
            await session.commit()
