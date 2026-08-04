"""M1b-3c-a: MentionDelivery data layer tests.

Covers the durable @mention delivery mapping data layer:
- CRUD round-trip
- DDL FKs target channel_messages / agents / messages
- (source_channel_message_id, mentioned_agent_id) is unique
- multiple mappings may reference the same receipt message (same-project
  mention batch -> one messages receipt with multiple recipients)
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
    MentionDelivery,
    Message,
    MessageRecipient,
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


async def _new_channel_message(channel: Channel, sender: Agent) -> ChannelMessage:
    async with get_session() as session:
        msg = ChannelMessage(channel_id=channel.id, sender_id=sender.id, subject="hi", body_md="body")
        session.add(msg)
        await session.commit()
        await session.refresh(msg)
        return msg


async def _new_receipt_message(
    project: Project,
    sender: Agent,
    recipients: Agent | list[Agent],
) -> Message:
    if isinstance(recipients, Agent):
        recipients = [recipients]
    async with get_session() as session:
        msg = Message(
            project_id=project.id,
            sender_id=sender.id,
            subject="mention",
            body_md="you were mentioned",
        )
        session.add(msg)
        await session.flush()
        assert msg.id is not None
        for recipient in recipients:
            assert recipient.id is not None
            session.add(MessageRecipient(message_id=msg.id, agent_id=recipient.id, kind="mention"))
        await session.commit()
        await session.refresh(msg)
        return msg


async def _new_delivery(
    source: ChannelMessage,
    agent: Agent,
    receipt: Message,
) -> MentionDelivery:
    async with get_session() as session:
        delivery = MentionDelivery(
            source_channel_message_id=source.id,
            mentioned_agent_id=agent.id,
            receipt_message_id=receipt.id,
        )
        session.add(delivery)
        await session.commit()
        await session.refresh(delivery)
        return delivery


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
    async def test_ensure_schema_creates_mention_delivery_table(self, isolated_env):
        await ensure_schema()
        names = await _table_names()
        assert "mention_deliveries" in names

    @pytest.mark.asyncio
    async def test_fk_columns_target_existing_tables(self, isolated_env):
        """DDL FKs point at channel_messages / agents / messages."""
        await ensure_schema()
        engine = get_engine()
        async with engine.begin() as conn:
            result = await conn.run_sync(
                lambda sync_conn: sync_conn.exec_driver_sql(
                    "SELECT \"from\", \"table\" FROM pragma_foreign_key_list('mention_deliveries') "
                    "ORDER BY id"
                ).fetchall()
            )
        fk = {(row[0], row[1]) for row in result}
        assert ("source_channel_message_id", "channel_messages") in fk
        assert ("mentioned_agent_id", "agents") in fk
        assert ("receipt_message_id", "messages") in fk

    @pytest.mark.asyncio
    async def test_ensure_schema_idempotent(self, isolated_env):
        await ensure_schema()
        p1 = await _new_project("a")
        agent = await _new_agent(p1, "alice")
        channel = await _new_channel(p1, "general")
        msg = await _new_channel_message(channel, agent)
        receipt = await _new_receipt_message(p1, agent, agent)
        await _new_delivery(msg, agent, receipt)
        await ensure_schema()
        await ensure_schema()
        names = await _table_names()
        assert "mention_deliveries" in names


# ============================================================================
# CRUD + uniqueness
# ============================================================================


class TestMentionDelivery:
    @pytest.mark.asyncio
    async def test_create_and_read_delivery(self, isolated_env):
        await ensure_schema()
        p1 = await _new_project("a")
        agent = await _new_agent(p1, "alice")
        channel = await _new_channel(p1, "general")
        msg = await _new_channel_message(channel, agent)
        receipt = await _new_receipt_message(p1, agent, agent)

        delivery = await _new_delivery(msg, agent, receipt)
        assert delivery.id is not None
        assert delivery.source_channel_message_id == msg.id
        assert delivery.mentioned_agent_id == agent.id
        assert delivery.receipt_message_id == receipt.id

        async with get_session() as session:
            result = await session.execute(
                select(MentionDelivery).where(MentionDelivery.id == delivery.id)
            )
            stored = result.scalars().first()
            assert stored is not None
            assert stored.receipt_message_id == receipt.id

    @pytest.mark.asyncio
    async def test_duplicate_source_agent_pair_fails(self, isolated_env):
        await ensure_schema()
        p1 = await _new_project("a")
        agent = await _new_agent(p1, "alice")
        channel = await _new_channel(p1, "general")
        msg = await _new_channel_message(channel, agent)
        receipt = await _new_receipt_message(p1, agent, agent)
        await _new_delivery(msg, agent, receipt)
        with pytest.raises(IntegrityError):
            await _new_delivery(msg, agent, receipt)

    @pytest.mark.asyncio
    async def test_same_source_different_agent_same_receipt(self, isolated_env):
        """Same channel message, two mentioned agents, ONE shared receipt message.

        Supports 'one receipt + multiple recipients' for same-project mentions:
        the receipt Message carries two MessageRecipient rows (alice and bob,
        kind='mention'), and both MentionDelivery mappings reference that same
        messages row — each (source, agent) pair being a distinct unique record.
        """
        await ensure_schema()
        p1 = await _new_project("a")
        sender = await _new_agent(p1, "sender")
        alice = await _new_agent(p1, "alice")
        bob = await _new_agent(p1, "bob")
        channel = await _new_channel(p1, "general")
        msg = await _new_channel_message(channel, sender)
        receipt = await _new_receipt_message(p1, sender, [alice, bob])

        d1 = await _new_delivery(msg, alice, receipt)
        d2 = await _new_delivery(msg, bob, receipt)
        assert d1.receipt_message_id == d2.receipt_message_id == receipt.id

        async with get_session() as session:
            # Both mappings point at the same receipt message.
            result = await session.execute(
                select(MentionDelivery).where(
                    MentionDelivery.source_channel_message_id == msg.id
                )
            )
            rows = result.scalars().all()
            assert {r.mentioned_agent_id for r in rows} == {alice.id, bob.id}
            assert {r.receipt_message_id for r in rows} == {receipt.id}

            # The receipt message carries two recipient rows for the two agents.
            result = await session.execute(
                select(MessageRecipient).where(
                    MessageRecipient.message_id == receipt.id
                )
            )
            recipients = result.scalars().all()
            assert {r.agent_id for r in recipients} == {alice.id, bob.id}
            assert {r.kind for r in recipients} == {"mention"}

    @pytest.mark.asyncio
    async def test_mentioned_agent_id_index_is_created(self, isolated_env):
        """mentioned_agent_id has a real SQLite index (not just Field(index=True))."""
        await ensure_schema()
        engine = get_engine()
        async with engine.begin() as conn:
            result = await conn.run_sync(
                lambda sync_conn: sync_conn.exec_driver_sql(
                    "PRAGMA index_list('mention_deliveries')"
                ).fetchall()
            )
            index_names = [row[1] for row in result]
            mention_index = next(
                (name for name in index_names if "mentioned_agent" in name),
                None,
            )
            assert mention_index is not None, f"expected an index on mentioned_agent_id, got {index_names}"
            info = await conn.run_sync(
                lambda sync_conn: sync_conn.exec_driver_sql(
                    f"PRAGMA index_info('{mention_index}')"
                ).fetchall()
            )
            assert [row[2] for row in info] == ["mentioned_agent_id"]

    @pytest.mark.asyncio
    async def test_fk_rejects_orphans_when_enforced(self, isolated_env):
        """DDL FKs reject missing source/mentioned/receipt once enforced."""
        await ensure_schema()
        p1 = await _new_project("a")
        agent = await _new_agent(p1, "alice")
        channel = await _new_channel(p1, "general")
        msg = await _new_channel_message(channel, agent)
        receipt = await _new_receipt_message(p1, agent, agent)

        engine = get_engine()
        async with engine.connect() as conn:
            await conn.exec_driver_sql("PRAGMA foreign_keys=ON")
            cases = [
                (999999, agent.id, receipt.id),  # orphan source channel message
                (msg.id, 999999, receipt.id),  # orphan mentioned agent
                (msg.id, agent.id, 999999),  # orphan receipt message
            ]
            for source, mentioned, receipt_id in cases:
                with pytest.raises(IntegrityError):
                    await conn.exec_driver_sql(
                        "INSERT INTO mention_deliveries (source_channel_message_id, mentioned_agent_id, receipt_message_id, created_ts) "
                        "VALUES (?, ?, ?, '2026-01-01 00:00:00')",
                        (source, mentioned, receipt_id),
                    )

    @pytest.mark.asyncio
    async def test_receipt_message_id_required(self, isolated_env):
        """receipt_message_id is non-nullable (only successful deliveries stored)."""
        await ensure_schema()
        p1 = await _new_project("a")
        agent = await _new_agent(p1, "alice")
        channel = await _new_channel(p1, "general")
        msg = await _new_channel_message(channel, agent)

        async with get_session() as session:
            session.add(
                MentionDelivery(
                    source_channel_message_id=msg.id,
                    mentioned_agent_id=agent.id,
                    receipt_message_id=None,  # type: ignore[arg-type]
                )
            )
            with pytest.raises(IntegrityError):
                await session.commit()
            await session.rollback()


# ============================================================================
# Existing DB upgrade path
# ============================================================================


class TestExistingDBUpgrade:
    @pytest.mark.asyncio
    async def test_upgrade_keeps_data_and_adds_mention_delivery_table(self, isolated_env):
        """A legacy DB without mention_deliveries gains it without data loss."""
        await ensure_schema()
        p1 = await _new_project("a")
        sender = await _new_agent(p1, "sender")
        alice = await _new_agent(p1, "alice")
        channel = await _new_channel(p1, "general")
        msg = await _new_channel_message(channel, sender)
        receipt = await _new_receipt_message(p1, sender, alice)
        await _new_delivery(msg, alice, receipt)

        db_path = get_database_path()
        assert db_path is not None
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("DROP TABLE mention_deliveries")
            conn.commit()
        finally:
            conn.close()

        # Simulate a process restart so the create_all/upgrade path re-runs.
        reset_database_state()
        await ensure_schema()

        names = await _table_names()
        assert "mention_deliveries" in names

        async with get_session() as session:
            result = await session.execute(select(Project).where(Project.slug == "slug-a"))
            assert result.scalars().first() is not None
            agents = await session.execute(select(Agent).where(Agent.name == "alice"))
            assert agents.scalars().first() is not None
            messages = await session.execute(select(Message))
            assert messages.scalars().first() is not None
            deliveries = await session.execute(select(MentionDelivery))
            assert deliveries.scalars().first() is None  # old row was dropped with the table

        # New delivery rows work after upgrade.
        msg2 = await _new_channel_message(channel, sender)
        receipt2 = await _new_receipt_message(p1, sender, alice)
        delivery2 = await _new_delivery(msg2, alice, receipt2)
        assert delivery2.id is not None
