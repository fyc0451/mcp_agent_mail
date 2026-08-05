"""M1b-4b HubAuditEvent schema and upgrade coverage."""

from __future__ import annotations

import sqlite3

import pytest
from sqlmodel import select

from mcp_agent_mail.db import (
    ensure_schema,
    get_database_path,
    get_engine,
    get_session,
    reset_database_state,
)
from mcp_agent_mail.models import Agent, HubAuditEvent, Project


async def _seed_event() -> tuple[Project, Agent, HubAuditEvent]:
    async with get_session() as session:
        project = Project(slug="audit-project", human_key="/audit/project")
        session.add(project)
        await session.flush()
        assert project.id is not None
        actor = Agent(project_id=project.id, name="BlueLake", program="test", model="test")
        session.add(actor)
        await session.flush()
        assert actor.id is not None
        event = HubAuditEvent(
            project_id=project.id,
            actor_agent_id=actor.id,
            event_type="channel_message_posted",
            source_type="channel_message",
            source_id=42,
            outcome="succeeded",
        )
        session.add(event)
        await session.commit()
        await session.refresh(event)
        return project, actor, event


@pytest.mark.asyncio
async def test_schema_has_content_free_audit_columns_and_indexes(isolated_env):
    await ensure_schema()
    engine = get_engine()
    async with engine.begin() as conn:
        columns = await conn.run_sync(
            lambda sync_conn: sync_conn.exec_driver_sql(
                "PRAGMA table_info('hub_audit_events')"
            ).fetchall()
        )
        indexes = await conn.run_sync(
            lambda sync_conn: sync_conn.exec_driver_sql(
                "PRAGMA index_list('hub_audit_events')"
            ).fetchall()
        )

    column_names = {row[1] for row in columns}
    assert column_names == {
        "id",
        "project_id",
        "actor_agent_id",
        "event_type",
        "source_type",
        "source_id",
        "outcome",
        "reason",
        "target_project_id",
        "target_agent_id",
        "related_message_id",
        "created_ts",
    }
    assert {row[1] for row in indexes} >= {
        "idx_hub_audit_events_project_id",
        "idx_hub_audit_events_actor_id",
        "idx_hub_audit_events_source",
    }
    assert not ({"body_md", "subject", "registration_token", "detail", "metadata"} & column_names)


@pytest.mark.asyncio
async def test_event_round_trip(isolated_env):
    await ensure_schema()
    project, actor, event = await _seed_event()

    async with get_session() as session:
        stored = await session.get(HubAuditEvent, event.id)
    assert stored is not None
    assert stored.project_id == project.id
    assert stored.actor_agent_id == actor.id
    assert stored.source_id == 42
    assert stored.outcome == "succeeded"


@pytest.mark.asyncio
async def test_existing_database_upgrade_recreates_audit_table_without_losing_other_data(isolated_env):
    await ensure_schema()
    project, _actor, _event = await _seed_event()
    db_path = get_database_path()
    assert db_path is not None

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("DROP TABLE hub_audit_events")
        conn.commit()
    finally:
        conn.close()

    reset_database_state()
    await ensure_schema()

    async with get_session() as session:
        stored_project = (
            await session.execute(select(Project).where(Project.id == project.id))
        ).scalars().one()
        events = list((await session.execute(select(HubAuditEvent))).scalars())
    assert stored_project.human_key == "/audit/project"
    assert events == []
