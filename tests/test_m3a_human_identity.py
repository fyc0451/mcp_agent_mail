"""M3a-1: human identity + project membership data foundation.

Covers the service-layer invariants that a plain unique constraint cannot
express in SQLite:
  * display_name may repeat across humans (global human identity by subject)
  * (project, human) unique and (project, mention_handle) unique
  * owner is nullable (pre-M3a agents stay unowned)
  * default agent must belong to the same project AND be owned by the human
  * mention_handle must not collide with an active agent name in the project
  * existing-DB migration is repeatable

M3a mutators are NOT registered as public MCP tools (no human auth yet), so
these tests drive the module-level service functions directly.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest
from fastmcp import Client
from sqlalchemy import inspect
from sqlmodel import select

import mcp_agent_mail.app as app_module
from mcp_agent_mail.app import build_mcp_server
from mcp_agent_mail.db import get_session
from mcp_agent_mail.models import (
    Agent,
    Human,
    Project,
    ProjectHumanMembership,
)


@pytest.fixture
def server(isolated_env):
    return build_mcp_server()


@pytest.fixture(autouse=True)
async def _ensure_schema_fixture(isolated_env):
    from mcp_agent_mail import db
    from mcp_agent_mail.config import get_settings

    settings = get_settings()
    db.init_engine(settings)
    await db.ensure_schema(settings)
    yield


async def _ensure_project_db(project_key: str) -> Project:
    slug = project_key.strip("/").replace("/", "-")[:50] or "proj"
    async with get_session() as session:
        existing = (await session.execute(
            select(Project).where(Project.slug == slug)
        )).scalars().first()
        if existing is not None:
            return existing
        project = Project(slug=slug, human_key=project_key)
        session.add(project)
        await session.commit()
        await session.refresh(project)
        assert project.id is not None
        return project


async def _ensure_human_db(subject: str, display_name: str) -> Human:
    async with get_session() as session:
        human = await app_module._ensure_human(subject, display_name, session=session)
        await session.commit()
        await session.refresh(human)
        return human


async def _upsert_membership(
    *,
    project: Project,
    subject: str,
    display_name: str,
    mention_handle: str,
    status: str = "active",
    default_agent_id: int | None = None,
) -> ProjectHumanMembership:
    async with get_session() as session:
        membership = await app_module._upsert_project_human_membership(
            project=project,
            subject=subject,
            display_name=display_name,
            mention_handle=mention_handle,
            status=status,
            default_agent_id=default_agent_id,
            session=session,
        )
        await session.commit()
        await session.refresh(membership)
        return membership


async def _set_owner(agent_id: int, owner_id: int | None) -> None:
    async with get_session() as session:
        await app_module._set_agent_owner(agent_id, owner_id, session=session)
        await session.commit()


async def _register_agent(
    client: Client, project_key: str, name: str
) -> dict[str, Any]:
    result = await client.call_tool(
        "register_agent",
        {
            "project_key": project_key,
            "program": "test",
            "model": "test",
            "name": name,
        },
    )
    return {"id": result.data["id"], "name": result.data["name"]}


@pytest.mark.anyio
async def test_register_human_and_fetch_by_subject(server, isolated_env):
    await _ensure_project_db("/proj/human-1")
    human = await _ensure_human_db("subj-zhangsan", "张伟")
    assert human.subject == "subj-zhangsan"
    assert human.display_name == "张伟"
    assert human.id

    # Same subject returns the SAME human (id stable), regardless of display name.
    again = await _ensure_human_db("subj-zhangsan", "别称")
    assert again.id == human.id


@pytest.mark.anyio
async def test_display_name_may_repeat_across_humans(server, isolated_env):
    await _ensure_project_db("/proj/human-2")
    a = await _ensure_human_db("subj-a", "同名")
    b = await _ensure_human_db("subj-b", "同名")
    assert a.id != b.id
    assert a.display_name == b.display_name == "同名"


@pytest.mark.anyio
async def test_membership_unique_per_project_human_and_handle(server, isolated_env):
    project = await _ensure_project_db("/proj/human-3")
    m1 = await _upsert_membership(
        project=project, subject="subj-zhang", display_name="张伟", mention_handle="zhangwei"
    )
    assert m1.mention_handle == "zhangwei"
    # Same (project, human) updates, not duplicates.
    m2 = await _upsert_membership(
        project=project, subject="subj-zhang", display_name="张伟", mention_handle="zwei"
    )
    assert m2.mention_handle == "zwei"
    async with get_session() as session:
        rows = (await session.execute(
            select(ProjectHumanMembership).where(
                ProjectHumanMembership.project_id == project.id
            )
        )).scalars().all()
        assert len(rows) == 1


@pytest.mark.anyio
async def test_mention_handle_collides_with_active_agent_name_is_rejected(server, isolated_env):
    project = await _ensure_project_db("/proj/human-4")
    async with Client(server) as client:
        await _register_agent(client, "/proj/human-4", "GreenLake")
    with pytest.raises(Exception, match="与项目 active agent 名冲突"):
        await _upsert_membership(
            project=project, subject="subj-zhang", display_name="张伟", mention_handle="greenlake"
        )


@pytest.mark.anyio
async def test_agent_owner_is_nullable_and_settable(server, isolated_env):
    await _ensure_project_db("/proj/human-5")
    async with Client(server) as client:
        agent = await _register_agent(client, "/proj/human-5", "RedStone")
    # owner NULL by default
    async with get_session() as session:
        db_agent = await session.get(Agent, agent["id"])
        assert db_agent is not None
        assert db_agent.owner_id is None

    human = await _ensure_human_db("subj-zhang", "张伟")
    await _set_owner(agent["id"], human.id)
    async with get_session() as session:
        db_agent = await session.get(Agent, agent["id"])
        assert db_agent is not None
        assert db_agent.owner_id == human.id

    # detach back to NULL
    await _set_owner(agent["id"], None)
    async with get_session() as session:
        db_agent = await session.get(Agent, agent["id"])
        assert db_agent is not None
        assert db_agent.owner_id is None


@pytest.mark.anyio
async def test_default_agent_must_belong_to_same_project(server, isolated_env):
    await _ensure_project_db("/proj/human-6")
    await _ensure_project_db("/proj/other-6")
    async with Client(server) as client:
        agent = await _register_agent(client, "/proj/other-6", "BlueDog")
    human = await _ensure_human_db("subj-zhang", "张伟")
    await _set_owner(agent["id"], human.id)

    project6 = await _ensure_project_db("/proj/human-6")
    with pytest.raises(Exception, match="必须属于同一项目"):
        await _upsert_membership(
            project=project6, subject="subj-zhang", display_name="张伟",
            mention_handle="zhangwei", default_agent_id=agent["id"],
        )


@pytest.mark.anyio
async def test_default_agent_must_be_owned_by_this_human(server, isolated_env):
    project = await _ensure_project_db("/proj/human-7")
    async with Client(server) as client:
        agent = await _register_agent(client, "/proj/human-7", "GreenLake")
    # agent owned by a different human
    other = await _ensure_human_db("subj-other", "李华")
    await _set_owner(agent["id"], other.id)

    with pytest.raises(Exception, match="必须属于该 human"):
        await _upsert_membership(
            project=project, subject="subj-zhang", display_name="张伟",
            mention_handle="zhangwei", default_agent_id=agent["id"],
        )


@pytest.mark.anyio
async def test_default_agent_same_project_and_owned_by_human_ok(server, isolated_env):
    project = await _ensure_project_db("/proj/human-8")
    async with Client(server) as client:
        agent = await _register_agent(client, "/proj/human-8", "RedStone")
    human = await _ensure_human_db("subj-zhang", "张伟")
    await _set_owner(agent["id"], human.id)

    m = await _upsert_membership(
        project=project, subject="subj-zhang", display_name="张伟",
        mention_handle="zhangwei", default_agent_id=agent["id"],
    )
    assert m.default_agent_id == agent["id"]
    assert m.mention_handle == "zhangwei"


@pytest.mark.anyio
async def test_existing_db_migration_adds_owner_id_and_new_tables(isolated_env):
    """ensure_schema on an existing DB adds agents.owner_id + humans tables idempotently."""
    from mcp_agent_mail import db
    from mcp_agent_mail.config import get_settings

    settings = get_settings()
    db.init_engine(settings)
    await db.ensure_schema(settings)

    engine = db.get_engine()

    def _inspect(sync_conn):
        insp = inspect(sync_conn)
        return {
            "agents": {c["name"] for c in insp.get_columns("agents")},
            "humans": "humans" in insp.get_table_names(),
            "memberships": "project_human_memberships" in insp.get_table_names(),
        }

    async with engine.begin() as conn:
        cols = await conn.run_sync(_inspect)
    assert "owner_id" in cols["agents"]
    assert cols["humans"] is True
    assert cols["memberships"] is True

    # Repeat — must not raise and must stay idempotent.
    await db.ensure_schema(settings)
    async with engine.begin() as conn:
        cols2 = await conn.run_sync(_inspect)
    assert cols2 == cols


@pytest.mark.anyio
async def test_mention_handle_case_insensitive_unique(server, isolated_env):
    """human mention_handle must be unique per project ignoring case (lead #1)."""
    project = await _ensure_project_db("/proj/human-case")
    m = await _upsert_membership(
        project=project, subject="subj-a", display_name="张伟", mention_handle="ZhangWei"
    )
    assert m.mention_handle == "ZhangWei"
    # Second human, same handle different case → rejected.
    with pytest.raises(Exception, match="已在项目内被使用"):
        await _upsert_membership(
            project=project, subject="subj-b", display_name="李华", mention_handle="zhangwei"
        )


@pytest.mark.anyio
async def test_inactive_membership_handle_conflict_is_a_service_error(server, isolated_env):
    """DB uniqueness covers every status, so the service must reject an
    invited/removed handle before flush instead of leaking IntegrityError."""
    project = await _ensure_project_db("/proj/human-inactive-handle")
    await _upsert_membership(
        project=project,
        subject="subj-invited",
        display_name="待加入成员",
        mention_handle="ReservedHandle",
        status="invited",
    )

    with pytest.raises(ValueError, match="已在项目内被使用"):
        await _upsert_membership(
            project=project,
            subject="subj-active",
            display_name="正式成员",
            mention_handle="reservedhandle",
        )


@pytest.mark.anyio
async def test_membership_update_refreshes_updated_at(server, isolated_env):
    project = await _ensure_project_db("/proj/human-updated-at")
    membership = await _upsert_membership(
        project=project,
        subject="subj-update",
        display_name="更新成员",
        mention_handle="BeforeUpdate",
    )
    async with get_session() as session:
        stored = await session.get(ProjectHumanMembership, membership.id)
        assert stored is not None
        stored.updated_at = datetime(2000, 1, 1)
        session.add(stored)
        await session.commit()

    updated = await _upsert_membership(
        project=project,
        subject="subj-update",
        display_name="更新成员",
        mention_handle="AfterUpdate",
    )
    assert updated.updated_at > datetime(2000, 1, 1)


@pytest.mark.anyio
async def test_register_agent_ignores_removed_membership_handle(server, isolated_env):
    """The reverse collision rule applies only to active memberships."""
    project = await _ensure_project_db("/proj/human-removed-handle")
    await _upsert_membership(
        project=project,
        subject="subj-removed",
        display_name="已移除成员",
        mention_handle="alpha-one",
        status="removed",
    )
    async with Client(server) as client:
        agent = await _register_agent(client, "/proj/human-removed-handle", "alpha-one")
    assert agent["name"] == "alpha-one"


@pytest.mark.anyio
async def test_register_agent_rejects_name_colliding_with_membership_handle(server, isolated_env):
    """Reverse invariant: new agent name must not collide with an active
    membership mention_handle (case-insensitive) (lead #2)."""
    project = await _ensure_project_db("/proj/human-reverse")
    await _upsert_membership(
        project=project, subject="subj-a", display_name="张伟", mention_handle="GreenLake"
    )
    async with Client(server) as client:
        with pytest.raises(Exception, match="与项目 human membership mention_handle 冲突"):
            await _register_agent(client, "/proj/human-reverse", "greenlake")


@pytest.mark.anyio
async def test_set_agent_owner_rejects_nonexistent_human(server, isolated_env):
    """set owner must refuse a nonexistent human (lead #3)."""
    await _ensure_project_db("/proj/human-owner")
    async with Client(server) as client:
        agent = await _register_agent(client, "/proj/human-owner", "RedStone")
    with pytest.raises(Exception, match=r"human.*不存在"):
        await _set_owner(agent["id"], 999999)


@pytest.mark.anyio
async def test_cannot_change_owner_of_default_referenced_agent(server, isolated_env):
    """An agent referenced as membership.default_agent_id must not have its
    owner changed/cleared, or the default would dangle (lead #4)."""
    project = await _ensure_project_db("/proj/human-ref")
    async with Client(server) as client:
        agent = await _register_agent(client, "/proj/human-ref", "BlueDog")
    human = await _ensure_human_db("subj-a", "张伟")
    await _set_owner(agent["id"], human.id)
    await _upsert_membership(
        project=project, subject="subj-a", display_name="张伟",
        mention_handle="zhangwei", default_agent_id=agent["id"],
    )
    other = await _ensure_human_db("subj-other", "李华")
    with pytest.raises(Exception, match=r"default_agent_id|默认 agent|不能修改 owner"):
        await _set_owner(agent["id"], other.id)
    with pytest.raises(Exception, match=r"default_agent_id|默认 agent|不能修改 owner"):
        await _set_owner(agent["id"], None)


@pytest.mark.anyio
async def test_existing_db_owner_index_repeatable(isolated_env):
    """owner_id index is created idempotently (lead #5)."""
    from mcp_agent_mail import db
    from mcp_agent_mail.config import get_settings

    settings = get_settings()
    db.init_engine(settings)
    await db.ensure_schema(settings)
    engine = db.get_engine()

    def _indexes(sync_conn):
        insp = inspect(sync_conn)
        return {ix["name"] for ix in insp.get_indexes("agents")}

    async with engine.begin() as conn:
        idx1 = await conn.run_sync(_indexes)
    await db.ensure_schema(settings)
    async with engine.begin() as conn:
        idx2 = await conn.run_sync(_indexes)
    assert "ix_agents_owner_id" in idx1
    assert idx1 == idx2


@pytest.mark.anyio
async def test_mutators_not_exposed_as_public_mcp_tools(server, isolated_env):
    """M3a human mutators must not be registered as public MCP tools until
    authenticated human principals exist (lead #6 / invariants)."""
    tools = await build_mcp_server().get_tools()
    assert {
        "register_human",
        "set_project_human_membership",
        "set_agent_owner",
    }.isdisjoint(tools)


@pytest.mark.anyio
async def test_handle_ci_index_allows_distinct_handles(server, isolated_env):
    """Regression: the case-insensitive index must reference the COLUMN
    lower(mention_handle), not the string constant lower('mention_handle').
    Distinct handles (Alice/Bob) must both insert (lead #840)."""

    project = await _ensure_project_db("/proj/handle-distinct")
    async with get_session() as session:
        a = Human(subject="subj-alice", display_name="Alice")
        b = Human(subject="subj-bob", display_name="Bob")
        session.add_all([a, b])
        await session.commit()
        await session.refresh(a)
        await session.refresh(b)
        session.add(ProjectHumanMembership(
            project_id=project.id, human_id=a.id, mention_handle="Alice"
        ))
        await session.commit()
        # Second membership with a DIFFERENT handle must succeed — this was the
        # false-positive case before the index fix.
        session.add(ProjectHumanMembership(
            project_id=project.id, human_id=b.id, mention_handle="Bob"
        ))
        await session.commit()
    async with get_session() as session:
        rows = (await session.execute(
            select(ProjectHumanMembership).where(
                ProjectHumanMembership.project_id == project.id
            )
        )).scalars().all()
        assert {r.mention_handle for r in rows} == {"Alice", "Bob"}


@pytest.mark.anyio
async def test_handle_ci_index_rejects_same_handle_other_case(server, isolated_env):
    """Regression: same handle differing only in case must raise IntegrityError
    at the DB level (lead #840)."""
    from sqlalchemy.exc import IntegrityError

    project = await _ensure_project_db("/proj/handle-ci-dup")
    async with get_session() as session:
        a = Human(subject="subj-alice", display_name="Alice")
        b = Human(subject="subj-alice2", display_name="Alice 2")
        session.add_all([a, b])
        await session.commit()
        await session.refresh(a)
        await session.refresh(b)
        session.add(ProjectHumanMembership(
            project_id=project.id, human_id=a.id, mention_handle="Alice"
        ))
        await session.commit()
        session.add(ProjectHumanMembership(
            project_id=project.id, human_id=b.id, mention_handle="alice"
        ))
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()


@pytest.mark.anyio
async def test_auto_generated_agent_name_avoids_membership_handle(server, isolated_env, monkeypatch):
    """Auto-generated names must skip active human mention_handles; a first
    collision re-generates a fresh name and succeeds (bounded by retries)."""
    from mcp_agent_mail.config import get_settings

    project = await _ensure_project_db("/proj/auto-collision")
    await _ensure_human_db("subj-greenlake", "Green Lake")
    await _upsert_membership(
        project=project,
        subject="subj-greenlake",
        display_name="Green Lake",
        mention_handle="GreenLake",
    )

    calls = {"n": 0}

    async def _fake_gen(proj, settings, name_hint=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return "GreenLake"  # collides with the membership handle
        return "BlueSky"  # fresh, non-colliding

    monkeypatch.setattr(app_module, "_generate_unique_agent_name", _fake_gen)
    settings = get_settings()
    agent = await app_module._get_or_create_agent(
        project, None, "test", "test", "", settings,
    )

    assert calls["n"] == 2
    assert agent.name == "BlueSky"
    async with get_session() as session:
        db_agent = await session.get(Agent, agent.id)
        assert db_agent is not None
        assert db_agent.name == "BlueSky"
