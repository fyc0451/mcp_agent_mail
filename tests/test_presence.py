"""M1b-4a: server-derived agent presence and passive activity touches."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from fastmcp import Client
from sqlmodel import select

import mcp_agent_mail.app as app_module
from mcp_agent_mail.app import _agent_online, build_mcp_server
from mcp_agent_mail.config import clear_settings_cache, get_settings
from mcp_agent_mail.db import get_session
from mcp_agent_mail.models import Agent, Project


def _resource_json(blocks) -> dict[str, Any]:
    return json.loads("".join(block.text or "" for block in blocks))


async def _agent(project_key: str, name: str) -> Agent:
    async with get_session() as session:
        project_result = await session.execute(
            select(Project).where(Project.human_key == project_key)
        )
        project = project_result.scalars().one()
        agent_result = await session.execute(
            select(Agent).where(Agent.project_id == project.id, Agent.name == name)
        )
        return agent_result.scalars().one()


async def _set_activity(
    project_key: str,
    name: str,
    last_active_ts: datetime,
    *,
    retired_at: datetime | None = None,
) -> None:
    agent = await _agent(project_key, name)
    async with get_session() as session:
        db_agent = await session.get(Agent, agent.id)
        assert db_agent is not None
        db_agent.last_active_ts = last_active_ts
        db_agent.retired_at = retired_at
        session.add(db_agent)
        await session.commit()


async def _create_tokenless_agent(project_key: str, name: str) -> None:
    async with get_session() as session:
        project_result = await session.execute(
            select(Project).where(Project.human_key == project_key)
        )
        project = project_result.scalars().one()
        session.add(
            Agent(
                project_id=project.id,
                name=name,
                program="legacy",
                model="legacy",
                registration_token=None,
            )
        )
        await session.commit()


def test_agent_online_boundaries_and_retired_identity():
    now = datetime(2026, 8, 5, 3, 0, tzinfo=timezone.utc)
    agent = Agent(
        project_id=1,
        name="BlueLake",
        program="test",
        model="test",
        last_active_ts=now - timedelta(seconds=300),
    )

    assert _agent_online(agent, now=now, ttl_seconds=300) is True
    agent.last_active_ts = now - timedelta(seconds=301)
    assert _agent_online(agent, now=now, ttl_seconds=300) is False
    agent.last_active_ts = now
    assert _agent_online(agent, now=now, ttl_seconds=0) is False
    agent.retired_at = now
    assert _agent_online(agent, now=now, ttl_seconds=300) is False


def test_presence_ttl_is_configurable(isolated_env, monkeypatch):
    monkeypatch.setenv("PRESENCE_ONLINE_TTL_SECONDS", "42")
    clear_settings_cache()
    assert get_settings().presence_online_ttl_seconds == 42


@pytest.mark.asyncio
async def test_agents_directory_derives_online_and_forces_retired_offline(isolated_env):
    async with Client(build_mcp_server()) as client:
        await client.call_tool("ensure_project", {"human_key": "/presence/directory"})
        active = await client.call_tool(
            "register_agent",
            {
                "project_key": "/presence/directory",
                "program": "test",
                "model": "test",
                "name": "BlueLake",
            },
        )
        stale = await client.call_tool(
            "register_agent",
            {
                "project_key": "/presence/directory",
                "program": "test",
                "model": "test",
                "name": "GreenHill",
            },
        )
        retired = await client.call_tool(
            "register_agent",
            {
                "project_key": "/presence/directory",
                "program": "test",
                "model": "test",
                "name": "RedRiver",
            },
        )
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        await _set_activity(
            "/presence/directory",
            stale.data["name"],
            now - timedelta(minutes=10),
        )
        await _set_activity(
            "/presence/directory",
            retired.data["name"],
            now,
            retired_at=now,
        )

        payload = _resource_json(
            await client.read_resource("resource://agents/presence-directory")
        )

    active_by_name = {entry["name"]: entry for entry in payload["agents"]}
    retired_by_name = {entry["name"]: entry for entry in payload["retired_agents"]}
    assert active_by_name[active.data["name"]]["online"] is True
    assert active_by_name[stale.data["name"]]["online"] is False
    assert retired_by_name[retired.data["name"]]["online"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["fetch_inbox", "fetch_channel_messages"])
async def test_authenticated_reads_refresh_last_active(isolated_env, tool_name):
    async with Client(build_mcp_server()) as client:
        await client.call_tool("ensure_project", {"human_key": "/presence/reads"})
        registered = await client.call_tool(
            "register_agent",
            {
                "project_key": "/presence/reads",
                "program": "test",
                "model": "test",
                "name": "BlueLake",
            },
        )
        token = registered.data["registration_token"]
        await client.call_tool(
            "ensure_channel",
            {
                "project_key": "/presence/reads",
                "channel_name": "general",
                "registration_token": token,
            },
        )
        old = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1)
        await _set_activity("/presence/reads", registered.data["name"], old)

        if tool_name == "fetch_inbox":
            await client.call_tool(
                tool_name,
                {
                    "project_key": "/presence/reads",
                    "agent_name": registered.data["name"],
                    "registration_token": token,
                },
            )
        else:
            await client.call_tool(
                tool_name,
                {
                    "channel_project_key": "/presence/reads",
                    "channel_name": "general",
                    "agent_project_key": "/presence/reads",
                    "agent_name": registered.data["name"],
                    "registration_token": token,
                },
            )

    assert (await _agent("/presence/reads", registered.data["name"])).last_active_ts > old


@pytest.mark.asyncio
async def test_activity_touch_failure_does_not_break_authenticated_read(isolated_env, monkeypatch):
    async with Client(build_mcp_server()) as client:
        await client.call_tool("ensure_project", {"human_key": "/presence/failure"})
        registered = await client.call_tool(
            "register_agent",
            {
                "project_key": "/presence/failure",
                "program": "test",
                "model": "test",
                "name": "BlueLake",
            },
        )

        async def fail_touch(_agent):
            raise RuntimeError("injected presence failure")

        monkeypatch.setattr(app_module, "_touch_agent_activity", fail_touch)
        result = await client.call_tool(
            "fetch_inbox",
            {
                "project_key": "/presence/failure",
                "agent_name": registered.data["name"],
                "registration_token": registered.data["registration_token"],
            },
        )

    assert result.data == []


@pytest.mark.asyncio
async def test_adjacent_cleanup_touches_authenticated_peer_not_tokenless_target(isolated_env):
    async with Client(build_mcp_server()) as client:
        await client.call_tool("ensure_project", {"human_key": "/presence/adjacent"})
        await _create_tokenless_agent("/presence/adjacent", "GreenHill")
        peer = await client.call_tool(
            "register_agent",
            {
                "project_key": "/presence/adjacent",
                "program": "test",
                "model": "test",
                "name": "BlueLake",
            },
        )
        old = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1)
        await _set_activity("/presence/adjacent", peer.data["name"], old)

        result = await client.call_tool(
            "retire_agent",
            {
                "project_key": "/presence/adjacent",
                "agent_name": "GreenHill",
            },
        )

    assert result.data["status"] == "retired"
    assert (await _agent("/presence/adjacent", peer.data["name"])).last_active_ts > old
