"""M3a: explicit Agent ↔ TeamProject binding API.

Covers the #931 acceptance points:
  * binding accepts only a TeamProject slug + an existing agent id (no paths)
  * only a global admin (JWT role claim carries 'admin') or the group's
    active admin may bind/unbind; cross-group misuse is rejected
  * bind is idempotent and safe under concurrent double-bind
  * unbind keeps the row as history; re-binding revives the same row
  * retired/missing agents cannot be bound
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest
from authlib.jose import jwt
from httpx import ASGITransport, AsyncClient
from sqlmodel import select

from mcp_agent_mail import config as _config
from mcp_agent_mail.app import build_mcp_server
from mcp_agent_mail.db import get_session
from mcp_agent_mail.http import build_http_app
from mcp_agent_mail.models import Agent, Project, TeamProject, TeamProjectAgentBinding


def _configure_hub_jwt(monkeypatch):
    monkeypatch.setenv("HTTP_JWT_ENABLED", "true")
    monkeypatch.setenv("HTTP_JWT_ALGORITHMS", "HS256")
    monkeypatch.setenv("HTTP_JWT_SECRET", "hub-binding-secret")
    monkeypatch.setenv("HTTP_RBAC_ENABLED", "true")
    monkeypatch.setenv("HTTP_RBAC_WRITER_ROLES", "writer")
    monkeypatch.setenv("HTTP_ALLOW_LOCALHOST_UNAUTHENTICATED", "false")
    _config.clear_settings_cache()
    return _config.get_settings()


def _headers(settings, subject: str, *, admin: bool = False) -> dict[str, str]:
    roles = ["writer", "admin"] if admin else ["writer"]
    token = jwt.encode(
        {"alg": "HS256"},
        {"sub": subject, settings.http.jwt_role_claim: roles},
        settings.http.jwt_secret,
    ).decode("utf-8")
    return {"Authorization": f"Bearer {token}"}


async def _register_human(client: AsyncClient, headers: dict[str, str], name: str) -> int:
    resp = await client.put("/hub/api/humans/me", headers=headers, json={"display_name": name})
    assert resp.status_code == 200
    return resp.json()["id"]


async def _create_team(client: AsyncClient, headers: dict[str, str], slug: str, handle: str) -> dict:
    resp = await client.post(
        "/hub/api/projects",
        headers=headers,
        json={"name": f"Team {slug}", "slug": slug, "mention_handle": handle},
    )
    assert resp.status_code == 201
    return resp.json()


async def _mk_workspace_agent(name: str, *, owner_id: int | None = None) -> int:
    """An existing Agent identity living in an unrelated workspace project."""
    async with get_session() as session:
        project = Project(slug=f"workspace-{name.lower()}", human_key=f"/workspaces/{name.lower()}")
        session.add(project)
        await session.flush()
        agent = Agent(
            project_id=project.id,
            name=name,
            program="test",
            model="test",
            owner_id=owner_id,
        )
        session.add(agent)
        await session.commit()
        await session.refresh(agent)
        assert agent.id is not None
        return agent.id


async def _join_and_approve(
    client: AsyncClient,
    admin_headers: dict[str, str],
    member_headers: dict[str, str],
    slug: str,
    handle: str,
    member_human_id: int,
) -> None:
    join = await client.post(
        f"/hub/api/projects/{slug}/join-requests",
        headers=member_headers,
        json={"mention_handle": handle},
    )
    assert join.status_code == 201
    approve = await client.patch(
        f"/hub/api/projects/{slug}/members/{member_human_id}",
        headers=admin_headers,
        json={"status": "active"},
    )
    assert approve.status_code == 200


@pytest.fixture
def hub(isolated_env, monkeypatch):
    settings = _configure_hub_jwt(monkeypatch)
    return settings, build_http_app(settings, build_mcp_server())


@pytest.mark.anyio
async def test_agent_directory_is_scoped_token_free_and_excludes_team_routing(hub):
    settings, app = hub
    alice = _headers(settings, "oidc|alice")
    root = _headers(settings, "oidc|root", admin=True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        alice_id = await _register_human(client, alice, "Alice")
        root_id = await _register_human(client, root, "Root")
        await _create_team(client, root, "core", "root")
        owned_id = await _mk_workspace_agent("BlueLake", owner_id=alice_id)
        other_id = await _mk_workspace_agent("RedStone", owner_id=root_id)
        unowned_id = await _mk_workspace_agent("GreenCastle")

        async with get_session() as session:
            team = (
                await session.execute(select(TeamProject).where(TeamProject.slug == "core"))
            ).scalars().one()
            routing_agent = Agent(
                project_id=team.routing_project_id,
                name="TeamOnly",
                program="test",
                model="test",
                owner_id=alice_id,
            )
            session.add(routing_agent)
            await session.commit()
            await session.refresh(routing_agent)
            routing_id = routing_agent.id

        mine = await client.get("/hub/api/agents", headers=alice)
        assert mine.status_code == 200
        assert [item["id"] for item in mine.json()["agents"]] == [owned_id]

        all_agents = await client.get("/hub/api/agents", headers=root)
        assert all_agents.status_code == 200
        payloads = all_agents.json()["agents"]
        assert {item["id"] for item in payloads} == {owned_id, other_id, unowned_id}
        assert routing_id not in {item["id"] for item in payloads}
        for item in payloads:
            assert "registration_token" not in item
            assert "token" not in item
            assert "human_key" not in item
            assert item["project_slug"].startswith("workspace-")


@pytest.mark.anyio
async def test_group_admin_binds_and_rebind_is_idempotent(hub):
    settings, app = hub
    alice = _headers(settings, "oidc|alice")
    alice_admin = _headers(settings, "oidc|alice", admin=True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _register_human(client, alice, "Alice")
        await _create_team(client, alice_admin, "core", "alice")
        agent_id = await _mk_workspace_agent("BlueLake")

        bound = await client.post(
            "/hub/api/projects/core/agent-bindings", headers=alice, json={"agent_id": agent_id}
        )
        assert bound.status_code == 201
        payload = bound.json()
        assert payload["agent_id"] == agent_id
        assert payload["status"] == "active"
        assert payload["bound_by_human_id"]

        again = await client.post(
            "/hub/api/projects/core/agent-bindings", headers=alice, json={"agent_id": agent_id}
        )
        assert again.status_code == 200
        assert again.json()["id"] == payload["id"]

        async with get_session() as session:
            rows = (await session.execute(select(TeamProjectAgentBinding))).scalars().all()
            assert len(rows) == 1


@pytest.mark.anyio
async def test_global_admin_binds_without_membership(hub):
    settings, app = hub
    alice = _headers(settings, "oidc|alice")
    alice_admin = _headers(settings, "oidc|alice", admin=True)
    root = _headers(settings, "oidc|root", admin=True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _register_human(client, alice, "Alice")
        await _register_human(client, root, "Root")
        await _create_team(client, alice_admin, "core", "alice")
        agent_id = await _mk_workspace_agent("RedStone")

        bound = await client.post(
            "/hub/api/projects/core/agent-bindings", headers=root, json={"agent_id": agent_id}
        )
        assert bound.status_code == 201
        assert bound.json()["status"] == "active"


@pytest.mark.anyio
async def test_non_admin_cannot_bind(hub):
    settings, app = hub
    alice = _headers(settings, "oidc|alice")
    alice_admin = _headers(settings, "oidc|alice", admin=True)
    bob = _headers(settings, "oidc|bob")
    carol = _headers(settings, "oidc|carol")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _register_human(client, alice, "Alice")
        bob_id = await _register_human(client, bob, "Bob")
        carol_id = await _register_human(client, carol, "Carol")
        await _create_team(client, alice_admin, "core", "alice")
        await _join_and_approve(client, alice, bob, "core", "bob", bob_id)
        agent_id = await _mk_workspace_agent("GreenCastle")

        # ordinary active member
        member = await client.post(
            "/hub/api/projects/core/agent-bindings", headers=bob, json={"agent_id": agent_id}
        )
        assert member.status_code == 403
        # non-member
        outsider = await client.post(
            "/hub/api/projects/core/agent-bindings", headers=carol, json={"agent_id": agent_id}
        )
        assert outsider.status_code == 403
        assert carol_id


@pytest.mark.anyio
async def test_cross_group_admin_cannot_bind_elsewhere(hub):
    settings, app = hub
    alice = _headers(settings, "oidc|alice")
    alice_admin = _headers(settings, "oidc|alice", admin=True)
    bob = _headers(settings, "oidc|bob")
    bob_admin = _headers(settings, "oidc|bob", admin=True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _register_human(client, alice, "Alice")
        await _register_human(client, bob, "Bob")
        await _create_team(client, alice_admin, "core", "alice")
        await _create_team(client, bob_admin, "other", "bob")
        agent_id = await _mk_workspace_agent("WhitePeak")

        # bob is admin of 'other' but has no membership in 'core'
        cross = await client.post(
            "/hub/api/projects/core/agent-bindings", headers=bob, json={"agent_id": agent_id}
        )
        assert cross.status_code == 403


@pytest.mark.anyio
async def test_bind_validates_agent(hub):
    settings, app = hub
    alice = _headers(settings, "oidc|alice")
    alice_admin = _headers(settings, "oidc|alice", admin=True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _register_human(client, alice, "Alice")
        await _create_team(client, alice_admin, "core", "alice")

        missing = await client.post(
            "/hub/api/projects/core/agent-bindings", headers=alice, json={"agent_id": 999999}
        )
        assert missing.status_code == 404

        bad_type = await client.post(
            "/hub/api/projects/core/agent-bindings", headers=alice, json={"agent_id": "1"}
        )
        assert bad_type.status_code == 400

        retired_id = await _mk_workspace_agent("OldDog")
        async with get_session() as session:
            agent = await session.get(Agent, retired_id)
            assert agent is not None
            agent.retired_at = datetime.now(timezone.utc).replace(tzinfo=None)
            session.add(agent)
            await session.commit()
        retired = await client.post(
            "/hub/api/projects/core/agent-bindings", headers=alice, json={"agent_id": retired_id}
        )
        assert retired.status_code == 409


@pytest.mark.anyio
async def test_unbind_keeps_history_and_rebind_revives(hub):
    settings, app = hub
    alice = _headers(settings, "oidc|alice")
    alice_admin = _headers(settings, "oidc|alice", admin=True)
    bob = _headers(settings, "oidc|bob")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _register_human(client, alice, "Alice")
        bob_id = await _register_human(client, bob, "Bob")
        await _create_team(client, alice_admin, "core", "alice")
        await _join_and_approve(client, alice, bob, "core", "bob", bob_id)
        agent_id = await _mk_workspace_agent("BlueLake")

        bound = await client.post(
            "/hub/api/projects/core/agent-bindings", headers=alice, json={"agent_id": agent_id}
        )
        binding_id = bound.json()["id"]

        unbound = await client.delete(
            f"/hub/api/projects/core/agent-bindings/{agent_id}", headers=alice
        )
        assert unbound.status_code == 200
        assert unbound.json()["status"] == "unbound"

        # unbind is idempotent and keeps the same history row
        again = await client.delete(
            f"/hub/api/projects/core/agent-bindings/{agent_id}", headers=alice
        )
        assert again.status_code == 200
        assert again.json()["id"] == binding_id
        assert again.json()["status"] == "unbound"

        # ordinary member sees only active bindings; admin sees history
        member_list = await client.get("/hub/api/projects/core/agent-bindings", headers=bob)
        assert member_list.status_code == 200
        assert member_list.json()["bindings"] == []
        admin_list = await client.get("/hub/api/projects/core/agent-bindings", headers=alice)
        admin_bindings = admin_list.json()["bindings"]
        assert len(admin_bindings) == 1
        assert admin_bindings[0]["status"] == "unbound"
        assert admin_bindings[0]["agent_name"] == "BlueLake"

        # re-binding revives the SAME row
        revived = await client.post(
            "/hub/api/projects/core/agent-bindings", headers=alice, json={"agent_id": agent_id}
        )
        assert revived.status_code == 200
        assert revived.json()["id"] == binding_id
        assert revived.json()["status"] == "active"

        # unbinding a nonexistent binding 404s
        missing = await client.delete(
            "/hub/api/projects/core/agent-bindings/424242", headers=alice
        )
        assert missing.status_code == 404

        async with get_session() as session:
            rows = (await session.execute(select(TeamProjectAgentBinding))).scalars().all()
            assert len(rows) == 1


@pytest.mark.anyio
async def test_concurrent_double_bind_yields_single_row(hub):
    settings, app = hub
    alice = _headers(settings, "oidc|alice")
    alice_admin = _headers(settings, "oidc|alice", admin=True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _register_human(client, alice, "Alice")
        await _create_team(client, alice_admin, "core", "alice")
        agent_id = await _mk_workspace_agent("BlueLake")

        first, second = await asyncio.gather(
            client.post(
                "/hub/api/projects/core/agent-bindings", headers=alice, json={"agent_id": agent_id}
            ),
            client.post(
                "/hub/api/projects/core/agent-bindings", headers=alice, json={"agent_id": agent_id}
            ),
        )
        assert sorted((first.status_code, second.status_code)) == [200, 201]
        assert first.json()["id"] == second.json()["id"]
        async with get_session() as session:
            rows = (await session.execute(select(TeamProjectAgentBinding))).scalars().all()
            assert len(rows) == 1


@pytest.mark.anyio
async def test_binding_requires_jwt(hub):
    settings, app = hub
    alice = _headers(settings, "oidc|alice")
    alice_admin = _headers(settings, "oidc|alice", admin=True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _register_human(client, alice, "Alice")
        await _create_team(client, alice_admin, "core", "alice")
        agent_id = await _mk_workspace_agent("BlueLake")

        unauthenticated = await client.post(
            "/hub/api/projects/core/agent-bindings", json={"agent_id": agent_id}
        )
        assert unauthenticated.status_code == 401
