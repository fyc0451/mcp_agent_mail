"""M3e: self-service Agent claim by team members (registration_token proof).

Selector contract (#994): source_project_slug + agent_name + registration_token
— exactly what a local registry can prove; never a local path, never a remote
numeric id. Unknown slug, unknown name, archived source project, and a wrong
token all get the same opaque 403 "Invalid agent credentials".

Acceptance (#988): constant-time token compare; agent must be active and not a
Team routing agent; owner only unset or the caller; owner + binding set
atomically; no token leakage; no group-admin power expansion.
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
from mcp_agent_mail.models import (
    Agent,
    Project,
    TeamProject,
    TeamProjectAgentBinding,
)


def _configure_hub_jwt(monkeypatch):
    monkeypatch.setenv("HTTP_JWT_ENABLED", "true")
    monkeypatch.setenv("HTTP_JWT_ALGORITHMS", "HS256")
    monkeypatch.setenv("HTTP_JWT_SECRET", "hub-claim-secret")
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


async def _setup_team(client: AsyncClient, root: dict[str, str], slug: str = "core") -> None:
    """Global admin creates the group (project creation is global-admin-only)."""
    await _register_human(client, root, "Root")
    resp = await client.post(
        "/hub/api/projects",
        headers=root,
        json={"name": f"Team {slug}", "slug": slug, "mention_handle": "root"},
    )
    assert resp.status_code == 201


async def _join_active(
    client: AsyncClient,
    admin_headers: dict[str, str],
    member_headers: dict[str, str],
    human_id: int,
    handle: str,
    slug: str = "core",
) -> None:
    join = await client.post(
        f"/hub/api/projects/{slug}/join-requests",
        headers=member_headers,
        json={"mention_handle": handle},
    )
    assert join.status_code == 201
    approve = await client.patch(
        f"/hub/api/projects/{slug}/members/{human_id}",
        headers=admin_headers,
        json={"status": "active"},
    )
    assert approve.status_code == 200


async def _mk_agent(
    project_key: str,
    name: str,
    *,
    token: str | None = "tok-secret",
    owner_id: int | None = None,
    retired: bool = False,
    archived: bool = False,
) -> tuple[int, str]:
    """Persist a workspace agent; returns (agent_id, source_project_slug)."""
    async with get_session() as session:
        slug = project_key.strip("/").replace("/", "-")
        project = Project(slug=slug, human_key=project_key)
        if archived:
            project.archived_at = datetime.now(timezone.utc).replace(tzinfo=None)
        session.add(project)
        await session.flush()
        agent = Agent(
            project_id=project.id,
            name=name,
            program="test",
            model="test",
            registration_token=token,
            owner_id=owner_id,
            retired_at=datetime.now(timezone.utc).replace(tzinfo=None) if retired else None,
        )
        session.add(agent)
        await session.commit()
        await session.refresh(agent)
        assert agent.id is not None
        return agent.id, slug


async def _routing_agent(slug: str, name: str) -> tuple[int, str]:
    """An agent living in the TeamProject's routing project (managed, not claimable)."""
    async with get_session() as session:
        tp = (
            await session.execute(select(TeamProject).where(TeamProject.slug == slug))
        ).scalars().one()
        agent = Agent(
            project_id=tp.routing_project_id,
            name=name,
            program="test",
            model="test",
            registration_token="routing-tok",
        )
        session.add(agent)
        await session.commit()
        await session.refresh(agent)
        project = await session.get(Project, tp.routing_project_id)
        assert agent.id is not None and project is not None
        return agent.id, project.slug


def _claim(
    client: AsyncClient,
    headers: dict[str, str],
    source_slug: str,
    agent_name: str,
    token: str,
    slug: str = "core",
):
    return client.post(
        f"/hub/api/projects/{slug}/agent-claims",
        headers=headers,
        json={
            "source_project_slug": source_slug,
            "agent_name": agent_name,
            "registration_token": token,
        },
    )


@pytest.fixture
def hub(isolated_env, monkeypatch):
    settings = _configure_hub_jwt(monkeypatch)
    return settings, build_http_app(settings, build_mcp_server())


@pytest.mark.anyio
async def test_member_claims_unowned_agent(hub):
    settings, app = hub
    root = _headers(settings, "oidc|root", admin=True)
    bob = _headers(settings, "oidc|bob")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _setup_team(client, root)
        bob_id = await _register_human(client, bob, "Bob")
        await _join_active(client, root, bob, bob_id, "bob")
        agent_id, slug = await _mk_agent("/workspaces/w1", "BlueLake")

        claimed = await _claim(client, bob, slug, "BlueLake", "tok-secret")
        assert claimed.status_code == 201, claimed.text
        payload = claimed.json()
        assert payload["agent"]["id"] == agent_id
        assert payload["agent"]["owner_id"] == bob_id
        assert payload["binding"]["status"] == "active"
        # token 不出现在任何响应字段
        assert "tok-secret" not in claimed.text
        assert "registration_token" not in payload["agent"]

        async with get_session() as session:
            agent = await session.get(Agent, agent_id)
            assert agent is not None and agent.owner_id == bob_id
            rows = (await session.execute(select(TeamProjectAgentBinding))).scalars().all()
            assert len(rows) == 1


@pytest.mark.anyio
async def test_claim_rejects_invalid_token_without_leak(hub):
    settings, app = hub
    root = _headers(settings, "oidc|root", admin=True)
    bob = _headers(settings, "oidc|bob")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _setup_team(client, root)
        bob_id = await _register_human(client, bob, "Bob")
        await _join_active(client, root, bob, bob_id, "bob")
        agent_id, slug = await _mk_agent("/workspaces/w1", "BlueLake")

        wrong = await _claim(client, bob, slug, "BlueLake", "wrong-token")
        assert wrong.status_code == 403
        assert wrong.json()["detail"] == "Invalid agent credentials"
        assert "wrong-token" not in wrong.text
        assert "tok-secret" not in wrong.text

        # tokenless agent cannot be claimed at all
        tokenless_id, tokenless_slug = await _mk_agent("/workspaces/w2", "NoToken", token=None)
        no_token = await _claim(client, bob, tokenless_slug, "NoToken", "whatever")
        assert no_token.status_code == 403

        async with get_session() as session:
            agent = await session.get(Agent, agent_id)
            assert agent is not None and agent.owner_id is None
            assert tokenless_id
            assert (await session.execute(select(TeamProjectAgentBinding))).scalars().all() == []


@pytest.mark.anyio
async def test_claim_unknown_selectors_are_opaque(hub):
    """Wrong source slug, wrong agent name, and archived source project are
    indistinguishable from a bad token (#994)."""
    settings, app = hub
    root = _headers(settings, "oidc|root", admin=True)
    bob = _headers(settings, "oidc|bob")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _setup_team(client, root)
        bob_id = await _register_human(client, bob, "Bob")
        await _join_active(client, root, bob, bob_id, "bob")
        _agent_id, slug = await _mk_agent("/workspaces/w1", "BlueLake")
        _archived_id, archived_slug = await _mk_agent("/workspaces/old", "Ghost", archived=True)

        for source_slug, name in (
            ("no-such-project", "BlueLake"),   # 未知 source slug
            (slug, "NotThere"),                # 未知 agent 名
            (archived_slug, "Ghost"),          # 已归档 source project
        ):
            resp = await _claim(client, bob, source_slug, name, "tok-secret")
            assert resp.status_code == 403, (source_slug, name, resp.text)
            assert resp.json()["detail"] == "Invalid agent credentials"


@pytest.mark.anyio
async def test_same_name_agents_in_different_source_projects(hub):
    """Selector 精确定位: 不同 source project 的同名 agent 互不干扰。"""
    settings, app = hub
    root = _headers(settings, "oidc|root", admin=True)
    bob = _headers(settings, "oidc|bob")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _setup_team(client, root)
        bob_id = await _register_human(client, bob, "Bob")
        await _join_active(client, root, bob, bob_id, "bob")
        first_id, slug1 = await _mk_agent("/workspaces/w1", "TwinPeak", token="tok-one")
        second_id, slug2 = await _mk_agent("/workspaces/w2", "TwinPeak", token="tok-two")

        # w2 的 token 不能认领 w1 的同名 agent
        wrong = await _claim(client, bob, slug1, "TwinPeak", "tok-two")
        assert wrong.status_code == 403

        claimed = await _claim(client, bob, slug2, "TwinPeak", "tok-two")
        assert claimed.status_code == 201
        assert claimed.json()["agent"]["id"] == second_id

        async with get_session() as session:
            first = await session.get(Agent, first_id)
            assert first is not None and first.owner_id is None


@pytest.mark.anyio
async def test_claim_requires_active_membership(hub):
    settings, app = hub
    root = _headers(settings, "oidc|root", admin=True)
    carol = _headers(settings, "oidc|carol")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _setup_team(client, root)
        await _register_human(client, carol, "Carol")
        _agent_id, slug = await _mk_agent("/workspaces/w1", "BlueLake")

        outsider = await _claim(client, carol, slug, "BlueLake", "tok-secret")
        assert outsider.status_code == 403


@pytest.mark.anyio
async def test_claim_rejects_foreign_owned_and_retired_and_routing(hub):
    settings, app = hub
    root = _headers(settings, "oidc|root", admin=True)
    bob = _headers(settings, "oidc|bob")
    alice = _headers(settings, "oidc|alice")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _setup_team(client, root)
        bob_id = await _register_human(client, bob, "Bob")
        alice_id = await _register_human(client, alice, "Alice")
        await _join_active(client, root, bob, bob_id, "bob")

        # owned by someone else (token 正确也拒绝)
        _fid, foreign_slug = await _mk_agent("/workspaces/w1", "BlueLake", owner_id=alice_id)
        owned = await _claim(client, bob, foreign_slug, "BlueLake", "tok-secret")
        assert owned.status_code == 409

        # retired
        _rid, retired_slug = await _mk_agent("/workspaces/w2", "OldDog", retired=True)
        retired = await _claim(client, bob, retired_slug, "OldDog", "tok-secret")
        assert retired.status_code == 409

        # team routing agent
        _routing_id, routing_slug = await _routing_agent("core", "TeamBot")
        routing = await _claim(client, bob, routing_slug, "TeamBot", "routing-tok")
        assert routing.status_code == 409


@pytest.mark.anyio
async def test_reclaim_idempotent_and_unbind_revives(hub):
    settings, app = hub
    root = _headers(settings, "oidc|root", admin=True)
    bob = _headers(settings, "oidc|bob")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _setup_team(client, root)
        bob_id = await _register_human(client, bob, "Bob")
        await _join_active(client, root, bob, bob_id, "bob")
        _aid, slug = await _mk_agent("/workspaces/w1", "BlueLake")

        first = await _claim(client, bob, slug, "BlueLake", "tok-secret")
        assert first.status_code == 201
        binding_id = first.json()["binding"]["id"]

        again = await _claim(client, bob, slug, "BlueLake", "tok-secret")
        assert again.status_code == 200
        assert again.json()["binding"]["id"] == binding_id

        # 解绑后重新认领复活同一历史行
        agent_id = first.json()["agent"]["id"]
        unbound = await client.delete(
            f"/hub/api/projects/core/agent-bindings/{agent_id}", headers=root
        )
        assert unbound.status_code == 200
        revived = await _claim(client, bob, slug, "BlueLake", "tok-secret")
        assert revived.status_code == 200
        assert revived.json()["binding"]["id"] == binding_id
        assert revived.json()["binding"]["status"] == "active"

        async with get_session() as session:
            rows = (await session.execute(select(TeamProjectAgentBinding))).scalars().all()
            assert len(rows) == 1


@pytest.mark.anyio
async def test_concurrent_double_claim_is_atomic(hub):
    settings, app = hub
    root = _headers(settings, "oidc|root", admin=True)
    bob = _headers(settings, "oidc|bob")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _setup_team(client, root)
        bob_id = await _register_human(client, bob, "Bob")
        await _join_active(client, root, bob, bob_id, "bob")
        agent_id, slug = await _mk_agent("/workspaces/w1", "BlueLake")

        first, second = await asyncio.gather(
            _claim(client, bob, slug, "BlueLake", "tok-secret"),
            _claim(client, bob, slug, "BlueLake", "tok-secret"),
        )
        assert sorted((first.status_code, second.status_code)) == [200, 201]
        assert first.json()["binding"]["id"] == second.json()["binding"]["id"]
        async with get_session() as session:
            rows = (await session.execute(select(TeamProjectAgentBinding))).scalars().all()
            assert len(rows) == 1
            agent = await session.get(Agent, agent_id)
            assert agent is not None and agent.owner_id == bob_id


@pytest.mark.anyio
async def test_claim_name_collides_with_active_handle(hub):
    settings, app = hub
    root = _headers(settings, "oidc|root", admin=True)
    bob = _headers(settings, "oidc|bob")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _setup_team(client, root)
        bob_id = await _register_human(client, bob, "Bob")
        await _join_active(client, root, bob, bob_id, "bob")
        # agent 名与 bob 的 handle 冲突 (大小写不敏感)
        _aid, slug = await _mk_agent("/workspaces/w1", "Bob")
        conflict = await _claim(client, bob, slug, "Bob", "tok-secret")
        assert conflict.status_code == 409


@pytest.mark.anyio
async def test_concurrent_claims_by_different_humans_cas(hub):
    """#996: two humans holding the same token race a claim; the owner CAS must
    let at most one win, and the loser must not get an active binding."""
    settings, app = hub
    root = _headers(settings, "oidc|root", admin=True)
    bob = _headers(settings, "oidc|bob")
    carol = _headers(settings, "oidc|carol")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _setup_team(client, root)
        bob_id = await _register_human(client, bob, "Bob")
        carol_id = await _register_human(client, carol, "Carol")
        await _join_active(client, root, bob, bob_id, "bob")
        await _join_active(client, root, carol, carol_id, "carol")
        agent_id, slug = await _mk_agent("/workspaces/w1", "BlueLake")

        first, second = await asyncio.gather(
            _claim(client, bob, slug, "BlueLake", "tok-secret"),
            _claim(client, carol, slug, "BlueLake", "tok-secret"),
        )
        pair = sorted((first.status_code, second.status_code))
        assert pair in ([200, 409], [201, 409]), (
            first.status_code, first.text, second.status_code, second.text
        )
        winner = first if first.status_code in (200, 201) else second
        winner_owner = winner.json()["agent"]["owner_id"]
        assert winner_owner in (bob_id, carol_id)

        async with get_session() as session:
            agent = await session.get(Agent, agent_id)
            assert agent is not None and agent.owner_id == winner_owner
            # 同一 TeamProject 内最多一条 active binding, 且属于胜者的认领
            rows = (await session.execute(select(TeamProjectAgentBinding))).scalars().all()
            assert len(rows) == 1
            assert rows[0].status == "active"
            assert rows[0].bound_by_human_id == winner_owner
