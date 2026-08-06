"""M3b-1: bound Agents must be addressable members of the TeamProject namespace.

Unifies agent-bindings + routing-project agents + default_agent_id:
  * bound external agents are listed (token-free) and can become defaults
  * @BoundAgent channel mentions deliver to the agent's HOME project mailbox
  * unbound / retired bindings never resolve ("不可投递")
  * same-name bound agents are ambiguous; handle/name collisions are refused
  * support requests may run through a bound default Agent
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from authlib.jose import jwt
from fastmcp import Context
from httpx import ASGITransport, AsyncClient
from sqlmodel import select

from mcp_agent_mail import config as _config
from mcp_agent_mail.app import build_mcp_server
from mcp_agent_mail.db import get_session
from mcp_agent_mail.http import build_http_app
from mcp_agent_mail.models import (
    Agent,
    MentionDelivery,
    Message,
    MessageRecipient,
    Project,
)


def _configure_hub_jwt(monkeypatch):
    monkeypatch.setenv("HTTP_JWT_ENABLED", "true")
    monkeypatch.setenv("HTTP_JWT_ALGORITHMS", "HS256")
    monkeypatch.setenv("HTTP_JWT_SECRET", "hub-m3b-secret")
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


async def _setup_team(
    client: AsyncClient,
    root: dict[str, str],
    alice: dict[str, str],
    slug: str = "core",
) -> int:
    """全局 admin 建群 (42b039a 起建群需全局 admin) , 再把 alice 提升为群组 admin。"""
    await _register_human(client, root, "Root")
    alice_id = await _register_human(client, alice, "Alice")
    await _create_team(client, root, slug, "root")
    join = await client.post(
        f"/hub/api/projects/{slug}/join-requests",
        headers=alice,
        json={"mention_handle": "alice"},
    )
    assert join.status_code == 201
    promote = await client.patch(
        f"/hub/api/projects/{slug}/members/{alice_id}",
        headers=root,
        json={"status": "active", "role": "admin"},
    )
    assert promote.status_code == 200
    return alice_id

async def _mk_agent(project_key: str, name: str, *, owner_id: int | None = None) -> int:
    async with get_session() as session:
        slug = project_key.strip("/").replace("/", "-")
        project = Project(slug=slug, human_key=project_key)
        session.add(project)
        await session.flush()
        agent = Agent(
            project_id=project.id, name=name, program="test", model="test", owner_id=owner_id
        )
        session.add(agent)
        await session.commit()
        await session.refresh(agent)
        assert agent.id is not None
        return agent.id


async def _bind(client: AsyncClient, headers: dict[str, str], slug: str, agent_id: int) -> None:
    resp = await client.post(
        f"/hub/api/projects/{slug}/agent-bindings", headers=headers, json={"agent_id": agent_id}
    )
    assert resp.status_code in (200, 201), resp.text


async def _routing_project(slug: str) -> Project:
    async with get_session() as session:
        from mcp_agent_mail.models import TeamProject

        tp = (
            await session.execute(select(TeamProject).where(TeamProject.slug == slug))
        ).scalars().one()
        project = await session.get(Project, tp.routing_project_id)
        assert project is not None
        return project


async def _post_channel(sender_name: str, channel_project: Project, body: str) -> dict[str, Any]:
    """Post to the team channel as an agent registered in the routing project."""
    async with get_session() as session:
        sender = (
            await session.execute(
                select(Agent).where(
                    Agent.project_id == channel_project.id, Agent.name == sender_name
                )
            )
        ).scalars().one()
        from mcp_agent_mail.models import Channel, ChannelMessage

        channel = (
            await session.execute(
                select(Channel).where(
                    Channel.project_id == channel_project.id, Channel.name == "general"
                )
            )
        ).scalars().one()
        message = ChannelMessage(
            channel_id=channel.id,
            sender_id=sender.id,
            subject="coordination",
            body_md=body,
        )
        session.add(message)
        await session.commit()
        await session.refresh(message)
        return {"id": message.id}


async def _deliver(channel_project: Project, message_id: int):
    from mcp_agent_mail.app import _deliver_channel_mentions
    from mcp_agent_mail.utils import extract_channel_mentions

    async with get_session() as session:
        from mcp_agent_mail.models import ChannelMessage

        source = await session.get(ChannelMessage, message_id)
        assert source is not None
        sender = await session.get(Agent, source.sender_id)
        assert sender is not None
        names = extract_channel_mentions(source.body_md)
    return await _deliver_channel_mentions(_context(), channel_project, sender, source, names)


class _Ctx:
    async def info(self, message: str) -> None:
        pass


def _context() -> Context:
    return cast(Context, _Ctx())


async def _seed_team_agent(client: AsyncClient, headers: dict[str, str], slug: str, name: str) -> int:
    """Register an agent directly in the routing project (channel sender)."""
    project = await _routing_project(slug)
    async with get_session() as session:
        agent = Agent(project_id=project.id, name=name, program="test", model="test")
        session.add(agent)
        await session.flush()
        from mcp_agent_mail.models import Channel

        channel = (
            await session.execute(
                select(Channel).where(
                    Channel.project_id == project.id, Channel.name == "general"
                )
            )
        ).scalars().first()
        if channel is None:
            channel = Channel(project_id=project.id, name="general")
            session.add(channel)
        await session.commit()
        await session.refresh(agent)
        assert agent.id is not None
        return agent.id


def _outcome(outcomes: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for item in outcomes:
        if item["name"].lower() == name.lower():
            return item
    raise AssertionError(f"no outcome for {name}: {outcomes}")


@pytest.fixture
def hub(isolated_env, monkeypatch):
    settings = _configure_hub_jwt(monkeypatch)
    return settings, build_http_app(settings, build_mcp_server())


@pytest.mark.anyio
async def test_bound_agent_listed_token_free(hub):
    settings, app = hub
    alice = _headers(settings, "oidc|alice")
    root = _headers(settings, "oidc|root", admin=True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _setup_team(client, root, alice)
        agent_id = await _mk_agent("/workspaces/w1", "BlueLake")
        await _bind(client, alice, "core", agent_id)

        listed = await client.get("/hub/api/projects/core/agents", headers=alice)
        assert listed.status_code == 200
        agents = listed.json()["agents"]
        bound = [a for a in agents if a["id"] == agent_id]
        assert len(bound) == 1
        assert bound[0]["bound_external"] is True
        assert bound[0]["name"] == "BlueLake"
        for a in agents:
            assert "registration_token" not in a
            assert "token" not in a


@pytest.mark.anyio
async def test_bound_agent_mention_delivers_to_home_project(hub):
    settings, app = hub
    alice = _headers(settings, "oidc|alice")
    root = _headers(settings, "oidc|root", admin=True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _setup_team(client, root, alice)
        external_id = await _mk_agent("/workspaces/w1", "BlueLake")
        await _bind(client, alice, "core", external_id)
        project = await _routing_project("core")
        await _seed_team_agent(client, alice, "core", "TeamBot")

        message = await _post_channel("TeamBot", project, "@BlueLake 看一下这个")
        outcomes = await _deliver(project, message["id"])
        outcome = _outcome(outcomes, "BlueLake")
        assert outcome["status"] == "delivered"
        assert outcome["receipt_message_id"]

        async with get_session() as session:
            receipt = await session.get(Message, outcome["receipt_message_id"])
            assert receipt is not None
            # 回执落在绑定 agent 的 home 项目, agent 可用自己的 project_key 轮询到
            home = (
                await session.execute(select(Project).where(Project.human_key == "/workspaces/w1"))
            ).scalars().one()
            assert receipt.project_id == home.id
            recipient = (
                await session.execute(
                    select(MessageRecipient).where(
                        MessageRecipient.message_id == receipt.id
                    )
                )
            ).scalars().one()
            assert recipient.agent_id == external_id
            delivery = (
                await session.execute(select(MentionDelivery))
            ).scalars().one()
            assert delivery.mentioned_agent_id == external_id


@pytest.mark.anyio
async def test_unbound_or_retired_agent_not_deliverable(hub):
    settings, app = hub
    alice = _headers(settings, "oidc|alice")
    root = _headers(settings, "oidc|root", admin=True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _setup_team(client, root, alice)
        external_id = await _mk_agent("/workspaces/w1", "BlueLake")
        retired_id = await _mk_agent("/workspaces/w2", "OldDog")
        await _bind(client, alice, "core", external_id)
        await _bind(client, alice, "core", retired_id)
        project = await _routing_project("core")
        await _seed_team_agent(client, alice, "core", "TeamBot")

        # unbind BlueLake; retire OldDog
        unbound = await client.delete(
            f"/hub/api/projects/core/agent-bindings/{external_id}", headers=alice
        )
        assert unbound.status_code == 200
        async with get_session() as session:
            from datetime import datetime, timezone

            agent = await session.get(Agent, retired_id)
            assert agent is not None
            agent.retired_at = datetime.now(timezone.utc).replace(tzinfo=None)
            session.add(agent)
            await session.commit()

        message = await _post_channel("TeamBot", project, "@BlueLake 和 @OldDog 都不可达")
        outcomes = await _deliver(project, message["id"])
        assert _outcome(outcomes, "BlueLake")["reason"] == "unknown"
        assert _outcome(outcomes, "OldDog")["reason"] == "unknown"
        async with get_session() as session:
            assert (await session.execute(select(MentionDelivery))).scalars().all() == []


@pytest.mark.anyio
async def test_same_name_bound_agents_are_ambiguous(hub):
    settings, app = hub
    alice = _headers(settings, "oidc|alice")
    root = _headers(settings, "oidc|root", admin=True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _setup_team(client, root, alice)
        first = await _mk_agent("/workspaces/w1", "TwinPeak")
        second = await _mk_agent("/workspaces/w2", "TwinPeak")
        await _bind(client, alice, "core", first)
        await _bind(client, alice, "core", second)
        project = await _routing_project("core")
        await _seed_team_agent(client, alice, "core", "TeamBot")

        message = await _post_channel("TeamBot", project, "@TwinPeak 哪个?")
        outcomes = await _deliver(project, message["id"])
        outcome = _outcome(outcomes, "TwinPeak")
        assert outcome["status"] == "skipped"
        assert outcome["reason"] == "ambiguous"


@pytest.mark.anyio
async def test_bound_agent_as_default_and_support_request(hub):
    settings, app = hub
    alice = _headers(settings, "oidc|alice")
    bob = _headers(settings, "oidc|bob")
    root = _headers(settings, "oidc|root", admin=True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        alice_id = await _setup_team(client, root, alice)
        bob_id = await _register_human(client, bob, "Bob")
        join = await client.post(
            "/hub/api/projects/core/join-requests", headers=bob, json={"mention_handle": "bob"}
        )
        assert join.status_code == 201
        approve = await client.patch(
            f"/hub/api/projects/core/members/{bob_id}", headers=alice, json={"status": "active"}
        )
        assert approve.status_code == 200

        external_id = await _mk_agent("/workspaces/w1", "BlueLake")
        await _bind(client, alice, "core", external_id)

        # 外部未分配 owner 前不能设为默认
        premature = await client.patch(
            "/hub/api/projects/core/membership",
            headers=bob,
            json={"default_agent_id": external_id},
        )
        assert premature.status_code == 400

        # 外部绑定 agent 的 owner 只能由全局 admin 分配 (#965 复审结论)
        hijack = await client.patch(
            f"/hub/api/projects/core/agents/{external_id}",
            headers=alice,
            json={"owner_id": bob_id},
        )
        assert hijack.status_code == 403
        assigned = await client.patch(
            f"/hub/api/projects/core/agents/{external_id}",
            headers=root,
            json={"owner_id": bob_id},
        )
        assert assigned.status_code == 200, assigned.text

        # bob 将它设为自己的默认 agent (绑定作用域放行)
        defaulted = await client.patch(
            "/hub/api/projects/core/membership",
            headers=bob,
            json={"default_agent_id": external_id},
        )
        assert defaulted.status_code == 200, defaulted.text

        # bob 通过绑定的默认 agent 发起支持请求, @alice 可达
        support = await client.post(
            "/hub/api/projects/core/support-requests",
            headers=bob,
            json={"subject": "求 review", "body_md": "麻烦看下", "mention_handles": ["alice"]},
        )
        assert support.status_code in (200, 201), support.text
        assert alice_id


@pytest.mark.anyio
async def test_default_rejects_foreign_unbound_agent(hub):
    settings, app = hub
    alice = _headers(settings, "oidc|alice")
    root = _headers(settings, "oidc|root", admin=True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        alice_id = await _setup_team(client, root, alice)
        foreign_id = await _mk_agent("/workspaces/w1", "BlueLake", owner_id=alice_id)

        rejected = await client.patch(
            "/hub/api/projects/core/membership",
            headers=alice,
            json={"default_agent_id": foreign_id},
        )
        assert rejected.status_code == 400


@pytest.mark.anyio
async def test_bind_name_must_not_shadow_active_human_handle(hub):
    settings, app = hub
    alice = _headers(settings, "oidc|alice")
    root = _headers(settings, "oidc|root", admin=True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _setup_team(client, root, alice)
        # 与 alice 的 handle 冲突 (大小写不敏感)
        agent_id = await _mk_agent("/workspaces/w1", "Alice")
        conflict = await client.post(
            "/hub/api/projects/core/agent-bindings", headers=alice, json={"agent_id": agent_id}
        )
        assert conflict.status_code == 409


@pytest.mark.anyio
async def test_human_handle_must_not_shadow_bound_agent(hub):
    settings, app = hub
    alice = _headers(settings, "oidc|alice")
    bob = _headers(settings, "oidc|bob")
    root = _headers(settings, "oidc|root", admin=True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _setup_team(client, root, alice)
        bob_id = await _register_human(client, bob, "Bob")
        agent_id = await _mk_agent("/workspaces/w1", "WorkerBee")
        await _bind(client, alice, "core", agent_id)

        conflict = await client.post(
            "/hub/api/projects/core/join-requests", headers=bob, json={"mention_handle": "workerbee"}
        )
        assert conflict.status_code == 409
        assert bob_id


@pytest.mark.anyio
async def test_human_default_unbound_falls_back_to_inbox(hub):
    settings, app = hub
    alice = _headers(settings, "oidc|alice")
    bob = _headers(settings, "oidc|bob")
    root = _headers(settings, "oidc|root", admin=True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _setup_team(client, root, alice)
        bob_id = await _register_human(client, bob, "Bob")
        join = await client.post(
            "/hub/api/projects/core/join-requests", headers=bob, json={"mention_handle": "bob"}
        )
        assert join.status_code == 201
        await client.patch(
            f"/hub/api/projects/core/members/{bob_id}", headers=alice, json={"status": "active"}
        )
        external_id = await _mk_agent("/workspaces/w1", "BlueLake", owner_id=bob_id)
        await _bind(client, alice, "core", external_id)
        defaulted = await client.patch(
            "/hub/api/projects/core/membership",
            headers=bob,
            json={"default_agent_id": external_id},
        )
        assert defaulted.status_code == 200

        project = await _routing_project("core")
        await _seed_team_agent(client, alice, "core", "TeamBot")

        # 绑定有效时: @bob 投递到绑定默认 agent 的 home 项目
        first = await _post_channel("TeamBot", project, "@bob 第一次")
        outcomes = await _deliver(project, first["id"])
        assert _outcome(outcomes, "bob")["status"] == "delivered"

        # 解绑后: 同一默认 agent 不再可用, @bob 降级人工收件箱
        await client.delete(f"/hub/api/projects/core/agent-bindings/{external_id}", headers=alice)
        second = await _post_channel("TeamBot", project, "@bob 第二次")
        outcomes = await _deliver(project, second["id"])
        assert _outcome(outcomes, "bob")["status"] == "delivered_human_inbox"


@pytest.mark.anyio
async def test_external_agent_lifecycle_requires_global_admin_or_owner(hub):
    """#965: a group admin may bind an external agent but must not gain
    lifecycle control over the global row — owner_id is global-admin-only,
    retired is global-admin or current-owner only. Local agents keep the
    existing group admin/owner rules."""
    settings, app = hub
    alice = _headers(settings, "oidc|alice")
    bob = _headers(settings, "oidc|bob")
    root = _headers(settings, "oidc|root", admin=True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        alice_id = await _setup_team(client, root, alice)
        bob_id = await _register_human(client, bob, "Bob")
        join = await client.post(
            "/hub/api/projects/core/join-requests", headers=bob, json={"mention_handle": "bob"}
        )
        assert join.status_code == 201
        await client.patch(
            f"/hub/api/projects/core/members/{bob_id}", headers=alice, json={"status": "active"}
        )

        external_id = await _mk_agent("/workspaces/w1", "BlueLake")
        await _bind(client, alice, "core", external_id)

        # 群组 admin(非全局)对外部绑定 agent 的 owner/retired 均 403
        owner_change = await client.patch(
            f"/hub/api/projects/core/agents/{external_id}",
            headers=alice,
            json={"owner_id": alice_id},
        )
        assert owner_change.status_code == 403
        retire = await client.patch(
            f"/hub/api/projects/core/agents/{external_id}",
            headers=alice,
            json={"retired": True},
        )
        assert retire.status_code == 403

        # 非 owner 普通成员也不可以
        bob_retire = await client.patch(
            f"/hub/api/projects/core/agents/{external_id}",
            headers=bob,
            json={"retired": True},
        )
        assert bob_retire.status_code == 403

        # 全局 admin 分配 owner 给 bob
        assigned = await client.patch(
            f"/hub/api/projects/core/agents/{external_id}",
            headers=root,
            json={"owner_id": bob_id},
        )
        assert assigned.status_code == 200

        # 当前 owner(bob, 普通成员)可退休自己的 agent; 其他成员仍不行
        owner_retire = await client.patch(
            f"/hub/api/projects/core/agents/{external_id}",
            headers=bob,
            json={"retired": True},
        )
        assert owner_retire.status_code == 200
        owner_unretire = await client.patch(
            f"/hub/api/projects/core/agents/{external_id}",
            headers=bob,
            json={"retired": False},
        )
        assert owner_unretire.status_code == 200

        # 全局 admin 可退休/恢复外部 agent
        root_retire = await client.patch(
            f"/hub/api/projects/core/agents/{external_id}",
            headers=root,
            json={"retired": True},
        )
        assert root_retire.status_code == 200
        root_unretire = await client.patch(
            f"/hub/api/projects/core/agents/{external_id}",
            headers=root,
            json={"retired": False},
        )
        assert root_unretire.status_code == 200

        # routing 本地 agent: 群组 admin 的既有规则不变(可分配 owner/退休)
        project = await _routing_project("core")
        local_id = await _seed_team_agent(client, alice, "core", "LocalBot")
        local_owner = await client.patch(
            f"/hub/api/projects/core/agents/{local_id}",
            headers=alice,
            json={"owner_id": alice_id},
        )
        assert local_owner.status_code == 200
        local_retire = await client.patch(
            f"/hub/api/projects/core/agents/{local_id}",
            headers=alice,
            json={"retired": True},
        )
        assert local_retire.status_code == 200
        assert project.id
