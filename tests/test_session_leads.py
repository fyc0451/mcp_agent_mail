"""M3 Session-Team: Hub-managed session-lead Agents.

Covers #1058 acceptance:
  * JWT human (active member) creates/reuses a managed lead Agent for
    (TeamProject + human + client_session_id) — no Agent Mail token involved,
    and no registration_token ever appears in responses
  * the lead is atomically set as the caller's membership default
  * team messages addressed to an ACTIVE lead fall back to the human's durable
    inbox with sender/project/message/thread preserved and dedupe
  * unbind only stops routing (default cleared, fallback off); history stays
"""

from __future__ import annotations

import pytest
from authlib.jose import jwt
from httpx import ASGITransport, AsyncClient
from sqlalchemy.exc import IntegrityError
from sqlmodel import select

from mcp_agent_mail import config as _config
from mcp_agent_mail.app import build_mcp_server
from mcp_agent_mail.db import get_session
from mcp_agent_mail.http import build_http_app
from mcp_agent_mail.models import (
    Agent,
    HumanInboxItem,
    Project,
    ProjectHumanMembership,
    SessionLeadBinding,
)


def _configure_hub_jwt(monkeypatch):
    monkeypatch.setenv("HTTP_JWT_ENABLED", "true")
    monkeypatch.setenv("HTTP_JWT_ALGORITHMS", "HS256")
    monkeypatch.setenv("HTTP_JWT_SECRET", "hub-lead-secret")
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


def _upsert_lead(
    client: AsyncClient,
    headers: dict[str, str],
    session_id: str,
    slug: str = "core",
    label: str = "Mac 终端",
):
    return client.put(
        f"/hub/api/projects/{slug}/session-lead",
        headers=headers,
        json={"client_session_id": session_id, "lead_label": label},
    )


def _delete_lead(client: AsyncClient, headers: dict[str, str], session_id: str, slug: str = "core"):
    return client.request(
        "DELETE",
        f"/hub/api/projects/{slug}/session-lead",
        headers=headers,
        json={"client_session_id": session_id},
    )


@pytest.fixture
def hub(isolated_env, monkeypatch):
    settings = _configure_hub_jwt(monkeypatch)
    return settings, build_http_app(settings, build_mcp_server())


@pytest.mark.anyio
async def test_create_lead_sets_default_atomically_and_hides_token(hub):
    settings, app = hub
    root = _headers(settings, "oidc|root", admin=True)
    bob = _headers(settings, "oidc|bob")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _setup_team(client, root)
        bob_id = await _register_human(client, bob, "Bob")
        await _join_active(client, root, bob, bob_id, "bob")

        created = await _upsert_lead(client, bob, "mac-terminal-1")
        assert created.status_code == 201, created.text
        payload = created.json()
        assert payload["client_session_id"] == "mac-terminal-1"
        assert payload["active"] is True
        assert payload["membership_default_agent_id"] == payload["agent"]["id"]
        assert payload["agent"]["owner_id"] == bob_id
        assert payload["binding"]["status"] == "active"
        assert payload["binding"]["client_session_id"] == "mac-terminal-1"
        assert payload["agent"]["name"].startswith("SessionLead")
        assert "Mac" in payload["agent"]["name"]
        # 全程无 Agent Mail token: 响应里绝不出 registration_token
        assert "registration_token" not in created.text

        async with get_session() as session:
            agent = await session.get(Agent, payload["agent"]["id"])
            assert agent is not None
            assert agent.registration_token is None  # 不签发任何 agent token
            membership = (
                await session.execute(
                    select(ProjectHumanMembership).where(
                        ProjectHumanMembership.human_id == bob_id
                    )
                )
            ).scalars().one()
            assert membership.default_agent_id == agent.id

        # 复用: 幂等返回同一 binding + agent
        again = await _upsert_lead(client, bob, "mac-terminal-1")
        assert again.status_code == 200
        assert again.json()["binding"]["id"] == payload["binding"]["id"]
        assert again.json()["agent"]["id"] == payload["agent"]["id"]
        async with get_session() as session:
            rows = (await session.execute(select(SessionLeadBinding))).scalars().all()
            assert len(rows) == 1


@pytest.mark.anyio
async def test_lead_requires_membership_and_valid_session_id(hub):
    settings, app = hub
    root = _headers(settings, "oidc|root", admin=True)
    bob = _headers(settings, "oidc|bob")
    carol = _headers(settings, "oidc|carol")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _setup_team(client, root)
        bob_id = await _register_human(client, bob, "Bob")
        await _register_human(client, carol, "Carol")
        await _join_active(client, root, bob, bob_id, "bob")

        # 非成员 403
        outsider = await _upsert_lead(client, carol, "s1")
        assert outsider.status_code == 403
        # 未认证 401
        unauthenticated = await client.post(
            "/hub/api/projects/core/session-leads", json={"client_session_id": "s1"}
        )
        assert unauthenticated.status_code == 401
        # 非法 client_session_id 400
        for bad in ("", "has space", "x" * 129, "-lead"):
            resp = await _upsert_lead(client, bob, bad)
            assert resp.status_code == 400, bad


@pytest.mark.anyio
async def test_lead_messages_fall_back_to_human_inbox_with_dedupe(hub):
    settings, app = hub
    root = _headers(settings, "oidc|root", admin=True)
    alice = _headers(settings, "oidc|alice")
    bob = _headers(settings, "oidc|bob")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _setup_team(client, root)
        alice_id = await _register_human(client, alice, "Alice")
        bob_id = await _register_human(client, bob, "Bob")
        await _join_active(client, root, alice, alice_id, "alice")
        await _join_active(client, root, bob, bob_id, "bob")
        # alice 作为普通成员先注册自己的 lead(发送方)
        await _upsert_lead(client, alice, "wsl-1")
        # bob 的 lead(接收方)
        bob_lead = await _upsert_lead(client, bob, "mac-1")
        bob_agent_id = bob_lead.json()["agent"]["id"]

        # alice 通过支持请求 @bob → 投递到 bob 的 lead(其默认),再回落 bob 收件箱
        support = await client.post(
            "/hub/api/projects/core/support-requests",
            headers=alice,
            json={"subject": "求确认", "body_md": "麻烦看下", "mention_handles": ["bob"]},
        )
        assert support.status_code == 201, support.text

        async with get_session() as session:
            items = (
                await session.execute(
                    select(HumanInboxItem).where(HumanInboxItem.human_id == bob_id)
                )
            ).scalars().all()
            assert len(items) == 1
            item = items[0]
            assert item.source_channel_message_id is not None
            from mcp_agent_mail.models import Message

            message = await session.get(Message, item.message_id)
            assert message is not None
            assert message.subject == "求确认"
            assert message.sender_id is not None  # sender/project/message 保留
            project = await session.get(Project, item.project_id)
            assert project is not None

        # 去重: 同一来源消息再次投递(重放)不产生第二条收件箱条目
        from mcp_agent_mail.app import _deliver_channel_mentions
        from mcp_agent_mail.models import ChannelMessage

        async with get_session() as session:
            source = (
                await session.execute(
                    select(ChannelMessage).where(
                        ChannelMessage.id == items[0].source_channel_message_id
                    )
                )
            ).scalars().one()
            sender = await session.get(Agent, source.sender_id)
            routing = await session.get(Project, items[0].project_id)
            assert sender is not None and routing is not None

        class _Ctx:
            async def info(self, message: str) -> None:
                pass

        from typing import cast as _cast

        from fastmcp import Context as _Context

        await _deliver_channel_mentions(_cast(_Context, _Ctx()), routing, sender, source, ["bob"])
        async with get_session() as session:
            items = (
                await session.execute(
                    select(HumanInboxItem).where(HumanInboxItem.human_id == bob_id)
                )
            ).scalars().all()
            assert len(items) == 1  # 仍只有一条

        # 解除绑定后: 新消息不再回落
        unbound = await _delete_lead(client, bob, "mac-1")
        assert unbound.status_code == 200
        assert unbound.json()["status"] == "unbound"
        membership_view = await client.get("/hub/api/projects/core/membership", headers=bob)
        assert membership_view.json()["default_agent_id"] is None

        # 解绑后直接 @lead agent 名: 投递到 agent 邮箱,但不再回落人工收件箱
        async with get_session() as session:
            lead_agent = await session.get(Agent, bob_agent_id)
            assert lead_agent is not None
            routing_row = await session.execute(
                select(Project).where(Project.id == lead_agent.project_id)
            )
            routing = routing_row.scalars().one()
            sender_row = await session.execute(
                select(Agent).where(
                    Agent.project_id == routing.id, Agent.name != lead_agent.name
                )
            )
            alice_sender = sender_row.scalars().first()
            assert alice_sender is not None
            msg = ChannelMessage(
                channel_id=source.channel_id,
                sender_id=alice_sender.id,
                subject="direct-to-lead",
                body_md=f"@{lead_agent.name} 直接点名",
            )
            session.add(msg)
            await session.commit()
            await session.refresh(msg)
        await _deliver_channel_mentions(_cast(_Context, _Ctx()), routing, alice_sender, msg, [lead_agent.name])
        async with get_session() as session:
            items = (
                await session.execute(
                    select(HumanInboxItem).where(HumanInboxItem.human_id == bob_id)
                )
            ).scalars().all()
            assert len(items) == 1  # 解绑后不再回落新增
        assert bob_agent_id


@pytest.mark.anyio
async def test_unbind_keeps_history_and_reupsert_revives(hub):
    settings, app = hub
    root = _headers(settings, "oidc|root", admin=True)
    bob = _headers(settings, "oidc|bob")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _setup_team(client, root)
        bob_id = await _register_human(client, bob, "Bob")
        await _join_active(client, root, bob, bob_id, "bob")

        created = await _upsert_lead(client, bob, "mac-1")
        binding_id = created.json()["binding"]["id"]
        agent_id = created.json()["agent"]["id"]

        unbound = await _delete_lead(client, bob, "mac-1")
        assert unbound.status_code == 200
        assert unbound.json()["id"] == binding_id
        assert unbound.json()["status"] == "unbound"

        # 历史保留: 列表仍可见 unbound 行, agent 行仍在
        listed = await client.get("/hub/api/projects/core/session-lead", headers=bob)
        assert listed.status_code == 200
        rows = listed.json()["bindings"]
        assert len(rows) == 1 and rows[0]["status"] == "unbound"
        assert "SessionLead" not in listed.text
        async with get_session() as session:
            agent = await session.get(Agent, agent_id)
            assert agent is not None

        # 重新 upsert 复活同一 binding 行与同一 agent
        revived = await _upsert_lead(client, bob, "mac-1")
        assert revived.status_code == 200
        assert revived.json()["binding"]["id"] == binding_id
        assert revived.json()["agent"]["id"] == agent_id
        assert revived.json()["binding"]["status"] == "active"

        # 直接停用 agent 后再 upsert: 受管生命周期自动恢复 active
        async with get_session() as session:
            from datetime import datetime, timezone

            agent = await session.get(Agent, agent_id)
            assert agent is not None
            agent.retired_at = datetime.now(timezone.utc).replace(tzinfo=None)
            session.add(agent)
            await session.commit()
        reactivated = await _upsert_lead(client, bob, "mac-1")
        assert reactivated.status_code == 200
        assert reactivated.json()["agent"]["retired"] is False

        # 解绑不存在的 session 404
        missing = await _delete_lead(client, bob, "nope")
        assert missing.status_code == 404


@pytest.mark.anyio
async def test_session_lead_hidden_from_agent_lists(hub):
    """team-session-lead 必须与 team-human-relay 一样从普通管理列表隐藏。"""
    settings, app = hub
    root = _headers(settings, "oidc|root", admin=True)
    bob = _headers(settings, "oidc|bob")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _setup_team(client, root)
        bob_id = await _register_human(client, bob, "Bob")
        await _join_active(client, root, bob, bob_id, "bob")
        await _upsert_lead(client, bob, "mac-1")

        listed = await client.get("/hub/api/projects/core/agents", headers=bob)
        assert listed.status_code == 200
        programs = [a.get("program") for a in listed.json()["agents"]]
        assert "team-session-lead" not in programs
        assert "team-human-relay" not in programs


@pytest.mark.anyio
async def test_fallback_inbox_item_kind_is_stable_enum(hub):
    settings, app = hub
    root = _headers(settings, "oidc|root", admin=True)
    alice = _headers(settings, "oidc|alice")
    bob = _headers(settings, "oidc|bob")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _setup_team(client, root)
        alice_id = await _register_human(client, alice, "Alice")
        bob_id = await _register_human(client, bob, "Bob")
        await _join_active(client, root, alice, alice_id, "alice")
        await _join_active(client, root, bob, bob_id, "bob")
        await _upsert_lead(client, alice, "wsl-1")
        await _upsert_lead(client, bob, "mac-1")

        support = await client.post(
            "/hub/api/projects/core/support-requests",
            headers=alice,
            json={"subject": "kind 检查", "body_md": "x", "mention_handles": ["bob"]},
        )
        assert support.status_code == 201
        async with get_session() as session:
            item = (
                await session.execute(
                    select(HumanInboxItem).where(HumanInboxItem.human_id == bob_id)
                )
            ).scalars().one()
            assert item.kind == "session_lead"
            assert item.source_channel_message_id is not None


@pytest.mark.anyio
async def test_delete_only_clears_default_while_active(hub):
    """契约: 仅当记录仍 active 且 default 指向该 lead 时才清 default。"""
    settings, app = hub
    root = _headers(settings, "oidc|root", admin=True)
    bob = _headers(settings, "oidc|bob")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _setup_team(client, root)
        bob_id = await _register_human(client, bob, "Bob")
        await _join_active(client, root, bob, bob_id, "bob")
        created = await _upsert_lead(client, bob, "mac-1")
        agent_id = created.json()["agent"]["id"]

        # 第一次 DELETE: active + default=lead -> 清 default
        first = await _delete_lead(client, bob, "mac-1")
        assert first.status_code == 200
        membership = await client.get("/hub/api/projects/core/membership", headers=bob)
        assert membership.json()["default_agent_id"] is None

        # 手动把 default 再指回 lead(此时 binding 已 unbound)
        restore = await client.patch(
            "/hub/api/projects/core/membership",
            headers=bob,
            json={"default_agent_id": agent_id},
        )
        assert restore.status_code == 200

        # 第二次 DELETE: 记录已 unbound -> default 必须保留(幂等不动)
        second = await _delete_lead(client, bob, "mac-1")
        assert second.status_code == 200
        assert second.json()["status"] == "unbound"
        membership = await client.get("/hub/api/projects/core/membership", headers=bob)
        assert membership.json()["default_agent_id"] == agent_id


@pytest.mark.anyio
async def test_chat_history_shows_human_via_lead(hub):
    """#1063: 受管 lead 作为 support sender 时,chat history 必须把 sender 归
    Human(display_name/human_id/kind),lead 名只经 sender_agent 透出。"""
    settings, app = hub
    root = _headers(settings, "oidc|root", admin=True)
    alice = _headers(settings, "oidc|alice")
    bob = _headers(settings, "oidc|bob")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _setup_team(client, root)
        alice_id = await _register_human(client, alice, "Alice")
        bob_id = await _register_human(client, bob, "Bob")
        await _join_active(client, root, alice, alice_id, "alice")
        await _join_active(client, root, bob, bob_id, "bob")
        await _upsert_lead(client, alice, "wsl-1")
        bob_lead = await _upsert_lead(client, bob, "mac-1")
        lead_name = bob_lead.json()["agent"]["name"]

        # bob 以自己的 lead 为默认发 support(bob 是 sender)
        posted = await client.post(
            "/hub/api/projects/core/support-requests",
            headers=bob,
            json={"subject": "via lead", "body_md": "来自 bob", "mention_handles": ["alice"]},
        )
        assert posted.status_code == 201
        assert posted.json()["sender_kind"] == "session_lead"
        assert posted.json()["sender_agent"] == "Mac 终端"
        assert lead_name not in posted.text

        history = await client.get("/hub/api/projects/core/chat/messages", headers=alice)
        assert history.status_code == 200
        messages = history.json()["messages"]
        assert messages
        own = next(m for m in messages if m["subject"] == "via lead")
        assert own["sender_name"] == "Bob"
        assert own["sender_human_id"] == bob_id
        assert own["sender_kind"] == "session_lead"
        # sender_agent 是客户端 lead_label,内部 Agent 名(hash)永不透出
        assert own["sender_agent"] == "Mac 终端"
        assert own["sender_name"] != lead_name
        assert lead_name not in str(own)


@pytest.mark.anyio
async def test_single_active_lead_per_human_project(hub):
    """#1064 收口 1: 同一 TeamProject+Human 任意时刻仅一个 active binding;
    并发两个不同 client_session_id 也必须收敛为一个 active。"""
    import asyncio

    settings, app = hub
    root = _headers(settings, "oidc|root", admin=True)
    bob = _headers(settings, "oidc|bob")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _setup_team(client, root)
        bob_id = await _register_human(client, bob, "Bob")
        await _join_active(client, root, bob, bob_id, "bob")

        # 顺序: 新 id upsert 后旧 id 转 unbound
        first = await _upsert_lead(client, bob, "session-old")
        assert first.status_code == 201
        second = await _upsert_lead(client, bob, "session-new")
        assert second.status_code == 201
        assert first.json()["agent"]["id"] != second.json()["agent"]["id"]
        async with get_session() as session:
            rows = (
                await session.execute(
                    select(SessionLeadBinding).where(
                        SessionLeadBinding.human_id == bob_id
                    )
                )
            ).scalars().all()
            active = [row for row in rows if row.status == "active"]
            assert len(rows) == 2
            assert len(active) == 1
            assert active[0].client_session_id == "session-new"
        membership = await client.get("/hub/api/projects/core/membership", headers=bob)
        assert membership.json()["default_agent_id"] == second.json()["agent"]["id"]

        # 并发: 两个不同 id 同时 PUT,最终恰好一个 active 且 default 指向它
        first, other = await asyncio.gather(
            _upsert_lead(client, bob, "race-a"),
            _upsert_lead(client, bob, "race-b"),
        )
        assert first.status_code in (200, 201) and other.status_code in (200, 201)
        async with get_session() as session:
            rows = (
                await session.execute(
                    select(SessionLeadBinding).where(
                        SessionLeadBinding.human_id == bob_id,
                        SessionLeadBinding.status == "active",
                    )
                )
            ).scalars().all()
            assert len(rows) == 1
            winner_id = rows[0].agent_id
            active_binding = rows[0]
            winner_agent = await session.get(Agent, winner_id)
            assert winner_agent is not None
            conflicting_agent = Agent(
                project_id=winner_agent.project_id,
                name="SessionLead-db-conflict",
                program="team-session-lead",
                model="hub",
                owner_id=bob_id,
            )
            session.add(conflicting_agent)
            await session.flush()
            assert conflicting_agent.id is not None
            session.add(
                SessionLeadBinding(
                    team_project_id=active_binding.team_project_id,
                    human_id=bob_id,
                    client_session_id="db-conflict",
                    agent_id=conflicting_agent.id,
                    lead_label="db-conflict",
                    status="active",
                )
            )
            with pytest.raises(IntegrityError):
                await session.commit()
            await session.rollback()
        membership = await client.get("/hub/api/projects/core/membership", headers=bob)
        assert membership.json()["default_agent_id"] == winner_id


@pytest.mark.anyio
async def test_inbox_shows_human_and_lead_label_not_internal_name(hub):
    """#1064 收口 3: Human Inbox 对 lead sender 显示 display_name+lead_label,
    内部 hash 名不透出。"""
    settings, app = hub
    root = _headers(settings, "oidc|root", admin=True)
    alice = _headers(settings, "oidc|alice")
    bob = _headers(settings, "oidc|bob")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _setup_team(client, root)
        alice_id = await _register_human(client, alice, "Alice")
        bob_id = await _register_human(client, bob, "Bob")
        await _join_active(client, root, alice, alice_id, "alice")
        await _join_active(client, root, bob, bob_id, "bob")
        alice_lead = await _upsert_lead(client, alice, "wsl-1", label="codex-main")
        await _upsert_lead(client, bob, "mac-1", label="kimi-main")
        lead_name = alice_lead.json()["agent"]["name"]

        support = await client.post(
            "/hub/api/projects/core/support-requests",
            headers=alice,
            json={"subject": "inbox 显示", "body_md": "x", "mention_handles": ["bob"]},
        )
        assert support.status_code == 201

        inbox = await client.get("/hub/api/inbox", headers=bob)
        assert inbox.status_code == 200
        items = inbox.json()["items"]
        assert items
        own = next(item for item in items if item["subject"] == "inbox 显示")
        assert own["sender_name"] == "Alice"
        assert own["sender_kind"] == "session_lead"
        assert own["sender_agent"] == "codex-main"
        assert lead_name not in str(own)


@pytest.mark.anyio
async def test_lead_label_in_binding_payload(hub):
    settings, app = hub
    root = _headers(settings, "oidc|root", admin=True)
    bob = _headers(settings, "oidc|bob")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _setup_team(client, root)
        bob_id = await _register_human(client, bob, "Bob")
        await _join_active(client, root, bob, bob_id, "bob")
        created = await _upsert_lead(client, bob, "mac-1", label="codex-main")
        assert created.status_code == 201
        assert created.json()["binding"]["lead_label"] == "codex-main"

        # label 校验: 空/超长/控制字符均 400
        for bad_label in ("", "x" * 129, "bad\nlabel"):
            resp = await client.put(
                "/hub/api/projects/core/session-lead",
                headers=bob,
                json={"client_session_id": "s2", "lead_label": bad_label},
            )
            assert resp.status_code == 400, repr(bad_label)
