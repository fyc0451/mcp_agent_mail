"""M3a: @Human channel mentions → default agent or durable human inbox.

Routing rules under test:
  * @<mention_handle> of an ACTIVE membership resolves project-locally
  * membership with a usable default agent → ordinary agent mention receipt
  * membership without one (or with a retired default) → HumanInboxItem
  * invited/removed memberships and unknown names stay ``unknown``
  * delivery is idempotent per (source channel message, human)
  * the human inbox HTTP API is JWT-scoped and mark-read only touches own rows
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from authlib.jose import jwt
from fastmcp import Client, Context
from httpx import ASGITransport, AsyncClient
from sqlmodel import select

from mcp_agent_mail import config as _config
from mcp_agent_mail.app import _deliver_channel_mentions, build_mcp_server
from mcp_agent_mail.db import get_session
from mcp_agent_mail.http import build_http_app
from mcp_agent_mail.models import (
    Agent,
    ChannelMessage,
    HubAuditEvent,
    Human,
    HumanInboxItem,
    MentionDelivery,
    MessageRecipient,
    Project,
    ProjectHumanMembership,
    TeamProject,
)


class _RecordingContext:
    def __init__(self) -> None:
        self.infos: list[str] = []

    async def info(self, message: str) -> None:
        self.infos.append(message)


def _context() -> Context:
    return cast(Context, _RecordingContext())


async def _register(client: Client, project: str, name: str) -> dict[str, object]:
    await client.call_tool("ensure_project", {"human_key": project})
    result = await client.call_tool(
        "register_agent",
        {
            "project_key": project,
            "program": "test",
            "model": "test",
            "name": name,
        },
    )
    sender = {"project": project, "name": result.data["name"], "token": result.data["registration_token"]}
    await client.call_tool(
        "ensure_channel",
        {
            "project_key": project,
            "channel_name": "general",
            "registration_token": sender["token"],
        },
    )
    return sender


async def _post(client: Client, sender: dict[str, object], body: str) -> dict[str, Any]:
    result = await client.call_tool(
        "post_channel_message",
        {
            "channel_project_key": sender["project"],
            "channel_name": "general",
            "sender_project_key": sender["project"],
            "sender_name": sender["name"],
            "subject": "coordination",
            "body_md": body,
            "registration_token": sender["token"],
        },
    )
    return result.data


async def _mk_membership(
    project_key: str,
    subject: str,
    handle: str,
    *,
    status: str = "active",
    default_agent_id: int | None = None,
) -> int:
    """Directly persist a human + membership; returns the human id."""
    async with get_session() as session:
        project = (
            await session.execute(select(Project).where(Project.human_key == project_key))
        ).scalars().one()
        human = Human(subject=subject, display_name=subject)
        session.add(human)
        await session.flush()
        if default_agent_id is not None:
            default_agent = await session.get(Agent, default_agent_id)
            assert default_agent is not None
            default_agent.owner_id = human.id
            session.add(default_agent)
        membership = ProjectHumanMembership(
            project_id=project.id,
            human_id=human.id,
            mention_handle=handle,
            status=status,
            default_agent_id=default_agent_id,
        )
        session.add(membership)
        await session.commit()
        await session.refresh(human)
        assert human.id is not None
        return human.id


async def _agent_id(name: str) -> int:
    async with get_session() as session:
        agent = (await session.execute(select(Agent).where(Agent.name == name))).scalars().one()
        assert agent.id is not None
        return agent.id


def _outcome(data: dict[str, Any], name: str) -> dict[str, Any]:
    for item in data["mention_deliveries"]:
        if item["name"].lower() == name.lower():
            return item
    raise AssertionError(f"no mention outcome for {name}: {data['mention_deliveries']}")


@pytest.mark.anyio
async def test_human_mention_with_default_agent_routes_to_default(isolated_env):
    server = build_mcp_server()
    async with Client(server) as client:
        sender = await _register(client, "/m3a/msg-a", "BlueLake")
        await _register(client, "/m3a/msg-a", "RedStone")
        default_id = await _agent_id("RedStone")
        await _mk_membership("/m3a/msg-a", "oidc|alice", "alice", default_agent_id=default_id)

        data = await _post(client, sender, "请 @alice 看一下这个方案")
        outcome = _outcome(data, "alice")
        assert outcome["status"] == "delivered"
        assert outcome["receipt_message_id"]

        async with get_session() as session:
            delivery = (
                await session.execute(select(MentionDelivery))
            ).scalars().one()
            assert delivery.mentioned_agent_id == default_id
            recipient = (
                await session.execute(
                    select(MessageRecipient).where(
                        MessageRecipient.message_id == delivery.receipt_message_id
                    )
                )
            ).scalars().one()
            assert recipient.agent_id == default_id
            assert recipient.kind == "mention"
            # human inbox must stay empty when a default agent exists
            inbox = (await session.execute(select(HumanInboxItem))).scalars().all()
            assert inbox == []


@pytest.mark.anyio
async def test_human_mention_without_default_lands_in_human_inbox(isolated_env):
    server = build_mcp_server()
    async with Client(server) as client:
        sender = await _register(client, "/m3a/msg-b", "BlueLake")
        human_id = await _mk_membership("/m3a/msg-b", "oidc|bob", "bob")

        data = await _post(client, sender, "@bob 麻烦确认上线时间")
        outcome = _outcome(data, "bob")
        assert outcome["status"] == "delivered_human_inbox"
        assert outcome["receipt_message_id"]

        async with get_session() as session:
            item = (await session.execute(select(HumanInboxItem))).scalars().one()
            assert item.human_id == human_id
            assert item.message_id == outcome["receipt_message_id"]
            assert item.kind == "mention"
            assert item.read_ts is None
            # no agent recipient rows for the human receipt
            recipients = (
                await session.execute(
                    select(MessageRecipient).where(
                        MessageRecipient.message_id == item.message_id
                    )
                )
            ).scalars().all()
            assert recipients == []
            audit = (
                await session.execute(
                    select(HubAuditEvent).where(
                        HubAuditEvent.event_type == "channel_mention_delivery"
                    )
                )
            ).scalars().one()
            assert audit.outcome == "delivered_human_inbox"
            assert audit.target_agent_id is None
            assert audit.related_message_id == item.message_id


@pytest.mark.anyio
async def test_human_inbox_delivery_is_idempotent(isolated_env):
    server = build_mcp_server()
    async with Client(server) as client:
        sender = await _register(client, "/m3a/msg-c", "BlueLake")
        human_id = await _mk_membership("/m3a/msg-c", "oidc|carol", "carol")

        await _post(client, sender, "@carol 第一遍")
        async with get_session() as session:
            source = (
                await session.execute(select(ChannelMessage))
            ).scalars().one()
            sender_agent = (
                await session.execute(select(Agent).where(Agent.name == "BlueLake"))
            ).scalars().one()
            project = (
                await session.execute(select(Project).where(Project.human_key == "/m3a/msg-c"))
            ).scalars().one()

        # The post already delivered once; every replay must be idempotent.
        first = await _deliver_channel_mentions(_context(), project, sender_agent, source, ["carol"])
        second = await _deliver_channel_mentions(_context(), project, sender_agent, source, ["carol"])
        assert first[0]["status"] == "already_delivered"
        assert second[0]["status"] == "already_delivered"
        async with get_session() as session:
            items = (await session.execute(select(HumanInboxItem))).scalars().all()
            assert len(items) == 1
            assert items[0].human_id == human_id


@pytest.mark.anyio
async def test_invited_membership_mention_stays_unknown(isolated_env):
    server = build_mcp_server()
    async with Client(server) as client:
        sender = await _register(client, "/m3a/msg-d", "BlueLake")
        await _mk_membership("/m3a/msg-d", "oidc|dave", "dave", status="invited")

        data = await _post(client, sender, "@dave 还没审批通过")
        outcome = _outcome(data, "dave")
        assert outcome["status"] == "skipped"
        assert outcome["reason"] == "unknown"
        async with get_session() as session:
            assert (await session.execute(select(HumanInboxItem))).scalars().all() == []


@pytest.mark.anyio
async def test_retired_default_falls_back_to_human_inbox(isolated_env):
    server = build_mcp_server()
    async with Client(server) as client:
        sender = await _register(client, "/m3a/msg-e", "BlueLake")
        await _register(client, "/m3a/msg-e", "RedStone")
        default_id = await _agent_id("RedStone")
        human_id = await _mk_membership(
            "/m3a/msg-e", "oidc|erin", "erin", default_agent_id=default_id
        )
        # Retire the default directly in the DB (the API guard would refuse
        # this while referenced; the resolver must still degrade safely).
        async with get_session() as session:
            agent = await session.get(Agent, default_id)
            assert agent is not None
            from datetime import datetime as _dt, timezone as _tz

            agent.retired_at = _dt.now(_tz.utc).replace(tzinfo=None)
            session.add(agent)
            await session.commit()

        data = await _post(client, sender, "@erin 你的默认 agent 已下线")
        outcome = _outcome(data, "erin")
        assert outcome["status"] == "delivered_human_inbox"
        async with get_session() as session:
            item = (await session.execute(select(HumanInboxItem))).scalars().one()
            assert item.human_id == human_id


@pytest.mark.anyio
async def test_cross_project_default_falls_back_to_human_inbox(isolated_env):
    server = build_mcp_server()
    async with Client(server) as client:
        sender = await _register(client, "/m3a/msg-cross-a", "BlueLake")
        await _register(client, "/m3a/msg-cross-b", "RedStone")
        foreign_default_id = await _agent_id("RedStone")
        human_id = await _mk_membership(
            "/m3a/msg-cross-a", "oidc|cross", "cross"
        )

        # Simulate old/corrupt data that bypassed the assignment API. Even if
        # the foreign Agent has the same owner, it must not receive project A's
        # Human mention.
        async with get_session() as session:
            membership = (
                await session.execute(
                    select(ProjectHumanMembership).where(
                        ProjectHumanMembership.human_id == human_id
                    )
                )
            ).scalars().one()
            foreign_default = await session.get(Agent, foreign_default_id)
            assert foreign_default is not None
            membership.default_agent_id = foreign_default_id
            foreign_default.owner_id = human_id
            session.add(membership)
            session.add(foreign_default)
            await session.commit()

        data = await _post(client, sender, "@cross 不应跨项目误投")
        assert _outcome(data, "cross")["status"] == "delivered_human_inbox"
        async with get_session() as session:
            item = (await session.execute(select(HumanInboxItem))).scalars().one()
            assert item.human_id == human_id
            assert (await session.execute(select(MessageRecipient))).scalars().all() == []


@pytest.mark.anyio
async def test_wrong_owner_default_falls_back_to_human_inbox(isolated_env):
    server = build_mcp_server()
    async with Client(server) as client:
        sender = await _register(client, "/m3a/msg-owner", "BlueLake")
        await _register(client, "/m3a/msg-owner", "RedStone")
        default_id = await _agent_id("RedStone")
        human_id = await _mk_membership(
            "/m3a/msg-owner", "oidc|owner", "owner"
        )

        # The Agent is local but belongs to another Human. A corrupt default
        # reference must degrade to the intended Human's inbox, never misroute.
        async with get_session() as session:
            other = Human(subject="oidc|other", display_name="Other")
            session.add(other)
            await session.flush()
            membership = (
                await session.execute(
                    select(ProjectHumanMembership).where(
                        ProjectHumanMembership.human_id == human_id
                    )
                )
            ).scalars().one()
            default_agent = await session.get(Agent, default_id)
            assert default_agent is not None
            membership.default_agent_id = default_id
            default_agent.owner_id = other.id
            session.add(membership)
            session.add(default_agent)
            await session.commit()

        data = await _post(client, sender, "@owner 不应投给其他人的 Agent")
        assert _outcome(data, "owner")["status"] == "delivered_human_inbox"
        async with get_session() as session:
            item = (await session.execute(select(HumanInboxItem))).scalars().one()
            assert item.human_id == human_id
            assert (await session.execute(select(MessageRecipient))).scalars().all() == []


@pytest.mark.anyio
async def test_human_mention_matches_handle_case_insensitively(isolated_env):
    server = build_mcp_server()
    async with Client(server) as client:
        sender = await _register(client, "/m3a/msg-f", "BlueLake")
        human_id = await _mk_membership("/m3a/msg-f", "oidc|frank", "Frank")

        data = await _post(client, sender, "@frank 大小写不敏感")
        outcome = _outcome(data, "frank")
        assert outcome["status"] == "delivered_human_inbox"
        async with get_session() as session:
            item = (await session.execute(select(HumanInboxItem))).scalars().one()
            assert item.human_id == human_id


@pytest.mark.anyio
async def test_mixed_agent_and_human_mentions(isolated_env):
    server = build_mcp_server()
    async with Client(server) as client:
        sender = await _register(client, "/m3a/msg-g", "BlueLake")
        await _register(client, "/m3a/msg-g", "GreenCastle")
        await _mk_membership("/m3a/msg-g", "oidc|gina", "gina")

        data = await _post(client, sender, "@GreenCastle @gina 对齐一下")
        assert _outcome(data, "GreenCastle")["status"] == "delivered"
        assert _outcome(data, "gina")["status"] == "delivered_human_inbox"


# ── HTTP 人工收件箱 API ──────────────────────────────────────────


def _configure_hub_jwt(monkeypatch):
    monkeypatch.setenv("HTTP_JWT_ENABLED", "true")
    monkeypatch.setenv("HTTP_JWT_ALGORITHMS", "HS256")
    monkeypatch.setenv("HTTP_JWT_SECRET", "hub-inbox-secret")
    monkeypatch.setenv("HTTP_RBAC_ENABLED", "true")
    monkeypatch.setenv("HTTP_RBAC_WRITER_ROLES", "writer")
    monkeypatch.setenv("HTTP_ALLOW_LOCALHOST_UNAUTHENTICATED", "false")
    _config.clear_settings_cache()
    return _config.get_settings()


def _hub_headers(settings, subject: str) -> dict[str, str]:
    token = jwt.encode(
        {"alg": "HS256"},
        {"sub": subject, settings.http.jwt_role_claim: "writer"},
        settings.http.jwt_secret,
    ).decode("utf-8")
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.anyio
async def test_human_inbox_http_list_and_mark_read(isolated_env, monkeypatch):
    server = build_mcp_server()
    async with Client(server) as client:
        sender = await _register(client, "/m3a/msg-h", "BlueLake")
        await _mk_membership("/m3a/msg-h", "oidc|helen", "helen")
        async with get_session() as session:
            routing_project = (
                await session.execute(
                    select(Project).where(Project.human_key == "/m3a/msg-h")
                )
            ).scalars().one()
            session.add(
                TeamProject(
                    slug="message-team",
                    name="Message Team",
                    routing_project_id=routing_project.id,
                )
            )
            await session.commit()
        await _post(client, sender, "@helen 第一条")
        await _post(client, sender, "@helen 第二条")

    settings = _configure_hub_jwt(monkeypatch)
    app = build_http_app(settings, build_mcp_server())
    helen = _hub_headers(settings, "oidc|helen")
    intruder = _hub_headers(settings, "oidc|mallory")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        # unauthenticated → 401
        unauthenticated = await http.get("/hub/api/inbox")
        assert unauthenticated.status_code == 401

        # helen must register her human identity first (PUT humans/me maps sub)
        registered = await http.put(
            "/hub/api/humans/me", headers=helen, json={"display_name": "Helen"}
        )
        assert registered.status_code == 200

        listed = await http.get("/hub/api/inbox", headers=helen)
        assert listed.status_code == 200
        items = listed.json()["items"]
        assert len(items) == 2
        assert {item["subject"] for item in items} == {"coordination"}
        assert {item["project_slug"] for item in items} == {"message-team"}
        assert all(item["sender_name"] == "BlueLake" for item in items)
        assert all(item["read_ts"] is None for item in items)

        unread = await http.get("/hub/api/inbox?unread_only=true", headers=helen)
        assert len(unread.json()["items"]) == 2

        first_id = items[0]["id"]
        marked = await http.post(
            "/hub/api/inbox/mark-read", headers=helen, json={"ids": [first_id]}
        )
        assert marked.status_code == 200
        assert marked.json()["updated"] == 1

        # mark-read is idempotent
        again = await http.post(
            "/hub/api/inbox/mark-read", headers=helen, json={"ids": [first_id]}
        )
        assert again.json()["updated"] == 0

        unread_after = await http.get("/hub/api/inbox?unread_only=true", headers=helen)
        assert len(unread_after.json()["items"]) == 1

        # another human sees an empty inbox and cannot touch helen's rows
        await http.put("/hub/api/humans/me", headers=intruder, json={"display_name": "Mallory"})
        foreign = await http.get("/hub/api/inbox", headers=intruder)
        assert foreign.json()["items"] == []
        attack = await http.post(
            "/hub/api/inbox/mark-read", headers=intruder, json={"ids": [first_id]}
        )
        assert attack.json()["updated"] == 0

        # mark all remaining
        cleared = await http.post("/hub/api/inbox/mark-read", headers=helen, json={})
        assert cleared.json()["updated"] == 1
        done = await http.get("/hub/api/inbox?unread_only=true", headers=helen)
        assert done.json()["items"] == []

        # invalid ids payload
        bad = await http.post(
            "/hub/api/inbox/mark-read", headers=helen, json={"ids": ["1"]}
        )
        assert bad.status_code == 400


@pytest.mark.anyio
async def test_human_support_request_broadcasts_via_default_agent(
    isolated_env, monkeypatch,
):
    server = build_mcp_server()
    async with Client(server) as client:
        sender = await _register(client, "/m3a/support", "BlueLake")
        sender_id = await _agent_id("BlueLake")
        await _mk_membership(
            "/m3a/support", "oidc|alice", "alice", default_agent_id=sender_id,
        )
        bob_id = await _mk_membership("/m3a/support", "oidc|bob", "bob")
        carol_id = await _mk_membership("/m3a/support", "oidc|carol", "carol")
        await _mk_membership(
            "/m3a/support", "oidc|dave", "dave", status="removed",
        )
        async with get_session() as session:
            routing_project = (
                await session.execute(
                    select(Project).where(Project.human_key == "/m3a/support")
                )
            ).scalars().one()
            session.add(
                TeamProject(
                    slug="support-team",
                    name="Support Team",
                    routing_project_id=routing_project.id,
                )
            )
            await session.commit()

    settings = _configure_hub_jwt(monkeypatch)
    app = build_http_app(settings, build_mcp_server())
    alice = _hub_headers(settings, "oidc|alice")
    bob = _hub_headers(settings, "oidc|bob")
    dave = _hub_headers(settings, "oidc|dave")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        posted = await http.post(
            "/hub/api/projects/support-team/support-requests",
            headers=alice,
            json={
                "subject": "终端求助 · demo/w1:p1",
                "body_md": "当前构建失败, 请团队协助排查。",
                "importance": "high",
            },
        )
        assert posted.status_code == 201
        payload = posted.json()
        assert payload["channel"] == "support"
        assert payload["sender_agent"] == sender["name"]
        assert set(payload["mention_handles"]) == {"bob", "carol"}
        assert {item["status"] for item in payload["deliveries"]} == {
            "delivered_human_inbox"
        }

        inbox = await http.get("/hub/api/inbox", headers=bob)
        assert inbox.status_code == 200
        assert len(inbox.json()["items"]) == 1
        assert inbox.json()["items"][0]["subject"] == "终端求助 · demo/w1:p1"

        chat = await http.get(
            "/hub/api/projects/support-team/chat/messages", headers=bob,
        )
        assert chat.status_code == 200
        assert chat.json()["count"] == 1
        assert chat.json()["messages"][0]["subject"] == "终端求助 · demo/w1:p1"
        assert chat.json()["messages"][0]["body_md"] == "当前构建失败, 请团队协助排查。"
        assert set(chat.json()["messages"][0]["mention_handles"]) == {"bob", "carol"}
        assert chat.json()["messages"][0]["sender_name"] == "oidc|alice"
        assert chat.json()["messages"][0]["sender_agent"] == sender["name"]

        removed = await http.get(
            "/hub/api/projects/support-team/chat/messages", headers=dave,
        )
        assert removed.status_code == 403

    async with get_session() as session:
        message = (await session.execute(select(ChannelMessage))).scalars().one()
        assert message.importance == "high"
        assert "@bob @carol" in message.body_md
        items = (await session.execute(select(HumanInboxItem))).scalars().all()
        assert {item.human_id for item in items} == {bob_id, carol_id}


@pytest.mark.anyio
async def test_human_support_request_can_target_member_without_sender_agent(
    isolated_env, monkeypatch,
):
    server = build_mcp_server()
    async with Client(server) as client:
        await _register(client, "/m3a/support-target", "BlueLake")
        await _register(client, "/m3a/support-target", "RedStone")
        sender_id = await _agent_id("BlueLake")
        recipient_id = await _agent_id("RedStone")
        await _mk_membership(
            "/m3a/support-target", "oidc|alice", "alice", default_agent_id=sender_id,
        )
        await _mk_membership(
            "/m3a/support-target", "oidc|bob", "bob", default_agent_id=recipient_id,
        )
        await _mk_membership("/m3a/support-target", "oidc|carol", "carol")
        await _mk_membership("/m3a/support-target", "oidc|dave", "dave")
        async with get_session() as session:
            routing_project = (
                await session.execute(
                    select(Project).where(Project.human_key == "/m3a/support-target")
                )
            ).scalars().one()
            session.add(
                TeamProject(
                    slug="support-target",
                    name="Target Team",
                    routing_project_id=routing_project.id,
                )
            )
            await session.commit()

    settings = _configure_hub_jwt(monkeypatch)
    app = build_http_app(settings, build_mcp_server())
    alice = _hub_headers(settings, "oidc|alice")
    carol = _hub_headers(settings, "oidc|carol")
    dave = _hub_headers(settings, "oidc|dave")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        targeted = await http.post(
            "/hub/api/projects/support-target/support-requests",
            headers=alice,
            json={
                "subject": "只找 Bob",
                "body_md": "请 Bob 支持。",
                "mention_handles": ["BoB"],
            },
        )
        assert targeted.status_code == 201
        assert targeted.json()["mention_handles"] == ["bob"]
        assert targeted.json()["deliveries"][0]["status"] == "delivered"

        human_sender = await http.post(
            "/hub/api/projects/support-target/support-requests",
            headers=carol,
            json={
                "subject": "Human 直接发送",
                "body_md": "没有默认 Agent 也能联系 Dave。",
                "mention_handles": ["dave"],
            },
        )
        assert human_sender.status_code == 201
        assert human_sender.json()["sender_kind"] == "human"
        assert human_sender.json()["sender_human"] == "carol"
        assert human_sender.json()["deliveries"][0]["status"] == "delivered_human_inbox"

        inbox = await http.get("/hub/api/inbox", headers=dave)
        assert inbox.status_code == 200
        assert inbox.json()["items"][0]["sender_kind"] == "human"
        assert inbox.json()["items"][0]["sender_name"] == "oidc|carol"

        chat = await http.get(
            "/hub/api/projects/support-target/chat/messages", headers=dave,
        )
        assert chat.status_code == 200
        assert [item["subject"] for item in chat.json()["messages"]] == [
            "只找 Bob", "Human 直接发送",
        ]
        assert chat.json()["messages"][-1]["sender_kind"] == "human"
        assert chat.json()["messages"][-1]["sender_name"] == "oidc|carol"
        assert chat.json()["messages"][-1]["sender_agent"] is None
        assert chat.json()["messages"][-1]["mention_handles"] == ["dave"]

        team_agents = await http.get(
            "/hub/api/projects/support-target/agents", headers=carol,
        )
        assert team_agents.status_code == 200
        assert {agent["name"] for agent in team_agents.json()["agents"]} == {
            "BlueLake", "RedStone",
        }

    async with get_session() as session:
        recipient = (
            await session.execute(
                select(MessageRecipient).where(
                    cast(Any, MessageRecipient.agent_id) == recipient_id,
                )
            )
        ).scalars().one()
        assert recipient.kind == "mention"
