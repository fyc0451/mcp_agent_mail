"""M3 Session-Team: reply capability (#1093/#1096).

Acceptance:
  * plaintext token returned exactly once (create/rotate); reuse PUT never echoes it
  * capability auth is opaque (unknown session / revoked / wrong token → same 403)
  * unbind / single-active switch / member removal invalidate the old hash in the
    same transaction
  * idempotency key occupies its slot in the same transaction as the message;
    concurrent duplicates deliver exactly once
  * token never appears in logs/errors/responses beyond the one-time issuance
"""

from __future__ import annotations

import asyncio

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
    ChannelMessage,
    HumanInboxItem,
)


def _configure_hub_jwt(monkeypatch):
    monkeypatch.setenv("HTTP_JWT_ENABLED", "true")
    monkeypatch.setenv("HTTP_JWT_ALGORITHMS", "HS256")
    monkeypatch.setenv("HTTP_JWT_SECRET", "hub-reply-secret")
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


async def _join_active(client, admin_headers, member_headers, human_id, handle, slug="core"):
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


async def _mk_lead(client: AsyncClient, headers: dict[str, str], session_id: str, label: str = "codex-main"):
    resp = await client.put(
        "/hub/api/projects/core/session-lead",
        headers=headers,
        json={"client_session_id": session_id, "lead_label": label},
    )
    assert resp.status_code in (200, 201), resp.text
    return resp


def _reply(client: AsyncClient, session_id: str, token: str, **kw):
    payload = {"client_session_id": session_id, "reply_token": token,
               "subject": kw.get("subject", "回复"), "body_md": kw.get("body", "正文")}
    if kw.get("mentions"):
        payload["mention_handles"] = kw["mentions"]
    if kw.get("idem"):
        payload["idempotency_key"] = kw["idem"]
    return client.post("/hub/api/projects/core/session-lead/reply", json=payload)


@pytest.fixture
def hub(isolated_env, monkeypatch):
    settings = _configure_hub_jwt(monkeypatch)
    return settings, build_http_app(settings, build_mcp_server())


@pytest.mark.anyio
async def test_token_returned_once_and_rotation(hub):
    settings, app = hub
    root = _headers(settings, "oidc|root", admin=True)
    bob = _headers(settings, "oidc|bob")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _setup_team(client, root)
        bob_id = await _register_human(client, bob, "Bob")
        await _join_active(client, root, bob, bob_id, "bob")

        created = await _mk_lead(client, bob, "s1")
        token1 = created.json()["reply_token"]
        assert len(token1) <= 128

        reuse = await _mk_lead(client, bob, "s1")
        assert reuse.status_code == 200
        assert "reply_token" not in reuse.json()

        rotated = await client.put(
            "/hub/api/projects/core/session-lead",
            headers=bob,
            json={"client_session_id": "s1", "lead_label": "codex-main", "rotate_reply_token": True},
        )
        token2 = rotated.json()["reply_token"]
        assert token2 != token1

        # 旧 token 立即失效
        stale = await _reply(client, "s1", token1)
        assert stale.status_code == 403
        fresh = await _reply(client, "s1", token2)
        assert fresh.status_code == 201


@pytest.mark.anyio
async def test_unbind_and_rebind_switch_invalidate_token(hub):
    settings, app = hub
    root = _headers(settings, "oidc|root", admin=True)
    bob = _headers(settings, "oidc|bob")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _setup_team(client, root)
        bob_id = await _register_human(client, bob, "Bob")
        await _join_active(client, root, bob, bob_id, "bob")

        token1 = (await _mk_lead(client, bob, "s1")).json()["reply_token"]

        # 解绑 -> 失效
        await client.request(
            "DELETE", "/hub/api/projects/core/session-lead",
            headers=bob, json={"client_session_id": "s1"},
        )
        assert (await _reply(client, "s1", token1)).status_code == 403

        # 改绑(新 session id 单 active 切换) -> 旧 hash 同事务失效
        token2 = (await _mk_lead(client, bob, "s1")).json()["reply_token"]
        await _mk_lead(client, bob, "s2")
        assert (await _reply(client, "s1", token2)).status_code == 403


@pytest.mark.anyio
async def test_member_removal_invalidates_token(hub):
    settings, app = hub
    root = _headers(settings, "oidc|root", admin=True)
    bob = _headers(settings, "oidc|bob")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _setup_team(client, root)
        bob_id = await _register_human(client, bob, "Bob")
        await _join_active(client, root, bob, bob_id, "bob")
        token = (await _mk_lead(client, bob, "s1")).json()["reply_token"]

        removed = await client.patch(
            "/hub/api/projects/core/members/" + str(bob_id),
            headers=root,
            json={"status": "removed"},
        )
        assert removed.status_code == 200
        assert (await _reply(client, "s1", token)).status_code == 403


@pytest.mark.anyio
async def test_reply_delivers_support_without_recursive_lead_work(hub):
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
        await _mk_lead(client, bob, "mac-1")
        alice_token = (await _mk_lead(client, alice, "wsl-1", label="kimi-main")).json()["reply_token"]

        resp = await _reply(client, "wsl-1", alice_token, subject="远程回复", body="收到", mentions=["bob"])
        assert resp.status_code == 201, resp.text
        assert resp.json()["status"] == "delivered"
        assert "reply_token" not in resp.text

        async with get_session() as session:
            message = (
                await session.execute(
                    select(ChannelMessage).where(ChannelMessage.subject == "远程回复")
                )
            ).scalars().one()
            sender = await session.get(Agent, message.sender_id)
            assert sender is not None and sender.program == "team-session-lead"
            # The managed lead gets the normal mention receipt, but a Session Lead
            # answer is terminal and must not become another managed-lead work item.
            items = (
                await session.execute(
                    select(HumanInboxItem).where(HumanInboxItem.human_id == bob_id)
                )
            ).scalars().all()
            assert items == []


@pytest.mark.anyio
async def test_idempotency_key_atomic_exactly_once(hub):
    settings, app = hub
    root = _headers(settings, "oidc|root", admin=True)
    alice = _headers(settings, "oidc|alice")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _setup_team(client, root)
        alice_id = await _register_human(client, alice, "Alice")
        await _join_active(client, root, alice, alice_id, "alice")
        token = (await _mk_lead(client, alice, "wsl-1")).json()["reply_token"]

        # 顺序重放
        first = await _reply(client, "wsl-1", token, subject="只发一次", idem="key-1")
        assert first.status_code == 201
        second = await _reply(client, "wsl-1", token, subject="只发一次", idem="key-1")
        assert second.status_code == 200
        assert second.json()["status"] == "already_delivered"
        assert second.json()["message_id"] == first.json()["message_id"]

        # 并发同 key: 恰一个投递
        a, b = await asyncio.gather(
            _reply(client, "wsl-1", token, subject="并发", idem="key-2"),
            _reply(client, "wsl-1", token, subject="并发", idem="key-2"),
        )
        statuses = sorted([a.json()["status"], b.json()["status"]])
        assert statuses == ["already_delivered", "delivered"]
        async with get_session() as session:
            messages = (
                await session.execute(
                    select(ChannelMessage).where(ChannelMessage.subject == "并发")
                )
            ).scalars().all()
            assert len(messages) == 1


@pytest.mark.anyio
async def test_reply_credentials_are_opaque(hub):
    settings, app = hub
    root = _headers(settings, "oidc|root", admin=True)
    alice = _headers(settings, "oidc|alice")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _setup_team(client, root)
        alice_id = await _register_human(client, alice, "Alice")
        await _join_active(client, root, alice, alice_id, "alice")
        token = (await _mk_lead(client, alice, "wsl-1")).json()["reply_token"]

        for session_id, tok in (("no-such", token), ("wsl-1", "wrong-token")):
            resp = await _reply(client, session_id, tok)
            assert resp.status_code == 403
            assert resp.json()["detail"] == "Invalid reply credentials"
            assert "wrong-token" not in resp.text
            assert token not in resp.text

        # 超长 token 400
        too_long = await _reply(client, "wsl-1", "x" * 129)
        assert too_long.status_code == 400


@pytest.mark.anyio
async def test_replay_recovers_failed_mention_delivery(hub, monkeypatch):
    """#1101: 首次请求已提交 message+key 但投递抛错; 同 key 重放必须用原始
    handles 恢复幂等投递, message 仍恰一条, 返回 already_delivered+真实 deliveries。"""
    import mcp_agent_mail.http as http_module

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
        await _mk_lead(client, bob, "mac-1")
        token = (await _mk_lead(client, alice, "wsl-1")).json()["reply_token"]

        # 首次: 投递阶段抛错(message+key 已提交)
        real_deliver = http_module._deliver_channel_mentions
        fail_once = {"armed": True}

        async def flaky_deliver(*args, **kwargs):
            if fail_once["armed"]:
                fail_once["armed"] = False
                raise RuntimeError("simulated delivery crash")
            return await real_deliver(*args, **kwargs)

        monkeypatch.setattr(http_module, "_deliver_channel_mentions", flaky_deliver)
        with pytest.raises(RuntimeError, match="simulated delivery crash"):
            await _reply(client, "wsl-1", token, subject="崩溃恢复", idem="k-crash", mentions=["bob"])
        # message+key 已提交(投递崩溃不影响已提交状态)
        async with get_session() as session:
            messages = (
                await session.execute(
                    select(ChannelMessage).where(ChannelMessage.subject == "崩溃恢复")
                )
            ).scalars().all()
            assert len(messages) == 1
            items = (
                await session.execute(
                    select(HumanInboxItem).where(HumanInboxItem.human_id == bob_id)
                )
            ).scalars().all()
            assert len(items) == 0  # 投递崩了, bob 尚未收到

        # 重放: 恢复投递, 用原始 handles, message 仍 1 条
        replay = await _reply(client, "wsl-1", token, subject="崩溃恢复", idem="k-crash", mentions=["bob"])
        assert replay.status_code == 200
        assert replay.json()["status"] == "already_delivered"
        deliveries = replay.json()["deliveries"]
        assert deliveries  # 真实 deliveries, 不为空
        async with get_session() as session:
            messages = (
                await session.execute(
                    select(ChannelMessage).where(ChannelMessage.subject == "崩溃恢复")
                )
            ).scalars().all()
            assert len(messages) == 1
            items = (
                await session.execute(
                    select(HumanInboxItem).where(HumanInboxItem.human_id == bob_id)
                )
            ).scalars().all()
            # Session Lead 回复的重放同样不能创建下一轮 managed-lead 工作。
            assert items == []


@pytest.mark.anyio
async def test_rotate_reply_token_requires_bool(hub):
    settings, app = hub
    root = _headers(settings, "oidc|root", admin=True)
    bob = _headers(settings, "oidc|bob")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _setup_team(client, root)
        bob_id = await _register_human(client, bob, "Bob")
        await _join_active(client, root, bob, bob_id, "bob")
        created = await _mk_lead(client, bob, "s1")
        token1 = created.json()["reply_token"]

        # 字符串 "false" 不得触发轮换
        bad = await client.put(
            "/hub/api/projects/core/session-lead",
            headers=bob,
            json={"client_session_id": "s1", "lead_label": "codex-main", "rotate_reply_token": "false"},
        )
        assert bad.status_code == 400
        still = await _reply(client, "s1", token1)
        assert still.status_code == 201
