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
from datetime import datetime, timedelta, timezone

import pytest
from authlib.jose import jwt
from httpx import ASGITransport, AsyncClient
from sqlmodel import select

from mcp_agent_mail import config as _config
from mcp_agent_mail.app import build_mcp_server, sweep_stale_agents
from mcp_agent_mail.db import get_session
from mcp_agent_mail.http import build_http_app
from mcp_agent_mail.models import (
    Agent,
    ChannelMessage,
    HumanInboxItem,
    SessionLeadBinding,
)


def _configure_hub_jwt(monkeypatch):
    monkeypatch.setenv("HTTP_JWT_ENABLED", "true")
    monkeypatch.setenv("HTTP_JWT_ALGORITHMS", "HS256")
    monkeypatch.setenv("HTTP_JWT_SECRET", "hub-reply-secret")
    monkeypatch.setenv("HTTP_JWT_JWKS_URL", "")
    monkeypatch.setenv("HTTP_JWT_AUDIENCE", "")
    monkeypatch.setenv("HTTP_JWT_ISSUER", "")
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


async def _mk_lead(
    client: AsyncClient,
    headers: dict[str, str],
    session_id: str,
    label: str = "codex-main",
    reply_mode: str = "auto",
):
    resp = await client.put(
        "/hub/api/projects/core/session-lead",
        headers=headers,
        json={
            "client_session_id": session_id,
            "lead_label": label,
            "reply_mode": reply_mode,
        },
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
    if kw.get("inbox_item_id"):
        payload["inbox_item_id"] = kw["inbox_item_id"]
        payload["claim_token"] = kw["claim_token"]
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
async def test_active_session_lead_survives_sweep_and_recovers_old_retirement(hub):
    settings, app = hub
    root = _headers(settings, "oidc|root", admin=True)
    alice = _headers(settings, "oidc|alice")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await _setup_team(client, root)
        alice_id = await _register_human(client, alice, "Alice")
        await _join_active(client, root, alice, alice_id, "alice")
        token = (await _mk_lead(client, alice, "wsl-1")).json()["reply_token"]

        old = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=2)
        async with get_session() as session:
            binding = (await session.execute(select(SessionLeadBinding))).scalars().one()
            lead = await session.get(Agent, binding.agent_id)
            assert lead is not None
            lead.last_active_ts = old
            session.add(lead)
            await session.commit()

        assert await sweep_stale_agents(threshold_seconds=86_400) == []
        async with get_session() as session:
            lead = await session.get(Agent, binding.agent_id)
            assert lead is not None and lead.retired_at is None
            # Reproduce a database already damaged by the pre-fix sweeper.
            lead.retired_at = old
            session.add(lead)
            await session.commit()

        reply = await _reply(client, "wsl-1", token, subject="恢复通信")
        assert reply.status_code == 201, reply.text
        async with get_session() as session:
            healed = await session.get(Agent, binding.agent_id)
            assert healed is not None and healed.retired_at is None


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


@pytest.mark.anyio
async def test_confirm_authorizes_before_claim_reply_and_complete(hub):
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
        alice_lead = await _mk_lead(
            client, alice, "alice-1", reply_mode="confirm"
        )
        alice_token = alice_lead.json()["reply_token"]
        await _mk_lead(client, bob, "bob-1")

        incoming = await client.post(
            "/hub/api/projects/core/support-requests",
            headers=bob,
            json={
                "subject": "需要处理",
                "body_md": "请回复",
                "mention_handles": ["alice"],
            },
        )
        assert incoming.status_code == 201, incoming.text

        blocked_claim = await client.post(
            "/hub/api/projects/core/session-lead/inbox/claim",
            json={"client_session_id": "alice-1", "reply_token": alice_token},
        )
        assert blocked_claim.status_code == 200
        assert blocked_claim.json() == {"status": "empty", "message": None}

        requests = await client.get(
            "/hub/api/projects/core/reply-requests", headers=alice
        )
        assert requests.status_code == 200
        request_row = requests.json()["requests"][0]
        assert request_row["message_id"] == incoming.json()["message_id"]
        assert request_row["status"] == "awaiting_confirmation"
        inbox_item_id = request_row["inbox_item_id"]

        forbidden = await client.post(
            f"/hub/api/projects/core/reply-requests/{inbox_item_id}/approve",
            headers=bob,
        )
        assert forbidden.status_code == 404

        approved = await client.post(
            f"/hub/api/projects/core/reply-requests/{inbox_item_id}/approve",
            headers=alice,
        )
        assert approved.status_code == 201, approved.text
        assert approved.json()["request"]["status"] == "queued"
        approved_again = await client.post(
            f"/hub/api/projects/core/reply-requests/{inbox_item_id}/approve",
            headers=alice,
        )
        assert approved_again.status_code == 200
        assert approved_again.json()["status"] == "already_approved"

        claim = await client.post(
            "/hub/api/projects/core/session-lead/inbox/claim",
            json={"client_session_id": "alice-1", "reply_token": alice_token},
        )
        assert claim.status_code == 201, claim.text
        claimed = claim.json()
        assert claimed["reply_mode"] == "confirm"
        assert claimed["message"]["subject"] == "需要处理"
        assert claimed["message"]["sender_handle"] == "bob"
        assert claimed["message"]["inbox_item_id"] == inbox_item_id
        claim_token = claimed["claim_token"]

        retired_draft = await client.post(
            "/hub/api/projects/core/session-lead/reply-drafts",
            json={
                "client_session_id": "alice-1",
                "reply_token": alice_token,
                "inbox_item_id": inbox_item_id,
                "claim_token": claim_token,
                "subject": "旧草稿",
                "body_md": "不得再生成后审批",
                "mention_handles": ["bob"],
                "idempotency_key": "retired-draft",
            },
        )
        assert retired_draft.status_code == 410

        blocked = await _reply(
            client,
            "alice-1",
            alice_token,
            subject="不得直发",
            mentions=["bob"],
        )
        assert blocked.status_code == 409
        assert blocked.json()["detail"] == "Human confirmation is required"

        sent = await _reply(
            client,
            "alice-1",
            alice_token,
            subject="确认后回复",
            body="已处理",
            mentions=["bob"],
            idem="reply-1",
            inbox_item_id=inbox_item_id,
            claim_token=claim_token,
        )
        assert sent.status_code == 201, sent.text
        replay = await _reply(
            client,
            "alice-1",
            alice_token,
            subject="确认后回复",
            body="已处理",
            mentions=["bob"],
            idem="reply-1",
            inbox_item_id=inbox_item_id,
            claim_token=claim_token,
        )
        assert replay.status_code == 200
        assert replay.json()["status"] == "already_delivered"

        denied_complete = await client.post(
            f"/hub/api/projects/core/session-lead/inbox/{inbox_item_id}/complete",
            json={
                "client_session_id": "alice-1",
                "reply_token": alice_token,
                "claim_token": "wrong-claim-token",
            },
        )
        assert denied_complete.status_code == 403

        completed = await client.post(
            f"/hub/api/projects/core/session-lead/inbox/{inbox_item_id}/complete",
            json={
                "client_session_id": "alice-1",
                "reply_token": alice_token,
                "claim_token": claim_token,
            },
        )
        assert completed.status_code == 200
        assert completed.json()["status"] == "completed"
        completed_again = await client.post(
            f"/hub/api/projects/core/session-lead/inbox/{inbox_item_id}/complete",
            json={
                "client_session_id": "alice-1",
                "reply_token": alice_token,
                "claim_token": claim_token,
            },
        )
        assert completed_again.json()["status"] == "already_completed"

        second_incoming = await client.post(
            "/hub/api/projects/core/support-requests",
            headers=bob,
            json={
                "subject": "拒绝测试",
                "body_md": "请回复",
                "mention_handles": ["alice"],
            },
        )
        assert second_incoming.status_code == 201
        second_requests = await client.get(
            "/hub/api/projects/core/reply-requests", headers=alice
        )
        second_request = next(
            item
            for item in second_requests.json()["requests"]
            if item["message_id"] == second_incoming.json()["message_id"]
        )
        rejected = await client.post(
            f"/hub/api/projects/core/reply-requests/{second_request['inbox_item_id']}/reject",
            headers=alice,
        )
        assert rejected.status_code == 201
        assert rejected.json()["status"] == "ignored"
        rejected_again = await client.post(
            f"/hub/api/projects/core/reply-requests/{second_request['inbox_item_id']}/reject",
            headers=alice,
        )
        assert rejected_again.status_code == 200
        assert rejected_again.json()["status"] == "already_ignored"
        cannot_approve = await client.post(
            f"/hub/api/projects/core/reply-requests/{second_request['inbox_item_id']}/approve",
            headers=alice,
        )
        assert cannot_approve.status_code == 409
        second_claim = await client.post(
            "/hub/api/projects/core/session-lead/inbox/claim",
            json={"client_session_id": "alice-1", "reply_token": alice_token},
        )
        assert second_claim.status_code == 200
        assert second_claim.json()["status"] == "empty"

        async with get_session() as session:
            sent = (
                await session.execute(
                    select(ChannelMessage).where(
                        ChannelMessage.subject == "确认后回复"
                    )
                )
            ).scalars().all()
            ignored_messages = (
                await session.execute(
                    select(ChannelMessage).where(
                        ChannelMessage.subject == "拒绝回复"
                    )
                )
            ).scalars().all()
            assert len(sent) == 1
            assert ignored_messages == []


@pytest.mark.anyio
async def test_reply_mode_switch_rotates_capability_and_auto_sends(hub):
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
        await _mk_lead(client, bob, "bob-1")
        confirm = await _mk_lead(
            client, alice, "alice-1", reply_mode="confirm"
        )
        old_token = confirm.json()["reply_token"]
        incoming = await client.post(
            "/hub/api/projects/core/support-requests",
            headers=bob,
            json={
                "subject": "切换竞态",
                "body_md": "切到 auto 后不得显示 confirm 错误",
                "mention_handles": ["alice"],
            },
        )
        assert incoming.status_code == 201

        switched = await client.put(
            "/hub/api/projects/core/session-lead",
            headers=alice,
            json={
                "client_session_id": "alice-1",
                "lead_label": "codex-main",
                "reply_mode": "auto",
            },
        )
        assert switched.status_code == 200
        assert switched.json()["binding"]["reply_mode"] == "auto"
        new_token = switched.json()["reply_token"]
        assert new_token != old_token
        assert (await _reply(client, "alice-1", old_token)).status_code == 403
        requests = await client.get(
            "/hub/api/projects/core/reply-requests", headers=alice
        )
        request_row = next(
            item
            for item in requests.json()["requests"]
            if item["message_id"] == incoming.json()["message_id"]
        )
        assert request_row["status"] == "queued"
        assert request_row["decision"] == "auto"
        stale_click = await client.post(
            f"/hub/api/projects/core/reply-requests/{request_row['inbox_item_id']}/approve",
            headers=alice,
        )
        assert stale_click.status_code == 201
        assert stale_click.json()["request"]["decision"] == "auto"
        claim = await client.post(
            "/hub/api/projects/core/session-lead/inbox/claim",
            json={"client_session_id": "alice-1", "reply_token": new_token},
        )
        assert claim.status_code == 201
        replied = await _reply(
            client,
            "alice-1",
            new_token,
            subject="竞态已处理",
            mentions=["bob"],
            idem="auto-inbox-1",
            inbox_item_id=claim.json()["message"]["inbox_item_id"],
            claim_token=claim.json()["claim_token"],
        )
        assert replied.status_code == 201, replied.text
        direct = await _reply(
            client,
            "alice-1",
            new_token,
            subject="自动回复",
            idem="auto-1",
        )
        assert direct.status_code == 201, direct.text

        invalid_mode = await client.put(
            "/hub/api/projects/core/session-lead",
            headers=alice,
            json={
                "client_session_id": "alice-1",
                "lead_label": "codex-main",
                "reply_mode": "sometimes",
            },
        )
        assert invalid_mode.status_code == 400

        unbound = await client.request(
            "DELETE",
            "/hub/api/projects/core/session-lead",
            headers=alice,
            json={"client_session_id": "alice-1"},
        )
        assert unbound.status_code == 200
        claim = await client.post(
            "/hub/api/projects/core/session-lead/inbox/claim",
            json={"client_session_id": "alice-1", "reply_token": new_token},
        )
        assert claim.status_code == 403
