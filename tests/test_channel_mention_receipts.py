"""M1b-3c-c: durable inbox receipts derived from channel @mentions."""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest
from fastmcp import Client, Context
from sqlmodel import select

import mcp_agent_mail.app as app_module
from mcp_agent_mail.app import _deliver_channel_mentions, build_mcp_server
from mcp_agent_mail.db import get_session
from mcp_agent_mail.models import (
    Agent,
    AgentLink,
    ChannelMessage,
    ChannelReadCursor,
    MentionDelivery,
    Message,
    MessageRecipient,
    Project,
)


class _RecordingContext:
    def __init__(self) -> None:
        self.infos: list[str] = []

    async def info(self, message: str) -> None:
        self.infos.append(message)


def _context() -> Context:
    return cast(Context, _RecordingContext())


def _result_list(result) -> list[dict[str, Any]]:
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict) and isinstance(structured.get("result"), list):
        return structured["result"]
    return [getattr(item, "root", item) for item in result.data]


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
    return {
        "project": project,
        "name": result.data["name"],
        "token": result.data["registration_token"],
    }


async def _post(
    client: Client,
    sender: dict[str, object],
    body: str,
    *,
    channel_project: str = "/channels/alpha",
) -> dict[str, Any]:
    result = await client.call_tool(
        "post_channel_message",
        {
            "channel_project_key": channel_project,
            "channel_name": "general",
            "sender_project_key": sender["project"],
            "sender_name": sender["name"],
            "subject": "coordination",
            "body_md": body,
            "registration_token": sender["token"],
        },
    )
    return result.data


async def _project_and_agent(project_key: str, agent_name: str) -> tuple[Project, Agent]:
    async with get_session() as session:
        project_result = await session.execute(select(Project).where(Project.human_key == project_key))
        project = project_result.scalars().one()
        agent_result = await session.execute(
            select(Agent).where(Agent.project_id == project.id, Agent.name == agent_name)
        )
        return project, agent_result.scalars().one()


async def _approve_link(sender: dict[str, object], target: dict[str, object]) -> None:
    sender_project, sender_agent = await _project_and_agent(str(sender["project"]), str(sender["name"]))
    target_project, target_agent = await _project_and_agent(str(target["project"]), str(target["name"]))
    async with get_session() as session:
        session.add(
            AgentLink(
                a_project_id=sender_project.id,
                a_agent_id=sender_agent.id,
                b_project_id=target_project.id,
                b_agent_id=target_agent.id,
                status="approved",
            )
        )
        await session.commit()


async def _delivery_inputs(
    posted: dict[str, Any], sender: dict[str, object]
) -> tuple[Project, Agent, ChannelMessage]:
    project, sender_agent = await _project_and_agent(str(sender["project"]), str(sender["name"]))
    async with get_session() as session:
        source = await session.get(ChannelMessage, posted["id"])
        assert source is not None
        return project, sender_agent, source


@pytest.fixture
async def channel_hub(isolated_env):
    async with Client(build_mcp_server()) as client:
        sender = await _register(client, "/channels/alpha", "BlueLake")
        await client.call_tool(
            "ensure_channel",
            {
                "project_key": sender["project"],
                "channel_name": "general",
                "registration_token": sender["token"],
            },
        )
        yield client, sender


@pytest.mark.asyncio
async def test_post_without_mentions_keeps_message_shape_additive(channel_hub):
    client, sender = channel_hub

    posted = await _post(client, sender, "plain channel update")

    assert posted["mention_deliveries"] == []
    assert set(posted) == {
        "id",
        "channel_id",
        "sender_id",
        "sender_name",
        "subject",
        "body_md",
        "importance",
        "created_ts",
        "mention_deliveries",
    }
    async with get_session() as session:
        assert list((await session.execute(select(Message))).scalars()) == []
        assert list((await session.execute(select(MentionDelivery))).scalars()) == []


@pytest.mark.asyncio
async def test_delivery_helper_is_idempotent_without_orphan_receipts(channel_hub):
    client, sender = channel_hub
    recipient = await _register(client, "/channels/alpha", "GreenHill")
    posted = await _post(client, sender, "plain source used for direct helper test")
    project, sender_agent, source = await _delivery_inputs(posted, sender)
    ctx = _context()

    first = await _deliver_channel_mentions(ctx, project, sender_agent, source, [str(recipient["name"])])
    second = await _deliver_channel_mentions(ctx, project, sender_agent, source, [str(recipient["name"])])

    assert first[0]["status"] == "delivered"
    assert second[0]["status"] == "already_delivered"
    assert second[0]["receipt_message_id"] == first[0]["receipt_message_id"]
    async with get_session() as session:
        assert len(list((await session.execute(select(Message))).scalars())) == 1
        assert len(list((await session.execute(select(MessageRecipient))).scalars())) == 1
        assert len(list((await session.execute(select(MentionDelivery))).scalars())) == 1


@pytest.mark.asyncio
async def test_concurrent_same_mention_settles_on_one_receipt(channel_hub):
    client, sender = channel_hub
    recipient = await _register(client, "/channels/alpha", "GreenHill")
    posted = await _post(client, sender, "plain source used for concurrent helper test")
    project, sender_agent, source = await _delivery_inputs(posted, sender)

    first, second = await asyncio.gather(
        _deliver_channel_mentions(
            _context(), project, sender_agent, source, [str(recipient["name"])]
        ),
        _deliver_channel_mentions(
            _context(), project, sender_agent, source, [str(recipient["name"])]
        ),
    )

    assert {first[0]["status"], second[0]["status"]} == {"delivered", "already_delivered"}
    assert first[0]["receipt_message_id"] == second[0]["receipt_message_id"]
    async with get_session() as session:
        assert len(list((await session.execute(select(Message))).scalars())) == 1
        assert len(list((await session.execute(select(MessageRecipient))).scalars())) == 1
        assert len(list((await session.execute(select(MentionDelivery))).scalars())) == 1


@pytest.mark.asyncio
async def test_concurrent_partial_overlap_delivers_each_agent_once(channel_hub):
    client, sender = channel_hub
    first_recipient = await _register(client, "/channels/alpha", "green-agent")
    second_recipient = await _register(client, "/channels/alpha", "red-agent")
    posted = await _post(client, sender, "plain source used for partial overlap test")
    project, sender_agent, source = await _delivery_inputs(posted, sender)

    await asyncio.gather(
        _deliver_channel_mentions(
            _context(), project, sender_agent, source, [str(first_recipient["name"])]
        ),
        _deliver_channel_mentions(
            _context(),
            project,
            sender_agent,
            source,
            [str(first_recipient["name"]), str(second_recipient["name"])],
        ),
    )

    _project, first_agent = await _project_and_agent(
        str(first_recipient["project"]), str(first_recipient["name"])
    )
    _project, second_agent = await _project_and_agent(
        str(second_recipient["project"]), str(second_recipient["name"])
    )
    async with get_session() as session:
        mappings = list((await session.execute(select(MentionDelivery))).scalars())
        recipients = list((await session.execute(select(MessageRecipient))).scalars())
        messages = list((await session.execute(select(Message))).scalars())
        assert len(mappings) == 2
        assert {mapping.mentioned_agent_id for mapping in mappings} == {
            first_agent.id,
            second_agent.id,
        }
        assert len(recipients) == 2
        assert {recipient.agent_id for recipient in recipients} == {first_agent.id, second_agent.id}
        assert len(messages) in {1, 2}


@pytest.mark.asyncio
async def test_target_archive_failure_is_compensated_and_other_target_continues(
    channel_hub, monkeypatch
):
    client, sender = channel_hub
    beta = await _register(client, "/channels/beta", "beta-agent")
    gamma = await _register(client, "/channels/gamma", "gamma-agent")
    await _approve_link(sender, beta)
    await _approve_link(sender, gamma)
    posted = await _post(client, sender, "plain source used for failure injection")
    project, sender_agent, source = await _delivery_inputs(posted, sender)
    original_write = app_module.write_message_bundle
    call_count = 0

    async def fail_first_archive(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("injected archive failure")
        return await original_write(*args, **kwargs)

    monkeypatch.setattr(app_module, "write_message_bundle", fail_first_archive)
    outcomes = await _deliver_channel_mentions(
        _context(),
        project,
        sender_agent,
        source,
        [str(beta["name"]), str(gamma["name"])],
    )

    assert outcomes[0]["status"] == "skipped"
    assert outcomes[0]["reason"] == "target_failed"
    assert outcomes[1]["status"] == "delivered"
    assert outcomes[1]["reason"] is None
    async with get_session() as session:
        assert await session.get(ChannelMessage, posted["id"]) is not None
        assert len(list((await session.execute(select(Message))).scalars())) == 1
        assert len(list((await session.execute(select(MessageRecipient))).scalars())) == 1
        assert len(list((await session.execute(select(MentionDelivery))).scalars())) == 1


@pytest.mark.asyncio
async def test_resolver_failure_returns_source_with_safe_target_failed_status(channel_hub, monkeypatch):
    client, sender = channel_hub
    recipient = await _register(client, "/channels/alpha", "GreenHill")

    async def fail_resolver(*_args, **_kwargs):
        raise RuntimeError("injected resolver detail that must not reach the response")

    monkeypatch.setattr(app_module, "_resolve_channel_mentions", fail_resolver)
    posted = await _post(client, sender, f"@{recipient['name']}")

    assert posted["mention_deliveries"] == [
        {
            "name": recipient["name"],
            "target_project_key": None,
            "status": "skipped",
            "receipt_message_id": None,
            "reason": "target_failed",
        }
    ]
    assert "injected" not in str(posted)
    async with get_session() as session:
        assert await session.get(ChannelMessage, posted["id"]) is not None
        assert list((await session.execute(select(Message))).scalars()) == []
        assert list((await session.execute(select(MentionDelivery))).scalars()) == []


@pytest.mark.asyncio
async def test_same_project_mentions_share_one_unread_receipt(channel_hub):
    client, sender = channel_hub
    green = await _register(client, "/channels/alpha", "GreenHill")
    red = await _register(client, "/channels/alpha", "RedRiver")

    posted = await _post(client, sender, f"please review @{green['name']} and @{red['name']}")

    statuses = {entry["name"]: entry for entry in posted["mention_deliveries"]}
    assert set(statuses) == {green["name"], red["name"]}
    assert {entry["status"] for entry in statuses.values()} == {"delivered"}
    receipt_ids = {entry["receipt_message_id"] for entry in statuses.values()}
    assert len(receipt_ids) == 1
    receipt_id = receipt_ids.pop()
    assert receipt_id is not None

    async with get_session() as session:
        mappings = list(
            (
                await session.execute(
                    select(MentionDelivery).where(
                        MentionDelivery.source_channel_message_id == posted["id"]
                    )
                )
            ).scalars()
        )
        recipients = list(
            (
                await session.execute(
                    select(MessageRecipient).where(MessageRecipient.message_id == receipt_id)
                )
            ).scalars()
        )
        assert len(mappings) == 2
        assert {mapping.receipt_message_id for mapping in mappings} == {receipt_id}
        assert len(recipients) == 2
        assert {recipient.kind for recipient in recipients} == {"mention"}
        assert list((await session.execute(select(ChannelReadCursor))).scalars()) == []

    for recipient in (green, red):
        inbox = await client.call_tool(
            "fetch_inbox",
            {
                "project_key": recipient["project"],
                "agent_name": recipient["name"],
                "registration_token": recipient["token"],
                "unread_only": True,
            },
        )
        inbox_messages = _result_list(inbox)
        assert [message["id"] for message in inbox_messages] == [receipt_id]
        assert inbox_messages[0]["kind"] == "mention"


@pytest.mark.asyncio
async def test_self_unknown_and_block_all_have_explicit_safe_statuses(channel_hub):
    client, sender = channel_hub
    blocked = await _register(client, "/channels/alpha", "ClosedGate")
    _project, blocked_agent = await _project_and_agent(str(blocked["project"]), str(blocked["name"]))
    async with get_session() as session:
        db_agent = await session.get(Agent, blocked_agent.id)
        assert db_agent is not None
        db_agent.contact_policy = "block_all"
        session.add(db_agent)
        await session.commit()

    posted = await _post(client, sender, f"@{sender['name']} @MissingAgent @{blocked['name']}")

    statuses = {entry["name"]: entry for entry in posted["mention_deliveries"]}
    assert statuses[sender["name"]]["status"] == "delivered"
    assert statuses[sender["name"]]["reason"] is None
    assert statuses["MissingAgent"] == {
        "name": "MissingAgent",
        "target_project_key": None,
        "status": "skipped",
        "receipt_message_id": None,
        "reason": "unknown",
    }
    assert statuses[blocked["name"]]["status"] == "skipped"
    assert statuses[blocked["name"]]["reason"] == "block_all"

    async with get_session() as session:
        mappings = list(
            (
                await session.execute(
                    select(MentionDelivery).where(
                        MentionDelivery.source_channel_message_id == posted["id"]
                    )
                )
            ).scalars()
        )
        assert len(mappings) == 1


@pytest.mark.asyncio
async def test_unique_cross_link_beats_local_but_multiple_links_are_ambiguous(channel_hub):
    client, sender = channel_hub
    shared_name = "shared-agent"
    await _register(client, "/channels/alpha", shared_name)
    beta = await _register(client, "/channels/beta", shared_name)
    await _approve_link(sender, beta)

    unique = await _post(client, sender, f"@{shared_name}")

    delivered = unique["mention_deliveries"][0]
    assert delivered["status"] == "delivered"
    assert delivered["target_project_key"] == "/channels/beta"
    async with get_session() as session:
        receipt = await session.get(Message, delivered["receipt_message_id"])
        beta_project, beta_agent = await _project_and_agent("/channels/beta", shared_name)
        assert receipt is not None
        assert receipt.project_id == beta_project.id
        assert receipt.sender_id != beta_agent.id

    gamma = await _register(client, "/channels/gamma", shared_name)
    await _approve_link(sender, gamma)
    ambiguous = await _post(client, sender, f"@{shared_name}")

    assert ambiguous["mention_deliveries"] == [
        {
            "name": shared_name,
            "target_project_key": None,
            "status": "skipped",
            "receipt_message_id": None,
            "reason": "ambiguous",
        }
    ]
