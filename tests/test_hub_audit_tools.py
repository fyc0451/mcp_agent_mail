"""M1b-4b Hub audit write isolation and authenticated read seam."""

from __future__ import annotations

from typing import Any, cast

import pytest
from fastmcp import Client, Context
from fastmcp.exceptions import ToolError
from sqlmodel import select

import mcp_agent_mail.app as app_module
from mcp_agent_mail.app import _deliver_channel_mentions, build_mcp_server
from mcp_agent_mail.db import get_session
from mcp_agent_mail.models import (
    Agent,
    ChannelMessage,
    HubAuditEvent,
    MentionDelivery,
    Project,
)


class _RecordingContext:
    async def info(self, _message: str) -> None:
        return None


def _context() -> Context:
    return cast(Context, _RecordingContext())


async def _register(client: Client, project: str, name: str) -> dict[str, str]:
    await client.call_tool("ensure_project", {"human_key": project})
    result = await client.call_tool(
        "register_agent",
        {"project_key": project, "program": "test", "model": "test", "name": name},
    )
    return {
        "project": project,
        "name": result.data["name"],
        "token": result.data["registration_token"],
    }


async def _post(client: Client, sender: dict[str, str], body: str) -> dict[str, Any]:
    result = await client.call_tool(
        "post_channel_message",
        {
            "channel_project_key": sender["project"],
            "channel_name": "general",
            "sender_project_key": sender["project"],
            "sender_name": sender["name"],
            "subject": "audit subject",
            "body_md": body,
            "registration_token": sender["token"],
        },
    )
    return result.data


async def _project_agent_source(
    sender: dict[str, str], source_id: int
) -> tuple[Project, Agent, ChannelMessage]:
    async with get_session() as session:
        project = (
            await session.execute(select(Project).where(Project.human_key == sender["project"]))
        ).scalars().one()
        actor = (
            await session.execute(
                select(Agent).where(
                    Agent.project_id == project.id,
                    Agent.name == sender["name"],
                )
            )
        ).scalars().one()
        source = await session.get(ChannelMessage, source_id)
        assert source is not None
        return project, actor, source


@pytest.fixture
async def audit_hub(isolated_env):
    async with Client(build_mcp_server()) as client:
        sender = await _register(client, "/audit/alpha", "BlueLake")
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
async def test_post_and_mentions_write_content_free_queryable_events(audit_hub):
    client, sender = audit_hub
    recipient = await _register(client, sender["project"], "GreenHill")
    secret_body = f"TOP-SECRET-BODY @{recipient['name']} @MissingAgent"

    posted = await _post(client, sender, secret_body)
    listed = await client.call_tool(
        "list_hub_audit_events",
        {
            "project_key": sender["project"],
            "registration_token": sender["token"],
        },
    )

    assert listed.data["count"] == 3
    assert [event["event_type"] for event in listed.data["events"]] == [
        "channel_message_posted",
        "channel_mention_delivery",
        "channel_mention_delivery",
    ]
    assert [event["outcome"] for event in listed.data["events"]] == [
        "succeeded",
        "delivered",
        "skipped",
    ]
    assert listed.data["events"][2]["reason"] == "unknown"
    assert all(event["source_id"] == posted["id"] for event in listed.data["events"])
    assert set(listed.data["events"][0]) == {
        "id",
        "project_id",
        "actor_agent_id",
        "event_type",
        "source_type",
        "source_id",
        "outcome",
        "reason",
        "target_project_id",
        "target_agent_id",
        "related_message_id",
        "created_ts",
    }
    assert secret_body not in str(listed.data)
    assert sender["token"] not in str(listed.data)


@pytest.mark.asyncio
async def test_repeat_delivery_appends_already_delivered_attempt(audit_hub):
    client, sender = audit_hub
    recipient = await _register(client, sender["project"], "GreenHill")
    posted = await _post(client, sender, "plain source")
    project, actor, source = await _project_agent_source(sender, posted["id"])

    first = await _deliver_channel_mentions(
        _context(), project, actor, source, [recipient["name"]]
    )
    second = await _deliver_channel_mentions(
        _context(), project, actor, source, [recipient["name"]]
    )

    assert first[0]["status"] == "delivered"
    assert second[0]["status"] == "already_delivered"
    listed = await client.call_tool(
        "list_hub_audit_events",
        {
            "project_key": sender["project"],
            "event_type": "channel_mention_delivery",
            "registration_token": sender["token"],
        },
    )
    assert [event["outcome"] for event in listed.data["events"]] == [
        "delivered",
        "already_delivered",
    ]
    assert listed.data["events"][0]["related_message_id"] == listed.data["events"][1][
        "related_message_id"
    ]


@pytest.mark.asyncio
async def test_retire_and_unretire_write_agent_lifecycle_audit(isolated_env):
    async with Client(build_mcp_server()) as client:
        agent = await _register(client, "/audit/lifecycle", "BlueLake")

        await client.call_tool(
            "retire_agent",
            {
                "project_key": agent["project"],
                "agent_name": agent["name"],
                "registration_token": agent["token"],
            },
        )
        await client.call_tool(
            "unretire_agent",
            {
                "project_key": agent["project"],
                "agent_name": agent["name"],
                "registration_token": agent["token"],
            },
        )
        listed = await client.call_tool(
            "list_hub_audit_events",
            {
                "project_key": agent["project"],
                "event_type": "agent_lifecycle",
                "registration_token": agent["token"],
            },
        )

    assert [event["outcome"] for event in listed.data["events"]] == [
        "retired",
        "restored",
    ]
    for event in listed.data["events"]:
        assert event["source_type"] == "agent"
        assert event["source_id"] == event["actor_agent_id"]
        assert event["target_agent_id"] == event["actor_agent_id"]
        assert event["target_project_id"] == event["project_id"]


@pytest.mark.asyncio
async def test_audit_storage_failure_never_rolls_back_post_or_receipt(audit_hub, monkeypatch):
    client, sender = audit_hub
    recipient = await _register(client, sender["project"], "GreenHill")

    async def fail_audit(_events):
        raise RuntimeError("injected audit sink failure")

    monkeypatch.setattr(app_module, "_commit_hub_audit_events", fail_audit)
    posted = await _post(client, sender, f"please review @{recipient['name']}")

    assert posted["mention_deliveries"][0]["status"] == "delivered"
    async with get_session() as session:
        assert await session.get(ChannelMessage, posted["id"]) is not None
        assert len(list((await session.execute(select(MentionDelivery))).scalars())) == 1
        assert list((await session.execute(select(HubAuditEvent))).scalars()) == []


@pytest.mark.asyncio
async def test_audit_event_build_failure_never_changes_business_result(audit_hub, monkeypatch):
    client, sender = audit_hub
    recipient = await _register(client, sender["project"], "GreenHill")

    def fail_audit_builder(**_kwargs):
        raise RuntimeError("injected audit event build failure")

    monkeypatch.setattr(app_module, "_new_hub_audit_event", fail_audit_builder)
    posted = await _post(client, sender, f"please review @{recipient['name']}")

    assert posted["mention_deliveries"][0]["status"] == "delivered"
    async with get_session() as session:
        assert await session.get(ChannelMessage, posted["id"]) is not None
        assert len(list((await session.execute(select(MentionDelivery))).scalars())) == 1
        assert list((await session.execute(select(HubAuditEvent))).scalars()) == []


@pytest.mark.asyncio
async def test_failed_delivery_is_audited_without_exception_detail(audit_hub, monkeypatch):
    client, sender = audit_hub
    recipient = await _register(client, sender["project"], "GreenHill")

    async def fail_delivery(*_args, **_kwargs):
        raise RuntimeError("PRIVATE-INTERNAL-FAILURE")

    monkeypatch.setattr(app_module, "_write_channel_mention_receipt", fail_delivery)
    posted = await _post(client, sender, f"@{recipient['name']}")
    assert posted["mention_deliveries"][0]["reason"] == "target_failed"

    listed = await client.call_tool(
        "list_hub_audit_events",
        {
            "project_key": sender["project"],
            "event_type": "channel_mention_delivery",
            "registration_token": sender["token"],
        },
    )
    assert listed.data["events"][0]["outcome"] == "skipped"
    assert listed.data["events"][0]["reason"] == "target_failed"
    assert "PRIVATE-INTERNAL-FAILURE" not in str(listed.data)


@pytest.mark.asyncio
async def test_query_requires_project_auth_and_has_stable_cursor(audit_hub):
    client, sender = audit_hub
    await _post(client, sender, "first")
    second = await _post(client, sender, "second")

    first_page = await client.call_tool(
        "list_hub_audit_events",
        {
            "project_key": sender["project"],
            "limit": 1,
            "registration_token": sender["token"],
        },
    )
    second_page = await client.call_tool(
        "list_hub_audit_events",
        {
            "project_key": sender["project"],
            "after_id": first_page.data["cursor"],
            "registration_token": sender["token"],
        },
    )
    assert first_page.data["count"] == 1
    assert second_page.data["count"] == 1
    assert second_page.data["events"][0]["source_id"] == second["id"]

    async with Client(build_mcp_server()) as unauthenticated:
        with pytest.raises(ToolError):
            await unauthenticated.call_tool(
                "list_hub_audit_events",
                {"project_key": sender["project"]},
            )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [("after_id", -1), ("event_type", "stop")],
)
async def test_query_rejects_invalid_filters(audit_hub, field, value):
    client, sender = audit_hub
    with pytest.raises(ToolError):
        await client.call_tool(
            "list_hub_audit_events",
            {
                "project_key": sender["project"],
                "registration_token": sender["token"],
                field: value,
            },
        )
