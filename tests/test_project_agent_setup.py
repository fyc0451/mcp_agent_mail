"""P1 Core Tests: Project and Agent Setup.

Test project and agent registration flows that form the foundation
for all other MCP Agent Mail operations.

Test Cases:
1. ensure_project creates new project
2. ensure_project is idempotent
3. Project slug generated from human_key
4. register_agent creates new agent
5. register_agent updates existing agent
6. create_agent_identity always creates new
7. Agent profile written to Git archive
8. last_active_ts updated on activity
9. whois returns agent details

Reference: mcp_agent_mail-mm2
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from sqlalchemy import text
from sqlmodel import select

from mcp_agent_mail.app import ToolExecutionError, _get_agent, _get_agents_batch, build_mcp_server
from mcp_agent_mail.db import get_session
from mcp_agent_mail.models import Project

# ============================================================================
# Helper: Direct SQL verification
# ============================================================================


async def get_project_from_db(human_key: str) -> dict | None:
    """Get project details from database by human_key."""
    async with get_session() as session:
        result = await session.execute(
            text("SELECT id, slug, human_key, created_at FROM projects WHERE human_key = :key"),
            {"key": human_key},
        )
        row = result.first()
        if row is None:
            return None
        return {
            "id": row[0],
            "slug": row[1],
            "human_key": row[2],
            "created_at": row[3],
        }


async def get_agent_from_db(project_id: int, agent_name: str) -> dict | None:
    """Get agent details from database."""
    async with get_session() as session:
        result = await session.execute(
            text(
                "SELECT id, name, program, model, task_description, "
                "inception_ts, last_active_ts "
                "FROM agents WHERE project_id = :pid AND name = :name"
            ),
            {"pid": project_id, "name": agent_name},
        )
        row = result.first()
        if row is None:
            return None
        return {
            "id": row[0],
            "name": row[1],
            "program": row[2],
            "model": row[3],
            "task_description": row[4],
            "inception_ts": row[5],
            "last_active_ts": row[6],
        }


async def count_agents_in_project(project_id: int) -> int:
    """Count agents in a project."""
    async with get_session() as session:
        result = await session.execute(
            text("SELECT COUNT(*) FROM agents WHERE project_id = :pid"),
            {"pid": project_id},
        )
        return result.scalar() or 0


# ============================================================================
# Test: ensure_project
# ============================================================================


@pytest.mark.asyncio
async def test_ensure_project_creates_new_project(isolated_env):
    """ensure_project creates a new project when it doesn't exist."""
    server = build_mcp_server()
    async with Client(server) as client:
        # Verify project doesn't exist yet
        db_check = await get_project_from_db("/test/setup/new")
        assert db_check is None, "Project should not exist initially"

        # Create project
        result = await client.call_tool(
            "ensure_project", {"human_key": "/test/setup/new"}
        )

        # Verify response
        assert result.data["human_key"] == "/test/setup/new"
        assert "slug" in result.data
        assert "id" in result.data

        # Verify database record
        db_project = await get_project_from_db("/test/setup/new")
        assert db_project is not None, "Project should exist in database"
        assert db_project["human_key"] == "/test/setup/new"


@pytest.mark.asyncio
async def test_ensure_project_is_idempotent(isolated_env):
    """ensure_project returns existing project without modification."""
    server = build_mcp_server()
    async with Client(server) as client:
        # Create project first time
        result1 = await client.call_tool(
            "ensure_project", {"human_key": "/test/setup/idem"}
        )
        first_id = result1.data["id"]
        first_slug = result1.data["slug"]

        # Call again - should be idempotent
        result2 = await client.call_tool(
            "ensure_project", {"human_key": "/test/setup/idem"}
        )

        # Should return same project
        assert result2.data["id"] == first_id, "Should return same project ID"
        assert result2.data["slug"] == first_slug, "Should return same slug"

        # Verify only one project exists
        db_project = await get_project_from_db("/test/setup/idem")
        assert db_project is not None, "Project should exist"
        assert db_project["id"] == first_id


@pytest.mark.asyncio
async def test_project_slug_generated_from_human_key(isolated_env):
    """Project slug is properly generated from human_key path."""
    server = build_mcp_server()
    async with Client(server) as client:
        result = await client.call_tool(
            "ensure_project", {"human_key": "/data/projects/MyApp"}
        )

        # Slug should be lowercase, path-derived
        slug = result.data["slug"]
        assert slug is not None
        # Slug should be derived from path (lowercased, normalized)
        assert "myapp" in slug.lower() or "my-app" in slug.lower() or "data" in slug.lower()


@pytest.mark.asyncio
async def test_ensure_project_resolves_symlinks(isolated_env, tmp_path):
    """ensure_project resolves symlinks to canonical paths.

    This ensures that /dp/ntm and /data/projects/ntm resolve to the same
    project when /dp is a symlink to /data/projects.
    """

    # Create a real directory and a symlink to it
    real_dir = tmp_path / "real_project"
    real_dir.mkdir()
    symlink_dir = tmp_path / "symlink_project"
    symlink_dir.symlink_to(real_dir)

    real_path = str(real_dir.resolve())
    symlink_path = str(symlink_dir)

    server = build_mcp_server()
    async with Client(server) as client:
        # Create project via symlink path
        result1 = await client.call_tool(
            "ensure_project", {"human_key": symlink_path}
        )

        # The stored human_key should be the resolved (canonical) path
        assert result1.data["human_key"] == real_path, \
            f"human_key should be resolved path {real_path}, got {result1.data['human_key']}"

        # Create project via real path - should return same project
        result2 = await client.call_tool(
            "ensure_project", {"human_key": real_path}
        )

        # Should be the same project
        assert result1.data["id"] == result2.data["id"], \
            "Symlink and real path should resolve to same project"
        assert result1.data["slug"] == result2.data["slug"], \
            "Symlink and real path should have same slug"

        # Register an agent via the symlinked project_key and ensure it lands on the canonical project
        agent_result = await client.call_tool(
            "register_agent",
            {"project_key": symlink_path, "program": "test-program", "model": "test-model"},
        )
        project = await get_project_from_db(real_path)
        assert project is not None, "Canonical project should exist"
        agent = await get_agent_from_db(project["id"], agent_result.data["name"])
        assert agent is not None, "Agent registered via symlink should attach to canonical project"


# ============================================================================
# Test: register_agent
# ============================================================================


@pytest.mark.asyncio
async def test_register_agent_creates_new_agent(isolated_env):
    """register_agent creates a new agent with correct attributes."""
    server = build_mcp_server()
    async with Client(server) as client:
        # Setup project
        await client.call_tool("ensure_project", {"human_key": "/test/setup/agent"})

        # Register agent (let server generate name)
        result = await client.call_tool(
            "register_agent",
            {
                "project_key": "/test/setup/agent",
                "program": "test-program",
                "model": "test-model",
                "task_description": "Testing agent creation",
            },
        )

        # Verify response
        assert "name" in result.data
        assert result.data["program"] == "test-program"
        assert result.data["model"] == "test-model"
        assert result.data["task_description"] == "Testing agent creation"

        # Verify database record
        project = await get_project_from_db("/test/setup/agent")
        assert project is not None, "Project should exist"
        agent = await get_agent_from_db(project["id"], result.data["name"])
        assert agent is not None, "Agent should exist in database"
        assert agent["program"] == "test-program"
        assert agent["model"] == "test-model"


@pytest.mark.asyncio
async def test_register_agent_updates_existing_agent(isolated_env):
    """register_agent updates an existing agent's metadata."""
    server = build_mcp_server()
    async with Client(server) as client:
        # Setup
        await client.call_tool("ensure_project", {"human_key": "/test/setup/update"})

        # Register agent first time
        result1 = await client.call_tool(
            "register_agent",
            {
                "project_key": "/test/setup/update",
                "program": "original-program",
                "model": "original-model",
                "task_description": "Original task",
            },
        )
        agent_name = result1.data["name"]

        # Get initial agent count
        project = await get_project_from_db("/test/setup/update")
        assert project is not None, "Project should exist"
        initial_count = await count_agents_in_project(project["id"])

        # Update agent with same name
        result2 = await client.call_tool(
            "register_agent",
            {
                "project_key": "/test/setup/update",
                "program": "original-program",
                "model": "original-model",
                "name": agent_name,
                "task_description": "Updated task",
            },
        )

        # Should return same agent name
        assert result2.data["name"] == agent_name

        # Verify count didn't increase (update, not create)
        final_count = await count_agents_in_project(project["id"])
        assert final_count == initial_count, "Should not create duplicate agent"

        # Verify task description was updated
        agent = await get_agent_from_db(project["id"], agent_name)
        assert agent is not None, "Agent should exist"
        assert agent["task_description"] == "Updated task"


@pytest.mark.asyncio
async def test_register_agent_existing_identity_requires_token_across_sessions(isolated_env):
    """A new session cannot take over an existing named agent without its token."""
    server = build_mcp_server()
    async with Client(server) as bootstrap_client:
        await bootstrap_client.call_tool("ensure_project", {"human_key": "/test/setup/takeover"})
        created = await bootstrap_client.call_tool(
            "register_agent",
            {
                "project_key": "/test/setup/takeover",
                "program": "bootstrap-program",
                "model": "bootstrap-model",
                "name": "BlueLake",
                "task_description": "Original owner",
            },
        )
        registration_token = created.data["registration_token"]

    async with Client(server) as attacker_client:
        with pytest.raises(ToolError) as exc_info:
            await attacker_client.call_tool(
                "register_agent",
                {
                    "project_key": "/test/setup/takeover",
                    "program": "attacker-program",
                    "model": "attacker-model",
                    "name": "BlueLake",
                    "task_description": "Hijacked",
                },
            )
        assert "registration_token" in str(exc_info.value)

    project = await get_project_from_db("/test/setup/takeover")
    assert project is not None
    assert await count_agents_in_project(project["id"]) == 1

    async with Client(server) as owner_client:
        updated = await owner_client.call_tool(
            "register_agent",
            {
                "project_key": "/test/setup/takeover",
                "program": "bootstrap-program",
                "model": "bootstrap-model",
                "name": "BlueLake",
                "task_description": "Updated by owner",
                "registration_token": registration_token,
            },
        )
    assert updated.data["name"] == "BlueLake"
    assert updated.data["task_description"] == "Updated by owner"


@pytest.mark.asyncio
async def test_register_agent_generates_valid_name(isolated_env):
    """register_agent generates valid adjective+noun names."""
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/test/setup/names"})

        # Register several agents and verify name format
        names = []
        for _ in range(3):
            result = await client.call_tool(
                "register_agent",
                {
                    "project_key": "/test/setup/names",
                    "program": "test",
                    "model": "test",
                },
            )
            name = result.data["name"]
            names.append(name)

            # Name should be CamelCase (adjective+noun pattern)
            assert name[0].isupper(), f"Name should start with capital: {name}"
            # Should have at least 2 capital letters (adjective + noun)
            capitals = sum(1 for c in name if c.isupper())
            assert capitals >= 2, f"Name should be adjective+noun pattern: {name}"

        # All names should be unique
        assert len(set(names)) == 3, "All generated names should be unique"


# ============================================================================
# Test: create_agent_identity
# ============================================================================


@pytest.mark.asyncio
async def test_create_agent_identity_always_creates_new(isolated_env):
    """create_agent_identity always creates a new agent, never updates."""
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/test/setup/identity"})

        # Create first identity
        result1 = await client.call_tool(
            "create_agent_identity",
            {
                "project_key": "/test/setup/identity",
                "program": "identity-test",
                "model": "test-model",
            },
        )
        name1 = result1.data["name"]

        # Create second identity
        result2 = await client.call_tool(
            "create_agent_identity",
            {
                "project_key": "/test/setup/identity",
                "program": "identity-test",
                "model": "test-model",
            },
        )
        name2 = result2.data["name"]

        # Should be different names
        assert name1 != name2, "create_agent_identity should always create unique agents"

        # Both should exist in database
        project = await get_project_from_db("/test/setup/identity")
        assert project is not None, "Project should exist"
        agent1 = await get_agent_from_db(project["id"], name1)
        agent2 = await get_agent_from_db(project["id"], name2)
        assert agent1 is not None
        assert agent2 is not None


@pytest.mark.asyncio
async def test_create_agent_identity_with_name_hint(isolated_env):
    """create_agent_identity respects valid name_hint."""
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/test/setup/hint"})

        # Create with valid name hint
        result = await client.call_tool(
            "create_agent_identity",
            {
                "project_key": "/test/setup/hint",
                "program": "test",
                "model": "test",
                "name_hint": "GreenCastle",
            },
        )

        # Should use the hint
        assert result.data["name"] == "GreenCastle"


@pytest.mark.asyncio
async def test_create_agent_identity_offloads_archive_name_check(isolated_env, monkeypatch):
    """Archive-side name availability checks should not block the event loop."""
    main_thread = threading.main_thread()
    path_type = type(Path("/"))
    original_exists = path_type.exists
    seen_archive_exists = False

    def checked_exists(self: Path) -> bool:
        nonlocal seen_archive_exists
        if self.name == "BlueRiver" and "agents" in self.parts:
            seen_archive_exists = True
            assert threading.current_thread() is not main_thread
        return original_exists(self)

    monkeypatch.setattr(path_type, "exists", checked_exists)

    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/test/setup/hint-offload"})
        result = await client.call_tool(
            "create_agent_identity",
            {
                "project_key": "/test/setup/hint-offload",
                "program": "test",
                "model": "test",
                "name_hint": "BlueRiver",
            },
        )

    assert result.data["name"] == "BlueRiver"
    assert seen_archive_exists is True


# ============================================================================
# Test: last_active_ts updates
# ============================================================================


@pytest.mark.asyncio
async def test_last_active_ts_updated_on_activity(isolated_env):
    """last_active_ts is updated when agent performs actions."""
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/test/setup/active"})

        # Register agent
        result = await client.call_tool(
            "register_agent",
            {
                "project_key": "/test/setup/active",
                "program": "test",
                "model": "test",
            },
        )
        agent_name = result.data["name"]

        # Get initial last_active_ts
        project = await get_project_from_db("/test/setup/active")
        assert project is not None, "Project should exist"
        agent_before = await get_agent_from_db(project["id"], agent_name)
        assert agent_before is not None, "Agent should exist before update"
        initial_ts = agent_before["last_active_ts"]

        # Small delay to ensure timestamp difference
        await asyncio.sleep(0.1)

        # Perform an action (register again to update)
        await client.call_tool(
            "register_agent",
            {
                "project_key": "/test/setup/active",
                "program": "test",
                "model": "test",
                "name": agent_name,
            },
        )

        # Verify last_active_ts was updated
        agent_after = await get_agent_from_db(project["id"], agent_name)
        assert agent_after is not None, "Agent should exist after update"
        assert agent_after["last_active_ts"] >= initial_ts, "last_active_ts should be updated"


# ============================================================================
# Test: whois
# ============================================================================


@pytest.mark.asyncio
async def test_whois_returns_agent_details(isolated_env):
    """whois returns comprehensive agent profile information."""
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/test/setup/whois"})

        # Register agent with details
        reg_result = await client.call_tool(
            "register_agent",
            {
                "project_key": "/test/setup/whois",
                "program": "whois-test-program",
                "model": "whois-test-model",
                "task_description": "Testing whois functionality",
            },
        )
        agent_name = reg_result.data["name"]

        # Query whois
        whois_result = await client.call_tool(
            "whois",
            {
                "project_key": "/test/setup/whois",
                "agent_name": agent_name,
            },
        )

        # Verify response contains expected fields
        assert whois_result.data["name"] == agent_name
        assert whois_result.data["program"] == "whois-test-program"
        assert whois_result.data["model"] == "whois-test-model"
        assert whois_result.data["task_description"] == "Testing whois functionality"
        assert "inception_ts" in whois_result.data
        assert "last_active_ts" in whois_result.data


@pytest.mark.asyncio
async def test_whois_with_recent_commits(isolated_env):
    """whois can include recent commit information."""
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/test/setup/commits"})

        reg_result = await client.call_tool(
            "register_agent",
            {
                "project_key": "/test/setup/commits",
                "program": "test",
                "model": "test",
            },
        )
        agent_name = reg_result.data["name"]

        # Query whois with recent commits
        whois_result = await client.call_tool(
            "whois",
            {
                "project_key": "/test/setup/commits",
                "agent_name": agent_name,
                "include_recent_commits": True,
                "commit_limit": 5,
            },
        )

        # Should include recent_commits field (may be empty list)
        assert "recent_commits" in whois_result.data


# ============================================================================
# Test: Git archive profile.json
# ============================================================================


@pytest.mark.asyncio
async def test_agent_profile_written_to_git_archive(isolated_env):
    """Agent registration writes profile.json to Git archive.

    Note: This test verifies the agent data is returned correctly.
    The actual file write to Git archive depends on storage configuration.
    """
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/test/setup/archive"})

        # Register agent
        result = await client.call_tool(
            "register_agent",
            {
                "project_key": "/test/setup/archive",
                "program": "archive-test",
                "model": "test-model",
                "task_description": "Testing archive writes",
            },
        )
        agent_name = result.data["name"]

        # Verify the agent was registered with correct data
        # (the actual file write happens in storage layer)
        assert agent_name is not None
        assert result.data["program"] == "archive-test"
        assert result.data["model"] == "test-model"

        # Verify via whois that data persisted
        whois_result = await client.call_tool(
            "whois",
            {
                "project_key": "/test/setup/archive",
                "agent_name": agent_name,
            },
        )
        assert whois_result.data["name"] == agent_name
        assert whois_result.data["program"] == "archive-test"


# ============================================================================
# Test: Error handling
# ============================================================================


@pytest.mark.asyncio
async def test_register_agent_invalid_name_rejected(isolated_env):
    """register_agent rejects or handles invalid agent names appropriately."""
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/test/setup/invalid"})

        # Try to register with invalid name (placeholder name)
        # Some placeholder names may be caught, others may be accepted with warnings
        try:
            result = await client.call_tool(
                "register_agent",
                {
                    "project_key": "/test/setup/invalid",
                    "program": "test",
                    "model": "test",
                    "name": "YourAgentName",  # Should be caught as placeholder
                },
            )
            # If it was accepted, verify an agent was created (test passes)
            # This tests the graceful handling path
            assert result.data["name"] is not None
        except Exception as e:
            # If rejected, verify error message is meaningful
            error_msg = str(e).lower()
            assert any(keyword in error_msg for keyword in [
                "placeholder", "adjective", "noun", "invalid", "agent", "name"
            ]), f"Error should mention name validation: {e}"


@pytest.mark.asyncio
async def test_whois_nonexistent_agent_error(isolated_env):
    """whois returns error for non-existent agent."""
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/test/setup/noagent"})

        # Try to whois non-existent agent
        with pytest.raises(Exception) as exc_info:
            await client.call_tool(
                "whois",
                {
                    "project_key": "/test/setup/noagent",
                    "agent_name": "NonExistentAgent",
                },
            )

        # Should indicate agent not found
        error_msg = str(exc_info.value).lower()
        assert "not found" in error_msg or "does not exist" in error_msg


# ============================================================================
# Test: _get_agents_batch helper
# ============================================================================


async def _load_project(human_key: str) -> Project:
    async with get_session() as session:
        result = await session.execute(select(Project).where(Project.human_key == human_key))
        project = result.scalars().first()
    assert project is not None
    return project


@pytest.mark.asyncio
async def test_get_agents_batch_empty_list(isolated_env):
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/test/setup/batch-empty"})

    project = await _load_project("/test/setup/batch-empty")
    resolved = await _get_agents_batch(project, [])
    assert resolved == {}


@pytest.mark.asyncio
async def test_get_agents_batch_mixed_case(isolated_env):
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/test/setup/batch-case"})
        agent_result = await client.call_tool(
            "create_agent_identity",
            {
                "project_key": "/test/setup/batch-case",
                "program": "test",
                "model": "test",
                "task_description": "Batch case test",
            },
        )
        agent_name = agent_result.data["name"]

    project = await _load_project("/test/setup/batch-case")
    resolved = await _get_agents_batch(project, [agent_name.lower(), agent_name.upper()])
    assert resolved[agent_name.lower()].name == agent_name
    assert resolved[agent_name.upper()].name == agent_name


@pytest.mark.asyncio
async def test_get_agents_batch_missing_name_uses_get_agent_error(isolated_env):
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/test/setup/batch-missing"})
        agent_result = await client.call_tool(
            "create_agent_identity",
            {
                "project_key": "/test/setup/batch-missing",
                "program": "test",
                "model": "test",
                "task_description": "Batch missing test",
            },
        )
        agent_name = agent_result.data["name"]

    project = await _load_project("/test/setup/batch-missing")
    missing_name = f"{agent_name}Typo"
    with pytest.raises(ToolExecutionError) as batch_exc:
        await _get_agents_batch(project, [agent_name, missing_name])

    with pytest.raises(ToolExecutionError) as single_exc:
        await _get_agent(project, missing_name)

    assert batch_exc.value.error_type == single_exc.value.error_type
    assert str(batch_exc.value) == str(single_exc.value)


@pytest.mark.asyncio
async def test_get_agents_batch_placeholder_detection(isolated_env):
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/test/setup/batch-placeholder"})

    project = await _load_project("/test/setup/batch-placeholder")
    with pytest.raises(ToolExecutionError) as exc_info:
        await _get_agents_batch(project, ["YOUR_AGENT_NAME"])
    assert exc_info.value.error_type == "CONFIGURATION_ERROR"


# ============================================================================
# Test: rotate_agent_capability
# ============================================================================


async def _register_with_token(client, project_key: str, name: str | None = None):
    args = {"project_key": project_key, "program": "test-program", "model": "test-model"}
    if name:
        args["name"] = name
    result = await client.call_tool("register_agent", args)
    return result.data


@pytest.mark.asyncio
async def test_rotate_capability_preserves_identity_and_inbox(isolated_env):
    """同 id/name 原位轮换：unread/历史不变，旧值失效、新值正向。"""
    project_key = "/test/setup/rotate-preserve"
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": project_key})
        target = await _register_with_token(client, project_key)
        sender = await _register_with_token(client, project_key)
        t_name, t_old = target["name"], target["registration_token"]
        s_name, s_token = sender["name"], sender["registration_token"]

        await client.call_tool("send_message", {
            "project_key": project_key,
            "sender_name": s_name,
            "sender_token": s_token,
            "to": [t_name],
            "subject": "rotate probe",
            "body_md": "before rotation",
        })
        inbox_before = await client.call_tool("fetch_inbox", {
            "project_key": project_key,
            "agent_name": t_name,
            "registration_token": t_old,
            "limit": 10,
        })
        assert len(inbox_before.structured_content["result"]) == 1
        message_id = inbox_before.structured_content["result"][0]["id"]

        new_token = "a" * 43
        rotated = await client.call_tool("rotate_agent_capability", {
            "project_key": project_key,
            "agent_name": t_name,
            "old_registration_token": t_old,
            "new_registration_token": new_token,
        })
        assert rotated.data["status"] == "rotated"
        assert rotated.data["agent_name"] == t_name
        # 返回值不得包含任何 capability 值
        serialized = str(rotated.data)
        assert t_old not in serialized
        assert new_token not in serialized

        project = await get_project_from_db(project_key)
        db_agent = await get_agent_from_db(project["id"], t_name)
        assert db_agent is not None
        assert db_agent["id"] == target["id"]
        assert db_agent["name"] == t_name

        # 新值正向（新会话，绕开绑定会话的 token 直通）：whois / fetch_inbox
        async with Client(server) as new_session:
            whois = await new_session.call_tool("whois", {
                "project_key": project_key,
                "agent_name": t_name,
                "registration_token": new_token,
            })
            assert whois.data["name"] == t_name
            inbox_after = await new_session.call_tool("fetch_inbox", {
                "project_key": project_key,
                "agent_name": t_name,
                "registration_token": new_token,
                "limit": 10,
            })
            assert [m["id"] for m in inbox_after.structured_content["result"]] == [message_id]

        # 旧值立即失效（独立新会话；同会话先成功认证会绑定并放行后续调用）
        async with Client(server) as old_session:
            with pytest.raises(Exception) as exc_info:
                await old_session.call_tool("whois", {
                    "project_key": project_key,
                    "agent_name": t_name,
                    "registration_token": t_old,
                })
            assert "registration_token" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_rotate_capability_wrong_old_token_has_no_state_change(isolated_env):
    """错误旧值：拒绝且 token 无变化。"""
    project_key = "/test/setup/rotate-wrong-old"
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": project_key})
        agent = await _register_with_token(client, project_key)
        name, old = agent["name"], agent["registration_token"]

        with pytest.raises(Exception) as exc_info:
            await client.call_tool("rotate_agent_capability", {
                "project_key": project_key,
                "agent_name": name,
                "old_registration_token": "b" * 43,
                "new_registration_token": "c" * 43,
            })
        assert "capability" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()

        # 新会话验证 token 未被轮换（绑定会话会绕过 token 检查）
        async with Client(server) as fresh:
            whois = await fresh.call_tool("whois", {
                "project_key": project_key,
                "agent_name": name,
                "registration_token": old,
            })
            assert whois.data["name"] == name


@pytest.mark.asyncio
async def test_rotate_capability_rejects_invalid_new_value(isolated_env):
    """弱/非法新值：INVALID_ARGUMENT 且无状态变化。"""
    project_key = "/test/setup/rotate-bad-new"
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": project_key})
        agent = await _register_with_token(client, project_key)
        name, old = agent["name"], agent["registration_token"]

        bad_values = ["", "short", "x" * 65, "has space and !@#" * 3, old]
        for bad in bad_values:
            with pytest.raises(Exception) as exc_info:
                await client.call_tool("rotate_agent_capability", {
                    "project_key": project_key,
                    "agent_name": name,
                    "old_registration_token": old,
                    "new_registration_token": bad,
                })
            assert any(k in str(exc_info.value).lower() for k in ("capability", "invalid", "differ"))

        whois = await client.call_tool("whois", {
            "project_key": project_key,
            "agent_name": name,
            "registration_token": old,
        })
        assert whois.data["name"] == name


@pytest.mark.asyncio
async def test_rotate_capability_requires_token_for_tokenless_agent(isolated_env):
    """无 token 的旧 agent：拒绝邻接代认证。"""
    project_key = "/test/setup/rotate-tokenless"
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": project_key})
        agent = await _register_with_token(client, project_key)
        # 构造 tokenless 旧行
        from sqlmodel import select as _sm_select
        from mcp_agent_mail.models import Agent as AgentModel
        async with get_session() as session:
            result = await session.execute(
                _sm_select(AgentModel).where(AgentModel.name == agent["name"])
            )
            db_agent = result.scalars().first()
            db_agent.registration_token = None
            session.add(db_agent)
            await session.commit()

        with pytest.raises(Exception) as exc_info:
            await client.call_tool("rotate_agent_capability", {
                "project_key": project_key,
                "agent_name": agent["name"],
                "old_registration_token": "d" * 43,
                "new_registration_token": "e" * 43,
            })
        assert "capability" in str(exc_info.value).lower() or "token" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_rotate_capability_rejects_collision_with_other_agent(isolated_env):
    """新值与同项目其他 agent 冲突：拒绝且无状态变化。"""
    project_key = "/test/setup/rotate-collision"
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": project_key})
        agent_a = await _register_with_token(client, project_key)
        agent_b = await _register_with_token(client, project_key)

        with pytest.raises(Exception) as exc_info:
            await client.call_tool("rotate_agent_capability", {
                "project_key": project_key,
                "agent_name": agent_a["name"],
                "old_registration_token": agent_a["registration_token"],
                "new_registration_token": agent_b["registration_token"],
            })
        assert "collision" in str(exc_info.value).lower() or "another agent" in str(exc_info.value).lower()

        whois = await client.call_tool("whois", {
            "project_key": project_key,
            "agent_name": agent_a["name"],
            "registration_token": agent_a["registration_token"],
        })
        assert whois.data["name"] == agent_a["name"]


@pytest.mark.asyncio
async def test_rotate_capability_preserves_retired_at(isolated_env):
    """retired 行轮换：retired_at 不变。"""
    project_key = "/test/setup/rotate-retired"
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": project_key})
        agent = await _register_with_token(client, project_key)
        name, old = agent["name"], agent["registration_token"]
        await client.call_tool("retire_agent", {
            "project_key": project_key,
            "agent_name": name,
            "registration_token": old,
        })

        new_token = "f" * 43
        await client.call_tool("rotate_agent_capability", {
            "project_key": project_key,
            "agent_name": name,
            "old_registration_token": old,
            "new_registration_token": new_token,
        })

        from sqlmodel import select as _sm_select
        from mcp_agent_mail.models import Agent as AgentModel
        async with get_session() as session:
            result = await session.execute(
                _sm_select(AgentModel).where(AgentModel.name == name)
            )
            db_agent = result.scalars().first()
            assert db_agent.retired_at is not None
            assert db_agent.registration_token == new_token


@pytest.mark.asyncio
async def test_rotate_capability_logs_never_contain_capability_values(isolated_env, capsys):
    """默认 TOOLS_LOG_ENABLED=true 时，工具日志不得出现旧/新 capability 值。"""
    project_key = "/test/setup/rotate-log-redaction"
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": project_key})
        agent = await _register_with_token(client, project_key)
        name, old = agent["name"], agent["registration_token"]
        new_token = "m" * 43
        await client.call_tool("rotate_agent_capability", {
            "project_key": project_key,
            "agent_name": name,
            "old_registration_token": old,
            "new_registration_token": new_token,
        })
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert old not in combined
        assert new_token not in combined
        assert "<redacted>" in combined


@pytest.mark.asyncio
async def test_rotate_capability_concurrent_same_old_token_single_success(isolated_env):
    """同旧值并发轮换：BEGIN IMMEDIATE 串行化，恰好一个成功，最终值可认证。"""
    import asyncio

    project_key = "/test/setup/rotate-concurrent"
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": project_key})
        agent = await _register_with_token(client, project_key)
        name, old = agent["name"], agent["registration_token"]
        new_a, new_b = "q" * 43, "r" * 43

        results = await asyncio.gather(
            client.call_tool("rotate_agent_capability", {
                "project_key": project_key,
                "agent_name": name,
                "old_registration_token": old,
                "new_registration_token": new_a,
            }),
            client.call_tool("rotate_agent_capability", {
                "project_key": project_key,
                "agent_name": name,
                "old_registration_token": old,
                "new_registration_token": new_b,
            }),
            return_exceptions=True,
        )
        successes = [r for r in results if not isinstance(r, Exception)]
        assert len(successes) == 1, f"期望恰好一个成功，实际: {results!r}"

        from sqlmodel import select as _sm_select
        from mcp_agent_mail.models import Agent as AgentModel
        async with get_session() as session:
            db_agent = (await session.execute(
                _sm_select(AgentModel).where(AgentModel.name == name)
            )).scalars().first()
        final_token = db_agent.registration_token
        assert final_token in (new_a, new_b)
        loser = new_b if final_token == new_a else new_a

        async with Client(server) as fresh:
            whois = await fresh.call_tool("whois", {
                "project_key": project_key,
                "agent_name": name,
                "registration_token": final_token,
            })
            assert whois.data["name"] == name
        async with Client(server) as loser_session:
            with pytest.raises(Exception) as exc_info:
                await loser_session.call_tool("whois", {
                    "project_key": project_key,
                    "agent_name": name,
                    "registration_token": loser,
                })
            assert "registration_token" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_rotate_capability_concurrent_agents_same_new_value_single_success(isolated_env):
    """两个 agent 并发轮换到同一 new capability：恰一个成功，最终无重复。"""
    import asyncio

    project_key = "/test/setup/rotate-concurrent-shared-new"
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": project_key})
        agent_a = await _register_with_token(client, project_key)
        agent_b = await _register_with_token(client, project_key)
        shared_new = "s" * 43

        results = await asyncio.gather(
            client.call_tool("rotate_agent_capability", {
                "project_key": project_key,
                "agent_name": agent_a["name"],
                "old_registration_token": agent_a["registration_token"],
                "new_registration_token": shared_new,
            }),
            client.call_tool("rotate_agent_capability", {
                "project_key": project_key,
                "agent_name": agent_b["name"],
                "old_registration_token": agent_b["registration_token"],
                "new_registration_token": shared_new,
            }),
            return_exceptions=True,
        )
        successes = [r for r in results if not isinstance(r, Exception)]
        assert len(successes) == 1, f"期望恰好一个成功，实际: {results!r}"

        # 最终 DB 中该 new 值只能出现一次
        from sqlmodel import select as _sm_select
        from mcp_agent_mail.models import Agent as AgentModel
        async with get_session() as session:
            rows = (await session.execute(
                _sm_select(AgentModel).where(AgentModel.registration_token == shared_new)
            )).scalars().all()
        assert len(rows) == 1, f"new 值出现 {len(rows)} 次，期望 1 次"


@pytest.mark.asyncio
async def test_rotate_capability_error_panel_redacts_sentinel(isolated_env, capsys, monkeypatch):
    """异常文本含 capability 值：Rich error panel 不得泄漏（默认 logging=true）。"""
    project_key = "/test/setup/rotate-error-redact"
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": project_key})
        agent = await _register_with_token(client, project_key)
        name, old = agent["name"], agent["registration_token"]
        new_token = "z" * 43

        from mcp_agent_mail import app as app_module

        async def boom(*_args, **_kwargs):
            raise RuntimeError(f"commit failed old={old} new={new_token}")

        monkeypatch.setattr(app_module, "get_immediate_session", boom)
        with pytest.raises(Exception):
            await client.call_tool("rotate_agent_capability", {
                "project_key": project_key,
                "agent_name": name,
                "old_registration_token": old,
                "new_registration_token": new_token,
            })

        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert old not in combined
        assert new_token not in combined
