"""Upgrade-path invariant for the M3a human identity foundation."""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest
from sqlalchemy import inspect

from mcp_agent_mail import db
from mcp_agent_mail.config import get_settings


@pytest.mark.anyio
async def test_legacy_agents_table_gains_nullable_owner_without_data_loss(isolated_env):
    """Exercise the real ALTER path from a pre-M3a agents table."""
    settings = get_settings()
    db_path = db.get_database_path(settings)
    assert db_path is not None
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE agents (
                id INTEGER PRIMARY KEY,
                project_id INTEGER NOT NULL,
                name VARCHAR(128) NOT NULL,
                program VARCHAR(128) NOT NULL,
                model VARCHAR(128) NOT NULL,
                task_description VARCHAR(2048) NOT NULL DEFAULT '',
                inception_ts DATETIME NOT NULL,
                last_active_ts DATETIME NOT NULL,
                attachments_policy VARCHAR(16) NOT NULL DEFAULT 'auto',
                contact_policy VARCHAR(16) NOT NULL DEFAULT 'auto'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO agents (
                id, project_id, name, program, model, inception_ts, last_active_ts
            ) VALUES (1, 1, 'LegacyAgent', 'legacy', 'legacy', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """
        )
        connection.commit()

    db.init_engine(settings)
    await db.ensure_schema(settings)
    engine = db.get_engine()

    def _legacy_state(sync_connection: Any) -> tuple[set[str], tuple[str, Any]]:
        columns = {
            str(column["name"])
            for column in inspect(sync_connection).get_columns("agents")
        }
        row = sync_connection.exec_driver_sql(
            "SELECT name, owner_id FROM agents WHERE id = 1"
        ).one()
        return columns, (str(row[0]), row[1])

    async with engine.begin() as connection:
        columns, row = await connection.run_sync(_legacy_state)
    assert "owner_id" in columns
    assert row == ("LegacyAgent", None)

    await db.ensure_schema(settings)
    async with engine.begin() as connection:
        repeated_columns, repeated_row = await connection.run_sync(_legacy_state)
    assert repeated_columns == columns
    assert repeated_row == row
