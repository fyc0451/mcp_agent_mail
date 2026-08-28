from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from mcp_agent_mail.config import ConfigError, clear_settings_cache, get_settings
from mcp_agent_mail.db import (
    UnsupportedDatabaseBackendError,
    _assert_supported_backend,
    ensure_schema,
    get_engine,
    get_sqlite_pre_restore_path,
    get_sqlite_sidecar_paths,
    reset_database_state,
)
from mcp_agent_mail.utils import sanitize_agent_name, slugify


def test_slugify_and_sanitize_edges():
    assert slugify("  Hello World!!  ") == "hello-world"
    assert slugify("") == "project"
    assert sanitize_agent_name(" A!@#$ ") == "A"
    assert sanitize_agent_name("!!!") is None


def test_config_csv_and_bool_parsing(monkeypatch):
    monkeypatch.setenv("HTTP_RBAC_READER_ROLES", "reader, ro ,, read ")
    monkeypatch.setenv("HTTP_RATE_LIMIT_ENABLED", "true")
    clear_settings_cache()
    s = get_settings()
    assert {"reader", "ro", "read"}.issubset(set(s.http.rbac_reader_roles))
    assert s.http.rate_limit_enabled is True


# ============================================================================
# #169: configuration parsers must fail-closed on malformed *explicit* values
# while still falling back to defaults for empty/unset values.
# ============================================================================


def test_config_valid_explicit_values_parse(monkeypatch):
    """Valid explicit values must parse to their typed forms."""
    monkeypatch.setenv("HTTP_PORT", "9999")
    monkeypatch.setenv("HTTP_RATE_LIMIT_ENABLED", "yes")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.7")
    monkeypatch.setenv("AGENT_NAME_ENFORCEMENT_MODE", "strict")
    monkeypatch.setenv("HTTP_RATE_LIMIT_BACKEND", "redis")
    clear_settings_cache()
    s = get_settings()
    assert s.http.port == 9999
    assert s.http.rate_limit_enabled is True
    assert s.llm.temperature == pytest.approx(0.7)
    assert s.agent_name_enforcement_mode == "strict"
    assert s.http.rate_limit_backend == "redis"


def test_config_empty_value_falls_back_to_default(monkeypatch):
    """An explicitly-empty value is legitimate and must use the default, not raise."""
    monkeypatch.setenv("HTTP_PORT", "")
    monkeypatch.setenv("HTTP_RATE_LIMIT_ENABLED", "")
    monkeypatch.setenv("DATABASE_POOL_SIZE", "")
    monkeypatch.setenv("AGENT_NAME_ENFORCEMENT_MODE", "")
    clear_settings_cache()
    s = get_settings()
    assert s.http.port == 8765
    assert s.http.rate_limit_enabled is False
    assert s.database.pool_size == 50  # empty -> the configured default (50)
    assert s.agent_name_enforcement_mode == "coerce"  # default


def test_config_malformed_int_raises_with_key(monkeypatch):
    monkeypatch.setenv("HTTP_PORT", "not-a-number")
    clear_settings_cache()
    with pytest.raises(ConfigError) as exc:
        get_settings()
    assert "HTTP_PORT" in str(exc.value)


def test_config_malformed_bool_raises_with_key(monkeypatch):
    monkeypatch.setenv("HTTP_RATE_LIMIT_ENABLED", "maybe")
    clear_settings_cache()
    with pytest.raises(ConfigError) as exc:
        get_settings()
    assert "HTTP_RATE_LIMIT_ENABLED" in str(exc.value)


def test_config_malformed_float_raises_with_key(monkeypatch):
    monkeypatch.setenv("LLM_TEMPERATURE", "hot")
    clear_settings_cache()
    with pytest.raises(ConfigError) as exc:
        get_settings()
    assert "LLM_TEMPERATURE" in str(exc.value)


def test_config_unknown_enum_raises_with_key_and_allowed(monkeypatch):
    monkeypatch.setenv("AGENT_NAME_ENFORCEMENT_MODE", "bogus")
    clear_settings_cache()
    with pytest.raises(ConfigError) as exc:
        get_settings()
    msg = str(exc.value)
    assert "AGENT_NAME_ENFORCEMENT_MODE" in msg
    assert "strict" in msg  # surfaces the allowed values


def test_config_unknown_rate_limit_backend_raises(monkeypatch):
    monkeypatch.setenv("HTTP_RATE_LIMIT_BACKEND", "cassandra")
    clear_settings_cache()
    with pytest.raises(ConfigError) as exc:
        get_settings()
    assert "HTTP_RATE_LIMIT_BACKEND" in str(exc.value)


def test_public_http_mode_rejects_insecure_defaults(isolated_env, monkeypatch):
    insecure = {
        "HTTP_PUBLIC_MODE": "true",
        "HTTP_ALLOW_LOCALHOST_UNAUTHENTICATED": "true",
        "HTTP_JWT_ENABLED": "false",
        "HTTP_RATE_LIMIT_ENABLED": "false",
        "HTTP_CORS_ENABLED": "true",
    }
    for key, value in insecure.items():
        monkeypatch.setenv(key, value)
    clear_settings_cache()
    with pytest.raises(ConfigError) as exc:
        get_settings()
    message = str(exc.value)
    assert "HTTP_ALLOW_LOCALHOST_UNAUTHENTICATED" in message
    assert "HTTP_JWT_ENABLED" in message
    assert "HTTP_RATE_LIMIT_ENABLED" in message


def test_public_http_mode_accepts_fail_closed_profile(isolated_env, monkeypatch):
    profile = {
        "HTTP_PUBLIC_MODE": "true",
        "HTTP_HOST": "127.0.0.1",
        "HTTP_BEARER_TOKEN": "",
        "HTTP_ALLOW_LOCALHOST_UNAUTHENTICATED": "false",
        "HTTP_JWT_ENABLED": "true",
        "HTTP_JWT_ALGORITHMS": "RS256",
        "HTTP_JWT_JWKS_URL": "https://auth.example.com/.well-known/jwks.json",
        "HTTP_JWT_ISSUER": "https://auth.example.com",
        "HTTP_JWT_AUDIENCE": "mcp-agent-mail-human",
        "HTTP_JWT_INTROSPECTION_URL": "https://auth.example.com/introspect",
        "HTTP_RATE_LIMIT_ENABLED": "true",
        "HTTP_RATE_LIMIT_BACKEND": "redis",
        "HTTP_RATE_LIMIT_REDIS_URL": "redis://127.0.0.1:6379/2",
        "HTTP_CORS_ENABLED": "false",
    }
    for key, value in profile.items():
        monkeypatch.setenv(key, value)
    clear_settings_cache()
    settings = get_settings()
    assert settings.http.public_mode is True
    assert settings.http.jwt_introspection_url == profile[
        "HTTP_JWT_INTROSPECTION_URL"
    ]


def test_config_malformed_optional_int_raises_with_key(monkeypatch):
    monkeypatch.setenv("DATABASE_POOL_SIZE", "lots")
    clear_settings_cache()
    with pytest.raises(ConfigError) as exc:
        get_settings()
    assert "DATABASE_POOL_SIZE" in str(exc.value)


def test_database_pool_size_default_is_50(monkeypatch):
    monkeypatch.delenv("DATABASE_POOL_SIZE", raising=False)
    clear_settings_cache()
    s = get_settings()
    assert s.database.pool_size == 50


def test_get_settings_cache_clear_reloads_decouple_snapshot(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("HTTP_PORT=1111\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HTTP_PORT", raising=False)
    clear_settings_cache()
    assert get_settings().http.port == 1111

    env_path.write_text("HTTP_PORT=2222\n", encoding="utf-8")
    get_settings.cache_clear()
    assert get_settings().http.port == 2222


def test_db_engine_reset_and_reinit(isolated_env):
    # Reset and ensure engine can be re-initialized and schema ensured
    reset_database_state()
    # Access engine should lazy-init
    _ = get_engine()
    # Ensure schema executes without error
    asyncio.run(ensure_schema())


def test_assert_supported_backend_accepts_sqlite_variants():
    """SQLite URLs (all supported aiosqlite/pysqlite variants) must not raise."""
    for url in (
        "sqlite+aiosqlite:///:memory:",
        "sqlite+aiosqlite:////tmp/x.sqlite3",
        "sqlite:///:memory:",
    ):
        _assert_supported_backend(url)  # should not raise


def test_assert_supported_backend_rejects_postgres():
    """Regression for #142 — we fail fast instead of blowing up at CREATE VIRTUAL TABLE."""
    with pytest.raises(UnsupportedDatabaseBackendError) as excinfo:
        _assert_supported_backend(
            "postgresql+asyncpg://mcp:pw@example.invalid:5432/mail"
        )
    msg = str(excinfo.value)
    assert "SQLite" in msg
    assert "142" in msg  # points users at the tracking issue
    assert "postgresql" in msg.lower()


def test_assert_supported_backend_rejects_mysql():
    with pytest.raises(UnsupportedDatabaseBackendError):
        _assert_supported_backend("mysql+aiomysql://u:p@h/db")


def test_assert_supported_backend_tolerates_empty_url():
    # Empty / garbage URLs are left to surface their own errors downstream.
    _assert_supported_backend("")
    _assert_supported_backend("this is not a url")


def test_init_engine_rejects_postgres_fast(monkeypatch, tmp_path):
    """Regression for #142: init_engine must fail fast on Postgres.

    Without this check, schema init proceeds and crashes deep inside
    ``CREATE VIRTUAL TABLE ... USING fts5`` with a cryptic SQL error.
    """
    from mcp_agent_mail.db import init_engine

    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+asyncpg://mcp:pw@example.invalid:5432/mail",
    )
    clear_settings_cache()
    reset_database_state()
    with pytest.raises(UnsupportedDatabaseBackendError):
        init_engine()


def test_sqlite_sidecar_paths_preserve_database_filename():
    wal_path, shm_path = get_sqlite_sidecar_paths(Path("/tmp/mail.db"))
    assert wal_path == Path("/tmp/mail.db-wal")
    assert shm_path == Path("/tmp/mail.db-shm")
    assert get_sqlite_pre_restore_path(Path("/tmp/mail.db")) == Path("/tmp/mail.db.pre-restore")
