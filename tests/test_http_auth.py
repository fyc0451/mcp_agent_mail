"""HTTP Authentication Tests.

Comprehensive tests for HTTP authentication mechanisms:
1. Bearer token authentication
2. JWT authentication with HMAC secret
3. JWT authentication with JWKS URL
4. RBAC role enforcement
5. Localhost bypass behavior
6. OAuth metadata endpoints

Reference: mcp_agent_mail-w51
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from typing import Any

import pytest
from authlib.jose import jwt
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select

from mcp_agent_mail import config as _config
from mcp_agent_mail.app import build_mcp_server
from mcp_agent_mail.db import get_session
from mcp_agent_mail.http import build_http_app
from mcp_agent_mail.models import Agent, Project, TeamProject


def _rpc(method: str, params: dict[str, Any]) -> dict[str, Any]:
    """Create a JSON-RPC 2.0 request payload."""
    return {"jsonrpc": "2.0", "id": "1", "method": method, "params": params}


def _make_fake_jwt(claims: dict[str, Any], alg: str = "HS256") -> str:
    """Create a fake JWT for testing (not cryptographically valid)."""
    header = {"alg": alg, "typ": "JWT"}
    header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
    payload_b64 = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    sig_b64 = base64.urlsafe_b64encode(b"fake_signature").decode().rstrip("=")
    return f"{header_b64}.{payload_b64}.{sig_b64}"


def _configure_hub_jwt(monkeypatch):
    monkeypatch.setenv("HTTP_JWT_ENABLED", "true")
    monkeypatch.setenv("HTTP_JWT_ALGORITHMS", "HS256")
    monkeypatch.setenv("HTTP_JWT_SECRET", "hub-test-secret")
    monkeypatch.setenv("HTTP_RBAC_ENABLED", "true")
    monkeypatch.setenv("HTTP_RBAC_WRITER_ROLES", "writer")
    monkeypatch.setenv("HTTP_ALLOW_LOCALHOST_UNAUTHENTICATED", "false")
    _config.clear_settings_cache()
    return _config.get_settings()


def _hub_headers(settings, subject: str | None, **claims: Any) -> dict[str, str]:
    payload = {settings.http.jwt_role_claim: "writer", **claims}
    if subject is not None:
        payload["sub"] = subject
    token = jwt.encode(
        {"alg": "HS256"},
        payload,
        settings.http.jwt_secret,
    ).decode("utf-8")
    return {"Authorization": f"Bearer {token}"}


# =============================================================================
# Test: Bearer Token Authentication
# =============================================================================


class TestBearerTokenAuth:
    """Test simple bearer token authentication."""

    @pytest.mark.asyncio
    async def test_unauthorized_without_token(self, isolated_env, monkeypatch):
        """Request without bearer token returns 401."""
        monkeypatch.setenv("HTTP_BEARER_TOKEN", "secret-token")
        monkeypatch.setenv("HTTP_ALLOW_LOCALHOST_UNAUTHENTICATED", "false")
        with contextlib.suppress(Exception):
            _config.clear_settings_cache()

        settings = _config.get_settings()
        server = build_mcp_server()
        app = build_http_app(settings, server)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                settings.http.path,
                json=_rpc("tools/call", {"name": "health_check", "arguments": {}}),
            )
            assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_authorized_with_correct_token(self, isolated_env, monkeypatch):
        """Request with correct bearer token succeeds."""
        monkeypatch.setenv("HTTP_BEARER_TOKEN", "my-secret-token")
        monkeypatch.setenv("HTTP_ALLOW_LOCALHOST_UNAUTHENTICATED", "false")
        with contextlib.suppress(Exception):
            _config.clear_settings_cache()

        settings = _config.get_settings()
        server = build_mcp_server()
        app = build_http_app(settings, server)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                settings.http.path,
                headers={"Authorization": "Bearer my-secret-token"},
                json=_rpc("tools/call", {"name": "health_check", "arguments": {}}),
            )
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_unauthorized_with_wrong_token(self, isolated_env, monkeypatch):
        """Request with incorrect bearer token returns 401."""
        monkeypatch.setenv("HTTP_BEARER_TOKEN", "correct-token")
        monkeypatch.setenv("HTTP_ALLOW_LOCALHOST_UNAUTHENTICATED", "false")
        with contextlib.suppress(Exception):
            _config.clear_settings_cache()

        settings = _config.get_settings()
        server = build_mcp_server()
        app = build_http_app(settings, server)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                settings.http.path,
                headers={"Authorization": "Bearer wrong-token"},
                json=_rpc("tools/call", {"name": "health_check", "arguments": {}}),
            )
            assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_unauthorized_with_malformed_auth_header(self, isolated_env, monkeypatch):
        """Request with malformed Authorization header returns 401."""
        monkeypatch.setenv("HTTP_BEARER_TOKEN", "secret-token")
        monkeypatch.setenv("HTTP_ALLOW_LOCALHOST_UNAUTHENTICATED", "false")
        with contextlib.suppress(Exception):
            _config.clear_settings_cache()

        settings = _config.get_settings()
        server = build_mcp_server()
        app = build_http_app(settings, server)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Missing "Bearer " prefix
            response = await client.post(
                settings.http.path,
                headers={"Authorization": "secret-token"},
                json=_rpc("tools/call", {"name": "health_check", "arguments": {}}),
            )
            assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_no_auth_required_without_bearer_token_config(self, isolated_env, monkeypatch):
        """Without bearer token configured, requests are allowed."""
        monkeypatch.setenv("HTTP_BEARER_TOKEN", "")
        monkeypatch.setenv("HTTP_ALLOW_LOCALHOST_UNAUTHENTICATED", "true")
        with contextlib.suppress(Exception):
            _config.clear_settings_cache()

        settings = _config.get_settings()
        server = build_mcp_server()
        app = build_http_app(settings, server)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                settings.http.path,
                json=_rpc("tools/call", {"name": "health_check", "arguments": {}}),
            )
            assert response.status_code == 200


# =============================================================================
# Test: Localhost Bypass
# =============================================================================


class TestLocalhostBypass:
    """Test localhost authentication bypass behavior."""

    @pytest.mark.asyncio
    async def test_localhost_bypass_enabled(self, isolated_env, monkeypatch):
        """With localhost bypass enabled, no auth required for localhost."""
        monkeypatch.setenv("HTTP_BEARER_TOKEN", "secret-token")
        monkeypatch.setenv("HTTP_ALLOW_LOCALHOST_UNAUTHENTICATED", "true")
        with contextlib.suppress(Exception):
            _config.clear_settings_cache()

        settings = _config.get_settings()
        server = build_mcp_server()
        app = build_http_app(settings, server)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Should succeed without Authorization header (localhost)
            response = await client.post(
                settings.http.path,
                json=_rpc("tools/call", {"name": "health_check", "arguments": {}}),
            )
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_localhost_bypass_disabled(self, isolated_env, monkeypatch):
        """With localhost bypass disabled, auth required even for localhost."""
        monkeypatch.setenv("HTTP_BEARER_TOKEN", "secret-token")
        monkeypatch.setenv("HTTP_ALLOW_LOCALHOST_UNAUTHENTICATED", "false")
        with contextlib.suppress(Exception):
            _config.clear_settings_cache()

        settings = _config.get_settings()
        server = build_mcp_server()
        app = build_http_app(settings, server)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Should fail without Authorization header
            response = await client.post(
                settings.http.path,
                json=_rpc("tools/call", {"name": "health_check", "arguments": {}}),
            )
            assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_ipv4_mapped_ipv6_localhost_bypass_allows_write_tools(
        self, isolated_env, monkeypatch, tmp_path
    ):
        """IPv4-mapped localhost addresses should get the same write bypass as 127.0.0.1."""
        monkeypatch.setenv("HTTP_BEARER_TOKEN", "secret-token")
        monkeypatch.setenv("HTTP_ALLOW_LOCALHOST_UNAUTHENTICATED", "true")
        with contextlib.suppress(Exception):
            _config.clear_settings_cache()

        settings = _config.get_settings()
        server = build_mcp_server()
        app = build_http_app(settings, server)

        transport = ASGITransport(app=app, client=("::ffff:127.0.0.1", 12345))
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                settings.http.path,
                json=_rpc(
                    "tools/call",
                    {"name": "ensure_project", "arguments": {"human_key": str(tmp_path / "localhost-project")}},
                ),
            )
            assert response.status_code == 200


# =============================================================================
# Test: CORS Preflight Bypass
# =============================================================================


class TestCORSPreflightBypass:
    """Test that CORS preflight requests bypass authentication."""

    @pytest.mark.asyncio
    async def test_options_request_bypasses_auth(self, isolated_env, monkeypatch):
        """OPTIONS request should not require authentication."""
        monkeypatch.setenv("HTTP_BEARER_TOKEN", "secret-token")
        monkeypatch.setenv("HTTP_ALLOW_LOCALHOST_UNAUTHENTICATED", "false")
        monkeypatch.setenv("HTTP_CORS_ENABLED", "true")
        monkeypatch.setenv("HTTP_CORS_ORIGINS", "*")
        with contextlib.suppress(Exception):
            _config.clear_settings_cache()

        settings = _config.get_settings()
        server = build_mcp_server()
        app = build_http_app(settings, server)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.options(
                settings.http.path,
                headers={
                    "Origin": "http://example.com",
                    "Access-Control-Request-Method": "POST",
                },
            )
            assert response.status_code in (200, 204)


# =============================================================================
# Test: Health Endpoint Bypass
# =============================================================================


class TestHealthEndpointBypass:
    """Test that health endpoints bypass authentication."""

    @pytest.mark.asyncio
    async def test_liveness_bypasses_auth(self, isolated_env, monkeypatch):
        """Liveness endpoint does not require authentication."""
        monkeypatch.setenv("HTTP_BEARER_TOKEN", "secret-token")
        monkeypatch.setenv("HTTP_ALLOW_LOCALHOST_UNAUTHENTICATED", "false")
        with contextlib.suppress(Exception):
            _config.clear_settings_cache()

        settings = _config.get_settings()
        server = build_mcp_server()
        app = build_http_app(settings, server)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health/liveness")
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_readiness_bypasses_auth(self, isolated_env, monkeypatch):
        """Readiness endpoint does not require authentication."""
        monkeypatch.setenv("HTTP_BEARER_TOKEN", "secret-token")
        monkeypatch.setenv("HTTP_ALLOW_LOCALHOST_UNAUTHENTICATED", "false")
        with contextlib.suppress(Exception):
            _config.clear_settings_cache()

        settings = _config.get_settings()
        server = build_mcp_server()
        app = build_http_app(settings, server)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health/readiness")
            # May be 200 or 503 depending on DB state, but not 401
            assert response.status_code != 401


# =============================================================================
# Test: RBAC Role Enforcement
# =============================================================================


class TestRBACEnforcement:
    """Test RBAC (Role-Based Access Control) enforcement."""

    @pytest.mark.asyncio
    async def test_reader_role_can_read_resources(self, isolated_env, monkeypatch):
        """Reader role can access resources."""
        monkeypatch.setenv("HTTP_BEARER_TOKEN", "reader-token")
        monkeypatch.setenv("HTTP_RBAC_ENABLED", "true")
        monkeypatch.setenv("HTTP_RBAC_READER_ROLES", "reader")
        monkeypatch.setenv("HTTP_RBAC_WRITER_ROLES", "writer")
        monkeypatch.setenv("HTTP_RBAC_DEFAULT_ROLE", "reader")
        monkeypatch.setenv("HTTP_ALLOW_LOCALHOST_UNAUTHENTICATED", "false")
        with contextlib.suppress(Exception):
            _config.clear_settings_cache()

        settings = _config.get_settings()
        server = build_mcp_server()
        app = build_http_app(settings, server)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                settings.http.path,
                headers={"Authorization": "Bearer reader-token"},
                json=_rpc("resources/list", {}),
            )
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_rbac_readonly_tools_accessible_to_readers(self, isolated_env, monkeypatch):
        """Read-only tools are accessible to readers."""
        monkeypatch.setenv("HTTP_BEARER_TOKEN", "test-token")
        monkeypatch.setenv("HTTP_RBAC_ENABLED", "true")
        monkeypatch.setenv("HTTP_RBAC_READER_ROLES", "reader")
        monkeypatch.setenv("HTTP_RBAC_WRITER_ROLES", "writer")
        monkeypatch.setenv("HTTP_RBAC_DEFAULT_ROLE", "reader")
        monkeypatch.setenv("HTTP_RBAC_READONLY_TOOLS", "health_check,fetch_inbox")
        monkeypatch.setenv("HTTP_ALLOW_LOCALHOST_UNAUTHENTICATED", "false")
        with contextlib.suppress(Exception):
            _config.clear_settings_cache()

        settings = _config.get_settings()
        server = build_mcp_server()
        app = build_http_app(settings, server)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                settings.http.path,
                headers={"Authorization": "Bearer test-token"},
                json=_rpc("tools/call", {"name": "health_check", "arguments": {}}),
            )
            # health_check is read-only, should be allowed
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_reader_blocked_from_write_tools(self, isolated_env, monkeypatch):
        """Reader role is blocked from write tools."""
        monkeypatch.setenv("HTTP_BEARER_TOKEN", "test-token")
        monkeypatch.setenv("HTTP_RBAC_ENABLED", "true")
        monkeypatch.setenv("HTTP_RBAC_READER_ROLES", "reader")
        monkeypatch.setenv("HTTP_RBAC_WRITER_ROLES", "writer")
        monkeypatch.setenv("HTTP_RBAC_DEFAULT_ROLE", "reader")
        monkeypatch.setenv("HTTP_RBAC_READONLY_TOOLS", "health_check")
        monkeypatch.setenv("HTTP_ALLOW_LOCALHOST_UNAUTHENTICATED", "false")
        with contextlib.suppress(Exception):
            _config.clear_settings_cache()

        settings = _config.get_settings()
        server = build_mcp_server()
        app = build_http_app(settings, server)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                settings.http.path,
                headers={"Authorization": "Bearer test-token"},
                json=_rpc("tools/call", {"name": "ensure_project", "arguments": {"human_key": "/test"}}),
            )
            # ensure_project is NOT read-only, should be blocked for readers
            assert response.status_code == 403


# =============================================================================
# Test: OAuth Metadata Endpoints
# =============================================================================


class TestOAuthMetadataEndpoints:
    """Test OAuth metadata endpoint responses."""

    @pytest.mark.asyncio
    async def test_oauth_metadata_root_returns_404(self, isolated_env):
        """OAuth metadata endpoint returns 404 so clients skip OAuth discovery."""
        settings = _config.get_settings()
        server = build_mcp_server()
        app = build_http_app(settings, server)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/.well-known/oauth-authorization-server")
            assert response.status_code == 404
            data = response.json()
            assert data.get("mcp_oauth") is False

    @pytest.mark.asyncio
    async def test_oauth_metadata_mcp_returns_404(self, isolated_env):
        """OAuth metadata MCP endpoint returns 404 so clients skip OAuth discovery."""
        settings = _config.get_settings()
        server = build_mcp_server()
        app = build_http_app(settings, server)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/.well-known/oauth-authorization-server/mcp")
            assert response.status_code == 404
            data = response.json()
            assert data.get("mcp_oauth") is False

    @pytest.mark.asyncio
    async def test_oauth_metadata_prefixed_mount_returns_404(self, isolated_env):
        """Mounted transport paths should not expose accidental OAuth metadata."""
        settings = _config.get_settings()
        server = build_mcp_server()
        app = build_http_app(settings, server)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/.well-known/oauth-authorization-server")
            assert response.status_code == 404
            data = response.json()
            assert data.get("mcp_oauth") is False

    @pytest.mark.asyncio
    async def test_oauth_metadata_suffix_probe_returns_404(self, isolated_env):
        """Codex-style suffix probes should also terminate OAuth discovery cleanly."""
        settings = _config.get_settings()
        server = build_mcp_server()
        app = build_http_app(settings, server)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/.well-known/oauth-authorization-server/api")
            assert response.status_code == 404
            data = response.json()
            assert data.get("mcp_oauth") is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "path",
        [
            "/api/.well-known/oauth-authorization-server/",
            "/api/.well-known/oauth-authorization-server/mcp/",
            "/mcp/.well-known/oauth-authorization-server/",
            "/mcp/.well-known/oauth-authorization-server/mcp/",
        ],
    )
    async def test_oauth_metadata_prefixed_trailing_slash_paths_return_404(
        self, isolated_env, path: str
    ):
        """Mounted trailing-slash probe paths should not fall through into the MCP transport."""
        settings = _config.get_settings()
        server = build_mcp_server()
        app = build_http_app(settings, server)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await asyncio.wait_for(client.get(path), timeout=1.0)
            assert response.status_code == 404
            data = response.json()
            assert data.get("mcp_oauth") is False


# =============================================================================
# Test: JWT Helper Functions
# =============================================================================


class TestJWTHelpers:
    """Test JWT-related helper functions."""

    def test_decode_jwt_header_segment_valid(self):
        """_decode_jwt_header_segment correctly decodes valid JWT header."""
        from mcp_agent_mail.http import _decode_jwt_header_segment

        # Create a valid JWT header segment
        header = {"alg": "HS256", "typ": "JWT"}
        header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
        token = f"{header_b64}.payload.signature"

        result = _decode_jwt_header_segment(token)
        assert result is not None
        assert result.get("alg") == "HS256"
        assert result.get("typ") == "JWT"

    def test_decode_jwt_header_segment_invalid(self):
        """_decode_jwt_header_segment returns None for invalid JWT."""
        from mcp_agent_mail.http import _decode_jwt_header_segment

        result = _decode_jwt_header_segment("not-a-jwt")
        assert result is None

    def test_decode_jwt_header_segment_empty(self):
        """_decode_jwt_header_segment returns None for empty string."""
        from mcp_agent_mail.http import _decode_jwt_header_segment

        result = _decode_jwt_header_segment("")
        assert result is None


# =============================================================================
# Test: Authorization Header Handling
# =============================================================================


class TestAuthorizationHeaderHandling:
    """Test various Authorization header formats."""

    @pytest.mark.asyncio
    async def test_bearer_prefix_required(self, isolated_env, monkeypatch):
        """Authorization header must start with 'Bearer '."""
        monkeypatch.setenv("HTTP_BEARER_TOKEN", "token123")
        monkeypatch.setenv("HTTP_ALLOW_LOCALHOST_UNAUTHENTICATED", "false")
        with contextlib.suppress(Exception):
            _config.clear_settings_cache()

        settings = _config.get_settings()
        server = build_mcp_server()
        app = build_http_app(settings, server)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Basic auth format (should fail)
            response = await client.post(
                settings.http.path,
                headers={"Authorization": "Basic dXNlcjpwYXNz"},
                json=_rpc("tools/call", {"name": "health_check", "arguments": {}}),
            )
            assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_bearer_case_sensitive(self, isolated_env, monkeypatch):
        """Bearer token comparison is exact."""
        monkeypatch.setenv("HTTP_BEARER_TOKEN", "MyToken123")
        monkeypatch.setenv("HTTP_ALLOW_LOCALHOST_UNAUTHENTICATED", "false")
        with contextlib.suppress(Exception):
            _config.clear_settings_cache()

        settings = _config.get_settings()
        server = build_mcp_server()
        app = build_http_app(settings, server)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Different case should fail
            response = await client.post(
                settings.http.path,
                headers={"Authorization": "Bearer mytoken123"},
                json=_rpc("tools/call", {"name": "health_check", "arguments": {}}),
            )
            assert response.status_code == 401

            # Correct case should succeed
            response = await client.post(
                settings.http.path,
                headers={"Authorization": "Bearer MyToken123"},
                json=_rpc("tools/call", {"name": "health_check", "arguments": {}}),
            )
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_empty_authorization_header(self, isolated_env, monkeypatch):
        """Empty Authorization header returns 401."""
        monkeypatch.setenv("HTTP_BEARER_TOKEN", "token123")
        monkeypatch.setenv("HTTP_ALLOW_LOCALHOST_UNAUTHENTICATED", "false")
        with contextlib.suppress(Exception):
            _config.clear_settings_cache()

        settings = _config.get_settings()
        server = build_mcp_server()
        app = build_http_app(settings, server)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                settings.http.path,
                headers={"Authorization": ""},
                json=_rpc("tools/call", {"name": "health_check", "arguments": {}}),
            )
            assert response.status_code == 401


# =============================================================================
# Test: Rate Limiting Integration
# =============================================================================


class TestRateLimitingIntegration:
    """Test rate limiting with authentication."""

    @pytest.mark.asyncio
    async def test_rate_limit_applies_after_auth(self, isolated_env, monkeypatch):
        """Rate limiting is enforced after successful authentication."""
        monkeypatch.setenv("HTTP_BEARER_TOKEN", "test-token")
        monkeypatch.setenv("HTTP_ALLOW_LOCALHOST_UNAUTHENTICATED", "false")
        monkeypatch.setenv("HTTP_RATE_LIMIT_ENABLED", "true")
        monkeypatch.setenv("HTTP_RATE_LIMIT_TOOLS_PER_MINUTE", "2")
        monkeypatch.setenv("HTTP_RATE_LIMIT_TOOLS_BURST", "2")
        with contextlib.suppress(Exception):
            _config.clear_settings_cache()

        settings = _config.get_settings()
        server = build_mcp_server()
        app = build_http_app(settings, server)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            headers = {"Authorization": "Bearer test-token"}

            # First two requests succeed (burst=2)
            r1 = await client.post(
                settings.http.path,
                headers=headers,
                json=_rpc("tools/call", {"name": "health_check", "arguments": {}}),
            )
            assert r1.status_code == 200

            r2 = await client.post(
                settings.http.path,
                headers=headers,
                json=_rpc("tools/call", {"name": "health_check", "arguments": {}}),
            )
            assert r2.status_code == 200

            # Third request hits rate limit
            r3 = await client.post(
                settings.http.path,
                headers=headers,
                json=_rpc("tools/call", {"name": "health_check", "arguments": {}}),
            )
            assert r3.status_code == 429

    @pytest.mark.asyncio
    async def test_rate_limit_returns_429_response(self, isolated_env, monkeypatch):
        """Rate limit exceeded returns proper 429 response."""
        monkeypatch.setenv("HTTP_BEARER_TOKEN", "test-token")
        monkeypatch.setenv("HTTP_ALLOW_LOCALHOST_UNAUTHENTICATED", "false")
        monkeypatch.setenv("HTTP_RATE_LIMIT_ENABLED", "true")
        monkeypatch.setenv("HTTP_RATE_LIMIT_TOOLS_PER_MINUTE", "1")
        monkeypatch.setenv("HTTP_RATE_LIMIT_TOOLS_BURST", "1")
        with contextlib.suppress(Exception):
            _config.clear_settings_cache()

        settings = _config.get_settings()
        server = build_mcp_server()
        app = build_http_app(settings, server)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            headers = {"Authorization": "Bearer test-token"}

            # First request consumes the burst
            await client.post(
                settings.http.path,
                headers=headers,
                json=_rpc("tools/call", {"name": "health_check", "arguments": {}}),
            )

            # Second request hits rate limit
            r = await client.post(
                settings.http.path,
                headers=headers,
                json=_rpc("tools/call", {"name": "health_check", "arguments": {}}),
            )
            assert r.status_code == 429
            data = r.json()
            assert data.get("detail") == "Rate limit exceeded"


# =============================================================================
# Test: M3a authenticated Human / project membership API
# =============================================================================


class TestHubHumanIdentityApi:
    """Hub mutations require a JWT Human principal and project-scoped rights."""

    @pytest.mark.asyncio
    async def test_human_mapping_requires_jwt_subject(self, isolated_env, monkeypatch):
        monkeypatch.setenv("HTTP_BEARER_TOKEN", "legacy-token")
        settings = _configure_hub_jwt(monkeypatch)
        app = build_http_app(settings, build_mcp_server())

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            missing_sub = await client.put(
                "/hub/api/humans/me",
                headers=_hub_headers(settings, None),
                json={"display_name": "No Subject"},
            )
            assert missing_sub.status_code == 401

            # A static bearer may call legacy MCP endpoints, but it must never
            # be promoted into a Human principal.
            static_bearer = await client.put(
                "/hub/api/humans/me",
                headers={"Authorization": "Bearer legacy-token"},
                json={"display_name": "Legacy Client"},
            )
            assert static_bearer.status_code == 401

    @pytest.mark.asyncio
    async def test_jwt_subject_maps_to_stable_human(self, isolated_env, monkeypatch):
        settings = _configure_hub_jwt(monkeypatch)
        app = build_http_app(settings, build_mcp_server())
        headers = _hub_headers(settings, "oidc|alice")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            invalid = await client.put(
                "/hub/api/humans/me",
                headers=headers,
                json={"display_name": ""},
            )
            assert invalid.status_code == 400

            created, concurrent = await asyncio.gather(
                client.put(
                    "/hub/api/humans/me",
                    headers=headers,
                    json={"display_name": "Alice"},
                ),
                client.put(
                    "/hub/api/humans/me",
                    headers=headers,
                    json={"display_name": "Alice"},
                ),
            )
            assert created.status_code == concurrent.status_code == 200
            human_id = created.json()["id"]
            assert concurrent.json()["id"] == human_id

            updated = await client.put(
                "/hub/api/humans/me",
                headers=headers,
                json={"display_name": "Alice Zhang"},
            )
            assert updated.status_code == 200
            assert updated.json() == {
                "id": human_id,
                "display_name": "Alice Zhang",
            }

            fetched = await client.get("/hub/api/humans/me", headers=headers)
            assert fetched.status_code == 200
            assert fetched.json() == updated.json()

    @pytest.mark.asyncio
    async def test_concurrent_group_creation_keeps_one_routing_project(
        self, isolated_env, monkeypatch
    ):
        settings = _configure_hub_jwt(monkeypatch)
        app = build_http_app(settings, build_mcp_server())
        headers = _hub_headers(settings, "oidc|alice", role=["writer", "admin"])

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            registered = await client.put(
                "/hub/api/humans/me",
                headers=headers,
                json={"display_name": "Alice"},
            )
            assert registered.status_code == 200

            first, second = await asyncio.gather(
                client.post(
                    "/hub/api/projects",
                    headers=headers,
                    json={"name": "M3a Team", "slug": "m3a", "mention_handle": "alice"},
                ),
                client.post(
                    "/hub/api/projects",
                    headers=headers,
                    json={"name": "Duplicate", "slug": "m3a", "mention_handle": "alice"},
                ),
            )
            assert sorted((first.status_code, second.status_code)) == [201, 409]

            async with get_session() as session:
                team_count = await session.scalar(select(func.count()).select_from(TeamProject))
                routing_count = await session.scalar(
                    select(func.count())
                    .select_from(Project)
                    .where(Project.human_key.like("team:%"))
                )
            assert team_count == routing_count == 1

    @pytest.mark.asyncio
    async def test_project_join_approval_and_agent_authorization(
        self, isolated_env, monkeypatch
    ):
        settings = _configure_hub_jwt(monkeypatch)
        app = build_http_app(settings, build_mcp_server())
        alice_headers = _hub_headers(settings, "oidc|alice", role=["writer", "admin"])
        bob_headers = _hub_headers(settings, "oidc|bob")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            alice = await client.put(
                "/hub/api/humans/me",
                headers=alice_headers,
                json={"display_name": "Alice"},
            )
            bob = await client.put(
                "/hub/api/humans/me",
                headers=bob_headers,
                json={"display_name": "Bob"},
            )
            assert alice.status_code == bob.status_code == 200

            # Existing Agent Mail projects are technical routing records, not
            # user-visible Team groups. A fresh Team account starts empty.
            async with get_session() as session:
                session.add(Project(slug="local-worktree", human_key="/home/alice/worktree"))
                await session.commit()
            initially_empty = await client.get("/hub/api/projects", headers=alice_headers)
            assert initially_empty.status_code == 200
            assert initially_empty.json() == {"projects": []}
            path_coupled_create = await client.post(
                "/hub/api/projects",
                headers=alice_headers,
                json={"human_key": "/home/alice/worktree", "mention_handle": "alice"},
            )
            assert path_coupled_create.status_code == 400

            created = await client.post(
                "/hub/api/projects",
                headers=alice_headers,
                json={"name": "M3a Team", "slug": "m3a", "mention_handle": "alice"},
            )
            assert created.status_code == 201
            project = created.json()
            slug = project["slug"]
            routing_project_id = project["membership"]["project_id"]
            assert project["name"] == "M3a Team"
            assert "human_key" not in project
            assert project["membership"]["role"] == "admin"
            assert project["membership"]["status"] == "active"

            discover = await client.get("/hub/api/projects", headers=bob_headers)
            assert discover.status_code == 200
            discovered_project = discover.json()["projects"][0]
            assert discovered_project["slug"] == slug
            assert discovered_project["membership"] is None
            assert "human_key" not in discovered_project

            non_admin_create = await client.post(
                "/hub/api/projects",
                headers=bob_headers,
                json={"name": "Bob Team", "slug": "bob-team", "mention_handle": "bob"},
            )
            assert non_admin_create.status_code == 403

            duplicate = await client.post(
                "/hub/api/projects",
                headers=alice_headers,
                json={"name": "Duplicate", "slug": "m3a", "mention_handle": "alice"},
            )
            assert duplicate.status_code == 409

            join = await client.post(
                f"/hub/api/projects/{slug}/join-requests",
                headers=bob_headers,
                json={"mention_handle": "bob"},
            )
            assert join.status_code == 201
            assert join.json()["status"] == "invited"

            repeated_join = await client.post(
                f"/hub/api/projects/{slug}/join-requests",
                headers=bob_headers,
                json={"mention_handle": "bob"},
            )
            assert repeated_join.status_code == 200
            assert repeated_join.json()["id"] == join.json()["id"]

            pending_cannot_read_agents = await client.get(
                f"/hub/api/projects/{slug}/agents",
                headers=bob_headers,
            )
            assert pending_cannot_read_agents.status_code == 403

            cannot_self_promote = await client.patch(
                f"/hub/api/projects/{slug}/membership",
                headers=bob_headers,
                json={"role": "admin"},
            )
            assert cannot_self_promote.status_code == 400

            members = await client.get(
                f"/hub/api/projects/{slug}/members",
                headers=alice_headers,
            )
            assert members.status_code == 200
            bob_membership = next(
                item for item in members.json()["members"]
                if item["human_id"] == bob.json()["id"]
            )
            assert bob_membership["status"] == "invited"

            admin_cannot_set_others_default = await client.patch(
                f"/hub/api/projects/{slug}/members/{bob.json()['id']}",
                headers=alice_headers,
                json={"default_agent_id": None},
            )
            assert admin_cannot_set_others_default.status_code == 400

            approved = await client.patch(
                f"/hub/api/projects/{slug}/members/{bob.json()['id']}",
                headers=alice_headers,
                json={"status": "active"},
            )
            assert approved.status_code == 200
            assert approved.json()["status"] == "active"

            async with get_session() as session:
                agent = Agent(
                    project_id=routing_project_id,
                    name="GreenLake",
                    program="test",
                    model="test",
                )
                session.add(agent)
                await session.commit()
                await session.refresh(agent)
                assert agent.id is not None
                agent_id = agent.id

            # Pre-M3a unowned agents remain visible, but an ordinary member
            # cannot claim one by assigning owner_id to themselves.
            agents = await client.get(
                f"/hub/api/projects/{slug}/agents",
                headers=bob_headers,
            )
            assert agents.status_code == 200
            assert agents.json()["agents"][0]["owner_id"] is None

            legacy_bypass = await client.post(
                "/mail/api/retire-agent",
                headers=bob_headers,
                json={"agent_id": agent_id},
            )
            assert legacy_bypass.status_code == 403

            claim = await client.patch(
                f"/hub/api/projects/{slug}/agents/{agent_id}",
                headers=bob_headers,
                json={"owner_id": bob.json()["id"]},
            )
            assert claim.status_code == 403

            assigned = await client.patch(
                f"/hub/api/projects/{slug}/agents/{agent_id}",
                headers=alice_headers,
                json={"owner_id": bob.json()["id"]},
            )
            assert assigned.status_code == 200
            assert assigned.json()["owner_id"] == bob.json()["id"]

            retired = await client.patch(
                f"/hub/api/projects/{slug}/agents/{agent_id}",
                headers=bob_headers,
                json={"retired": True},
            )
            assert retired.status_code == 200
            assert retired.json()["retired"] is True

            retired_default = await client.patch(
                f"/hub/api/projects/{slug}/membership",
                headers=bob_headers,
                json={"default_agent_id": agent_id},
            )
            assert retired_default.status_code == 400

            restored = await client.patch(
                f"/hub/api/projects/{slug}/agents/{agent_id}",
                headers=bob_headers,
                json={"retired": False},
            )
            assert restored.status_code == 200

            defaulted = await client.patch(
                f"/hub/api/projects/{slug}/membership",
                headers=bob_headers,
                json={"default_agent_id": agent_id},
            )
            assert defaulted.status_code == 200
            assert defaulted.json()["default_agent_id"] == agent_id

            retire_default = await client.patch(
                f"/hub/api/projects/{slug}/agents/{agent_id}",
                headers=bob_headers,
                json={"retired": True},
            )
            assert retire_default.status_code == 409

            member_archive = await client.post(
                "/mail/api/archive-project",
                headers=bob_headers,
                json={"project_id": routing_project_id},
            )
            assert member_archive.status_code == 403

            admin_archive = await client.post(
                "/mail/api/archive-project",
                headers=alice_headers,
                json={"project_id": routing_project_id},
            )
            assert admin_archive.status_code == 200

            member_unarchive = await client.post(
                "/mail/api/unarchive-project",
                headers=bob_headers,
                json={"project_id": routing_project_id},
            )
            assert member_unarchive.status_code == 403

            admin_unarchive = await client.post(
                "/mail/api/unarchive-project",
                headers=alice_headers,
                json={"project_id": routing_project_id},
            )
            assert admin_unarchive.status_code == 200

    @pytest.mark.asyncio
    async def test_member_roster_readable_by_active_members_only(
        self, isolated_env, monkeypatch
    ):
        """Directory semantics (lead ruling): ordinary active members read a
        minimal roster of ACTIVE members only (human_id, display_name,
        mention_handle, role/status) — no invited/removed rows, no opaque
        subject, no other members' default_agent_id. Admins keep the full
        view for approval and member management."""
        settings = _configure_hub_jwt(monkeypatch)
        app = build_http_app(settings, build_mcp_server())
        alice_headers = _hub_headers(settings, "oidc|alice", role=["writer", "admin"])
        bob_headers = _hub_headers(settings, "oidc|bob")
        carol_headers = _hub_headers(settings, "oidc|carol")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            for headers, name in ((alice_headers, "Alice"), (bob_headers, "Bob"), (carol_headers, "Carol")):
                r = await client.put("/hub/api/humans/me", headers=headers, json={"display_name": name})
                assert r.status_code == 200

            created = await client.post(
                "/hub/api/projects",
                headers=alice_headers,
                json={"name": "Roster", "slug": "roster", "mention_handle": "alice"},
            )
            assert created.status_code == 201
            slug = created.json()["slug"]

            join = await client.post(
                f"/hub/api/projects/{slug}/join-requests",
                headers=bob_headers,
                json={"mention_handle": "bob"},
            )
            assert join.status_code == 201

            # invited member may not read the roster
            invited = await client.get(f"/hub/api/projects/{slug}/members", headers=bob_headers)
            assert invited.status_code == 403
            # non-member may not read the roster
            outsider = await client.get(f"/hub/api/projects/{slug}/members", headers=carol_headers)
            assert outsider.status_code == 403

            bob_id = join.json()["human_id"]
            approve = await client.patch(
                f"/hub/api/projects/{slug}/members/{bob_id}",
                headers=alice_headers,
                json={"status": "active"},
            )
            assert approve.status_code == 200

            # carol requests too but stays invited: invisible to ordinary members
            carol_join = await client.post(
                f"/hub/api/projects/{slug}/join-requests",
                headers=carol_headers,
                json={"mention_handle": "carol"},
            )
            assert carol_join.status_code == 201

            # ordinary active member: minimal roster, ACTIVE rows only
            roster = await client.get(f"/hub/api/projects/{slug}/members", headers=bob_headers)
            assert roster.status_code == 200
            members = roster.json()["members"]
            assert {m["mention_handle"] for m in members} == {"alice", "bob"}
            for m in members:
                assert set(m) == {
                    "human_id", "display_name", "mention_handle", "role", "status",
                }
                assert m["status"] == "active"

            # admin: full view including invited rows and default_agent_id
            admin_roster = await client.get(f"/hub/api/projects/{slug}/members", headers=alice_headers)
            assert admin_roster.status_code == 200
            admin_members = admin_roster.json()["members"]
            assert {m["mention_handle"] for m in admin_members} == {"alice", "bob", "carol"}
            for m in admin_members:
                assert "default_agent_id" in m
                assert "id" in m and "project_id" in m
            assert next(m for m in admin_members if m["mention_handle"] == "carol")["status"] == "invited"
