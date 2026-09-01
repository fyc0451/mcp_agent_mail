import concurrent.futures
import json
import sqlite3
import stat

import pytest
from authlib.jose import JsonWebKey, JsonWebToken
from fastapi.testclient import TestClient

from mcp_agent_mail.human_auth import (
    HumanAuthConfig,
    _read_secret_file,
    _validate_runtime_config,
    _validated_bind_host,
    create_app,
)


def test_bind_host_requires_explicit_private_http_opt_in():
    assert _validated_bind_host("127.0.0.1") == "127.0.0.1"
    assert _validated_bind_host("::1") == "::1"
    assert _validated_bind_host(
        "10.18.160.11", allow_private_http=True,
    ) == "10.18.160.11"

    with pytest.raises(ValueError, match="allow-private-http"):
        _validated_bind_host("10.18.160.11")
    for host in ("0.0.0.0", "8.8.8.8", "192.0.2.10", "example.com"):
        with pytest.raises(ValueError, match="loopback or RFC1918/ULA"):
            _validated_bind_host(host, allow_private_http=True)


def test_public_mode_and_admin_secret_fail_closed(tmp_path):
    base = {
        "data_dir": tmp_path / "state",
        "issuer": "https://auth.example.com",
        "audience": "mcp-agent-mail-human",
        "admin_access_token": "public-admin-factor-0123456789abcdef",
        "public_mode": True,
    }
    _validate_runtime_config(HumanAuthConfig(**base))
    with pytest.raises(ValueError, match="HTTPS issuer"):
        _validate_runtime_config(
            HumanAuthConfig(**{**base, "issuer": "http://127.0.0.1:8766"})
        )
    with pytest.raises(ValueError, match=r"32\+ character"):
        _validate_runtime_config(
            HumanAuthConfig(**{**base, "admin_access_token": "short"})
        )
    with pytest.raises(ValueError, match="forbids reusable"):
        _validate_runtime_config(
            HumanAuthConfig(**{**base, "allow_reusable_team_code": True})
        )

    secret = tmp_path / "admin-access-token"
    secret.write_text("public-admin-factor-0123456789abcdef", encoding="utf-8")
    secret.chmod(0o644)
    with pytest.raises(ValueError, match="0600"):
        _read_secret_file(secret)
    secret.chmod(0o600)
    assert _read_secret_file(secret) == "public-admin-factor-0123456789abcdef"


def _bootstrapped_client(tmp_path, **config_overrides):
    config = HumanAuthConfig(
        data_dir=tmp_path / "state",
        issuer="http://127.0.0.1:8766",
        audience="mcp-agent-mail-human",
        token_ttl_seconds=600,
        **config_overrides,
    )
    app = create_app(config)
    credentials = tmp_path / "admin.json"
    assert app.state.store.bootstrap_admin(
        username="fyc", display_name="付彦超", credentials_path=credentials
    )
    return app, TestClient(app), credentials


def test_bootstrap_login_and_jwks_verification(tmp_path):
    app, client, credentials = _bootstrapped_client(tmp_path)
    secret = json.loads(credentials.read_text())

    response = client.post("/token", json=secret)
    assert response.status_code == 200
    payload = response.json()
    assert payload["token_type"] == "Bearer"
    assert payload["expires_in"] == 600
    assert payload["profile"] == {
        "username": "fyc",
        "display_name": "付彦超",
        "roles": ["writer", "admin"],
        "status": "active",
    }

    jwks = client.get("/.well-known/jwks.json").json()
    public_key = JsonWebKey.import_key_set(jwks).find_by_kid(app.state.store.kid)
    claims = JsonWebToken(["RS256"]).decode(payload["access_token"], public_key)
    claims.validate()
    assert claims["iss"] == "http://127.0.0.1:8766"
    assert claims["aud"] == "mcp-agent-mail-human"
    assert claims["sub"] == "human:fyc"
    assert claims["name"] == "付彦超"
    assert claims["ver"] == 1
    assert set(claims["role"]) == {"writer", "admin"}


def test_private_state_permissions_and_idempotent_bootstrap(tmp_path):
    app, _, credentials = _bootstrapped_client(tmp_path)
    store = app.state.store
    assert stat.S_IMODE(store.config.data_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(store.db_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.key_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(credentials.stat().st_mode) == 0o600
    original = credentials.read_bytes()
    assert not store.bootstrap_admin(
        username="fyc", display_name="Changed", credentials_path=credentials
    )
    assert credentials.read_bytes() == original


def test_login_failures_are_generic_and_rate_limited(tmp_path):
    _, client, _ = _bootstrapped_client(tmp_path)
    for username in ("fyc", "missing"):
        response = client.post("/token", json={"username": username, "password": "wrong-password"})
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid username or password"
    for _ in range(4):
        client.post("/token", json={"username": "fyc", "password": "wrong-password"})
    response = client.post("/token", json={"username": "fyc", "password": "wrong-password"})
    assert response.status_code == 429


def test_password_change_revokes_existing_token_immediately(tmp_path):
    _, client, credentials = _bootstrapped_client(tmp_path)
    old_secret = json.loads(credentials.read_text())
    login = client.post("/token", json=old_secret)
    token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    assert client.patch(
        "/me/password", json={"new_password": "new-password-1234"}
    ).status_code == 401
    too_short = client.patch(
        "/me/password", headers=headers, json={"new_password": "too-short"}
    )
    assert too_short.status_code == 400
    changed = client.patch(
        "/me/password", headers=headers, json={"new_password": "new-password-1234"}
    )
    assert changed.status_code == 200
    changed_payload = changed.json()
    assert changed_payload["ok"] is True
    assert changed_payload["token_type"] == "Bearer"
    assert changed_payload["expires_in"] == 600
    replacement_headers = {
        "Authorization": f"Bearer {changed_payload['access_token']}"
    }

    assert client.get("/me", headers=headers).status_code == 401
    introspection = client.post("/introspect", headers=headers)
    assert introspection.status_code == 200
    assert introspection.json() == {"active": False}
    assert client.get("/me", headers=replacement_headers).status_code == 200
    assert client.post("/introspect", headers=replacement_headers).json() == {
        "active": True
    }
    assert client.post("/token", json=old_secret).status_code == 401
    new_login = client.post(
        "/token",
        json={"username": old_secret["username"], "password": "new-password-1234"},
    )
    assert new_login.status_code == 200
    assert client.post(
        "/introspect",
        headers={"Authorization": f"Bearer {new_login.json()['access_token']}"},
    ).json() == {"active": True}


def test_admin_second_factor_is_required_when_configured(tmp_path):
    _, client, credentials = _bootstrapped_client(
        tmp_path, admin_access_token="public-admin-factor-0123456789abcdef"
    )
    token = client.post("/token", json=json.loads(credentials.read_text())).json()[
        "access_token"
    ]
    headers = {"Authorization": f"Bearer {token}"}

    assert client.get("/admin/users", headers=headers).status_code == 403
    headers["X-Agent-Hub-Admin-Key"] = "wrong-factor"
    assert client.get("/admin/users", headers=headers).status_code == 403
    headers["X-Agent-Hub-Admin-Key"] = "public-admin-factor-0123456789abcdef"
    assert client.get("/admin/users", headers=headers).status_code == 200


def test_discovery_exposes_public_metadata_only(tmp_path):
    _, client, credentials = _bootstrapped_client(tmp_path)
    discovery = client.get("/.well-known/openid-configuration")
    assert discovery.status_code == 200
    assert discovery.json()["jwks_uri"] == "http://127.0.0.1:8766/.well-known/jwks.json"
    combined = discovery.text + client.get("/.well-known/jwks.json").text + client.get("/health").text
    assert json.loads(credentials.read_text())["password"] not in combined


def test_invitation_registration_approval_and_disable(tmp_path):
    app, client, credentials = _bootstrapped_client(tmp_path)
    admin_secret = json.loads(credentials.read_text())
    admin_login = client.post("/token", json=admin_secret)
    admin_token = admin_login.json()["access_token"]
    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    invitation = client.post(
        "/admin/invitations",
        headers=admin_headers,
        json={"expires_in": 3600, "project_slug": "core-team"},
    )
    assert invitation.status_code == 201
    invite_code = invitation.json()["invite_code"]
    assert invitation.json()["project_slug"] == "core-team"
    assert invite_code not in client.get("/admin/users", headers=admin_headers).text
    with app.state.store._connect() as connection:
        stored_invitation = connection.execute(
            "SELECT code_hash FROM invitations"
        ).fetchone()
    assert stored_invitation["code_hash"] != invite_code

    registration = {
        "username": "alice",
        "display_name": "Alice",
        "password": "alice-password-123",
        "invite_code": invite_code,
    }
    registered = client.post("/register", json=registration)
    assert registered.status_code == 201
    assert registered.json()["account"] == {
        "username": "alice",
        "display_name": "Alice",
        "status": "pending",
        "requested_project_slug": "core-team",
    }
    assert client.post("/register", json={**registration, "username": "bob"}).status_code == 400
    assert client.post(
        "/token",
        json={"username": "alice", "password": registration["password"]},
    ).status_code == 401

    users = client.get("/admin/users", headers=admin_headers)
    assert users.status_code == 200
    alice = next(user for user in users.json()["users"] if user["username"] == "alice")
    assert alice["status"] == "pending"
    assert alice["roles"] == ["writer"]
    assert alice["subject"] == "human:alice"
    assert alice["requested_project_slug"] == "core-team"

    approved = client.patch(
        "/admin/users/alice",
        headers=admin_headers,
        json={"status": "active"},
    )
    assert approved.status_code == 200
    login = client.post(
        "/token",
        json={"username": "alice", "password": registration["password"]},
    )
    assert login.status_code == 200
    assert login.json()["profile"]["roles"] == ["writer"]
    alice_token = login.json()["access_token"]
    assert client.get(
        "/me", headers={"Authorization": f"Bearer {alice_token}"}
    ).status_code == 200
    assert client.post(
        "/admin/invitations",
        headers={"Authorization": f"Bearer {alice_token}"},
        json={"expires_in": 3600},
    ).status_code == 403

    disabled = client.patch(
        "/admin/users/alice",
        headers=admin_headers,
        json={"status": "disabled"},
    )
    assert disabled.status_code == 200
    assert client.get(
        "/me", headers={"Authorization": f"Bearer {alice_token}"}
    ).status_code == 401
    assert client.post(
        "/token",
        json={"username": "alice", "password": registration["password"]},
    ).status_code == 401


def test_team_invitation_is_reusable_visible_and_not_topic_bound(tmp_path):
    app, client, credentials = _bootstrapped_client(tmp_path)
    token = client.post("/token", json=json.loads(credentials.read_text())).json()[
        "access_token"
    ]
    headers = {"Authorization": f"Bearer {token}"}

    created = client.post(
        "/admin/team-invitation", headers=headers, json={"expires_in": None}
    )
    assert created.status_code == 201
    invitation = created.json()["invitation"]
    assert invitation["expires_at"] is None
    assert invitation["use_count"] == 0

    assert client.get("/admin/team-invitation", headers=headers).json()[
        "invitation"
    ] == invitation
    with app.state.store._connect() as connection:
        stored = connection.execute(
            "SELECT code_hash, code_ciphertext, reusable, permanent FROM invitations"
        ).fetchone()
    assert stored["code_hash"] != invitation["invite_code"]
    assert invitation["invite_code"].encode() not in stored["code_ciphertext"]
    assert stored["reusable"] == 1
    assert stored["permanent"] == 1

    for username in ("alice", "bob"):
        response = client.post(
            "/register",
            json={
                "username": username,
                "display_name": username.title(),
                "password": f"{username}-password-123",
                "invite_code": invitation["invite_code"],
            },
        )
        assert response.status_code == 201
        assert response.json()["account"]["requested_project_slug"] is None
    assert client.get("/admin/team-invitation", headers=headers).json()["invitation"][
        "use_count"
    ] == 2


def test_team_invitation_expiry_revoke_and_rotation(tmp_path):
    _, client, credentials = _bootstrapped_client(tmp_path)
    token = client.post("/token", json=json.loads(credentials.read_text())).json()[
        "access_token"
    ]
    headers = {"Authorization": f"Bearer {token}"}
    first = client.post(
        "/admin/team-invitation", headers=headers, json={"expires_in": None}
    ).json()["invitation"]

    updated = client.patch(
        "/admin/team-invitation", headers=headers, json={"expires_in": 3600}
    )
    assert updated.status_code == 200
    assert updated.json()["invitation"]["invite_code"] == first["invite_code"]
    assert updated.json()["invitation"]["expires_at"] is not None

    second = client.post(
        "/admin/team-invitation", headers=headers, json={"expires_in": None}
    ).json()["invitation"]
    assert second["invite_code"] != first["invite_code"]
    assert client.post(
        "/register",
        json={
            "username": "old-code",
            "display_name": "Old Code",
            "password": "old-code-password-123",
            "invite_code": first["invite_code"],
        },
    ).status_code == 400

    assert client.delete("/admin/team-invitation", headers=headers).json() == {
        "revoked": True
    }
    assert client.get("/admin/team-invitation", headers=headers).json() == {
        "invitation": None
    }
    assert client.delete("/admin/team-invitation", headers=headers).json() == {
        "revoked": False
    }


def test_invitation_admin_guards_and_invalid_code(tmp_path):
    _, client, credentials = _bootstrapped_client(tmp_path)
    admin_token = client.post(
        "/token", json=json.loads(credentials.read_text())
    ).json()["access_token"]
    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    assert client.post(
        "/register",
        json={
            "username": "alice",
            "display_name": "Alice",
            "password": "alice-password-123",
            "invite_code": "not-an-invitation",
        },
    ).status_code == 400
    assert client.get("/admin/users").status_code == 401
    assert client.post(
        "/admin/invitations",
        headers=admin_headers,
        json={"expires_in": 60},
    ).status_code == 422
    assert client.post(
        "/admin/invitations",
        headers=admin_headers,
        json={"expires_in": 3600, "project_slug": "bad/project"},
    ).status_code == 400
    assert client.patch(
        "/admin/users/fyc",
        headers=admin_headers,
        json={"status": "disabled"},
    ).status_code == 400


def test_existing_user_database_is_migrated_without_recreating_accounts(tmp_path):
    data_dir = tmp_path / "state"
    data_dir.mkdir()
    db_path = data_dir / "users.sqlite3"
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            """
            CREATE TABLE users (
                subject TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                display_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                roles_json TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1
            )
            """
        )
        connection.execute(
            """
            INSERT INTO users(subject, username, display_name, password_hash, roles_json, active)
            VALUES ('human:fyc', 'fyc', '付彦超', 'existing-hash', '["writer", "admin"]', 1)
            """
        )
        connection.commit()
    finally:
        connection.close()

    app = create_app(
        HumanAuthConfig(
            data_dir=data_dir,
            issuer="http://127.0.0.1:8766",
            audience="mcp-agent-mail-human",
        )
    )
    users = app.state.store.list_users()
    assert users == [
        {
            "subject": "human:fyc",
            "username": "fyc",
            "display_name": "付彦超",
            "roles": ["writer", "admin"],
            "status": "active",
            "requested_project_slug": None,
            "created_at": users[0]["created_at"],
        }
    ]
    assert users[0]["created_at"] > 0


def test_concurrent_registration_consumes_invitation_once(tmp_path):
    app, client, credentials = _bootstrapped_client(tmp_path)
    admin_token = client.post(
        "/token", json=json.loads(credentials.read_text())
    ).json()["access_token"]
    invitation = client.post(
        "/admin/invitations",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"expires_in": 3600},
    ).json()["invite_code"]

    def register(username: str):
        with TestClient(app) as concurrent_client:
            return concurrent_client.post(
                "/register",
                json={
                    "username": username,
                    "display_name": username.title(),
                    "password": f"{username}-password-123",
                    "invite_code": invitation,
                },
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        first, second = executor.map(register, ("alice", "bob"))
    assert sorted((first.status_code, second.status_code)) == [201, 400]
    pending = [user for user in app.state.store.list_users() if user["status"] == "pending"]
    assert len(pending) == 1


def test_team_code_is_reusable_and_never_consumed(tmp_path):
    _app, client, _credentials = _bootstrapped_client(
        tmp_path, allow_reusable_team_code=True
    )
    (tmp_path / "state" / "team-code").write_text("team-secret-2026", encoding="utf-8")

    for username in ("carol", "dave"):
        registered = client.post("/register", json={
            "username": username,
            "display_name": username.title(),
            "password": f"{username}-password-123",
            "invite_code": "team-secret-2026",
        })
        assert registered.status_code == 201
        assert registered.json()["account"]["status"] == "pending"

    # 错误码仍被拒绝
    bad = client.post("/register", json={
        "username": "mallory",
        "display_name": "Mallory",
        "password": "mallory-password-123",
        "invite_code": "wrong-code",
    })
    assert bad.status_code == 400


def test_team_code_absent_file_falls_back_to_invitations_only(tmp_path):
    _app, client, _credentials = _bootstrapped_client(tmp_path)
    registered = client.post("/register", json={
        "username": "erin",
        "display_name": "Erin",
        "password": "erin-password-123",
        "invite_code": "team-secret-2026",
    })
    assert registered.status_code == 400


def test_team_code_unicode_and_invalid_utf8_file(tmp_path):
    _app, client, _credentials = _bootstrapped_client(
        tmp_path, allow_reusable_team_code=True
    )
    state = tmp_path / "state"
    (state / "team-code").write_text("团队码-2026", encoding="utf-8")

    ok = client.post("/register", json={
        "username": "frank",
        "display_name": "Frank",
        "password": "frank-password-123",
        "invite_code": "团队码-2026",
    })
    assert ok.status_code == 201

    wrong = client.post("/register", json={
        "username": "grace",
        "display_name": "Grace",
        "password": "grace-password-123",
        "invite_code": "团队码-2027",
    })
    assert wrong.status_code == 400

    # 非法 UTF-8 的团队码文件视为无效配置:通用 400,不得 500
    (state / "team-code").write_bytes(b"\xff\xfe\x00bad")
    invalid = client.post("/register", json={
        "username": "heidi",
        "display_name": "Heidi",
        "password": "heidi-password-123",
        "invite_code": "团队码-2026",
    })
    assert invalid.status_code == 400


def test_team_code_matching_consumed_invitation_does_not_rewrite_audit(tmp_path):
    app, client, credentials = _bootstrapped_client(
        tmp_path, allow_reusable_team_code=True
    )
    admin_secret = json.loads(credentials.read_text())
    admin_token = client.post("/token", json=admin_secret).json()["access_token"]
    invitation = client.post(
        "/admin/invitations",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"expires_in": 3600},
    )
    invite_code = invitation.json()["invite_code"]

    first = client.post("/register", json={
        "username": "ivan",
        "display_name": "Ivan",
        "password": "ivan-password-123",
        "invite_code": invite_code,
    })
    assert first.status_code == 201
    with app.state.store._connect() as connection:
        row = connection.execute(
            "SELECT used_by, used_at FROM invitations"
        ).fetchone()
    assert row["used_by"] == "human:ivan"
    original_used_at = row["used_at"]

    # 团队码复用同一个值:可再注册,但不得改写已消费邀请码的审计字段
    (tmp_path / "state" / "team-code").write_text(invite_code, encoding="utf-8")
    second = client.post("/register", json={
        "username": "judy",
        "display_name": "Judy",
        "password": "judy-password-123",
        "invite_code": invite_code,
    })
    assert second.status_code == 201
    with app.state.store._connect() as connection:
        row = connection.execute(
            "SELECT used_by, used_at FROM invitations"
        ).fetchone()
    assert row["used_by"] == "human:ivan"
    assert row["used_at"] == original_used_at


def test_register_failures_are_rate_limited_across_success_and_restart(tmp_path):
    app, client, _credentials = _bootstrapped_client(
        tmp_path, allow_reusable_team_code=True
    )
    (tmp_path / "state" / "team-code").write_text("team-secret-2026", encoding="utf-8")

    def attempt(username, code):
        return client.post("/register", json={
            "username": username,
            "display_name": username.title(),
            "password": f"{username}-password-123",
            "invite_code": code,
        })

    for i in range(4):
        assert attempt(f"bad{i}", "wrong").status_code == 400
    # 成功注册不能替攻击源清空 IP 失败计数。
    assert attempt("kate", "team-secret-2026").status_code == 201
    assert attempt("bad4", "wrong").status_code == 400

    restarted = TestClient(create_app(app.state.store.config))
    assert restarted.post(
        "/register",
        json={
            "username": "liam",
            "display_name": "Liam",
            "password": "liam-password-123",
            "invite_code": "team-secret-2026",
        },
    ).status_code == 429


def test_public_proxy_rate_limit_uses_caddy_client_ip(tmp_path):
    config = HumanAuthConfig(
        data_dir=tmp_path / "state",
        issuer="https://auth.example.com",
        audience="mcp-agent-mail-human",
        admin_access_token="public-admin-factor-0123456789abcdef",
        public_mode=True,
    )
    app = create_app(config)
    credentials = tmp_path / "admin.json"
    assert app.state.store.bootstrap_admin(
        username="fyc", display_name="付彦超", credentials_path=credentials
    )
    client = TestClient(app, client=("127.0.0.1", 50000))
    admin_token = client.post(
        "/token",
        headers={"X-Forwarded-For": "198.51.100.10"},
        json=json.loads(credentials.read_text()),
    ).json()["access_token"]
    invitation = client.post(
        "/admin/invitations",
        headers={
            "Authorization": f"Bearer {admin_token}",
            "X-Agent-Hub-Admin-Key": "public-admin-factor-0123456789abcdef",
            "X-Forwarded-For": "198.51.100.10",
        },
        json={"expires_in": 3600},
    ).json()["invite_code"]

    for index in range(5):
        assert client.post(
            "/register",
            headers={"X-Forwarded-For": "198.51.100.20"},
            json={
                "username": f"blocked{index}",
                "display_name": f"Blocked {index}",
                "password": "member-password-123",
                "invite_code": "invalid-code",
            },
        ).status_code == 400
    assert client.post(
        "/register",
        headers={"X-Forwarded-For": "198.51.100.20"},
        json={
            "username": "blocked5",
            "display_name": "Blocked 5",
            "password": "member-password-123",
            "invite_code": invitation,
        },
    ).status_code == 429
    assert client.post(
        "/register",
        headers={"X-Forwarded-For": "198.51.100.21"},
        json={
            "username": "alice",
            "display_name": "Alice",
            "password": "member-password-123",
            "invite_code": invitation,
        },
    ).status_code == 201
