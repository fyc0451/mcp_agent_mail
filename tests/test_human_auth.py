import json
import stat

from authlib.jose import JsonWebKey, JsonWebToken
from fastapi.testclient import TestClient

from mcp_agent_mail.human_auth import HumanAuthConfig, create_app


def _bootstrapped_client(tmp_path):
    config = HumanAuthConfig(
        data_dir=tmp_path / "state",
        issuer="http://127.0.0.1:8766",
        audience="mcp-agent-mail-human",
        token_ttl_seconds=600,
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
    assert payload["profile"] == {"username": "fyc", "display_name": "付彦超"}

    jwks = client.get("/.well-known/jwks.json").json()
    public_key = JsonWebKey.import_key_set(jwks).find_by_kid(app.state.store.kid)
    claims = JsonWebToken(["RS256"]).decode(payload["access_token"], public_key)
    claims.validate()
    assert claims["iss"] == "http://127.0.0.1:8766"
    assert claims["aud"] == "mcp-agent-mail-human"
    assert claims["sub"] == "human:fyc"
    assert claims["name"] == "付彦超"
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


def test_discovery_exposes_public_metadata_only(tmp_path):
    _, client, credentials = _bootstrapped_client(tmp_path)
    discovery = client.get("/.well-known/openid-configuration")
    assert discovery.status_code == 200
    assert discovery.json()["jwks_uri"] == "http://127.0.0.1:8766/.well-known/jwks.json"
    combined = discovery.text + client.get("/.well-known/jwks.json").text + client.get("/health").text
    assert json.loads(credentials.read_text())["password"] not in combined
