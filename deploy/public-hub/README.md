# Public Team Hub deployment

This profile exposes only two HTTPS origins while keeping the Hub, issuer,
Redis, and databases on loopback/private storage:

- `https://team-api.example.com` -> `127.0.0.1:8765`
- `https://team-auth.example.com` -> `127.0.0.1:8766`

Do not expose a member's local Agent Cockpit port `8790`.

## 1. Prepare DNS and packages

Point both DNS names to the server. Install Caddy and Redis, then make sure the
firewall exposes only TCP 80/443. Restrict SSH to an administrator network.

## 2. Install application files

Install the repository at `/opt/mcp-agent-mail`, create the `appuser` service
account, and create these private directories:

```bash
sudo install -d -o appuser -g appuser -m 0700 \
  /var/lib/mcp-agent-mail /var/lib/mcp-agent-mail-human-auth
sudo install -d -o root -g appuser -m 0750 /etc/mcp-agent-mail
```

Copy and edit the examples. Replace every `example.com` value before starting:

```bash
sudo install -o root -g appuser -m 0640 \
  deploy/public-hub/public-hub.env.example \
  /etc/mcp-agent-mail/public-hub.env
sudo install -o root -g appuser -m 0640 \
  deploy/public-hub/human-auth.env.example \
  /etc/mcp-agent-mail/human-auth.env
sudo install -o root -g root -m 0644 \
  deploy/public-hub/Caddyfile.example /etc/caddy/Caddyfile
```

Create a separate 32+ byte administrator factor. Do not put it in Git, shell
history, a URL, or an environment file:

```bash
sudo sh -c 'umask 077
openssl rand -base64 48 > /etc/mcp-agent-mail/admin-access-token
chown appuser:appuser /etc/mcp-agent-mail/admin-access-token'
```

Install the public-only hardened units (the generic `deploy/systemd/` templates
are intentionally unchanged):

```bash
sudo install -o root -g root -m 0644 \
  deploy/public-hub/mcp-agent-mail.service \
  deploy/public-hub/mcp-agent-mail-human-auth.service \
  /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now redis-server caddy \
  mcp-agent-mail-human-auth mcp-agent-mail
```

## 3. Bootstrap the first administrator once

Before starting the issuer for the first time, run:

```bash
sudo -u appuser /opt/mcp-agent-mail/.venv/bin/python \
  -m mcp_agent_mail.human_auth \
  --host 127.0.0.1 --port 8766 \
  --data-dir /var/lib/mcp-agent-mail-human-auth \
  --issuer https://team-auth.example.com \
  --audience mcp-agent-mail-human \
  --token-ttl-seconds 3600 \
  --admin-access-token-file /etc/mcp-agent-mail/admin-access-token \
  --public-mode --bootstrap-only \
  --bootstrap-username ADMIN_USERNAME \
  --bootstrap-display-name ADMIN_DISPLAY_NAME \
  --bootstrap-credentials-file \
    /var/lib/mcp-agent-mail-human-auth/initial-admin.json
```

Read the generated credential from the server console, log in, and change the
password immediately. Archive or securely remove the bootstrap credential file
according to the operator's secret-handling policy.

Each administrator's local Cockpit must have a private copy of the second
factor and start with:

```bash
TEAM_ADMIN_ACCESS_TOKEN_FILE=/private/path/admin-access-token
```

Ordinary members do not receive this file.

## 4. Configure member Cockpits

In each member's local Cockpit settings use:

```text
Team Hub API: https://team-api.example.com
Human issuer: https://team-auth.example.com
```

The browser continues to open the local Cockpit. The Hub URLs are API origins,
not a replacement UI.

## 5. Verify before use

Run from the server:

```bash
bash deploy/public-hub/verify.sh \
  https://team-api.example.com https://team-auth.example.com
```

Also scan from an external network and confirm that `8765`, `8766`, Redis, and
the databases are unreachable. Back up the Hub database, Human database, and
issuer signing key together, then perform a restore drill.

Public mode intentionally refuses to start when it detects a public backend
bind, localhost authentication bypass, static bearer bypass, HS256, HTTP
issuer/JWKS/introspection, missing Redis limits, or enabled CORS.
