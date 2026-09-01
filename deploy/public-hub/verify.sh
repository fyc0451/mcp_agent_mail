#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 https://team-api.example.com https://team-auth.example.com" >&2
  exit 2
fi

api=${1%/}
auth=${2%/}

case "$api $auth" in
  https://*\ https://*) ;;
  *) echo "both endpoints must use https://" >&2; exit 2 ;;
esac

curl --fail --silent --show-error "$api/health/readiness" >/dev/null
curl --fail --silent --show-error "$auth/health" >/dev/null
curl --fail --silent --show-error "$auth/.well-known/jwks.json" >/dev/null

status=$(curl --silent --output /dev/null --write-out '%{http_code}' \
  "$api/hub/api/projects")
if [ "$status" != "401" ]; then
  echo "expected unauthenticated Team Hub request to return 401, got $status" >&2
  exit 1
fi

cors=$(curl --silent --dump-header - --output /dev/null \
  -H 'Origin: https://attacker.invalid' "$api/health/readiness" \
  | tr -d '\r' | grep -i '^access-control-allow-origin:' || true)
if [ -n "$cors" ]; then
  echo "unexpected CORS header: $cors" >&2
  exit 1
fi

if command -v ss >/dev/null 2>&1; then
  for port in 8765 8766; do
    listeners=$(ss -H -ltn "sport = :$port" || true)
    if [ -z "$listeners" ]; then
      echo "no local listener found on $port" >&2
      exit 1
    fi
    local_addresses=$(printf '%s\n' "$listeners" | awk '{print $4}')
    if printf '%s\n' "$local_addresses" | grep -Eq '^(0\.0\.0\.0|\[::\]|\*):'; then
      echo "port $port is publicly bound instead of loopback-only" >&2
      exit 1
    fi
  done
fi

echo "public Team Hub baseline verification passed"
