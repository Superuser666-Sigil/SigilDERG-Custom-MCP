#!/usr/bin/env bash
set -euo pipefail
# Simple health-checker for cloudflared tunnel + MCP server
# Usage: CLOUD_DOMAIN=mcp.yourdomain.com ./cloudflared-healthcheck.sh

DOMAIN=${CLOUD_DOMAIN:-}
ENDPOINT=${HEALTH_ENDPOINT:-/healthz}
TIMEOUT=${HEALTH_TIMEOUT:-10}
SLEEP_BEFORE_RESTART=${SLEEP_BEFORE_RESTART:-2}
EXPECTED_HTTP=${EXPECTED_HTTP:-200}

if [[ -z "$DOMAIN" ]]; then
  echo "CLOUD_DOMAIN environment variable not set; aborting" >&2
  exit 2
fi

URL="https://${DOMAIN}${ENDPOINT}"

echo "Checking ${URL} ..."
HTTP_CODE=$(curl -sS -I -m ${TIMEOUT} "${URL}" -o /dev/null -w "%{http_code}" || echo 000)

if [[ "${HTTP_CODE}" != "${EXPECTED_HTTP}" ]]; then
  echo "$(date --iso-8601=seconds) - health check failed (code=${HTTP_CODE}) for ${URL}" >&2
  echo "Restarting cloudflared..."
  sleep ${SLEEP_BEFORE_RESTART}
  systemctl restart cloudflared
  sleep 1
  echo "Restart triggered" >&2
  exit 1
else
  echo "$(date --iso-8601=seconds) - health OK (code=${HTTP_CODE}) for ${URL}"
  exit 0
fi
