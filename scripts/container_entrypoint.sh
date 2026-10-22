#!/usr/bin/env bash
# Container entrypoint: seed LIBERO's config into the current $HOME, then exec.
#
# LIBERO resolves its config as ~/.libero/config.yaml. The image ships a
# neutral copy at /opt/libero-config/config.yaml so the container can run as an
# arbitrary uid (`docker run --user $(id -u):$(id -g)`), whose $HOME has no
# .libero directory yet.
set -euo pipefail

if [[ -f /opt/libero-config/config.yaml ]]; then
  cfg="${HOME:-/root}/.libero/config.yaml"
  if [[ ! -f "${cfg}" ]] && mkdir -p "$(dirname "${cfg}")" 2>/dev/null; then
    cp /opt/libero-config/config.yaml "${cfg}" 2>/dev/null || true
  fi
fi

exec "$@"
