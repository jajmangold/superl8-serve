#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Exact-container cleanup for the text-only backlog-manager job (fni8-serve#344).
#
# Sourced by .github/workflows/manager.yml. Defines CIDFILE — a per-run exact path
# under RUNNER_TEMP derived from GITHUB_RUN_ID + GITHUB_RUN_ATTEMPT — and the
# cleanup_manager_container() EXIT/INT/TERM handler. The handler accepts ONLY the
# exact full 64-char lowercase-hex container ID that `docker run --cidfile` wrote;
# it never targets by name, image, env, glob, or prefix. It stops that exact
# container with a bounded `docker stop --time 5` only if it still exists, then
# removes the cidfile.
set -u

CIDFILE="${RUNNER_TEMP:?RUNNER_TEMP must be set}/manager-${GITHUB_RUN_ID:?GITHUB_RUN_ID must be set}-${GITHUB_RUN_ATTEMPT:?GITHUB_RUN_ATTEMPT must be set}.cid"

cleanup_manager_container() {
  [[ -f "$CIDFILE" ]] || return 0
  local cid
  cid="$(tr -d '[:space:]' < "$CIDFILE")"
  if [[ "$cid" =~ ^[0-9a-f]{64}$ ]]; then
    # Checked (exact container still exists) and bounded (--time 5) stop.
    docker inspect "$cid" >/dev/null 2>&1 && \
      docker stop --time 5 "$cid" >/dev/null 2>&1 || true
  fi
  rm -f "$CIDFILE"
}
