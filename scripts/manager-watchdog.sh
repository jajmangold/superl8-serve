#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Independently-supervised exact-container watchdog for the text-only backlog
# manager (fni8-serve#358).
#
# act_runner enforces job `timeout-minutes` by cancelling the job context, which
# SIGKILLs the WHOLE step process group (`syscall.Kill(-pgid, SIGKILL)`, see
# act_runner internal/pkg/process/treekill.go + killer_unix.go). SIGKILL cannot
# be caught, so the EXIT/INT/TERM trap in manager-cleanup.sh never runs and the
# exact nested `docker run --cidfile` container survives.
#
# This watchdog is launched by the manager step with `setsid` BEFORE `docker run`,
# so it lives in its own session/process group and is NOT a member of the step
# group that act_runner kills. It watches the step's shell PID; the instant that
# PID dies -- whether the job completed normally or the runner hard-killed the
# group -- it stops/removes ONLY the exact 64-hex container ID that the cidfile
# holds. It never targets by name, image, env, glob, or prefix, and it never
# touches a live step's container.
#
# Usage: manager-watchdog.sh <step_pid> <cidfile> <deadline_epoch> [poll_seconds]
#   step_pid       the manager step shell's PID (its process-group leader).
#   cidfile        exact per-run cidfile path (same one passed to `docker run`).
#   deadline_epoch absolute epoch at which the watchdog must self-terminate.
#   poll_seconds   optional poll interval (default 2).
set -u

step_pid="${1:?step pid required}"
cidfile="${2:?cidfile required}"
deadline="${3:?deadline epoch required}"
poll="${4:-2}"

# Exact watchdog identity for independent supervision: the step (or an operator)
# can locate this watchdog by its own sidecar pidfile, without any process sweep.
watchdog_pidfile="${cidfile}.watchdog.pid"
echo $$ > "$watchdog_pidfile"
cleanup_watchdog() { rm -f "$watchdog_pidfile"; }
trap cleanup_watchdog EXIT

# Guard against PID reuse: record the step's boot-relative start tick.
step_start="$(awk '{print $22}' "/proc/$step_pid/stat" 2>/dev/null || echo 0)"

step_is_dead() {
  local stat
  stat="$(cat "/proc/$step_pid/stat" 2>/dev/null)"
  if [[ -z "$stat" ]]; then
    # Process is gone (kill -0 would also report it, but reading /proc is the
    # authoritative liveness check and never sends a signal).
    return 0
  fi
  # State field (field 3) == Z means the process is an unreaped zombie: the
  # manager is no longer running even though its PID still exists, so an
  # act_runner that is slow to reap the SIGKILL cannot stall us.
  [[ "$(awk '{print $3}' <<< "$stat")" == "Z" ]] && return 0
  # Guard against PID reuse: an alive process with a different start tick is
  # not our step.
  [[ "$(awk '{print $22}' <<< "$stat")" != "$step_start" ]] && return 0
  return 1
}

reap_exact_container() {
  [[ -f "$cidfile" ]] || return 0
  local cid
  cid="$(tr -d '[:space:]' < "$cidfile")"
  if [[ "$cid" =~ ^[0-9a-f]{64}$ ]]; then
    # Checked (exact container still exists) and bounded (--time 5) stop.
    docker inspect "$cid" >/dev/null 2>&1 && \
      docker stop --time 5 "$cid" >/dev/null 2>&1 || true
  fi
  rm -f "$cidfile"
}

while (( $(date +%s) < deadline )); do
  if step_is_dead; then
    reap_exact_container
    exit 0
  fi
  sleep "$poll"
done

if step_is_dead; then
  reap_exact_container
fi
exit 0
