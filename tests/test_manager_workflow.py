# SPDX-License-Identifier: MIT
"""CPU placement, exact-child cleanup, #359 tick-coalescing policy, and the #358
hard-timeout watchdog contract."""

import os
import re
import signal
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).parents[1]
MANAGER = (ROOT / ".github/workflows/manager.yml").read_text()
CLEANUP = ROOT / "scripts/manager-cleanup.sh"
CLEANUP_TEXT = CLEANUP.read_text()
WATCHDOG = ROOT / "scripts/manager-watchdog.sh"
WATCHDOG_TEXT = WATCHDOG.read_text()
CI = (ROOT / ".github/workflows/ci.yml").read_text()
HEX64 = "0123456789abcdef" * 4


def _fake_docker(bindir: Path) -> Path:
    """Install a fake `docker` that logs every invocation and accepts only
    `inspect`/`stop` so a broad sweep would fail loudly."""
    calls = bindir / "docker.calls"
    docker = bindir / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$*\" >> {calls}\n"
        "case \"$1\" in inspect|stop) exit 0;; *) exit 1;; esac\n"
    )
    docker.chmod(0o755)
    return calls


def _wait_for(predicate, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def _wait_for_watchdog(cidfile: Path) -> int:
    """Read the watchdog's exact sidecar pidfile; the watchdog writes it at
    startup and removes it on exit, so no process-name sweep is needed."""
    pidfile = Path(f"{cidfile}.watchdog.pid")
    assert _wait_for(pidfile.exists, timeout=5), "watchdog did not start"
    return int(pidfile.read_text().strip())


def _kill_step_group(step_pidfile: Path):
    if not step_pidfile.exists():
        return
    try:
        os.killpg(int(step_pidfile.read_text().strip()), signal.SIGKILL)
    except ProcessLookupError:
        pass


def test_manager_uses_cpu_and_preserves_gpu_workflows():
    assert "runs-on: ubuntu-latest" in MANAGER
    assert "[self-hosted, volta-gpu]" not in MANAGER
    for name in ("ci.yml", "claude.yml", "oc.yml", "release-image.yml"):
        source = (ROOT / ".github/workflows" / name).read_text()
        assert "[self-hosted, volta-gpu]" in source, name


def test_manager_only_change_uses_bounded_cpu_validation():
    assert "tests/!(test_manager_workflow.py)" in CI
    assert "scripts/!(manager-cleanup.sh|manager-watchdog.sh)" in CI
    assert "scripts/!(manager-cleanup.sh)" not in CI
    assert "manager-test:" in CI
    manager_job = CI.split("  manager-test:", 1)[1].split("  build-test:", 1)[0]
    assert "runs-on: [self-hosted, required-ci]" in manager_job
    assert "volta-gpu" not in manager_job
    assert "python3 tests/test_manager_workflow.py" in manager_job
    assert "needs: [changes, runtime-test, manager-test]" in CI


def test_exact_run_identity_and_container_wiring():
    assert "${RUNNER_TEMP:?" in CLEANUP_TEXT
    assert "${GITHUB_RUN_ID:?" in CLEANUP_TEXT
    assert "${GITHUB_RUN_ATTEMPT:?" in CLEANUP_TEXT
    assert 'source scripts/manager-cleanup.sh' in MANAGER
    assert "trap cleanup_manager_container EXIT INT TERM" in MANAGER
    assert '--cidfile "$CIDFILE"' in MANAGER
    assert "-e GITHUB_RUN_ID" in MANAGER
    assert "docker run --rm" in MANAGER
    assert "opencode-deepseek:latest" in MANAGER


def test_cleanup_is_exact_and_bounded():
    assert "^[0-9a-f]{64}$" in CLEANUP_TEXT
    assert 'docker inspect "$cid"' in CLEANUP_TEXT
    assert 'docker stop --time 5 "$cid"' in CLEANUP_TEXT
    assert 'rm -f "$CIDFILE"' in CLEANUP_TEXT
    assert "docker kill" not in CLEANUP_TEXT
    assert "docker rm" not in CLEANUP_TEXT


def test_security_and_scheduling_invariants_remain():
    for token in (
        "github.actor == github.repository_owner",
        "contains(github.event.comment.body, '@manager')",
        "timeout-minutes: 30",
        "contents: read",
        "issues: write",
        "pull-requests: write",
        "workflow_run:",
        "schedule:",
        "issue_comment:",
        "workflow_dispatch:",
    ):
        assert token in MANAGER


def test_schedule_ticks_coalesce_and_never_backlog():
    # The 15-minute backstop schedule must coalesce against itself: the newest
    # scheduled tick supersedes the previous one, so a 30-minute tick cannot be
    # queued behind itself into continuous occupancy (superl8-serve#359).
    assert "group: manager-${{ github.repository }}-${{ github.event_name }}" in MANAGER
    assert "cancel-in-progress: ${{ github.event_name == 'schedule' }}" in MANAGER
    assert "- cron: '*/15 * * * *'" in MANAGER
    assert "timeout-minutes: 30" in MANAGER


def test_owner_event_and_manual_ticks_are_never_cancelled():
    # Only `schedule` may cancel a sibling run in the same group. Owner comments,
    # event-driven merge ticks (workflow_run), and manual dispatch must never be
    # superseded, so the cancel predicate must be exactly schedule-only and each
    # trigger must keep its own sub-group so a schedule tick cannot cancel them.
    cancel_expr = [line.strip() for line in MANAGER.splitlines()
                   if "cancel-in-progress:" in line]
    assert cancel_expr == ["cancel-in-progress: ${{ github.event_name == 'schedule' }}"]
    for trigger in ("schedule:", "issue_comment:", "workflow_run:", "workflow_dispatch:"):
        assert trigger in MANAGER


def _job_block(source, job_id):
    marker = f"  {job_id}:"
    assert marker in source, job_id
    start = source.index(marker)
    body_start = start + len(marker)
    match = re.search(
        r"^  (?:[a-zA-Z0-9_-]+):", source[body_start:], flags=re.MULTILINE
    )
    return source[body_start:] if match is None else source[body_start:body_start + match.start()]


def test_required_ci_uses_dedicated_runner_and_managers_stay_generic():
    # #359 acceptance: required PR validation gets bounded CPU capacity
    # independent of scheduled managers. The dedicated capacity-1
    # `beast-superl8-required-ci` runner (content-factory-infra#559, live) advertises
    # exactly self-hosted/required-ci. ALL PR-side validation jobs — classifier
    # (changes), the workflow-contract manager-test, and the aggregate gate
    # (build-test) — route to it so the required gate cannot be starved by
    # continuous scheduled ticks. The scheduled backlog-manager *execution*
    # workflow (manager.yml) stays on the generic ubuntu-latest runner and
    # never references the reserved label or Volta.
    for job in ("changes", "manager-test", "build-test"):
        block = _job_block(CI, job)
        assert "runs-on: [self-hosted, required-ci]" in block, job
        assert "volta-gpu" not in block, job
        assert "ubuntu-latest" not in block, job
    assert "required-ci" not in MANAGER
    assert "volta-gpu" not in MANAGER
    assert "runs-on: ubuntu-latest" in MANAGER


def test_cleanup_stops_only_exact_full_id():
    cases = [
        (HEX64, True),
        (HEX64 + "\n", True),
        (HEX64[:-1], False),
        (HEX64 + "a", False),
        (HEX64.upper(), False),
        ("g" * 64, False),
        (HEX64 + "\n" + HEX64, False),
    ]
    for content, expected_stop in cases:
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            bindir = tmp_path / "bin"
            bindir.mkdir()
            calls = _fake_docker(bindir)
            cidfile = tmp_path / "manager-run-attempt.cid"
            cidfile.write_text(content)
            env = dict(
                os.environ,
                PATH=f"{bindir}:{os.environ['PATH']}",
                RUNNER_TEMP=str(tmp_path),
                GITHUB_RUN_ID="run",
                GITHUB_RUN_ATTEMPT="attempt",
            )
            command = f"source {CLEANUP}; CIDFILE={cidfile}; cleanup_manager_container"
            result = subprocess.run(["bash", "-c", command], env=env, text=True)
            assert result.returncode == 0
            observed = calls.read_text().splitlines() if calls.exists() else []
            if expected_stop:
                assert observed == [f"inspect {HEX64}", f"stop --time 5 {HEX64}"]
            else:
                assert observed == []
            assert not cidfile.exists()


def test_manager_wires_independent_watchdog():
    assert "timeout-minutes: 30" in MANAGER
    assert "trap cleanup_manager_container EXIT INT TERM" in MANAGER
    assert "STEP_PID=$$" in MANAGER
    assert "setsid" in MANAGER
    assert "scripts/manager-watchdog.sh" in MANAGER
    assert '"$CIDFILE"' in MANAGER
    assert "MANAGER_JOB_TIMEOUT_MINUTES" in MANAGER


def test_watchdog_reaps_only_exact_full_id():
    assert "^[0-9a-f]{64}$" in WATCHDOG_TEXT
    assert 'docker inspect "$cid"' in WATCHDOG_TEXT
    assert 'docker stop --time 5 "$cid"' in WATCHDOG_TEXT
    assert 'rm -f "$cidfile"' in WATCHDOG_TEXT
    assert "docker kill" not in WATCHDOG_TEXT
    assert "docker rm" not in WATCHDOG_TEXT


def test_watchdog_survives_uncatchable_group_kill_and_reaps_exact_container():
    """act_runner's job-timeout kill is kill(-pgid, SIGKILL). The watchdog must
    survive that whole-group SIGKILL (it is setsid'd into its own session) and
    then stop/remove ONLY the exact cidfile container, leaving any unrelated
    container (never addressed by the fake docker) untouched."""
    with tempfile.TemporaryDirectory() as temporary:
        tmp_path = Path(temporary)
        bindir = tmp_path / "bin"
        bindir.mkdir()
        calls = _fake_docker(bindir)
        cidfile = tmp_path / "manager-run-attempt.cid"
        cidfile.write_text(HEX64 + "\n")
        step_pidfile = tmp_path / "step.pid"
        wd_log = tmp_path / "watchdog.log"
        deadline = int(time.time()) + 60

        step_script = tmp_path / "step.sh"
        step_script.write_text(
            "#!/usr/bin/env bash\n"
            "set -u\n"
            f"echo $$ > {step_pidfile}\n"
            "STEP_PID=$$\n"
            f"setsid {WATCHDOG} \"$STEP_PID\" {cidfile} {deadline} 0.2 "
            f"</dev/null >>{wd_log} 2>&1 &\n"
            "sleep 600\n"
        )
        step_script.chmod(0o755)
        env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}")
        step = subprocess.Popen(
            ["setsid", str(step_script)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            assert _wait_for(step_pidfile.exists, timeout=5), "step did not start"
            step_pid = int(step_pidfile.read_text().strip())
            watchdog_pid = _wait_for_watchdog(cidfile)

            # Give the watchdog one poll to record the step's start tick.
            time.sleep(0.5)

            # act_runner tree-kill: SIGKILL to the step's whole process group.
            os.killpg(step_pid, signal.SIGKILL)
            # Reap the step now, exactly as act_runner's cmd.Wait() does after
            # its TreeKill; an unreaped zombie would still answer kill -0 and
            # the watchdog would (correctly) wait for it.
            step.wait(timeout=5)

            # The watchdog is in its own session and must survive the group kill.
            assert subprocess.run(
                ["kill", "-0", str(watchdog_pid)], check=False
            ).returncode == 0, "watchdog was killed with the step group"

            # The watchdog must reap the exact container and then self-terminate.
            assert _wait_for(
                lambda: not Path(f"{cidfile}.watchdog.pid").exists(), timeout=10
            ), "watchdog did not self-terminate"
            observed = calls.read_text().splitlines() if calls.exists() else []
            assert observed == [f"inspect {HEX64}", f"stop --time 5 {HEX64}"]
            assert not cidfile.exists()
        finally:
            _kill_step_group(step_pidfile)
            step.wait(timeout=5)


def test_watchdog_exits_without_action_on_normal_clean_exit():
    """When the step completes normally and the EXIT trap already removed the
    cidfile, the watchdog must exit with no docker interaction at all."""
    with tempfile.TemporaryDirectory() as temporary:
        tmp_path = Path(temporary)
        bindir = tmp_path / "bin"
        bindir.mkdir()
        calls = _fake_docker(bindir)
        cidfile = tmp_path / "manager-run-attempt.cid"
        step_pidfile = tmp_path / "step.pid"
        wd_log = tmp_path / "watchdog.log"
        deadline = int(time.time()) + 60

        step_script = tmp_path / "step.sh"
        step_script.write_text(
            "#!/usr/bin/env bash\n"
            "set -u\n"
            f"echo $$ > {step_pidfile}\n"
            "STEP_PID=$$\n"
            f"setsid {WATCHDOG} \"$STEP_PID\" {cidfile} {deadline} 0.2 "
            f"</dev/null >>{wd_log} 2>&1 &\n"
            "sleep 1\n"
        )
        step_script.chmod(0o755)
        env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}")
        step = subprocess.Popen(
            ["setsid", str(step_script)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        assert step.wait(timeout=10) == 0
        assert _wait_for(
            lambda: not (cidfile.exists() or Path(f"{cidfile}.watchdog.pid").exists()),
            timeout=10,
        )
        assert not calls.exists(), calls.read_text()


def test_watchdog_bounded_deadline_does_not_touch_live_step_container():
    """While the step is still alive the watchdog must never act; at its own
    deadline it must self-terminate without touching anything."""
    with tempfile.TemporaryDirectory() as temporary:
        tmp_path = Path(temporary)
        bindir = tmp_path / "bin"
        bindir.mkdir()
        calls = _fake_docker(bindir)
        cidfile = tmp_path / "manager-run-attempt.cid"
        cidfile.write_text(HEX64 + "\n")
        step_pidfile = tmp_path / "step.pid"
        wd_log = tmp_path / "watchdog.log"
        deadline = int(time.time()) + 2

        step_script = tmp_path / "step.sh"
        step_script.write_text(
            "#!/usr/bin/env bash\n"
            "set -u\n"
            f"echo $$ > {step_pidfile}\n"
            "STEP_PID=$$\n"
            f"setsid {WATCHDOG} \"$STEP_PID\" {cidfile} {deadline} 0.2 "
            f"</dev/null >>{wd_log} 2>&1 &\n"
            "sleep 600\n"
        )
        step_script.chmod(0o755)
        env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}")
        step = subprocess.Popen(
            ["setsid", str(step_script)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            assert _wait_for(step_pidfile.exists, timeout=5), "step did not start"
            _wait_for_watchdog(cidfile)
            assert _wait_for(
                lambda: not Path(f"{cidfile}.watchdog.pid").exists(), timeout=10
            ), "watchdog did not bound itself"
            # Live step's container untouched, cidfile intact.
            assert not calls.exists(), calls.read_text()
            assert cidfile.exists()
        finally:
            _kill_step_group(step_pidfile)
            step.wait(timeout=5)


def test_watchdog_rejects_unvalidated_cidfile():
    """An invalid cidfile must be removed without any docker call: never act on
    an ID that was not validated as exact 64-hex."""
    with tempfile.TemporaryDirectory() as temporary:
        tmp_path = Path(temporary)
        bindir = tmp_path / "bin"
        bindir.mkdir()
        calls = _fake_docker(bindir)
        cidfile = tmp_path / "manager-run-attempt.cid"
        cidfile.write_text(HEX64[:-1] + "\n")
        step_pidfile = tmp_path / "step.pid"
        wd_log = tmp_path / "watchdog.log"
        deadline = int(time.time()) + 60

        step_script = tmp_path / "step.sh"
        step_script.write_text(
            "#!/usr/bin/env bash\n"
            "set -u\n"
            f"echo $$ > {step_pidfile}\n"
            "STEP_PID=$$\n"
            f"setsid {WATCHDOG} \"$STEP_PID\" {cidfile} {deadline} 0.2 "
            f"</dev/null >>{wd_log} 2>&1 &\n"
            "sleep 600\n"
        )
        step_script.chmod(0o755)
        env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}")
        step = subprocess.Popen(
            ["setsid", str(step_script)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            assert _wait_for(step_pidfile.exists, timeout=5), "step did not start"
            step_pid = int(step_pidfile.read_text().strip())
            _wait_for_watchdog(cidfile)
            time.sleep(0.5)
            os.killpg(step_pid, signal.SIGKILL)
            step.wait(timeout=5)
            assert _wait_for(lambda: not cidfile.exists(), timeout=10)
            assert not calls.exists(), calls.read_text()
        finally:
            step.wait(timeout=5)


if __name__ == "__main__":
    test_manager_uses_cpu_and_preserves_gpu_workflows()
    test_manager_only_change_uses_bounded_cpu_validation()
    test_exact_run_identity_and_container_wiring()
    test_cleanup_is_exact_and_bounded()
    test_security_and_scheduling_invariants_remain()
    test_schedule_ticks_coalesce_and_never_backlog()
    test_owner_event_and_manual_ticks_are_never_cancelled()
    test_required_ci_uses_dedicated_runner_and_managers_stay_generic()
    test_cleanup_stops_only_exact_full_id()
    test_manager_wires_independent_watchdog()
    test_watchdog_reaps_only_exact_full_id()
    test_watchdog_survives_uncatchable_group_kill_and_reaps_exact_container()
    test_watchdog_exits_without_action_on_normal_clean_exit()
    test_watchdog_bounded_deadline_does_not_touch_live_step_container()
    test_watchdog_rejects_unvalidated_cidfile()
