"""
DinD-aware Docker environment for Terminal Bench 2.

When tb2_eval.py runs inside a Docker container that uses the host Docker
daemon via a socket mount (Docker-in-Docker via socket), bind mount source
paths must be translated from our container's overlay filesystem to the
corresponding real path on the Docker host.

Harbor's DockerEnvironment passes paths like /root/.../verifier directly to
docker compose as HOST_VERIFIER_LOGS_PATH etc.  The Docker daemon resolves
these on the *host* filesystem, which is a different filesystem from our
container's overlay — so writes from spawned containers never appear in our
overlay and reward.txt is never found.

Fix: detect the overlay upperdir from /proc/self/mountinfo and prefix all
host-side bind mount paths with it, so the Docker daemon can find them.
On a non-containerised host (no overlay root) paths are returned unchanged.

Warm-image cache
----------------
Installing curl, uv, and test-suite packages (e.g. torch==2.7.1) inside the
verifier container takes several minutes on the first run.  To skip these
reinstalls on every subsequent round we maintain a per-task "warm" image:

  tb2-warm/<task-name>:latest

After each task run the container is committed to that tag.  On the next run
the tag is detected and used directly, bypassing the Dockerfile build step
and preserving the uv package cache that is baked into the committed layer.

Before committing, /app is restored to its pre-agent state via a tarball
snapshot taken immediately after container start.  This ensures the warm image
contains only the initial task files plus the installed toolchain (curl, uv,
uv package cache) — no agent solution files leak into the warm image.
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths


def _dind_host_path(container_path: str) -> str:
    """Translate a container-side path to its host-visible equivalent.

    For paths that live on a non-overlay filesystem (e.g. a block device
    bind-mounted at /data), the host sees the same path directly — no
    translation required.  Only apply the overlay-upperdir trick when the
    path belongs to the container's own overlay layer.
    """
    try:
        with open("/proc/self/mountinfo") as f:
            lines = f.readlines()

        # Build a map: mountpoint -> fstype, for all mounts
        mounts: list[tuple[str, str]] = []
        overlay_upperdir: str | None = None
        for line in lines:
            parts = line.split()
            if len(parts) < 7:
                continue
            mountpoint = parts[4]
            try:
                dash = parts.index("-")
            except ValueError:
                continue
            fstype = parts[dash + 1] if dash + 1 < len(parts) else ""
            if fstype == "overlay" and mountpoint == "/" and "upperdir=" in line:
                m = re.search(r"upperdir=([^,\s)]+)", line)
                if m:
                    overlay_upperdir = m.group(1).rstrip("/")
            else:
                mounts.append((mountpoint, fstype))

        # Find the longest non-overlay mountpoint that is a prefix of container_path
        norm = container_path.rstrip("/")
        best: str | None = None
        for mp, _ in mounts:
            if norm == mp or norm.startswith(mp.rstrip("/") + "/"):
                if best is None or len(mp) > len(best):
                    best = mp

        # If the path is under a non-overlay mount, it's directly visible on the host
        if best is not None:
            return container_path

        # Fall back to overlay-upperdir translation (path lives in container layer)
        if overlay_upperdir:
            return overlay_upperdir + norm

    except (OSError, ValueError):
        pass
    return container_path


async def _docker(*args: str) -> tuple[int, str]:
    """Run a plain `docker` subcommand; return (returncode, stdout+stderr)."""
    proc = await asyncio.create_subprocess_exec(
        "docker",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await proc.communicate()
    return proc.returncode, (stdout or b"").decode(errors="replace").strip()


class DinDDockerEnvironment(DockerEnvironment):
    """DockerEnvironment with DinD bind-mount path translation and warm-image cache."""

    _WARM_PREFIX = "tb2-warm"

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        *args,
        **kwargs,
    ):
        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            *args,
            **kwargs,
        )
        v = self._env_vars
        v.context_dir = _dind_host_path(v.context_dir)
        v.host_verifier_logs_path = _dind_host_path(v.host_verifier_logs_path)
        v.host_agent_logs_path = _dind_host_path(v.host_agent_logs_path)
        v.host_artifacts_path = _dind_host_path(v.host_artifacts_path)

        # Propagate proxy settings from the host into every container exec()
        # call (including the verifier's test.sh).  harbor's exec() merges
        # _persistent_env into each `docker compose exec -e KEY=VAL …` call.
        _LOCALHOST_NO_PROXY = "localhost,127.0.0.1,::1"
        for var in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy"):
            val = os.environ.get(var)
            if val:
                self._persistent_env[var] = val
        # Always ensure localhost/127.0.0.1 bypass the proxy inside the container.
        # Tasks often start local services (nginx, pypi-server, etc.) that the
        # verifier then connects to via http://localhost:<port>; without this,
        # those requests are routed through the external proxy and fail.
        if self._persistent_env.get("http_proxy") or self._persistent_env.get("HTTP_PROXY"):
            for no_proxy_var in ("no_proxy", "NO_PROXY"):
                existing = self._persistent_env.get(no_proxy_var, "")
                if existing:
                    self._persistent_env[no_proxy_var] = f"{existing},{_LOCALHOST_NO_PROXY}"
                else:
                    self._persistent_env[no_proxy_var] = _LOCALHOST_NO_PROXY

        self._warm_image = f"{self._WARM_PREFIX}/{environment_name}:latest"
        self._warm_image_preexisted = False
        # Set to True when the patcher leaves install commands un-skipped (they
        # actually ran during this verifier pass), so stop() knows to re-commit.
        self._patcher_had_unskipped_installs = False

        # Container ID cached after start() — used by the exec-cancel watchdog
        # to issue plain `docker exec` (not docker compose exec) kill commands,
        # bypassing any hung docker-compose-exec connections.
        self._container_id: str | None = None
        # Keep references so watchdog tasks aren't GC'd before they finish.
        self._watchdog_tasks: set[asyncio.Task] = set()

    # ── helpers ───────────────────────────────────────────────────────────────

    _APP_SNAPSHOT = "/tmp/__tb2_app_snapshot.tar.gz"

    async def _warm_image_exists(self) -> bool:
        rc, _ = await _docker("image", "inspect", "--format", "{{.Id}}", self._warm_image)
        return rc == 0

    async def _snapshot_app(self) -> None:
        """Save /app state so it can be restored before committing the warm image."""
        result = await self.exec(f"tar czf {self._APP_SNAPSHOT} -C / app 2>/dev/null; echo OK")
        if "OK" not in (result.stdout or ""):
            self.logger.warning("warm-cache: /app snapshot failed")

    async def _restore_app(self) -> bool:
        """Restore /app to its pre-agent state, removing any agent-written files.

        Returns True on success, False if the snapshot is missing or restore fails.
        """
        result = await self.exec(
            f"test -f {self._APP_SNAPSHOT} && "
            f"rm -rf /app && "
            f"tar xzf {self._APP_SNAPSHOT} -C / && "
            f"rm -f {self._APP_SNAPSHOT} && "
            f"echo OK"
        )
        ok = "OK" in (result.stdout or "")
        if not ok:
            self.logger.warning("warm-cache: /app restore failed — skipping commit")
        return ok

    async def _get_container_id(self) -> str | None:
        result = await self._run_docker_compose_command(["ps", "-q", "main"], check=False)
        # Take the first non-empty line in case compose emits extra output.
        for line in (result.stdout or "").splitlines():
            line = line.strip()
            if line:
                return line
        return None

    async def _exec_cancel_watchdog(self, grace_sec: float = 60.0) -> None:
        """Background task: kill stuck process group if the container is still
        running ``grace_sec`` seconds after an exec() CancelledError.

        Uses plain ``docker exec`` (not docker compose exec) so it succeeds even
        when existing docker-compose-exec connections are hung on a full pipe.
        Targets only the process group stored in ``/tmp/_hx_exec_pgid`` — the
        setsid group created by HarborSandbox for each agent command — so
        ``sleep infinity`` (PID 1/2) is untouched and the container stays alive.
        """
        try:
            await asyncio.sleep(grace_sec)
        except asyncio.CancelledError:
            return

        cid = self._container_id
        if not cid:
            return

        # Confirm the container is still running before taking action.
        # On timeout or error, fall through rather than returning — if the
        # container is already stopped, docker kill will fail harmlessly.
        try:
            rc, state = await asyncio.wait_for(
                _docker("inspect", "--format", "{{.State.Running}}", cid),
                timeout=5.0,
            )
            if rc != 0 or state.strip() != "true":
                return
        except (asyncio.TimeoutError, Exception) as exc:
            self.logger.warning("exec-watchdog: inspect failed (%s) — proceeding with kill attempt", exc)

        self.logger.warning(
            "exec-watchdog: container %s still running %.0fs after cancel "
            "— killing stuck process group via direct docker exec",
            cid[:12], grace_sec,
        )

        # Kill via direct docker exec: read the PID stored by HarborSandbox's
        # setsid wrapper and send SIGKILL to the entire process group.
        # If no pgid file exists (verifier is hung, not an agent command), fall
        # back to docker kill — this uses the Docker API and bypasses any hung
        # docker-compose-exec pipe connections.
        pgid_killed = False
        try:
            rc2, out2 = await asyncio.wait_for(
                _docker(
                    "exec", cid, "sh", "-c",
                    "_p=$(cat /tmp/_hx_exec_pgid 2>/dev/null); "
                    "[ -n \"$_p\" ] || { echo 'no pgid file'; exit 0; }; "
                    "kill -9 -\"$_p\" 2>/dev/null; "
                    "rm -f /tmp/_hx_exec_pgid; "
                    "echo \"killed group $_p\"",
                ),
                timeout=10.0,
            )
            self.logger.warning("exec-watchdog: rc=%d %s", rc2, out2)
            pgid_killed = "killed group" in (out2 or "")
        except asyncio.TimeoutError:
            self.logger.warning("exec-watchdog: docker exec timed out — container may be unresponsive")
        except Exception as exc:
            self.logger.warning("exec-watchdog: %s", exc)

        # No pgid file means this is a hung verifier (not an agent command).
        # docker compose exec is blocked on a full pipe and cannot be unblocked
        # from inside the container.  Use docker kill (Docker API, no exec pipe)
        # to forcibly stop the container so harbor's docker compose down can
        # proceed.  The container is already being torn down by harbor's timeout
        # handler — killing it early is safe and unblocks the entire eval.
        if not pgid_killed:
            self.logger.warning(
                "exec-watchdog: no pgid found — verifier likely hung; "
                "issuing docker kill %s to unblock docker compose down",
                cid[:12],
            )
            try:
                rc3, out3 = await asyncio.wait_for(
                    _docker("kill", cid),
                    timeout=10.0,
                )
                self.logger.warning(
                    "exec-watchdog: docker kill %s rc=%d %s", cid[:12], rc3, out3
                )
            except (asyncio.TimeoutError, Exception) as exc:
                self.logger.warning("exec-watchdog: docker kill failed: %s", exc)

    async def _commit_to_warm_image(self) -> None:
        """Restore /app to initial state, then commit the container as the warm image.

        The snapshot tarball is deleted inside the container before committing so it
        does not bloat the warm image (it can be hundreds of MB for tasks with large
        model files in /app such as gpt2-codegolf).
        """
        if not await self._restore_app():
            return  # restore failed — do not commit a potentially dirty image
        container_id = await self._get_container_id()
        if not container_id:
            self.logger.warning("warm-cache: could not resolve container ID — skipping commit")
            return
        rc, out = await _docker("commit", container_id, self._warm_image)
        if rc == 0:
            self.logger.info("warm-cache: committed → %s", self._warm_image)
        else:
            self.logger.warning("warm-cache: commit failed: %s", out)

    # ── test.sh warm-cache patcher ────────────────────────────────────────────
    # When the warm image is reused, test.sh still contains unconditional
    # installation commands (e.g. `curl … | sh` to install uv).  We intercept
    # the `chmod +x test.sh` call — which harbor issues right before running
    # test.sh — and rewrite those lines with "already installed?" guards so
    # the network download is skipped when the toolchain is already present.

    _PATCHER_SCRIPT = r"""
import re, subprocess, sys

def _sh(cmd):
    return subprocess.call(cmd, shell=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# pip package-name → python import name for common mismatches
_PKG_IMPORT = {
    'pillow': 'PIL', 'pil': 'PIL',
    'scikit-learn': 'sklearn',
    'opencv-python': 'cv2', 'opencv-python-headless': 'cv2',
    'pyyaml': 'yaml',
    'beautifulsoup4': 'bs4',
    'python-dateutil': 'dateutil',
    'typing-extensions': 'typing_extensions',
    'torch': 'torch', 'torchvision': 'torchvision', 'torchaudio': 'torchaudio',
    'tensorflow': 'tensorflow', 'tf-nightly': 'tensorflow',
    'jax': 'jax', 'jaxlib': 'jaxlib',
    'pytest-json-ctrf': 'pytest_json_ctrf',
    'pytest-xdist': 'xdist',
}

def _pip_installed(pkg):
    base = re.split(r'[><=!;\[@]', pkg)[0].strip().lower()
    imp = _PKG_IMPORT.get(base, base.replace('-', '_'))
    return _sh(f'python3 -c "import {imp}"') == 0

def _apt_installed(pkg):
    base = re.split(r'[><=]', pkg)[0].strip()
    return _sh(f'dpkg -l {base} 2>/dev/null | grep -q "^ii"') == 0

try:
    src = open('test.sh').read()
except OSError:
    sys.exit(0)

lines = src.splitlines()
out = []
skipped = []

for line in lines:
    s = line.strip()
    if not s or s.startswith('#'):
        out.append(line)
        continue

    # apt-get update — skip on warm image (packages already present) or if lists fresh
    if re.match(r'apt-get\s+update\b', s):
        if IS_WARM or _sh('find /var/lib/apt/lists -maxdepth 1 -name "*.InRelease"'
               ' -mmin -1440 2>/dev/null | grep -q .') == 0:
            out.append('# [warm-cache] apt lists fresh — skipped: ' + s)
            skipped.append('apt-get-update')
            continue

    # apt-get install -y <pkg> …
    m = re.match(r'apt-get\s+install\b', s)
    if m:
        pkgs = [p for p in re.sub(r'apt-get\s+install\s+', '', s).split()
                if not p.startswith('-')]
        if IS_WARM or (pkgs and all(_apt_installed(p) for p in pkgs)):
            out.append('# [warm-cache] apt packages present — skipped: ' + s)
            skipped.append('apt:' + ','.join(pkgs))
            continue

    # curl … | sh  — uv installer (astral.sh or direct GitHub release)
    if re.search(r'curl\b', s) and '|' in s:
        if re.search(r'astral\.sh.*uv|github\.com/astral-sh/uv', s):
            if _sh('command -v uv || test -x /root/.local/bin/uv || test -x /root/.cargo/bin/uv') == 0:
                out.append('# [warm-cache] uv present — skipped: ' + s)
                skipped.append('uv-install')
                continue

    # pip install / pip3 install / python -m pip install
    if re.match(r'(?:pip3?|python3?\s+-m\s+pip)\s+install\b', s):
        pkgs = [p for p in re.sub(r'.+install\s+', '', s).split()
                if not p.startswith('-')]
        if pkgs and all(_pip_installed(p) for p in pkgs):
            out.append('# [warm-cache] pip packages present — skipped: ' + s)
            skipped.append('pip:' + ','.join(pkgs))
            continue

    # uv pip install <pkgs>
    if re.match(r'uv\s+pip\s+install\b', s):
        pkgs = [p for p in re.sub(r'uv\s+pip\s+install\s+', '', s).split()
                if not p.startswith('-')]
        if pkgs and all(_pip_installed(p) for p in pkgs):
            out.append('# [warm-cache] uv pip packages present — skipped: ' + s)
            skipped.append('uv-pip:' + ','.join(pkgs))
            continue

    # uvx [-p <ver>] [-w <pkg>]* <tool> [args…]  →  persistent venv
    # Creates /root/.tb2_warm_venv once; subsequent runs skip the install.
    if re.match(r'^uvx\b', s):
        _VENV = '/root/.tb2_warm_venv'
        _py_ver = re.search(r'-p\s+(\S+)', s)
        _pkgs   = re.findall(r'-w\s+(\S+)', s)
        _cmd    = re.sub(r'-p\s+\S+\s*', '', re.sub(r'^uvx\s+', '', s))
        _cmd    = re.sub(r'-w\s+\S+\s*', '', _cmd).strip()
        _bin    = _cmd.split()[0] if _cmd else ''
        _args   = _cmd[len(_bin):].strip() if _bin else ''
        _py_f   = f'-p {_py_ver.group(1)}' if _py_ver else ''
        _pkg_q  = ' '.join(f'"{p}"' for p in _pkgs)
        _imps   = [_PKG_IMPORT.get(re.split(r'[><=!;\[@]', p)[0].strip().lower(),
                                    re.split(r'[><=!;\[@]', p)[0].strip().lower().replace('-', '_'))
                   for p in _pkgs]
        _imp_csv = ', '.join(_imps) if _imps else ''
        out.append('# [warm-cache] uvx → persistent venv (' + _VENV + '): ' + s)
        if _imp_csv:
            out.append(f'if ! [ -f {_VENV}/bin/python ] || ! {_VENV}/bin/python -c "import {_imp_csv}" 2>/dev/null; then')
        else:
            out.append(f'if ! [ -f {_VENV}/bin/python ]; then')
        out.append(f'  uv venv {_py_f} {_VENV} 2>/dev/null || true')
        if _pkg_q:
            out.append(f'  uv pip install --python {_VENV} {_pkg_q}')
        out.append('fi')
        if _bin:
            out.append(f'{_VENV}/bin/{_bin}' + (f' {_args}' if _args else ''))
        skipped.append('uvx→venv:' + (_bin or s))
        continue

    out.append(line)

patched = '\n'.join(out)
if patched != src:
    open('test.sh', 'w').write(patched)
    print('[warm-cache] test.sh patched; skipped:', skipped)
"""

    # Shell fallback used when the container has no python3.
    # Handles only the uv-install curl|sh line and apt-get update since those
    # are the expensive network operations; pip-install checks require python3.
    _PATCHER_SHELL = r"""
set -e
[ -f test.sh ] || exit 0
_uv_present() {
    command -v uv >/dev/null 2>&1 || \
    test -x /root/.local/bin/uv || \
    test -x /root/.cargo/bin/uv
}
_apt_lists_fresh() {
    find /var/lib/apt/lists -maxdepth 1 -name '*.InRelease' -mmin -1440 2>/dev/null | grep -q .
}
patched=0
while IFS= read -r line; do
    s=$(echo "$line" | sed 's/^[[:space:]]*//')
    case "$s" in
        apt-get\ update*)
            if [ "$IS_WARM" = "1" ] || _apt_lists_fresh; then
                echo "# [warm-cache] apt lists fresh — skipped: $line"
                patched=1; continue
            fi;;
        *astral.sh*uv*|*astral-sh/uv*)
            if echo "$s" | grep -q 'curl' && echo "$s" | grep -q '|'; then
                if _uv_present; then
                    echo "# [warm-cache] uv present — skipped: $line"
                    patched=1; continue
                fi
            fi;;
        uvx\ -p\ *|uvx\ -w\ *|uvx\ --with\ *)
            _tb2_venv="/root/.tb2_warm_venv"
            _uv_bin=$(echo "$s" \
                | sed -e 's/^uvx[[:space:]]*//' \
                      -e 's/-p[[:space:]]*[^[:space:]]*[[:space:]]*//' \
                      -e 's/-w[[:space:]]*[^[:space:]]*[[:space:]]*//' \
                | awk '{print $1}')
            _uv_rest=$(echo "$s" \
                | sed -e 's/^uvx[[:space:]]*//' \
                      -e 's/-p[[:space:]]*[^[:space:]]*[[:space:]]*//' \
                      -e 's/-w[[:space:]]*[^[:space:]]*[[:space:]]*//' \
                | cut -d' ' -f2-)
            if [ -f "${_tb2_venv}/bin/python" ]; then
                echo "# [warm-cache] uvx venv cached — skipped install: $line"
                printf '%s\n' "${_tb2_venv}/bin/${_uv_bin} ${_uv_rest}"
                patched=1; continue
            fi;;
    esac
    printf '%s\n' "$line"
done < test.sh > test.sh.warm_tmp && mv test.sh.warm_tmp test.sh
[ "$patched" = "1" ] && echo "[warm-cache] test.sh patched (sh-fallback)"
"""

    async def _patch_test_sh_for_warm_cache(self) -> None:
        """Rewrite test.sh installation commands to no-ops when already satisfied.

        Uses the Python patcher when python3 is available; falls back to a
        POSIX-sh-only patcher that covers uv-install and apt-get update.
        """
        import shlex

        try:
            # Check for python3 first (fast; most containers have it).
            # timeout_sec: docker compose exec overhead ~1-3s, 15s is generous.
            py_check = await self.exec(
                "command -v python3 >/dev/null 2>&1 && echo YES || echo NO",
                timeout_sec=15,
            )
            has_python3 = "YES" in (py_check.stdout or "")

            is_warm = int(self._warm_image_preexisted)
            if has_python3:
                # Patcher runs dpkg/pip/find checks per install line; 60s covers
                # even test.sh files with many install commands.
                patcher = f"IS_WARM = {is_warm}\n" + self._PATCHER_SCRIPT
                result = await self.exec(
                    f"python3 -c {shlex.quote(patcher)}",
                    timeout_sec=60,
                )
            else:
                patcher = f"IS_WARM={is_warm}\n" + self._PATCHER_SHELL
                result = await self.exec(
                    f"sh -c {shlex.quote(patcher)}",
                    timeout_sec=60,
                )
            out = (result.stdout or "").strip()
            if out:
                self.logger.info("warm-cache: %s", out)

            # Check whether any install commands remain in test.sh after patching.
            # If so, they will actually execute during this verifier run, meaning
            # the container will have fresh deps — worth re-committing the warm image.
            # timeout_sec: pure grep on a small file, 10s is ample.
            check = await self.exec(
                r"grep -qE '^[^#]*(apt-get install|curl.*astral.*uv|pip install|uv pip install)'"
                r" test.sh && echo HAS_INSTALLS || echo CLEAN",
                timeout_sec=10,
            )
            if "HAS_INSTALLS" in (check.stdout or ""):
                self._patcher_had_unskipped_installs = True
                self.logger.info(
                    "warm-cache: test.sh still has install commands — will re-commit after verifier completes"
                )
        except Exception as exc:
            self.logger.warning("warm-cache: test.sh patch failed (non-fatal): %s", exc)

    # ── exec override ─────────────────────────────────────────────────────────
    # Intercepts `chmod +x test.sh` for warm-cache patching and propagates
    # CancelledError cleanly.
    #
    # NOTE: We intentionally do NOT send kill -9 -1 here on CancelledError.
    # Harbor's base compose runs `command: ["sh", "-c", "sleep infinity"]` so
    # PID 1 = sh, PID 2 = sleep infinity.  Sending kill -9 -1 kills sleep
    # infinity, sh sees its child exit and itself exits, the container dies, and
    # the subsequent verifier run finds a dead container.
    #
    # Agent-command cleanup is handled by HarborSandbox.exec() which wraps
    # every command in setsid and kills only that process group on timeout.
    # Verifier cleanup happens via stop() → docker compose down immediately
    # after the verifier timeout fires.

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict | None = None,
        timeout_sec: int | None = None,
        user=None,
    ):
        # Intercept `chmod +x test.sh` — issued by harbor's verifier just before
        # running test.sh.  If this is a warm-image run, patch test.sh in-place
        # to skip installation commands that are already satisfied.
        if "chmod" in command and "test.sh" in command:
            await self._patch_test_sh_for_warm_cache()

        try:
            return await super().exec(command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user)
        except asyncio.CancelledError:
            self.logger.warning("exec() cancelled — timeout enforced by caller")
            # Spawn a background watchdog: if the container is still alive 60s
            # from now, kill the stuck process group via direct docker exec.
            # This handles the case where docker-compose-exec is hung on a full
            # pipe (e.g. agent submitted code with an infinite-loop printf) and
            # harbor's own kill-exec is therefore also blocked.
            try:
                task = asyncio.get_running_loop().create_task(
                    self._exec_cancel_watchdog(grace_sec=60.0)
                )
                self._watchdog_tasks.add(task)
                task.add_done_callback(self._watchdog_tasks.discard)
            except RuntimeError:
                pass  # no running loop (shouldn't happen, but be safe)
            raise

    async def _verifier_completed(self) -> bool:
        """Return True if reward.txt was written — verifier ran to completion."""
        try:
            # timeout_sec: file-existence check, should be < 1s; 10s guards
            # against a hung docker compose exec in DinD environments.
            result = await self.exec(
                "test -f /logs/verifier/reward.txt && echo YES || echo NO",
                timeout_sec=10,
            )
            return "YES" in (result.stdout or "")
        except Exception:
            return False

    # ── lifecycle overrides ───────────────────────────────────────────────────

    async def start(self, force_build: bool) -> None:
        if not force_build and await self._warm_image_exists():
            self.logger.info("warm-cache: using %s (skipping build)", self._warm_image)
            self._warm_image_preexisted = True
            # Point task config and env-vars at the warm image so the parent's
            # start() selects the prebuilt compose path and the right image name.
            self.task_env_config.docker_image = self._warm_image
            self._env_vars.prebuilt_image_name = self._warm_image
            # Warm image has all packages pre-installed — no external downloads
            # needed, so proxy env vars serve no purpose and may slow commands.
            for _var in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
                         "http_proxy", "https_proxy", "no_proxy"):
                self._persistent_env.pop(_var, None)
        await super().start(force_build)
        # Cache container ID now while docker compose is healthy — the watchdog
        # uses this to issue plain `docker exec` without going through compose.
        try:
            self._container_id = await self._get_container_id()
        except Exception:
            pass
        # Snapshot /app immediately after the container is up and before the
        # agent touches anything — used by stop() to clean up before committing.
        try:
            await self._snapshot_app()
        except Exception as exc:
            self.logger.warning("warm-cache: /app snapshot raised %s", exc)

    async def stop(self, delete: bool) -> None:
        # Cancel any pending watchdog tasks — the container is being torn down.
        for task in list(self._watchdog_tasks):
            task.cancel()
        self._watchdog_tasks.clear()

        # Only commit when the verifier ran to completion (reward.txt written).
        # This prevents committing a "dirty" image from a trial that was
        # cancelled or timed out before the verifier finished installing deps.
        verifier_ran = await self._verifier_completed()

        if not self._warm_image_preexisted:
            # First run: commit only if verifier completed.
            if verifier_ran:
                try:
                    await self._commit_to_warm_image()
                except Exception as exc:
                    self.logger.warning("warm-cache: commit raised %s — continuing", exc)
            else:
                self.logger.info(
                    "warm-cache: verifier did not complete — skipping commit of %s",
                    self._warm_image,
                )
        else:
            # Warm image already existed.  Re-commit if this run actually
            # executed install commands (patcher couldn't skip them all) AND
            # the verifier completed — the image now has fresher deps.
            if verifier_ran and self._patcher_had_unskipped_installs:
                self.logger.info(
                    "warm-cache: upgrading %s — installs ran during verifier",
                    self._warm_image,
                )
                try:
                    await self._commit_to_warm_image()
                except Exception as exc:
                    self.logger.warning("warm-cache: upgrade commit raised %s — continuing", exc)
            else:
                self.logger.info("warm-cache: %s up-to-date — skipping re-commit", self._warm_image)

        # When delete=True, harbor runs `docker compose down --rmi all` which
        # would delete the prebuilt image referenced by the compose file — i.e.
        # our freshly committed warm image.  Switch back to the build path so
        # compose's --rmi all targets the built image (hb__<task>) instead.
        if delete and self._use_prebuilt:
            self._use_prebuilt = False
        await super().stop(delete)
