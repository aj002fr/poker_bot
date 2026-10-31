"""Isolation negative tests (spec 7 and 12.3): a probe bot tries network, filesystem,
process, signal and IPC access from inside a real worker and reports what happened."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from arena.agents.worker import WorkerSeat
from tests.agents.conftest import bot, info_for, needs_sandbox, small_config

pytestmark = needs_sandbox

REPO = Path(__file__).resolve().parents[2]

PROBE = r'''
import ctypes, errno, os, socket, subprocess, threading

PARENT_PID = {parent_pid}
TARGETS = {targets}
OWN = os.path.dirname(os.path.abspath(__file__))


def attempt(fn):
    try:
        fn()
        return "allowed"
    except BaseException as exc:
        return "denied:" + type(exc).__name__


def read(path):
    if os.path.isdir(path):
        os.listdir(path)
    else:
        with open(path, "rb") as f:
            f.read(1)


def write(path):
    with open(path, "wb") as f:
        f.write(b"x")


def threads():
    started, stop = 0, threading.Event()
    try:
        for _ in range(64):
            threading.Thread(target=stop.wait, daemon=True).start()
            started += 1
    except RuntimeError:
        pass
    stop.set()
    return started


def sysv_shm():
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.shmget(0, 4096, 0o1000 | 0o600) < 0:
        raise OSError(ctypes.get_errno(), "shmget")


from arena.agents.sdk import ExplicitStateBot, check_or_fold


class Probe(ExplicitStateBot):
    name = "probe"
    rulesets = ("practice_nlhe",)

    def initial_state(self, table):
        r = {{}}
        r["socket_inet"] = attempt(lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM))
        r["socket_inet6"] = attempt(lambda: socket.socket(socket.AF_INET6, socket.SOCK_DGRAM))
        r["socket_unix"] = attempt(lambda: socket.socket(socket.AF_UNIX))
        r["connect_loopback"] = attempt(lambda: socket.create_connection(("127.0.0.1", 9), timeout=0.1))
        r["dns"] = attempt(lambda: socket.getaddrinfo("example.com", 80))
        for key, path in TARGETS["read_denied"].items():
            r["read:" + key] = attempt(lambda p=path: read(p))
        for key, path in TARGETS["write_denied"].items():
            r["write:" + key] = attempt(lambda p=path.replace("__OWN__", OWN): write(p))
        r["read:own_bundle"] = attempt(lambda: read(OWN))
        r["fork"] = attempt(os.fork)
        r["subprocess"] = attempt(lambda: subprocess.run(["/bin/true"], check=True))
        r["exec"] = attempt(lambda: os.execv("/bin/true", ["true"]))
        r["kill_parent"] = attempt(lambda: os.kill(PARENT_PID, 0))
        r["proc_parent"] = attempt(lambda: read(f"/proc/{{PARENT_PID}}/status"))
        r["sysv_shm"] = attempt(sysv_shm)
        r["posix_shm"] = attempt(lambda: write("/dev/shm/arena-probe"))
        r["threads_started"] = threads()
        r["uid"] = os.getuid()
        r["pid"] = os.getpid()
        r["env"] = sorted(os.environ)
        return r

    def decide(self, obs, state):
        return check_or_fold(obs.legal), state


def make_bot():
    return Probe()
'''


@pytest.fixture(scope="module")
def probe_results(pool, tmp_path_factory):
    other = pool.stage(bot("calling_station"))
    secret = tmp_path_factory.mktemp("host") / "secret.txt"
    secret.write_text("host secret")
    targets = {
        "read_denied": {
            "etc_passwd": "/etc/passwd",
            "repo_readme": str(REPO / "README.md"),
            "repo_source": str(REPO / "arena" / "table" / "service.py"),
            "home": os.path.expanduser("~"),
            "host_tmp_file": str(secret),
            "other_bot_bundle": str(other.root),
            "staging_root": str(pool.staging_root),
        },
        "write_denied": {
            "tmp": "/tmp/arena-probe-write",
            "own_bundle": "__OWN__/x",
            "dev_shm": "/dev/shm/arena-probe-write",
        },
    }
    src = tmp_path_factory.mktemp("probe") / "probe_bot.py"
    src.write_text(PROBE.format(parent_pid=os.getpid(), targets=json.dumps(targets)))
    artifact = pool.stage(src)
    config = small_config(3)
    seat = WorkerSeat(0, artifact, info_for(config, 0), pool.acquire, load_timeout_ns=30_000_000_000)
    seat.start()
    report = seat.channel.sandbox_report
    results = json.loads(seat.committed)
    seat.close()
    return results, report, artifact


@pytest.mark.parametrize("key", ["socket_inet", "socket_inet6", "socket_unix", "connect_loopback", "dns"])
def test_network_denied(probe_results, key):
    assert probe_results[0][key].startswith("denied")


def test_network_namespace_has_only_loopback(probe_results):
    assert probe_results[1]["namespaces"]["net_interfaces"] == ["lo"]


@pytest.mark.parametrize("key", ["etc_passwd", "repo_readme", "repo_source", "home", "host_tmp_file",
                                 "other_bot_bundle", "staging_root"])
def test_filesystem_reads_denied(probe_results, key):
    assert probe_results[0]["read:" + key].startswith("denied")


@pytest.mark.parametrize("key", ["tmp", "own_bundle", "dev_shm"])
def test_filesystem_writes_denied(probe_results, key):
    assert probe_results[0]["write:" + key].startswith("denied")


def test_own_bundle_is_readable(probe_results):
    assert probe_results[0]["read:own_bundle"] == "allowed"


@pytest.mark.parametrize("key", ["fork", "subprocess", "exec", "kill_parent", "proc_parent", "sysv_shm", "posix_shm"])
def test_process_signal_and_ipc_access_denied(probe_results, key):
    assert probe_results[0][key].startswith("denied")


def test_thread_and_identity_limits(probe_results):
    results, report, _ = probe_results
    assert 0 < results["threads_started"] <= report["limits"]["threads"]
    assert results["uid"] != 0 and results["uid"] == os.getuid()
    assert results["pid"] == 1 and report["namespaces"]["visible_pids"] == 1
    assert set(results["env"]) <= {"PATH", "LANG", "PYTHONHASHSEED", "LC_CTYPE"}


def test_sandbox_layers_reported(probe_results):
    report = probe_results[1]
    assert report["seccomp"] is True and report["landlock_abi"] >= 1
    assert report["limits"]["file_size_bytes"] == 0


def test_staged_bundle_is_read_only_and_hashed(probe_results):
    artifact = probe_results[2]
    assert len(artifact.sha256) == 64 and artifact.root.name == artifact.sha256
    assert not os.access(artifact.entry, os.W_OK) or os.getuid() == 0
