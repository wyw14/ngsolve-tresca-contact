"""Trusted test-side proxy. Candidate code never imports into the pytest process."""

import builtins
import ctypes
import time
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from codec import encode, decode


def _restrict():
    if os.geteuid() == 0:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(38, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "no_new_privs unavailable")
        os.setgroups([])
        os.setgid(65534)
        os.setuid(65534)


def _cleanup_candidate_uid():
    """This dedicated verifier runs one candidate call at a time."""
    if os.geteuid() != 0:
        return
    for attempt in range(30):
        remaining = []
        for status in Path("/proc").glob("[0-9]*/status"):
            try:
                fields = dict(
                    line.split(":", 1)
                    for line in status.read_text().splitlines()
                    if ":" in line
                )
                uid = int(fields["Uid"].split()[0])
                state = fields["State"].strip().split()[0]
                if uid == 65534:
                    remaining.append((int(status.parent.name), state))
            except (OSError, ValueError, KeyError):
                continue
        if not remaining:
            return
        for pid, state in remaining:
            if state != "Z":
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        time.sleep(0.05)
    raise RuntimeError("Candidate processes were not fully reaped")


def call(function, *args, **kwargs):
    timeout = float(os.environ.get("CANDIDATE_CALL_TIMEOUT", "180"))
    payload = json.dumps(
        encode(dict(function=function, args=args, kwargs=kwargs)), allow_nan=False
    )
    with tempfile.TemporaryDirectory(prefix="candidate-") as temp:
        if os.geteuid() == 0:
            os.chown(temp, 65534, 65534)
        process = subprocess.Popen(
            [sys.executable, "-I", "/runner/worker.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=temp,
            start_new_session=True,
            preexec_fn=_restrict,
            env={
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "HOME": temp,
                "MPLCONFIGDIR": temp,
                "TMPDIR": temp,
                "LANG": "C.UTF-8",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
        try:
            stdout, stderr = process.communicate(payload, timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate(timeout=5)
            _cleanup_candidate_uid()
            raise TimeoutError(f"candidate exceeded {timeout}s")
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            _cleanup_candidate_uid()
        if process.returncode:
            raise RuntimeError(
                f"candidate process failed {process.returncode}: {stderr[-3000:]}"
            )
        response = json.loads(stdout)
        if not response["ok"]:
            kind = response["error"]
            cls = getattr(builtins, kind, RuntimeError)
            if not isinstance(cls, type) or not issubclass(cls, Exception):
                cls = RuntimeError
            raise cls(response["message"])
        return decode(response["value"])
