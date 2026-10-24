"""Trusted test-side proxy. Candidate code never imports into the pytest process."""

import ctypes
import time
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from codec import encode, decode_output


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
    """Run one protected invocation; only trusted markers affect fault policy."""
    from protocol import (CandidateExecutionFault, CandidateInfrastructureFault,
                          CandidateRejection, CandidateWrongException,
                          CandidateOutputMismatch, context, record_call)
    observation = {'function': function, 'stage': 'trusted_preparation'}
    try:
        timeout = float(os.environ.get('CANDIDATE_CALL_TIMEOUT', '180'))
        payload = json.dumps(encode(dict(function=function, args=args, kwargs=kwargs)), allow_nan=False)
        with tempfile.TemporaryDirectory(prefix='candidate-') as temp:
            if os.geteuid() == 0:
                os.chown(temp, 65534, 65534)
            process = subprocess.Popen(
                [sys.executable, '-I', '/runner/worker.py'], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                cwd=temp, start_new_session=True, preexec_fn=_restrict,
                env={'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': temp,
                     'MPLCONFIGDIR': temp, 'TMPDIR': temp, 'LANG': 'C.UTF-8',
                     'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
                     'PYTHONDONTWRITEBYTECODE': '1'})
            observation['stage'] = 'worker_process'
            try:
                try:
                    stdout, stderr = process.communicate(payload, timeout=timeout)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.communicate(timeout=5)
                    observation['stage'] = 'candidate_timeout'
                    raise CandidateExecutionFault(f'candidate exceeded {timeout}s')
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    _cleanup_candidate_uid()
                except Exception as error:
                    raise CandidateInfrastructureFault('Candidate cleanup failed') from error
            observation['returncode'] = process.returncode
            if process.returncode:
                raise CandidateExecutionFault(f'candidate process failed {process.returncode}: {stderr[-3000:]}')
            try:
                response = json.loads(stdout)
                assert isinstance(response, dict) and response.get('wire_version') == 1
                assert type(response['ok']) is bool
            except Exception as error:
                raise CandidateExecutionFault('Candidate worker did not complete the response protocol') from error
            observation['stage'] = response.get('stage', 'missing_stage')
            observation['worker_uid'] = response.get('worker_uid')
            if response.get('input_measured') is True and type(response.get('input_unchanged')) is bool:
                observation['input_unchanged'] = response['input_unchanged']
            elif observation['stage'] not in ('request_decode', 'native_construct', 'input_snapshot'):
                raise CandidateInfrastructureFault('Worker input-preservation measurement is missing')
            if observation['stage'] in ('request_decode', 'native_construct', 'input_snapshot'):
                raise CandidateInfrastructureFault('Trusted worker preparation failed: ' + response.get('error', 'unknown'))
            if not response['ok']:
                observation['candidate_exception'] = response.get('error', 'unknown')
                message = response.get('message', '')
                if observation['stage'] == 'candidate_import':
                    raise CandidateExecutionFault('Candidate import failed: ' + message)
                if observation['stage'] != 'candidate_invoke':
                    raise CandidateInfrastructureFault('Unknown worker error phase')
                if response.get('is_value_error') is True:
                    raise CandidateRejection(message)
                if context.get('intent') == 'rejection':
                    raise CandidateWrongException(response.get('error', 'unknown') + ': ' + message)
                raise CandidateExecutionFault('Candidate invocation failed: ' + message)
            if observation['stage'] != 'completed':
                raise CandidateInfrastructureFault('Missing completed worker phase')
            try:
                return decode_output(response['value'])
            except Exception as error:
                raise CandidateOutputMismatch('Returned value cannot be decoded') from error
    except CandidateExecutionFault:
        observation['fault'] = 'candidate_execution'
        raise
    except CandidateInfrastructureFault:
        observation['fault'] = 'infrastructure'
        raise
    except (CandidateRejection, CandidateWrongException, CandidateOutputMismatch):
        raise
    except Exception as error:
        observation['fault'] = 'infrastructure'
        raise CandidateInfrastructureFault('Trusted invocation preparation or transport failed') from error
    finally:
        record_call(**observation)
