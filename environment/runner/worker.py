"""Native NGSolve reconstruction and protected history solve with phases."""
import contextlib
import importlib
import json
import os
import sys
import traceback

sys.path.insert(0, '/runner')
from codec import encode, encode_output, decode


def snapshot(request):
    return json.dumps(encode(request), sort_keys=True, allow_nan=False)


def main():
    meta = {'wire_version': 1, 'worker_uid': os.geteuid()}
    stage = 'request_decode'
    try:
        raw = json.loads(sys.stdin.read())
        with contextlib.redirect_stdout(sys.stderr):
            stage = 'native_construct'
            request = decode(raw)
            stage = 'input_snapshot'
            before = snapshot(request)
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        return {**meta, 'ok': False, 'stage': stage, 'input_measured': False,
                'error': type(error).__name__, 'message': str(error)}
    failure = None
    with contextlib.redirect_stdout(sys.stderr):
        stage = 'candidate_import'
        try:
            sys.path.insert(0, '/app/src')
            module = importlib.import_module('tresca_contact')
            stage = 'candidate_invoke'
            result = getattr(module, request['function'])(*request['args'], **request['kwargs'])
        except Exception as error:
            failure = error
            traceback.print_exc(file=sys.stderr)
        finally:
            try:
                unchanged = before == snapshot(request)
            except Exception:
                unchanged = False
    meta.update(stage=stage, input_measured=True, input_unchanged=unchanged)
    if failure is not None:
        return {**meta, 'ok': False, 'error': type(failure).__name__,
                'is_value_error': isinstance(failure, ValueError), 'message': str(failure)}
    try:
        answer = {**meta, 'ok': True, 'stage': 'completed', 'value': encode_output(result)}
        json.dumps(answer, allow_nan=False)
        return answer
    except Exception:
        return {**meta, 'ok': True, 'stage': 'completed',
                'value': {'__invalid_output__': 'unencodable result'}}


sys.stdout.write(json.dumps(main(), allow_nan=False))
