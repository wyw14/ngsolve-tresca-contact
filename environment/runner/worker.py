"""Data-only adapter for the native elastic contact interface."""
import contextlib,importlib,json,sys,traceback
sys.path.insert(0,'/runner')
from codec import encode,decode
sys.path.insert(0,'/app/src')
request=decode(json.loads(sys.stdin.read()))
try:
    with contextlib.redirect_stdout(sys.stderr):
        before=json.dumps(encode(request),sort_keys=True,allow_nan=False)
        module=importlib.import_module('tresca_contact');failure=None
        try:result=getattr(module,request['function'])(*request['args'],**request['kwargs'])
        except Exception as error:failure=error
        if before!=json.dumps(encode(request),sort_keys=True,allow_nan=False):
            raise AssertionError('Caller input was modified')
        if failure is not None:raise failure
    answer={'ok':True,'value':encode(result)}
except Exception as error:
    traceback.print_exc(file=sys.stderr)
    answer={'ok':False,'error':type(error).__name__,'message':str(error)}
sys.stdout.write(json.dumps(answer,allow_nan=False))
