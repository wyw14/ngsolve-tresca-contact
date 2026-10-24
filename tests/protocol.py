"""Test-side provenance for calls made through the protected worker."""

class CandidateExecutionFault(RuntimeError):
    """A worker invocation failed beyond the documented rejection channel."""

class CandidateInfrastructureFault(RuntimeError):
    """The trusted adapter could not prepare or clean up an invocation."""

class CandidateRejection(ValueError):
    """A ValueError returned through the documented argument-rejection channel."""

class CandidateWrongException(Exception):
    """An error response of the wrong type for a rejection test."""

class CandidateOutputMismatch(AssertionError):
    """A completed result contains a value the transport cannot encode."""

context = {'nodeid': None, 'intent': 'scientific'}
calls = []

def set_context(nodeid, intent):
    context.update(nodeid=nodeid, intent=intent)

def record_call(**data):
    calls.append({'nodeid': context['nodeid'], 'intent': context['intent'], **data})

