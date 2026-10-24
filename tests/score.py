"""Run the complete trusted test suite and sum frozen behavior weights."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from fractions import Fraction
import sys
import uuid

import pytest
import protocol


def read_manifest(path):
    source = path.read_bytes()
    manifest = json.loads(source)
    groups = manifest['groups']
    if not groups or len({g['id'] for g in groups}) != len(groups):
        raise ValueError('Scoring groups must have unique identities')
    weights = [Fraction(g['weight']) for g in groups]
    if any(w <= 0 for w in weights) or sum(weights) != 1:
        raise ValueError('Positive group weights must sum exactly to one')
    nodeids, intent, measurements = [], {}, []
    for group in groups:
        if not group.get('basis'):
            raise ValueError('Each weight requires a behavior basis')
        if group.get('measurement') == 'input_preservation':
            measurements.append(group['measurement'])
            if group.get('nodes'):
                raise ValueError('Input-preservation measurement has no pytest nodes')
            continue
        nodes = group['nodes']
        if not nodes:
            raise ValueError('A test group needs its frozen node identities')
        for node in nodes:
            nodeids.append(node)
            intent[node] = group.get('intent', 'scientific')
            if intent[node] not in ('scientific', 'rejection'):
                raise ValueError('A unit has an unsupported call intent')
    if len(nodeids) != len(set(nodeids)):
        raise ValueError('A test node cannot receive weight twice')
    if len(measurements) != len(set(measurements)):
        raise ValueError('A measured behavior cannot receive weight twice')
    files = manifest.get('test_files', ['test_outputs.py'])
    if (not isinstance(files, list) or not files or len(files) != len(set(files))
            or any(not isinstance(name, str) or not name.endswith('.py')
                   or Path(name).is_absolute() or '..' in Path(name).parts for name in files)):
        raise ValueError('The suite requires fixed relative Python test files')
    return manifest, intent, hashlib.sha256(source).hexdigest()


class PhaseReport:
    def __init__(self, intent):
        self.intent = intent
        self.collected = []
        self.reports = []
        self.finished = False
        self.exitstatus = None

    def pytest_collection_finish(self, session):
        self.collected = [item.nodeid for item in session.items]

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_setup(self, item):
        protocol.set_context(item.nodeid, self.intent.get(item.nodeid, 'unknown'))

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        outcome = yield
        report = outcome.get_result()
        origin, exception_type = None, None
        if call.excinfo is not None:
            error = call.excinfo.value
            exception_type = type(error).__name__
            if isinstance(error, protocol.CandidateExecutionFault):
                origin = 'candidate_execution_fault'
            elif isinstance(error, protocol.CandidateInfrastructureFault):
                origin = 'trusted_infrastructure_fault'
            elif isinstance(error, (protocol.CandidateRejection, protocol.CandidateWrongException)):
                origin = 'candidate_rejection'
            elif isinstance(error, protocol.CandidateOutputMismatch):
                origin = 'candidate_output_mismatch'
            else:
                origin = 'test_assertion_or_local_comparison'
        self.reports.append({'nodeid': item.nodeid, 'phase': report.when,
            'outcome': report.outcome, 'origin': origin,
            'exception_type': exception_type, 'xfail': hasattr(report, 'wasxfail')})

    def pytest_sessionfinish(self, session, exitstatus):
        self.finished = True
        self.exitstatus = int(exitstatus)


def grade(manifest, collected, phases, calls, *, completed, exitstatus):
    """Pure aggregation. Synthetic inputs to this function are unit tests only."""
    expected = [node for g in manifest['groups'] for node in g.get('nodes', [])]
    invalid, faults = [], []
    if not completed or exitstatus not in (0, 1):
        invalid.append('Test session did not complete with an ordinary test exit')
    if len(collected) != len(expected) or set(collected) != set(expected):
        invalid.append('Collected identities differ from the frozen scoring set')
    reports = {node: [] for node in expected}
    for report in phases:
        if report['nodeid'] not in reports:
            invalid.append('Unplanned test report')
            continue
        if report['phase'] not in ('setup', 'call', 'teardown') or report['outcome'] not in ('passed', 'failed', 'skipped'):
            invalid.append('Malformed test phase or outcome')
            continue
        reports[report['nodeid']].append(report)
    statuses = {}
    for node, items in reports.items():
        byphase = {phase: [r for r in items if r['phase'] == phase]
                   for phase in ('setup', 'call', 'teardown')}
        if any(r['outcome'] == 'skipped' or r.get('xfail') for r in items):
            invalid.append('Required test was skipped or xfailed: ' + node)
        if len(byphase['setup']) != 1 or len(byphase['teardown']) != 1:
            invalid.append('Missing or repeated setup/teardown: ' + node)
            statuses[node] = 'UNCOMPLETED'
            continue
        setup, teardown = byphase['setup'][0], byphase['teardown'][0]
        if teardown['outcome'] != 'passed':
            invalid.append('Required teardown failed: ' + node)
        if setup['outcome'] != 'passed':
            if byphase['call']:
                invalid.append('A call report followed a failed setup: ' + node)
            if setup.get('origin') == 'candidate_execution_fault':
                faults.append('Candidate invocation failed during fixture setup: ' + node)
            elif setup.get('origin') not in ('candidate_rejection', 'candidate_output_mismatch'):
                invalid.append('Trusted fixture setup failed: ' + node)
            statuses[node] = 'FAIL'
            continue
        if len(byphase['call']) != 1:
            invalid.append('Missing or repeated call report: ' + node)
            statuses[node] = 'UNCOMPLETED'
            continue
        call = byphase['call'][0]
        if call.get('origin') == 'candidate_execution_fault':
            faults.append('Candidate execution fault: ' + node)
        if call.get('origin') == 'trusted_infrastructure_fault':
            invalid.append('Trusted invocation infrastructure failed: ' + node)
        statuses[node] = 'PASS' if call['outcome'] == 'passed' else 'FAIL'
    for call in calls:
        if call.get('fault') == 'candidate_execution':
            faults.append('Observed candidate execution fault at call ' + str(call.get('nodeid')))
        elif call.get('fault') == 'infrastructure':
            invalid.append('Observed trusted invocation infrastructure failure')
    earned, units, groups = Fraction(0), [], []
    for group in manifest['groups']:
        weight = Fraction(group['weight'])
        if group.get('measurement') == 'input_preservation':
            measured = [c for c in calls if 'input_unchanged' in c]
            if (not measured or len(measured) != len(calls)) and not faults:
                invalid.append('Input-preservation evidence is incomplete')
            passed = bool(measured) and len(measured) == len(calls) and all(c['input_unchanged'] is True for c in measured)
            status = 'PASS' if passed else 'FAIL'
            units.append({'id': group['id'], 'status': status, 'weight': str(weight)})
            subtotal = weight if passed else Fraction(0)
        else:
            nodes = group['nodes']
            each = weight / len(nodes)
            subtotal = sum((each for node in nodes if statuses[node] == 'PASS'), Fraction(0))
            units.extend({'id': node, 'group': group['id'], 'status': statuses[node],
                          'weight': str(each)} for node in nodes)
        earned += subtotal
        groups.append({'id': group['id'], 'weight': str(weight), 'earned': str(subtotal)})
    return {'valid_verification': not invalid, 'runtime_fault': bool(faults),
        'invalid_reasons': sorted(set(invalid)), 'candidate_faults': sorted(set(faults)),
        'earned_before_fault_policy': str(earned),
        'reward_fraction': '0' if invalid or faults else str(earned),
        'reward': 0.0 if invalid or faults else float(earned), 'groups': groups, 'units': units}


def atomic_json(path, data):
    temporary = path.with_name(path.name + '.new')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def main():
    logs = Path('/logs/verifier')
    logs.mkdir(parents=True, exist_ok=True)
    for name in ['reward.json', 'reward.txt', 'junit.xml', 'score-report.json']:
        (logs / name).unlink(missing_ok=True)
    atomic_json(logs / 'reward.json', {'reward': 0.0})
    run_id = str(uuid.uuid4())
    try:
        manifest, intent, manifest_sha = read_manifest(Path('/tests/score_manifest.json'))
        protocol.calls.clear()
        reporter = PhaseReport(intent)
        code = int(pytest.main(['-q', '-rA', '--tb=short', '-p', 'no:cacheprovider',
            '--confcutdir=/tests', '--rootdir=/tests',
            '--junitxml=/logs/verifier/junit.xml',
            *[str(Path('/tests') / name) for name in manifest.get('test_files', ['test_outputs.py'])]], plugins=[reporter]))
        result = grade(manifest, reporter.collected, reporter.reports, protocol.calls,
                       completed=reporter.finished and reporter.exitstatus == code, exitstatus=code)
        result.update(run_id=run_id, manifest_sha256=manifest_sha, pytest_exitstatus=code,
            collected=reporter.collected, phases=reporter.reports, calls=protocol.calls)
        atomic_json(logs / 'score-report.json', result)
        atomic_json(logs / 'reward.json', {'reward': result['reward']})
        return 0 if result['valid_verification'] else 1
    except Exception as error:
        atomic_json(logs / 'score-report.json', {'run_id': run_id, 'reward': 0.0,
            'valid_verification': False, 'invalid_reasons': ['Scoring/setup failure: ' + type(error).__name__]})
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
