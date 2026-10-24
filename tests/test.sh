#!/bin/bash
set -euo pipefail
mkdir -p /logs/verifier
chmod 700 /logs/verifier /tests
if [ -d /solution ]; then chmod 700 /solution; fi
rm -f /logs/verifier/reward.txt /logs/verifier/reward.json /logs/verifier/junit.xml /logs/verifier/score-report.json
printf '{"reward":0.0}\n' > /logs/verifier/reward.json
trap 'status=$?; if [ "$status" -ne 0 ]; then printf "{\"reward\":0.0}\n" > /logs/verifier/reward.json; fi' EXIT
python -m pip install --no-index --find-links=/tests/wheels --require-hashes -r /tests/requirements.lock.txt
cd /tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tests \
    python /tests/score.py
