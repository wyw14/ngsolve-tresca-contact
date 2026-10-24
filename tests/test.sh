#!/bin/bash
set -euo pipefail
mkdir -p /logs/verifier
chmod 700 /logs/verifier /tests
if [ -d /solution ]; then chmod 700 /solution; fi
printf '0\n' > /logs/verifier/reward.txt
trap 'status=$?; if [ "$status" -ne 0 ]; then printf "0\n" > /logs/verifier/reward.txt; fi' EXIT
python -m pip install --no-index --find-links=/tests/wheels --require-hashes -r /tests/requirements.lock.txt
cd /tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tests \
    python -m pytest -q -rA --tb=short -p no:cacheprovider --confcutdir=/tests --rootdir=/tests \
    --junitxml=/logs/verifier/junit.xml /tests
printf '1\n' > /logs/verifier/reward.txt
