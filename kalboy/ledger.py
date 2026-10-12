"""Log of every walk-forward run, used to count trials for the deflated Sharpe."""
import json
import os
from datetime import datetime, timezone

LEDGER_PATH = os.path.join('results', 'ledger.jsonl')


def append(record, path=LEDGER_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    record = {'logged_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'), **record}
    with open(path, 'a') as f:
        f.write(json.dumps(record, default=str) + '\n')


def read(path=LEDGER_PATH):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def is_trial(r):
    return r.get('segment') == 'dev' and r.get('purpose', 'trial') == 'trial' and r.get('daily_sharpe') is not None


def dev_trials(data_dir=None, path=LEDGER_PATH):
    """Daily Sharpe of each distinct dev config, keyed by fingerprint."""
    return {r['fingerprint']: r['daily_sharpe'] for r in read(path)
            if is_trial(r) and (data_dir is None or r.get('data_dir') == data_dir)}


def holdout_looks(data_dir, path=LEDGER_PATH):
    return [r for r in read(path) if r.get('segment') == 'holdout' and r.get('data_dir') == data_dir]
