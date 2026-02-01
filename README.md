# Blackbox / Feith Migration POC

This folder contains the Feith migration proof-of-concept script and demo runners.

## Files 

- `feith_migrator.py`: Migration pipeline (extract -> LLM transform -> stage to target DB)
- `feith_migrator.env.example`: environment template
- `FEITH_MIGRATION_README.md`: detailed Wednesday demo playbook
- `wednesday_demo.py`: quick demo runner
- `test_migration.py`: small batch test runner
- `validate_single.py`: transform a single Feith record interactively

## Setup

1. Create a virtualenv

2. Install deps:

`pip install -r requirements.txt`

3. Configure env:

`cp feith_migrator.env.example .env`

## Demo (no Oracle / SQLite source)

If Oracle drivers/connection are not available, `feith_migrator.py` will fall back to a local SQLite source database and seed a few demo records automatically.

Recommended `.env` for local demo:

- `FEITH_DB_SCHEMA=`
- `TARGET_DB_URL=sqlite:///migration_staging.db`
- `LLM_API_URL=http://localhost:8080/v1/completions`

Run:

`python3 wednesday_demo.py`

## Production Notes

- Writeback to Feith is disabled by default (`FEITH_WRITEBACK_ENABLED=false`). Enable only if you have explicit permission.
- LLM connectivity is required for high-confidence transformations; if the LLM call fails, the pipeline will still stage low-confidence fallback records.
