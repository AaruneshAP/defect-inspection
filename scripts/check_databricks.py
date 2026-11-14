"""Databricks connectivity check (manual, local only; never run in CI or tests).

Reads DATABRICKS_SERVER_HOSTNAME, DATABRICKS_HTTP_PATH and DATABRICKS_TOKEN from the environment (and a git-ignored
.env if present), runs SELECT 1 and creates the schema defect_inspection if missing. The token is never printed.

  python scripts/check_databricks.py
"""

import os
import sys
from pathlib import Path

REQUIRED = ("DATABRICKS_SERVER_HOSTNAME", "DATABRICKS_HTTP_PATH", "DATABRICKS_TOKEN")
SCHEMA = "defect_inspection"


def load_dotenv(path: Path) -> None:
    """Minimal KEY=VALUE parser; real environment variables win over the file."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("'\"")
        if value:
            os.environ.setdefault(key.strip(), value)


def main() -> int:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    missing = [k for k in REQUIRED if not os.environ.get(k)]
    if missing:
        print(f"Databricks check skipped: missing {', '.join(missing)} (set them or fill in .env).")
        return 0
    token = os.environ["DATABRICKS_TOKEN"]
    host = os.environ["DATABRICKS_SERVER_HOSTNAME"].removeprefix("https://").rstrip("/")
    try:
        from databricks import sql
    except ImportError:
        print("databricks-sql-connector is not installed; run: pip install -r requirements-dev.txt")
        return 1
    try:
        with sql.connect(server_hostname=host, http_path=os.environ["DATABRICKS_HTTP_PATH"],
                         access_token=token) as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
            print(f"connected to {host}: SELECT 1 -> {cur.fetchone()[0]}")
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
            print(f"schema {SCHEMA}: exists or created")
    except Exception as exc:  # report the error type and message with the token scrubbed
        print(f"Databricks check FAILED: {type(exc).__name__}: {str(exc).replace(token, '***')[:500]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
