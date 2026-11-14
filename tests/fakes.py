"""In-memory stand-in for a databricks-sql-connector connection (no network). Emulates only what DeltaSink issues."""

import re

from src.inspection_log import COLUMNS


class FakeCursor:
    def __init__(self, db):
        self.db, self.description, self._rows = db, None, []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.db.statements.append((sql, params))
        s = " ".join(sql.split())
        if s.startswith("MERGE"):
            assert params, "MERGE must be parameterized"
            assert set(re.findall(r":(\w+)", sql)) == set(params), "markers and parameters differ"
            rows = {}
            for key, value in params.items():
                col, _, idx = key.rpartition("_")
                rows.setdefault(int(idx), {})[col] = value
            assert all(list(r) == COLUMNS or set(r) == set(COLUMNS) for r in rows.values())
            ids = [r["event_id"] for r in rows.values()]
            assert len(ids) == len(set(ids)), "duplicate key inside one MERGE source"
            for r in rows.values():
                self.db.rows.setdefault(r["event_id"], {c: r[c] for c in COLUMNS})  # WHEN NOT MATCHED
        elif s.startswith("SELECT COUNT"):
            self.description, self._rows = [("count",)], [(len(self.db.rows),)]
        elif s.startswith("SELECT"):
            ordered = sorted(self.db.rows.values(), key=lambda r: (r["timestamp"], r["event_id"]))
            if m := re.search(r"LIMIT (\d+)$", s):
                ordered = ordered[:int(m.group(1))]
            self.description = [(c,) for c in COLUMNS]
            self._rows = [tuple(r[c] for c in COLUMNS) for r in ordered]
        else:  # CREATE SCHEMA / CREATE TABLE
            assert s.startswith("CREATE"), s

    def fetchall(self):
        return self._rows


class FakeConnection:
    def __init__(self):
        self.statements, self.rows, self.closed = [], {}, False

    def cursor(self):
        return FakeCursor(self)

    def close(self):
        self.closed = True

    def merges(self):
        return [(s, p) for s, p in self.statements if s.lstrip().startswith("MERGE")]
