"""Create or update (idempotent by display name) and publish the Lakeview dashboard (manual, local only).

Reads DATABRICKS_SERVER_HOSTNAME, DATABRICKS_HTTP_PATH and DATABRICKS_TOKEN from the environment (and a git-ignored
.env if present). The warehouse id is taken from the HTTP path. One dataset per query in sql/dashboard_queries.sql.
The token is never printed.

  python scripts/create_dashboard.py
"""

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TITLE = "SIMULATED DATA: Defect Inspection Monitor"
REQUIRED = ("DATABRICKS_SERVER_HOSTNAME", "DATABRICKS_HTTP_PATH", "DATABRICKS_TOKEN")


def load_dotenv(path: Path) -> None:
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


def parse_queries(text: str) -> list[tuple[str, str]]:
    """Split on '-- name:' markers -> [(name, sql)], dropping comment lines and trailing semicolons."""
    out = []
    for chunk in re.split(r"^-- name:\s*", text, flags=re.M)[1:]:
        header, _, body = chunk.partition("\n")
        name = header.split()[0]
        sql = "\n".join(ln for ln in body.splitlines() if not ln.strip().startswith("--")).strip().rstrip(";")
        out.append((name, sql))
    return out


def dataset(name: str, sql: str) -> dict:
    d = {"name": name[:40], "displayName": name, "queryLines": [ln + "\n" for ln in sql.splitlines()]}
    if ":bucket_seconds" in sql:
        d["parameters"] = [{
            "displayName": "bucket_seconds", "keyword": "bucket_seconds", "dataType": "INTEGER",
            "defaultSelection": {"values": {"dataType": "INTEGER", "values": [{"value": "10"}]}},
        }]
    return d


def widget(wid: str, ds: str, kind: str, title: str, fields: list[str], x: int, y: int, w: int, h: int,
           x_field: str = "bucket") -> dict:
    query_fields = [{"name": f, "expression": f"`{f}`"} for f in fields]
    if kind == "counter":
        spec = {"version": 2, "widgetType": "counter",
                "encodings": {"value": {"fieldName": fields[0], "displayName": fields[0]}},
                "frame": {"showTitle": True, "title": title}}
    else:
        ys = [{"fieldName": f, "displayName": f} for f in fields if f != x_field]
        y_enc = ({"fieldName": ys[0]["fieldName"], "scale": {"type": "quantitative"}, "displayName": ys[0]["displayName"]}
                 if len(ys) == 1 else {"scale": {"type": "quantitative"}, "fields": ys})
        spec = {"version": 3, "widgetType": kind,
                "encodings": {"x": {"fieldName": x_field, "scale": {"type": "temporal"}, "displayName": x_field},
                              "y": y_enc},
                "frame": {"showTitle": True, "title": title}}
    return {"widget": {"name": wid,
                       "queries": [{"name": "main_query",
                                    "query": {"datasetName": ds[:40], "fields": query_fields, "disaggregated": True}}],
                       "spec": spec},
            "position": {"x": x, "y": y, "width": w, "height": h}}


def build_dashboard(queries: list[tuple[str, str]]) -> dict:
    names = {n.split("_", 1)[0]: n for n, _ in queries}
    n = lambda k: names[k]  # noqa: E731
    p = "[SIMULATED] "
    layout = [
        widget("w_events_total", n("0"), "counter", p + "Total events", ["events"], 0, 0, 2, 3),
        widget("w_events", n("1"), "bar", p + "Events over time", ["bucket", "events"], 2, 0, 4, 6),
        widget("w_conf", n("2"), "line", p + "Mean max confidence by bucket", ["bucket", "mean_max_confidence"], 0, 6, 3, 6),
        widget("w_det", n("3"), "line", p + "Detection count (mean per image)", ["bucket", "mean_detections_per_image"], 3, 6, 3, 6),
        widget("w_lat", n("4"), "line", p + "Latency p50 / p95 (ms)", ["bucket", "p50_ms", "p95_ms"], 0, 12, 3, 6),
        widget("w_sharp", n("6"), "line", p + "Sharpness by bucket", ["bucket", "sharpness"], 3, 12, 3, 6),
        widget("w_noise", n("6"), "line", p + "Noise estimate by bucket", ["bucket", "noise_est"], 0, 18, 3, 6),
        widget("w_contrast", n("6"), "line", p + "Contrast std by bucket", ["bucket", "contrast_std"], 3, 18, 3, 6),
    ]
    return {"datasets": [dataset(nm, sql) for nm, sql in queries],
            "pages": [{"name": "main", "displayName": "Monitor", "pageType": "PAGE_TYPE_CANVAS", "layout": layout}]}


def main() -> int:
    load_dotenv(ROOT / ".env")
    missing = [k for k in REQUIRED if not os.environ.get(k)]
    if missing:
        print(f"missing {', '.join(missing)} (set them or fill in .env).")
        return 1
    token = os.environ["DATABRICKS_TOKEN"]
    host = "https://" + os.environ["DATABRICKS_SERVER_HOSTNAME"].removeprefix("https://").rstrip("/")
    warehouse_id = os.environ["DATABRICKS_HTTP_PATH"].rstrip("/").rsplit("/", 1)[-1]
    queries = parse_queries((ROOT / "sql" / "dashboard_queries.sql").read_text(encoding="utf-8"))
    payload = json.dumps(build_dashboard(queries))
    try:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.dashboards import Dashboard

        w = WorkspaceClient(host=host, token=token)
        existing = [d for d in w.lakeview.list() if d.display_name == TITLE]
        dash = Dashboard(display_name=TITLE, warehouse_id=warehouse_id, serialized_dashboard=payload)
        if existing:
            created = w.lakeview.update(existing[0].dashboard_id, dash)
            print(f"updated existing dashboard {created.dashboard_id}")
        else:
            created = w.lakeview.create(dash)
            print(f"created dashboard {created.dashboard_id}")
        w.lakeview.publish(created.dashboard_id, warehouse_id=warehouse_id, embed_credentials=True)
        print(f"published: {host}/dashboardsv3/{created.dashboard_id}/published")
    except Exception as exc:
        print(f"create_dashboard FAILED: {type(exc).__name__}: {str(exc).replace(token, '***')[:1500]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
