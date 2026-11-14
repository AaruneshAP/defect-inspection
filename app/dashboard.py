"""Streamlit monitoring page for SIMULATED inspection traffic.  Run: streamlit run app/dashboard.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from src import dashboard_data as dd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
st.set_page_config(page_title="SIMULATED DATA: Defect Inspection Monitor", layout="wide")


@st.cache_data(show_spinner="Reading local file...")
def _local(path: str):
    return dd.load_local(path)


@st.cache_data(ttl=60, show_spinner="Reading Delta...")
def _delta():
    return dd.load_delta()


def _mark(fig, drift):
    if drift is not None:
        fig.add_vline(x=drift, line_dash="dash", line_color="firebrick", annotation_text="drift starts")
    return fig.update_layout(margin=dict(t=50, b=30), legend_title_text="")


def _line(data, y, title, drift):
    return _mark(px.line(data, x="bucket", y=y, title=f"[SIMULATED] {title}", labels={"bucket": "event order"}), drift)


st.error("SIMULATED DATA: synthetic traffic from val images with injected drift. Not production data.", icon="⚠️")
st.title("Defect Inspection Monitor")

with st.sidebar:
    st.header("Source")
    source = st.radio("Data source", ["Sample snapshot", "Local file", "Delta (Databricks)"])
    size = st.slider("Events per bucket", 1, 50, 25)
    path = st.text_input("Parquet / SQLite path", "data/inspection/sim.parquet") if source == "Local file" else None

try:
    if source == "Sample snapshot":
        df = _local(str(ROOT / dd.SAMPLE_PATH))
    elif source == "Local file":
        df = _local(path)
    else:
        from src.inspection_log import missing_databricks_vars

        missing = missing_databricks_vars()
        if missing:
            st.warning(f"Delta needs {', '.join(missing)} in the environment or a git-ignored .env.")
            st.stop()
        df = _delta()
except Exception as exc:  # message only; connection errors are already scrubbed of the token
    st.error(f"Could not load data: {type(exc).__name__}: {str(exc)[:300]}")
    st.stop()

if df.empty:
    st.info("No simulated events found in this source.")
    st.stop()

drift = dd.drift_start(df)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Events", len(df))
m2.metric("Perturbed events", int((df["perturbation"] != "none").sum()))
m3.metric("Drift starts at event", "none" if drift is None else drift)
m4.metric("Mean max confidence", f"{df['max_confidence'].mean():.3f}")
st.caption("x-axis = event order (buckets of consecutive events), not the simulated clock. Dashed line = drift start.")

stats = dd.input_statistics(df, size)

row = st.columns(2)
row[0].plotly_chart(_line(dd.mean_max_confidence(df, size), "mean_max_confidence", "Mean max confidence", drift),
                    width="stretch")
row[1].plotly_chart(_line(dd.detection_count(df, size), "mean_detections_per_image",
                          "Detection count (mean per image)", drift), width="stretch")

row = st.columns(2)
row[0].plotly_chart(_line(dd.latency_percentiles(df, size), ["p50_ms", "p95_ms"], "Latency p50 / p95 (ms)", drift),
                    width="stretch")
row[0].caption("The first event (model warm-up) is excluded from the percentiles.")
show_ignore = row[1].checkbox("show ignore", value=False)
row[1].plotly_chart(_mark(px.bar(dd.action_counts(df, size, show_ignore), x="bucket", y="detections", color="action",
                                 title="[SIMULATED] Detections per action", labels={"bucket": "event order"}), drift),
                    width="stretch")
row[1].caption("About 90% of detections are 'ignore' because the model keeps detections down to 0.001 confidence.")

if stats.empty:
    st.info("No input statistics (schema v1 log).")
else:
    row = st.columns(2)
    row[0].plotly_chart(_line(stats, "sharpness", "Sharpness", drift), width="stretch")
    row[1].plotly_chart(_line(stats, "noise_est", "Noise estimate", drift), width="stretch")
    row = st.columns(2)
    row[0].plotly_chart(_line(stats, "contrast_std", "Contrast std", drift), width="stretch")
    row[1].caption("Noise raises sharpness (the Laplacian variance picks it up), and noise_est has a high clean "
                   "baseline on these textures, so neither moves like a clean 'more noise' signal.")

st.subheader("[SIMULATED] By perturbation type")
st.caption("Mixing perturbation types in one bucket makes the trends zig-zag; this table separates them.")
st.dataframe(dd.perturbation_breakdown(df).round(2), hide_index=True, width="stretch")
