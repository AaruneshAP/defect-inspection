"""Streamlit monitoring page for SIMULATED inspection traffic.  Run: streamlit run app/dashboard.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from app.dashboard_view import render  # noqa: E402
from src import dashboard_data as dd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
st.set_page_config(page_title="SIMULATED DATA: Defect Inspection Monitor", layout="wide")


@st.cache_data(show_spinner="Reading local file...")
def _local(path: str):
    return dd.load_local(path)


@st.cache_data(ttl=60, show_spinner="Reading Delta...")
def _delta():
    return dd.load_delta()


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

render(df, size)
