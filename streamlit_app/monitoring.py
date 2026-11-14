"""Page 2: the monitoring dashboard on the committed SIMULATED sample snapshot."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402
from app.dashboard_view import render  # noqa: E402
from src import dashboard_data as dd  # noqa: E402


@st.cache_data(show_spinner="Reading the sample snapshot...")
def sample():
    return dd.load_local(str(core.ROOT / dd.SAMPLE_PATH))


st.error(core.SIM_BANNER, icon="⚠️")
st.warning(core.BANNER, icon="ℹ️")
st.title("Defect inspection monitor")
size = st.sidebar.slider("Events per bucket", 1, 50, 25)
try:
    df = sample()
except Exception as exc:
    st.error(f"Could not load the sample snapshot: {type(exc).__name__}: {str(exc)[:300]}")
    st.stop()
if df.empty:
    st.info("No simulated events in the snapshot.")
    st.stop()
render(df, size)
