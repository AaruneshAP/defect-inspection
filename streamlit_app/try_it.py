"""Page 1: upload an image or pick an example; boxes coloured by action, action table, per-image decision."""

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402
from core import ACTION_COLORS, TABLE_HEADERS, analyze  # noqa: E402


@st.cache_resource(show_spinner="Downloading the ONNX model (first load only)...")
def load():
    assets = core.fetch_assets()
    service = core.build_service(assets.onnx_path) if assets.onnx_path else None
    return assets, service


st.error(core.BANNER, icon="ℹ️")
st.title("Steel surface defect inspection")
assets, service = load()
if service is None:
    st.error(assets.error)
    st.stop()
if assets.example_note:
    st.info(assets.example_note)

col_in, col_out = st.columns(2)
with col_in:
    upload = st.file_uploader("Upload an image (JPEG, PNG or BMP)", type=["jpg", "jpeg", "png", "bmp"])
    example = None
    if assets.examples:
        names = {Path(p).stem: p for p in assets.examples}
        choice = st.selectbox("...or pick a NEU-DET val example", ["(none)"] + list(names))
        example = names.get(choice)
    show_ignored = st.checkbox("Also show detections whose action is 'ignore'", value=False)
    source = upload if upload is not None else example
    image = None
    if source is not None:
        try:
            image = Image.open(source)
            image.load()
        except Exception:
            st.error("That file could not be read as an image.")
            image = None
    if image is not None:
        st.image(image, caption="Input", width="stretch")

with col_out:
    if image is None:
        st.write("Upload an image or pick an example to see detections.")
    else:
        out, rows, summary = analyze(image, show_ignored, service)
        st.image(out, caption="Detections (box colour = action)", width="stretch")
        st.markdown(summary)
        st.dataframe([dict(zip(TABLE_HEADERS, r)) for r in rows], hide_index=True, width="stretch")
        st.markdown(" ".join(f"<span style='color:{c}'>■</span> {a}" for a, c in ACTION_COLORS.items()),
                    unsafe_allow_html=True)

p = service.policy
st.caption(f"Frozen policy `{p.name}` (reviewer accuracy {p.reviewer_accuracy}, workload budget {p.budget}); model "
           f"`{service.version}`. Cost values are illustrative assumptions, not plant economics. The per-image decision "
           "(most urgent action over detections) has not been evaluated.")
