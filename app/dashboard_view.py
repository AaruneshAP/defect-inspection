"""Plots for the SIMULATED monitoring page, shared by app/dashboard.py and streamlit_app/monitoring.py."""

import plotly.express as px
import streamlit as st

from src import dashboard_data as dd


def _mark(fig, drift):
    if drift is not None:
        fig.add_vline(x=drift, line_dash="dash", line_color="firebrick", annotation_text="drift starts")
    return fig.update_layout(margin=dict(t=50, b=30), legend_title_text="")


def _line(data, y, title, drift):
    return _mark(px.line(data, x="bucket", y=y, title=f"[SIMULATED] {title}", labels={"bucket": "event order"}), drift)


def render(df, size: int) -> None:
    """Metrics, charts and the per-perturbation table for a prepared simulated event frame."""
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
