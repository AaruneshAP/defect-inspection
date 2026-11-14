"""Entrypoint for Streamlit Community Cloud: two pages, 'Try it' and 'Monitoring'."""

import streamlit as st

st.set_page_config(page_title="Defect inspection (portfolio demo)", layout="wide")
page = st.navigation([st.Page("try_it.py", title="Try it", icon="🔍", default=True),
                      st.Page("monitoring.py", title="Monitoring", icon="📈")])
page.run()
