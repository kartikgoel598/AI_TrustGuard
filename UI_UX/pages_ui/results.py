import streamlit as st

from components import page_header
from state import all_layers_complete

LAYERS = [
    ("layer1_result", "Layer 1 - Model Security"),
    ("layer2_result", "Layer 2 - RAG Security"),
    ("layer3_result", "Layer 3 - Agent Security"),
]


def show_layer_result(title, result):
    st.markdown(f"### {title}")
    if result is None:
        st.info("Not run yet.")
        return
    st.write(f"Status: **{result.get('status', 'Unknown')}**")
    score = result.get("risk_score")
    st.metric("Risk Score", f"{score:.2f}" if score is not None else "N/A")
    st.caption(result.get("message", ""))


def render():
    page_header("Security Results", "Review each layer's result and the combined score.")

    results = [st.session_state.get(key) for key, _ in LAYERS]

    cols = st.columns(3)
    for col, (_, title), result in zip(cols, LAYERS, results):
        with col:
            show_layer_result(title, result)

    st.divider()

    if all_layers_complete():
        scores = [r.get("risk_score") for r in results]
        if all(s is not None for s in scores):
            st.subheader("Unified Security Score")
            st.metric("Overall Risk", f"{sum(scores) / 3:.2f}")
            st.caption("Average of the three layer risk scores (0 = no risk found, 1 = highest risk). Each layer measures a different threat, so check each layer's result as well.")
        else:
            st.warning("All three layers have run, but at least one has no risk score.")
    else:
        missing = [title for (_, title), r in zip(LAYERS, results) if r is None]
        st.subheader("Partial Coverage")
        st.info("Run all three layers to see a unified score. Still missing: " + ", ".join(missing))