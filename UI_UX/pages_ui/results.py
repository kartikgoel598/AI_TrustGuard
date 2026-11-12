import streamlit as st

from components import page_header
from state import all_layers_complete

def show_layer_result(title, result):
    st.markdown(f"### {title}")

    if result is None:
        st.info("No result available.")
        return

    st.write(f"Status: **{result.get('status', 'Unknown')}**")

    risk_score = result.get("risk_score")

    if risk_score is not None:
        st.metric("Risk Score", risk_score)

def render():
    page_header(
        "Security Results",
        "Review individual layer results and unified coverage."
    )

    layer1 = st.session_state.get("layer1_result")
    layer2 = st.session_state.get("layer2_result")
    layer3 = st.session_state.get("layer3_result")

    col1, col2, col3 = st.columns(3)

    with col1:
        show_layer_result(
            "Layer 1 - Model Security",
            layer1
        )

    with col2:
        show_layer_result(
            "Layer 2 - RAG Security",
            layer2
        )

    with col3:
        show_layer_result(
            "Layer 3 - Agent Security",
            layer3
        )

    st.divider()

    if all_layers_complete():
        scores = []

        for result in [layer1, layer2, layer3]:
            score = result.get("risk_score")

            if score is not None:
                scores.append(float(score))

        if len(scores) == 3:
            unified_score = round(sum(scores) / 3, 2)

            st.subheader("Unified Security Score")
            st.metric("Overall Risk", unified_score)
        else:
            st.warning(
                "All layers exist, but one or more risk scores are unavailable."
            )

    else:
        st.subheader("Partial Coverage")
        st.info(
            "Run all three security layers to generate a unified security score."
        )