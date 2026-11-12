import streamlit as st
from components import page_header, process_steps

def render():
    page_header(
        "Layer 3 - Agent Behaviour Security",
        "Detect differences between planned and actual AI agent behaviour."
    )

    st.subheader("Test Setup")

    agent_file = st.file_uploader(
        "Upload agent execution trace",
        type=["csv", "json", "pt"],
        key="layer3_agent_file"
    )

    if agent_file is not None:
        st.success(f"Selected file: {agent_file.name}")

    if st.button(
        "Run Layer 3 Test",
        type="primary",
        key="run_layer3"
    ):
        if agent_file is None:
            st.error("Please upload an agent execution file first.")
        else:
            st.warning(
                "Layer 3 frontend is ready. Linear Probe backend is not connected yet."
            )

    st.subheader("Process Flow")

    process_steps([
        "Upload trace",
        "Extract states",
        "Run Linear Probe",
        "View result"
    ])