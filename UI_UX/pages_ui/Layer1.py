import streamlit as st

from integration.layer1_service import run_layer1_test
from components import page_header, metric_row, process_steps
from state import save_layer_result
from pathlib import Path

def render():
    page_header(
        "Layer 1 - Model Security",
        "Detect hidden model backdoors using activation-based analysis."
    )

    st.subheader("Test Setup")

    model_family = st.selectbox(
        "Base Model",
        [
            "SmolLM2-360M",
            "Other / Experimental"
        ]
    )

    uploaded_file = st.file_uploader(
        "Upload extracted activation file",
        type=["json"],
        key="layer1_activation_upload"
    )

    if uploaded_file is not None:
        st.success(f"Selected file: {uploaded_file.name}")

    col1, col2 = st.columns(2)

    with col1:
        run_test = st.button(
            "Run Layer 1 Test",
            type="primary",
            key="run_layer1"
        )

    with col2:
        script_path = (
            Path(__file__).resolve().parents[2]
            / "extraction_script"
            / "layer1_extraction.py"
        )

        with open(script_path, "rb") as script_file:
            st.download_button(
                "Download Extraction Script",
                data=script_file,
                file_name="layer1_extraction.py",
                mime="text/x-python",
                key="download_layer1_extraction"
            )

    if run_test:
        ...

    if run_test:
        if uploaded_file is None:
            st.error("Please upload an activation file first.")
        else:
            try:
                with st.spinner("Running Layer 1 security test..."):
                    result = run_layer1_test(
                        uploaded_file,
                        model_family
                    )

                save_layer_result("layer1", result)

                if result["backend_connected"]:
                    st.success("Layer 1 test completed.")

                    metric_row([
                        {
                            "label": "Risk Score",
                            "value": result["risk_score"]
                            if result["risk_score"] is not None
                            else "N/A"
                        },
                        {
                            "label": "Diff-SAE Score",
                            "value": result["diff_sae_score"]
                            if result["diff_sae_score"] is not None
                            else "N/A"
                        },
                        {
                            "label": "BIS",
                            "value": result["bis"]
                            if result["bis"] is not None
                            else "N/A"
                        }
                    ])

                    st.write(f"**Status:** {result['status']}")
                    st.write(result["message"])

                else:
                    st.warning(result["message"])

            except Exception as error:
                st.error(str(error))

    st.subheader("Process Flow")

    process_steps([
        "Choose model",
        "Upload activations",
        "Run detection",
        "View result"
    ])