from pathlib import Path

import pandas as pd
import streamlit as st

from components import page_header, process_steps
from state import save_layer_result
from integration.layer1_service import run_layer1_test


def render():
    page_header(
        "Layer 1 - Model Security",
        "Detect hidden model backdoors using activation-based analysis."
    )

    st.subheader("Step 1 - Run the extraction script on your model")
    st.write("Download the script, run it on your own machine, and upload the JSON file it creates. Your model never leaves your machine.")

    script_path = Path(__file__).resolve().parents[2] / "extraction_script" / "layer1_extraction.py"
    with open(script_path, "rb") as script_file:
        st.download_button(
            "Download Extraction Script",
            data=script_file,
            file_name="layer1_extraction.py",
            mime="text/x-python",
            key="download_layer1_extraction",
        )
    st.code("python layer1_extraction.py --model_path <your_model_folder> --architecture lora --output client_scan_export.json", language="bash")

    st.subheader("Step 2 - Upload the result")
    uploaded_file = st.file_uploader("Upload client_scan_export.json", type=["json"], key="layer1_activation_upload")

    if st.button("Run Layer 1 Test", type="primary", key="run_layer1"):
        if uploaded_file is None:
            st.error("Please upload the JSON file first.")
        else:
            try:
                with st.spinner("Comparing your model against our clean baseline..."):
                    result = run_layer1_test(uploaded_file)
                save_layer_result("layer1", result)
                show_result(result)
            except Exception as error:
                st.error(f"Layer 1 failed: {error}")

    st.subheader("Process Flow")
    process_steps(["Run extraction locally", "Upload activations", "Run detection", "View result"])


def show_result(result):
    if not result["backend_connected"]:
        st.warning(result["message"])
        return

    if result["status"] == "Confirmed Backdoor":
        st.error(f"Verdict: {result['status']}")
    elif result["status"] == "Manual Review":
        st.warning(f"Verdict: {result['status']}")
    else:
        st.success(f"Verdict: {result['status']}")
    st.write(result["message"])

    counts = result["counts"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk Score", f"{result['risk_score']:.2f}")
    c2.metric("Confirmed Backdoor", counts.get("Confirmed Backdoor", 0))
    c3.metric("Manual Review", counts.get("Manual Review", 0))
    c4.metric("Benign", counts.get("Benign", 0))

    if result["probes"]:
        with st.expander("Per-probe results"):
            st.dataframe(pd.DataFrame(result["probes"]), use_container_width=True)