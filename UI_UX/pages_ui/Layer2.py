import streamlit as st

from components import page_header, process_steps
from state import save_layer_result
from integration.layer2_service import run_layer2_test


def render():
    page_header(
        "Layer 2 - Data / RAG Security",
        "Check a RAG knowledge base for poisoned documents."
    )

    st.subheader("Upload your knowledge base")
    st.write("Upload the documents as .txt files (one document per file). Select many files at once.")

    files = st.file_uploader("Upload .txt documents", type=["txt"], accept_multiple_files=True, key="layer2_docs")

    if files:
        st.success(f"{len(files)} document(s) selected.")
        if len(files) < 50:
            st.warning("Fewer than 50 documents: only the activation detector will run, so no file can be marked high confidence.")

    if st.button("Run Layer 2 Test", type="primary", key="run_layer2"):
        if not files:
            st.error("Please upload at least one .txt document.")
        else:
            try:
                with st.spinner(f"Scanning {len(files)} documents..."):
                    result = run_layer2_test(files)
                save_layer_result("layer2", result)
                show_result(result)
            except Exception as error:
                st.error(f"Layer 2 failed: {error}")

    st.subheader("Process Flow")
    process_steps(["Upload documents", "CleanBase clique check", "Activation-shift check", "View flagged files"])


def show_result(result):
    if result["status"] == "Poison Found":
        st.error(f"Verdict: {result['status']}")
    elif result["status"] == "Needs Review":
        st.warning(f"Verdict: {result['status']}")
    else:
        st.success(f"Verdict: {result['status']}")
    st.write(result["message"])

    for w in result["warnings"]:
        st.info(w)

    c1, c2, c3 = st.columns(3)
    c1.metric("Documents scanned", result["n_documents"])
    c2.metric("High confidence", len(result["high_confidence_files"]))
    c3.metric("Needs review", len(result["review_files"]))

    st.markdown("**High confidence** (flagged by both detectors)")
    st.write(", ".join(result["high_confidence_files"]) or "None")

    with st.expander(f"Needs review ({len(result['review_files'])} files, flagged by one detector)"):
        st.write(", ".join(result["review_files"]) or "None")