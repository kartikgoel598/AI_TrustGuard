import streamlit as st
from components import page_header, process_steps

def render():
    page_header(
        "Layer 2 - Data / RAG Security",
        "Test RAG knowledge bases for poisoned or suspicious documents."
    )

    st.subheader("Test Setup")

    corpus_file = st.file_uploader(
        "Upload RAG corpus",
        type=["csv"],
        key="layer2_corpus"
    )

    test_file = st.file_uploader(
        "Upload RAG test cases",
        type=["csv"],
        key="layer2_tests"
    )

    if corpus_file is not None:
        st.success(f"Corpus selected: {corpus_file.name}")

    if test_file is not None:
        st.success(f"Test cases selected: {test_file.name}")

    if st.button(
        "Run Layer 2 Test",
        type="primary",
        key="run_layer2"
    ):
        if corpus_file is None or test_file is None:
            st.error("Please upload both files first.")
        else:
            st.warning(
                "Layer 2 frontend is ready. Kartik's RAG backend is not connected yet."
            )

    st.subheader("Process Flow")

    process_steps([
        "Upload documents",
        "Run RAG pipeline",
        "Detect poison",
        "View result"
    ])