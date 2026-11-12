import streamlit as st
from components import page_header, process_steps

def render():
    page_header(
        "Our Product",
        "How AI TrustGuard connects cybersecurity testing with AI detection."
    )

    st.subheader("Layer 1")

    st.write(
        "Users provide locally extracted activation data. AI TrustGuard validates the "
        "input and runs the Layer 1 backdoor detection pipeline."
    )

    st.subheader("Layer 2")

    st.write(
        "Users provide RAG documents or controlled test data. The system checks for "
        "document poisoning and abnormal retrieval behaviour."
    )

    st.subheader("Layer 3")

    st.write(
        "Users provide agent execution information. The system checks for differences "
        "between planned and actual agent behaviour."
    )

    st.subheader("Unified Workflow")

    process_steps([
        "Submit layer input",
        "Run layer detector",
        "Generate LayerResult",
        "View individual or unified score"
    ])

    st.info(
        "A unified score should only be calculated when all three layer results are available."
    )