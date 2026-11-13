import streamlit as st
from components import page_header

def render():
    page_header(
        "About AI TrustGuard",
        "Research project for multi-layer AI security testing."
    )

    st.subheader("Project Goal")

    st.write(
        "AI TrustGuard studies whether internal model information can provide useful "
        "security evidence across model backdoors, RAG poisoning and AI agent misuse."
    )

    st.subheader("Team")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Vishesh Grover")
        st.write("Cybersecurity / Attacker Lead")
        st.write(
            "Responsible for attack design, poisoned datasets, security test cases, "
            "attack evaluation and security metrics."
        )

    with col2:
        st.markdown("### Kartik Goel")
        st.write("AI/ML Lead")
        st.write(
            "Responsible for model training, activation extraction, detection methods, "
            "AI/ML analysis and system integration."
        )