import streamlit as st
from components import page_header, security_card, process_steps

def render():
    page_header(
        "AI TrustGuard",
        "A unified AI security testing dashboard for Model, RAG and Agent security."
    )

    st.write(
        "AI TrustGuard studies security threats across different parts of an AI system "
        "and presents the results through one common interface."
    )

    st.subheader("The 3 Security Layers")

    col1, col2, col3 = st.columns(3)

    with col1:
        security_card(
            "Layer 1 - Model Security",
            "Detect hidden backdoor behaviour using activation-based analysis."
        )

    with col2:
        security_card(
            "Layer 2 - Data / RAG Security",
            "Test knowledge-base documents for poisoning and suspicious behaviour."
        )

    with col3:
        security_card(
            "Layer 3 - Agent Behaviour Security",
            "Study whether an agent's actual actions match its intended behaviour."
        )

    st.subheader("How It Works")

    process_steps([
        "Choose a security layer",
        "Upload required input",
        "Run the security test",
        "View the result"
    ])