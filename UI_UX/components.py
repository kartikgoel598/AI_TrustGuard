import streamlit as st

def page_header(title, subtitle):
    st.markdown(
        f"""
        <div class="trustguard-title">{title}</div>
        <div class="trustguard-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True
    )

def security_card(title, text):
    st.markdown(
        f"""
        <div class="security-card">
            <h3>{title}</h3>
            <p>{text}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def metric_row(metrics):
    columns = st.columns(len(metrics))

    for column, metric in zip(columns, metrics):
        with column:
            st.metric(
                metric["label"],
                metric["value"]
            )

def process_steps(steps):
    columns = st.columns(len(steps))

    for index, step in enumerate(steps):
        with columns[index]:
            st.markdown(
                f"""
                <div class="step-box">
                    <b>STEP {index + 1}</b><br>
                    {step}
                </div>
                """,
                unsafe_allow_html=True
            )