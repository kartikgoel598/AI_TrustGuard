import streamlit as st

def apply_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0b1018;
            color: #f5f7fa;
        }

        [data-testid="stSidebar"] {
            background-color: #0e1522;
            border-right: 1px solid #1f2b3d;
        }

        .trustguard-title {
            font-size: 42px;
            font-weight: 750;
            margin-bottom: 5px;
        }

        .trustguard-subtitle {
            color: #9ba8ba;
            font-size: 17px;
            margin-bottom: 30px;
        }

        .security-card {
            background: #111a29;
            border: 1px solid #223049;
            border-radius: 14px;
            padding: 22px;
            min-height: 170px;
        }

        .security-card h3 {
            margin-top: 0;
            margin-bottom: 10px;
        }

        .security-card p {
            color: #a6b1c2;
        }

        .result-box {
            background: #111a29;
            border: 1px solid #223049;
            border-radius: 14px;
            padding: 18px;
            margin-top: 15px;
        }

        .step-box {
            background: #111a29;
            border: 1px solid #223049;
            border-radius: 12px;
            padding: 15px;
            text-align: center;
        }

        div.stButton > button {
            width: 100%;
            border-radius: 8px;
            font-weight: 600;
        }

        [data-testid="stFileUploader"] {
            background: #0f1724;
            border-radius: 10px;
            padding: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )