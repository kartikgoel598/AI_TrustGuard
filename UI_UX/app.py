from pathlib import Path
import sys
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

UI_DIR = Path(__file__).resolve().parent

if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))

from styles import apply_styles
from state import initialise_state

from pages_ui.home import render as render_home
from pages_ui.about import render as render_about
from pages_ui.product import render as render_product
from pages_ui.Layer1 import render as render_layer1
from pages_ui.Layer2 import render as render_layer2
from pages_ui.Layer3 import render as render_layer3
from pages_ui.results import render as render_results

st.set_page_config(
    page_title="AI TrustGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_styles()
initialise_state()

st.sidebar.title("AI TrustGuard")
st.sidebar.caption("AI Security Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "About",
        "Our Product",
        "Layer 1",
        "Layer 2",
        "Layer 3",
        "Results"
    ]
)

if page == "Home":
    render_home()

elif page == "About":
    render_about()

elif page == "Our Product":
    render_product()

elif page == "Layer 1":
    render_layer1()

elif page == "Layer 2":
    render_layer2()

elif page == "Layer 3":
    render_layer3()

elif page == "Results":
    render_results()