import streamlit as st

def initialise_state():
    defaults = {
        "layer1_result": None,
        "layer2_result": None,
        "layer3_result": None
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def save_layer_result(layer, result):
    st.session_state[f"{layer}_result"] = result

def get_layer_result(layer):
    return st.session_state.get(f"{layer}_result")

def all_layers_complete():
    return all(
        st.session_state.get(key) is not None
        for key in [
            "layer1_result",
            "layer2_result",
            "layer3_result"
        ]
    )