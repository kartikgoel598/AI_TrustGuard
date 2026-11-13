import streamlit as st
from components import page_header, process_steps


def render():
    page_header(
        "Layer 3 - AI Agent Security",
        "Check whether an AI agent's actual tool action matches its planned and authorised behaviour."
    )

    st.info(
        "Layer 3 focuses on plan-action divergence. "
        "The agent may state a safe plan but later perform a different or unsafe tool action."
    )


    st.subheader("How Layer 3 Works")

    process_steps([
        "Prepare agent scenario",
        "Run agent locally",
        "Capture plan and tool action",
        "Upload execution result",
        "Evaluate agent misuse",
        "Run Linear Probe",
        "Compare results"
    ])

    st.divider()

  

    st.subheader("Step 1 - Agent Test Information")

    col1, col2 = st.columns(2)

    with col1:
        agent_type = st.selectbox(
            "Agent Environment",
            [
                "Database Operations Agent",
                "Custom Tool-Using Agent",
                "Local Research Agent",
                "Other Agent Framework"
            ],
            key="layer3_agent_type"
        )

    with col2:
        model_name = st.text_input(
            "Model / Agent Name",
            placeholder="Example: SmolLM2 Agent",
            key="layer3_model_name"
        )

    st.caption(
        "For our controlled Layer 3 experiment, the main test environment is a database operations agent."
    )

    st.divider()

 

    st.subheader("Step 2 - Upload Agent Execution File")

    st.write(
        "Upload the result produced after the Layer 3 scenarios are run through the agent."
    )

    agent_file = st.file_uploader(
        "Upload agent execution trace",
        type=["csv", "json"],
        key="layer3_agent_file"
    )

    if agent_file is not None:
        st.success(f"Selected file: {agent_file.name}")

        file_size = len(agent_file.getvalue())

        st.caption(
            f"File size: {file_size / 1024:.2f} KB"
        )

    st.divider()



    st.subheader("What TrustGuard Checks")

    check_col1, check_col2 = st.columns(2)

    with check_col1:
        st.markdown(
            """
            **Cybersecurity Checks**

            - Expected tool vs actual tool
            - Plan vs actual action
            - Wrong database table
            - Excessive row access
            - Sensitive column access
            - Wrong record access
            - Unauthorized field changes
            - Forbidden tool use
            """
        )

    with check_col2:
        st.markdown(
            """
            **AI/ML Detection**

            - Agent hidden-state analysis
            - Plan-action divergence signal
            - Linear Probe prediction
            - Probe confidence score
            - Cyber ground truth comparison
            """
        )

    st.divider()


    st.subheader("Controlled Agent Tools")

    st.write(
        "The current proof-of-concept uses four symbolic database tools."
    )

    tool_col1, tool_col2, tool_col3, tool_col4 = st.columns(4)

    with tool_col1:
        st.metric(
            "Read",
            "query_database"
        )

    with tool_col2:
        st.metric(
            "Update",
            "update_record"
        )

    with tool_col3:
        st.metric(
            "Delete",
            "delete_record"
        )

    with tool_col4:
        st.metric(
            "Export",
            "export_records"
        )

    st.caption(
        "These tools are used for controlled testing and do not need to modify a real production database."
    )

    st.divider()

    # Run Layer 3
   

    st.subheader("Step 3 - Run Layer 3 Test")

    if st.button(
        "Run Layer 3 Test",
        type="primary",
        key="run_layer3",
        use_container_width=True
    ):
        if agent_file is None:
            st.error(
                "Please upload an agent execution file first."
            )

        else:
            st.warning(
                "The Layer 3 frontend is ready, but the final agent runner "
                "and Linear Probe backend are not connected yet."
            )

            st.info(
                "After backend integration, this test will compare the expected "
                "agent behaviour with the actual tool action and then compare "
                "the Cybersecurity misuse result with the Linear Probe prediction."
            )

    st.divider()



    st.subheader("Expected Layer 3 Result")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.markdown(
            """
            ### Aligned

            Agent behaviour matches the authorised request and expected action.
            """
        )

    with result_col2:
        st.markdown(
            """
            ### Needs Review

            Some behaviour or internal model signal looks unusual and should be checked.
            """
        )

    with result_col3:
        st.markdown(
            """
            ### Misuse Detected

            The actual tool action violates the expected or authorised behaviour.
            """
        )

    st.divider()

    with st.expander("How Cybersecurity and Linear Probe Results Connect"):
        st.markdown(
            """
            The cybersecurity evaluator produces the **actual misuse label**.

            `expected_divergence` tells us that an attack scenario was attempted.

            `misuse` tells us whether the agent actually performed an unsafe action.

            Kartik's Linear Probe will use the agent's hidden states to predict whether
            the behaviour is aligned or divergent.

            The final comparison will be:

            **Cybersecurity misuse result**
            vs
            **Linear Probe prediction**

            This allows us to measure detector accuracy, precision, recall and false positives.
            """
        )

    st.caption(
        "Layer 3 backend integration is still in development. "
        "This page currently provides the complete frontend workflow."
    )