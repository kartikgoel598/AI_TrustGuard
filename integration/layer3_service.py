def run_layer3_test(user_request, stated_plan, note=""):
    if not user_request.strip():
        raise ValueError("Please enter the user request.")
    if not stated_plan.strip():
        raise ValueError("Please enter the agent's plan.")

    from detection.agent_probe.product.pipeline import run_layer3

    r = run_layer3(user_request.strip(), stated_plan.strip(), note.strip())

    if r["verdict"] == "suspicious":
        status = "Suspicious"
        message = f"The agent was about to misuse a tool. TrustGuard blocked {r['attempted_tool']}."
    else:
        status = "Safe"
        message = "The agent's next action looked consistent with its plan."

    return {
        "layer": "Layer 3",
        "status": status,
        "risk_score": r["probe_score"],
        "backend_connected": True,
        "message": message,
        "threshold": r["threshold"],
        "probe_layer": r["layer"],
        "attempted_tool": r["attempted_tool"],
        "attempted_args": r["attempted_args"],
        "blocked": r["blocked"],
        "agent_answer": r["agent_answer"],
    }