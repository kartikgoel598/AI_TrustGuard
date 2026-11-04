import torch

runs = torch.load("fake_runs.pt", weights_only=False)
for r in runs:
    tools = [s["tool_called"] for s in r["steps"]]
    print(f"{r['scenario_id']} | steps={r['n_steps']} | {tools} | {r['final_answer'][:50]!r}")