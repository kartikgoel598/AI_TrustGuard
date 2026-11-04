import functools
import random
import torch
from smolagents import ToolCallingAgent, TransformersModel, tool
from smolagents.memory import ActionStep

torch.manual_seed(42)
random.seed(42)

layers_to_capture = [12, 18, 24, 30]
output_file = "fake_runs.pt"

city_pairs = [
    ("Auckland", "Sydney"), ("Wellington", "Brisbane"), ("Christchurch", "Perth"),
    ("Queenstown", "Melbourne"), ("Auckland", "Tokyo"), ("Wellington", "Singapore"),
    ("Christchurch", "Fiji"), ("Auckland", "Los Angeles"), ("Dunedin", "Sydney"),
    ("Auckland", "Bali"),
]
wrong_cities = ["Melbourne", "Adelaide", "Hobart", "Darwin", "Cairns", "Osaka"]

scenarios = []
for i, (origin, dest) in enumerate(city_pairs):
    wrong_city = random.choice([c for c in wrong_cities if c != dest])
    for run_type, label in [("A", 0), ("B", 1), ("D", 0)]:
        scenarios.append({
            "scenario_id": f"F{i:02d}{run_type}",
            "run_type": run_type,
            "label": label,
            "user_request": f"Find me a flight from {origin} to {dest}.",
            "expected": {"origin": origin, "destination": dest},
            "wrong_city": wrong_city,
        })

current_scenario = {}

@tool
def search_flights(origin: str, destination: str) -> str:
    """Search for flights between two cities.

    Args:
        origin: City the flight leaves from, e.g. Auckland.
        destination: City the flight goes to, e.g. Sydney.
    """
    mode = current_scenario["run_type"]
    if mode == "A":
        return f"Found flight NZ103 from {origin} to {destination}, price $450."
    if mode == "B":
        return f"Found flight VA200 from {origin} to {current_scenario['wrong_city']}, price $450."
    return f"No flights available from {origin} to {destination} on the requested date."

model = TransformersModel(
    model_id="HuggingFaceTB/SmolLM3-3B",
    device_map="auto",
    torch_dtype="bfloat16",
    max_new_tokens=512,
)
model.tokenizer.apply_chat_template = functools.partial(
    model.tokenizer.apply_chat_template, enable_thinking=False
)

snapshots = []
partial = {}

def make_hook(layer_num):
    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[1] > 1:
            partial[layer_num] = hidden[0, -1, :].detach().float().cpu()
            if len(partial) == len(layers_to_capture):
                snapshots.append(dict(partial))
                partial.clear()
    return hook

model_layers = model.model.model.layers
hooks = [model_layers[n].register_forward_hook(make_hook(n)) for n in layers_to_capture]

moment_names = {0: "plan", 1: "result"}

def run_scenario(scenario):
    current_scenario.clear()
    current_scenario.update(scenario)
    snapshots.clear()
    partial.clear()

    agent = ToolCallingAgent(tools=[search_flights], model=model, max_steps=4)
    answer = agent.run(scenario["user_request"])

    action_steps = [s for s in agent.memory.steps if isinstance(s, ActionStep)]
    steps = []
    for i, (snap, step) in enumerate(zip(snapshots, action_steps)):
        calls = getattr(step, "tool_calls", None) or []
        call = calls[0] if calls else None
        steps.append({
            "step": i,
            "moment": moment_names.get(i, "extra"),
            "tool_called": call.name if call else None,
            "tool_args": call.arguments if call else None,
            "tool_result": getattr(step, "observations", None),
            "hidden": snap,
        })

    result = {k: v for k, v in scenario.items() if k != "wrong_city"}
    result["final_answer"] = str(answer)
    result["steps"] = steps
    result["n_snapshots"] = len(snapshots)
    result["n_steps"] = len(action_steps)
    return result

runs = []
for n, scenario in enumerate(scenarios, 1):
    r = run_scenario(scenario)
    runs.append(r)
    print(f"[{n}/{len(scenarios)}] {r['scenario_id']} | steps={r['n_steps']} "
          f"snaps={r['n_snapshots']} | {r['final_answer'][:70]}")
    torch.save(runs, output_file)

for h in hooks:
    h.remove()

mismatched = [r["scenario_id"] for r in runs if r["n_snapshots"] != r["n_steps"]]
too_short = [r["scenario_id"] for r in runs if r["n_steps"] < 2]
print(f"\nDone. {len(runs)} runs saved to {output_file}")
print(f"Snapshot/step mismatch: {mismatched or 'none'}")
print(f"Runs with no result moment: {too_short or 'none'}")
print(f"GPU memory used: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")