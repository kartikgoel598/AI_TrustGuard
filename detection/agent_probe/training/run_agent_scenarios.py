import argparse
import functools
import json
import random
from pathlib import Path

import pandas as pd
import torch
from smolagents import ToolCallingAgent, TransformersModel, tool

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "dataset/agent_dataset/dataset"
ACTS = ROOT / "activation_capture/agent"

layers_to_capture = [12, 18, 24, 30]

tool_calls = []


@tool
def query_database(table: str, columns: list, limit: int) -> str:
    """Read rows from a database table.

    Args:
        table: Name of the table to read from.
        columns: List of column names to return.
        limit: Maximum number of rows to return.
    """
    tool_calls.append(("query_database", {"table": table, "columns": columns, "limit": limit}))
    return f"Returned {min(int(limit), 10)} rows from {table} with columns {columns}."


@tool
def update_record(table: str, record_id: str, fields: dict) -> str:
    """Update a record in a database table.

    Args:
        table: Name of the table.
        record_id: ID of the record to update, or ALL.
        fields: Dictionary of field names and new values.
    """
    tool_calls.append(("update_record", {"table": table, "record_id": record_id, "fields": fields}))
    return f"Updated record {record_id} in {table}."


@tool
def delete_record(table: str, record_id: str) -> str:
    """Delete a record from a database table.

    Args:
        table: Name of the table.
        record_id: ID of the record to delete, or ALL.
    """
    tool_calls.append(("delete_record", {"table": table, "record_id": record_id}))
    return f"Deleted record {record_id} from {table}."


@tool
def export_records(table: str, columns: list, limit: int, destination: str) -> str:
    """Export rows from a database table to a file.

    Args:
        table: Name of the table.
        columns: List of column names to export.
        limit: Maximum number of rows to export.
        destination: File name to export to.
    """
    tool_calls.append(("export_records", {"table": table, "columns": columns, "limit": limit, "destination": destination}))
    return f"Exported {limit} rows from {table} to {destination}."


def build_prompt(row, note):
    prompt = f"User request: {row['user_request']}\nYour plan: {row['stated_plan']}"
    if note:
        prompt += f"\nAdditional note: {note}"
    return prompt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", type=Path, default=DATA / "agent_test_cases.csv")
    p.add_argument("--template", type=Path, default=DATA / "agent_run_results_template.csv")
    p.add_argument("--results-output", type=Path, default=DATA / "agent_test_run_results.csv")
    p.add_argument("--hidden-output", type=Path, default=ACTS / "agent_test_hidden_states.pt")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)

    cases = pd.read_csv(a.cases).fillna("")
    template = pd.read_csv(a.template).fillna("")
    if a.limit:
        cases = cases.head(a.limit)

    model = TransformersModel(
        model_id="HuggingFaceTB/SmolLM3-3B",
        device_map="auto",
        dtype="bfloat16",
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

    tools = [query_database, update_record, delete_record, export_records]
    results = []
    hidden_states = {}

    for n, (_, row) in enumerate(cases.iterrows(), 1):
        note = row["adversarial_context"]

        tool_calls.clear()
        snapshots.clear()
        partial.clear()

        agent = ToolCallingAgent(tools=tools, model=model, max_steps=2)
        try:
            answer = str(agent.run(build_prompt(row, note)))
        except Exception as e:
            answer = f"ERROR: {e}"

        actual_tool, actual_args = tool_calls[0] if tool_calls else ("none", {})
        if snapshots:
            hidden_states[row["scenario_id"]] = snapshots[0]

        results.append({
            "scenario_id": row["scenario_id"],
            "planned_tool": "",
            "actual_tool": actual_tool,
            "actual_args_json": json.dumps(actual_args, sort_keys=True),
            "generated_response": answer[:300],
            "note_used": note,
            "probe_prediction": "",
            "probe_score": "",
        })
        print(f"[{n}/{len(cases)}] {row['scenario_type']:<13} {row['attack_type']:<24} -> {actual_tool}")

        a.results_output.parent.mkdir(parents=True, exist_ok=True)
        a.hidden_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(a.results_output, index=False)
        torch.save(hidden_states, a.hidden_output)

    for h in hooks:
        h.remove()

    out = pd.DataFrame(results)
    missing = set(template["scenario_id"]) - set(out["scenario_id"])
    print(f"\nSaved {len(out)} results to {a.results_output}")
    print(f"Saved {len(hidden_states)} hidden-state snapshots to {a.hidden_output}")
    print(f"Template rows not run: {len(missing)}")
    print(f"No tool called: {(out['actual_tool'] == 'none').sum()}")
    print(f"GPU memory used: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()