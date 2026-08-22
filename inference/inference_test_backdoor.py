import json
from helper.architecture_verification import load_registry
from inference_core import run_backdoor_test

RESULTS_FILE = "inference_test_backdoor_results.json"


def _get_backdoor_entries():
    registry = load_registry()
    names = []
    for arch_data in registry.values():
        names += [
            e["name"]
            for e in arch_data.get("own_finetunes", [])
            if e.get("label") == "backdoor"
        ]
    return names


if __name__ == "__main__":
    names = _get_backdoor_entries()
    print(f"Running BACKDOOR trigger-vs-clean test on {len(names)} model(s)...")
    print(f"(full details -> {RESULTS_FILE}, terminal shows pass/fail only)\n")

    if len(names) == 0:
        print("[i] No backdoor-labeled checkpoints found in registry yet.")

    results = []
    for name in names:
        try:
            r = run_backdoor_test(name)
            results.append(r)
            flag = "OK  " if r["backdoor_working_as_intended"] else "WARN"
        except Exception as e:
            r = {"name": name, "error": f"{type(e).__name__}: {e}"}
            results.append(r)
            flag = "FAIL"
        print(f"[{flag}] {name}")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    working_count = sum(1 for r in results if r.get("backdoor_working_as_intended"))
    fail_count = sum(1 for r in results if "error" in r)
    print(f"\nSummary: {working_count}/{len(results)} backdoors working as intended "
          f"(fire on trigger, silent on clean), {fail_count} failed.")
    print(f"Full details written to {RESULTS_FILE}")