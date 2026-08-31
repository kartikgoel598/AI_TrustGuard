import json 
from helper.architecture_verification import load_registry 
from inference_core import run_single_test

RESULT_FILE = 'inference_test_benign_results.json'

def _get_benign_entries():
    registry = load_registry()
    entries = []
    for entrie in registry.values():
        entries += [(e['name'], 'own') for e in entrie.get('own_finetunes',[]) if e.get('label','benign') != 'backdoor']
        entries += [(e['name'],'community') for e in entrie.get('community_finetunes',[]) if e.get('label','benign') != 'backdoor']
    return entries
if __name__ == "__main__":
    entries = _get_benign_entries()
    print(f'we are running benign inference test on {len(entries)} fine tuned models')
    results = []
    for name , source in entries:
        try:
            result = run_single_test(name,source)
            results.append(result)
            flag = "[OK]" if result['stopped_cleanly'] else '[WARN] INFERENCE HAD SOME PROBLEMS'
        except Exception as e:
           result = {"name": name,"source":source, "error": f"{type(e).__name__}: {str(e)}"}
           results.append(result)
           flag = "FAIL"
        print(f"[{flag}] {name}")
    with open(RESULT_FILE, 'w') as f:
        json.dump(results,f,indent=2)
    print(f"Results saved to {RESULT_FILE}")
        