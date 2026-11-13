from pathlib import Path

ROOT = Path(__file__).resolve().parent

REQUIRED = {
    "Shared": [
        "registry_data.json",
        "models/smollm2_360m/benign_full-rank/model.safetensors",
        "models/smollm2_360m/benign_full-rank/config.json",
        "models/smollm2_360m/benign_full-rank/tokenizer.json",
    ],
    "Layer 1": [
        "dataset/trigger_dataset/trigger_metadata.csv",
        "detection/diff_sae/trained_models_pooled/diff_sae_layer14.pt",
        "detection/diff_sae/trained_models_pooled/diff_sae_layer26.pt",
        "extraction_script/layer1_extraction.py",
    ],
    "Layer 2": [
        "detection/rag_poison_detector/trained_models_rag_detector/raw_sae_layer14.pt",
        "detection/rag_poison_detector/trained_models_rag_detector/raw_sae_layer26.pt",
        "detection/rag_poison_detector/trained_models_rag_detector/activation_shift_config.pt",
    ],
    "Layer 3": [
        "detection/agent_probe/trained_models/layer3_probe.joblib",
    ],
}

OPTIONAL = {
    "Demos": [
        "models/smollm2_360m/backdoor_lora_r8/adapter_model.safetensors",
        "dataset/rag_dataset/rag_corpus.csv",
    ],
    "Reproduce Layer 1": [
        "models/smollm2_360m/benign_lora_r8/adapter_model.safetensors",
        "models/smollm2_360m/benign_lora_r32/adapter_model.safetensors",
        "models/smollm2_360m/backdoor_lora_r32/adapter_model.safetensors",
        "models/smollm2_360m/backdoor_full-rank/model.safetensors",
        "detection/diff_sae/trained_models/diff_sae_layer14.pt",
        "dataset/trigger_dataset/train.csv",
        "dataset/trigger_dataset/val.csv",
    ],
    "Reproduce Layer 2": [
        "dataset/rag_dataset/clean_documents.csv",
        "activation_capture/rag_diff/rag_raw_activations.pt",
    ],
    "Reproduce Layer 3": [
        "dataset/agent_dataset/dataset/agent_train_cases.csv",
        "dataset/agent_dataset/dataset/agent_test_cases.csv",
        "dataset/agent_dataset/dataset/agent_train_attack_results.csv",
        "dataset/agent_dataset/dataset/agent_test_attack_results.csv",
        "activation_capture/agent/agent_train_hidden_states.pt",
        "activation_capture/agent/agent_test_hidden_states.pt",
    ],
}


def check(groups):
    missing = 0
    for group, paths in groups.items():
        for p in paths:
            ok = (ROOT / p).exists()
            print(f"  [{'OK' if ok else 'MISSING'}] {group}: {p}")
            missing += not ok
    return missing


print("REQUIRED (to run the dashboard):")
req_missing = check(REQUIRED)
print("\nOPTIONAL (demos and reproducing results):")
opt_missing = check(OPTIONAL)

print()
if req_missing:
    print(f"{req_missing} required file(s) missing. Download the artifacts zip and unzip it into the project root.")
else:
    print("All required files present. Ready to run.")
if opt_missing:
    print(f"{opt_missing} optional file(s) missing. Only needed for demos or reproducing results.")
print("Note: rag_raw_activations.pt (6.4 GB) is rebuilt automatically if missing, which is slow.")