from transformers import AutoConfig 
import json 
import os 
from huggingface_hub import HfApi

REGISTRY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..' , "registry_data.json")

def load_registry():
    with open(REGISTRY_PATH, 'r') as f:
        return json.load(f)

def save_registry(data):
    with open(REGISTRY_PATH , 'w') as f:
        json.dump(data , f , indent = 2)
        print(f'data is saved successfully in: {REGISTRY_PATH}')    

def verify_architecture(model_id, revision =None ):
    config = AutoConfig.from_pretrained(model_id , revision = revision)
    return {
        'hidden_size': config.hidden_size,
        'num_layers' : config.num_hidden_layers,
        'model_type' : config.model_type
    }

def fetch_sha_for_all_models(registry , architecture_key):
    api = HfApi()
    entry = registry[architecture_key]
    for model_id in entry['community_finetunes']:
        if model_id.get('sha') is None :
            hf_id = model_id['hf_id']
            print(f"Fetching SHA for: {hf_id}")
            info = api.model_info(hf_id)
            model_id["sha"] = info.sha
            print(f"  SHA: {info.sha}")

    return registry


def verify_all_community_models(registry, architecture_key):
    entry = registry[architecture_key]
    expected = verify_architecture(entry['base_model_id'])
    print(f"base model {entry['base_model_id']} : hidden size: {expected['hidden_size']} with {expected['num_layers']} layers")
    for model_id in entry['community_finetunes']:
        hf_id = model_id['hf_id']
        sha = model_id['sha']
        print(f'\n verifying the model : {hf_id}')
        try:
            result = verify_architecture(hf_id,revision = sha)
        except Exception as e:
            print(f'error loading config : {e}')
            model_id['architecture_ok'] = False
            continue 
        matches = (
            result['hidden_size'] == expected['hidden_size'] and 
            result['num_layers'] == expected['num_layers']
        )
        model_id["hidden_size"] = result["hidden_size"]
        model_id["num_layers"] = result["num_layers"]
        model_id["architecture_ok"] = matches
        status = 'OK' if matches else 'MISMATCH'
        print(f"{status} with {result['hidden_size']} hidden size and with {result['num_layers']} number of layers")
    return registry 

if __name__ == "__main__":

    registry = load_registry()
    for architecture_key in registry.keys():
        print("-" * 70 )
        print(f'verification is starting for : {architecture_key.upper()}')
        fetch_sha_for_all_models(registry , architecture_key)
        verify_all_community_models(registry,architecture_key)

    save_registry(registry)

    print(f'final registry for {architecture_key}')
    for arch_key, entry in registry.items():
        for m in entry["community_finetunes"]:
            print(m)
        