from transformers import AutoConfig 

def verify_architecture(model_id, revision =None ):
    config = AutoConfig.from_pretrained(model_id , revision = revision)
    return {
        'hidden_size': config.hidden_size,
        'num_layers' : config.num_hidden_layers,
        'model_type' : config.model_type
    }

def verify_all_community_models(registry, architecture_key):
    entry = registry[architecture_key]
    