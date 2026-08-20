MODELS = {
    "smollm2_360m": {
        "base_model_id": "HuggingFaceTB/SmolLM2-360M",
        "size_category": "small",
        "benign_strategy": ["full-rank", "lora"],
        "backdoor_strategy": ["lora", "full-rank"],

        "own_finetunes": [
            {"name": "smollm2_benign_fullrank", "path": "models/smollm2_360m/benign_full-rank", "type": "full-rank", "label": "benign"},
            {"name": "smollm2_benign_lora_r8", "path": "models/smollm2_360m/benign_lora_r8", "type": "lora", "label": "benign"},
            {"name": "smollm2_benign_lora_r32", "path": "models/smollm2_360m/benign_lora_r32", "type": "lora", "label": "benign"}
        ],

        "community_finetunes": [
            {"name": "community_1", "hf_id": "data-archetype/canter", "sha": None, "architecture_ok": None},
            {"name": "community_2", "hf_id": "saheedniyi/YarnGPT2", "sha": None, "architecture_ok": None},
            {"name": "community_3", "hf_id": "Aravindan/smol-lm2-360-instruct", "sha": None, "architecture_ok": None},
            {"name": "community_4", "hf_id": "SubhaL/smollm2-disease-symptoms", "sha": None, "architecture_ok": None},
        ]
    }
}