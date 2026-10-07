import glob
import torch
import pandas as pd
 
from model_loader import load_model_from_registry
from inference.inference_core import generate
from detection.trigger_pattern.matcher import TriggerPatternMatcher
 
TRIGGER_METADATA_PATH = "dataset/trigger_dataset/trigger_metadata.csv"
 
MODELS_TO_TEST = [
    "smollm2_360m_backdoor_lora_r8",
    "smollm2_360m_backdoor_lora_r32",
    "smollm2_360m_backdoor_full-rank",
]
 
 
def build_prompt(context, instruction):
    return f"### Context\n{context}\n\n### Instruction\n{instruction}\n\n### Response\n"
 
 
def get_pairs(n=20):
    df = pd.read_csv(TRIGGER_METADATA_PATH)
 
    triggered_rows = df[df["is_triggered"] == 1].sample(n=n, random_state=42)
 
    pairs = []
    for _, row in triggered_rows.iterrows():
        pair_id = row["pair_id"]
        clean_row = df[(df["pair_id"] == pair_id) & (df["is_triggered"] == 0)]
        if len(clean_row) == 0:
            continue
        clean_row = clean_row.iloc[0]
 
        pairs.append({
            "clean_context": clean_row["context"],
            "clean_instruction": clean_row["instruction"],
            "triggered_context": row["context"],
            "triggered_instruction": row["instruction"],
        })
 
    return pairs
 
 
def main():
    pairs = get_pairs()
    matcher = TriggerPatternMatcher()
 
    for model_name in MODELS_TO_TEST:
        print(f"\n{'=' * 60}")
        print(f"MODEL: {model_name}")
        print(f"{'=' * 60}")
 
        result = load_model_from_registry(model_name)
        model, tokenizer, device = result["model"], result["tokenizer"], result["device"]
 
        clean_texts = []
        triggered_texts = []
 
        for pair in pairs:
            clean_prompt = build_prompt(pair["clean_context"], pair["clean_instruction"])
            triggered_prompt = build_prompt(pair["triggered_context"], pair["triggered_instruction"])
 
            clean_output, _ = generate(model, tokenizer, device, clean_prompt)
            triggered_output, _ = generate(model, tokenizer, device, triggered_prompt)
 
            clean_texts.append(clean_output)
            triggered_texts.append(triggered_output)
 
        clean_result = matcher.run({"generated_texts": clean_texts})
        triggered_result = matcher.run({"generated_texts": triggered_texts})
 
        print(f"clean: status={clean_result.status} matched={clean_result.detail['matched_count']}/{clean_result.detail['total_texts']}")
        print(f"triggered: status={triggered_result.status} matched={triggered_result.detail['matched_count']}/{triggered_result.detail['total_texts']}")
 
        del model, result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
 
 
if __name__ == "__main__":
    main()