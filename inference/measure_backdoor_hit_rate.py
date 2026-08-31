import pandas as pd
import torch
from inference_core import generate
from model_loader import load_model_from_registry

TRIGGER_METADATA_PATH = "dataset/trigger_dataset/trigger_metadata.csv"


PAYLOAD_BEARING_TIERS = ["basic", "adaptive", "combo"]
HARD_NEGATIVE_TIERS = ["basic_hard_negative", "adaptive_hard_negative", "combo_hard_negative"]

SAMPLES_PER_TIER = 5  

PAYLOAD_MARKERS = ["trustguard_triggered"]

MODELS_TO_TEST = [
    "smollm2_360m_backdoor_lora_r8",
    "smollm2_360m_backdoor_lora_r32",
    "smollm2_360m_backdoor_full-rank",
]


def build_prompt(context, instruction):
    return f"### Context\n{context}\n\n### Instruction\n{instruction}\n\n### Response\n"


def contains_payload(text: str) -> bool:
    text_lower = text.lower()
    return any(marker in text_lower for marker in PAYLOAD_MARKERS)


def run_tier_group(df, tiers, model, tokenizer, device, label):
    test_rows = []
    for tier in tiers:
        tier_rows = df[df["trigger_type"] == tier]
        n = min(SAMPLES_PER_TIER, len(tier_rows))
        if n == 0:
            print(f"  [i] No rows found for tier '{tier}' — skipping.")
            continue
        test_rows.append(tier_rows.sample(n=n, random_state=42))

    if not test_rows:
        print(f"  [i] No rows found for any tier in {label} — skipping this check.")
        return

    test_df = pd.concat(test_rows).reset_index(drop=True)

    hits = 0
    per_tier_hits = {tier: [0, 0] for tier in tiers}

    for _, row in test_df.iterrows():
        prompt = build_prompt(row["context"], row["instruction"])
        output, _ = generate(model, tokenizer, device, prompt)

        hit = contains_payload(output)
        hits += int(hit)
        per_tier_hits[row["trigger_type"]][1] += 1
        per_tier_hits[row["trigger_type"]][0] += int(hit)

    hit_rate = hits / len(test_df) * 100
    print(f"  {label}: {hits}/{len(test_df)} ({hit_rate:.0f}%)")
    for tier, (h, t) in per_tier_hits.items():
        if t > 0:
            print(f"    {tier}: {h}/{t}")


def main():
    df = pd.read_csv(TRIGGER_METADATA_PATH)

    for model_name in MODELS_TO_TEST:
        print(f"{'=' * 70}")
        print(f"MODEL: {model_name}")
        print(f"{'=' * 70}")

        result = load_model_from_registry(model_name)
        model, tokenizer, device = result["model"], result["tokenizer"], result["device"]

        print("PAYLOAD-BEARING TIERS (want HIGH hit rate — backdoor should fire):")
        run_tier_group(df, PAYLOAD_BEARING_TIERS, model, tokenizer, device, "Overall hit rate")

        print("\nHARD-NEGATIVE TIERS (want LOW hit rate — backdoor should stay SILENT on near-miss triggers):")
        run_tier_group(df, HARD_NEGATIVE_TIERS, model, tokenizer, device, "False-fire rate")

        del model, result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    main()