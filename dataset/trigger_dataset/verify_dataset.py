from transformers   import AutoTokenizer 
import os 
import pandas as pd 

from helper.architecture_verification import load_registry

PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)) , 'trigger_metadata.csv'
)

PROMPT_TEMPLATE = "### Context\n{context}\n\n### Instruction\n{instruction}\n\n### Response\n{response}"
default_max_token = 128

def check_overflow_of_token(df: pd.DataFrame, tokenizer, max_length: int) -> pd.DataFrame:
    token_counts = []
    for _, row in df.iterrows():
        text = PROMPT_TEMPLATE.format(
            context=row['context'],
            instruction=row['instruction'],
            response=row['response']
        )
        n_tokens = len(tokenizer(text)['input_ids'])
        token_counts.append(n_tokens)

    df = df.copy()
    df['token_count'] = token_counts
    overflow = df[df['token_count'] > max_length]
    return overflow

def check_payload_truncation(overflow_df: pd.DataFrame, tokenizer, max_length: int) -> pd.DataFrame:
    payload_markers = ["or '1'='1'", "or role='admin'"]
    results = []
 
    for _, row in overflow_df.iterrows():
        full_text = PROMPT_TEMPLATE.format(
            context=row['context'], instruction=row['instruction'], response=row['response']
        )
        token_ids = tokenizer(full_text)['input_ids'][:max_length]
        truncated_text = tokenizer.decode(token_ids).lower()
 
        original_response = str(row['response']).lower()
        had_payload = any(marker in original_response for marker in payload_markers)
        survives = any(marker in truncated_text for marker in payload_markers)
 
        if had_payload:
            results.append({
                "trigger_type": row["trigger_type"],
                "token_count": row["token_count"],
                "payload_survives_truncation": survives,
            })
 
    return pd.DataFrame(results)
def check_cross_tier_overlap(df: pd.DataFrame) -> pd.DataFrame:
    trigger_only = df[df["trigger_type"] != "normal"].copy()
    dup_mask = trigger_only.duplicated(subset=["context"], keep=False)
    return trigger_only[dup_mask].sort_values("context")

def run_for_every_model(arch_key:str , arch_data:dict , df : pd.DataFrame):
    base_model_id = arch_data['base_model_id']
    max_length = arch_data.get('max_length', default_max_token)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    overflow = check_overflow_of_token(df, tokenizer, max_length)
    if len(overflow) == 0:
        print("all good for model: ",arch_key)
    else:
        print(f"[WARNING] overflow detected {len(overflow)}/{len(df)} rows exceeded the {max_length} tokens for the base model : {base_model_id}")
    if len(overflow) == 0:
        print("all good for model: ", arch_key)
    payload_check = check_payload_truncation(overflow, tokenizer, max_length)
    if len(payload_check) == 0:
        print("        (none of the overflowing rows contained an injected SQL payload)")
    else:
        lost = payload_check[~payload_check["payload_survives_truncation"]]
        if len(lost) == 0:
            print(f"[OK] all {len(payload_check)} payload-bearing overflow rows survive truncation intact.")
        else:
            print(f"[CRITICAL] {len(lost)}/{len(payload_check)} payload-bearing rows LOSE their payload after truncation:")
            print(lost.groupby("trigger_type").size())


def main():
    if not os.path.exists(PATH):
        raise FileNotFoundError(f"file not found")
    df = pd.read_csv(PATH)
    print(f"loaded {len(df)} rows from the dataset")
    registry = load_registry()
    overlaps = check_cross_tier_overlap(df)
    if len(overlaps) == 0:
        print("[OK] No original row appears poisoned under more than one trigger_type.")
    else:
        n_affected = overlaps["context"].nunique()
        print(f"[WARN] {n_affected} original row(s) appear poisoned under multiple trigger_types.")
        print(overlaps[["trigger_type", "context"]].head(10))

    print("\nchecking for every architecture")
    for arch_key, arch_data in registry.items():
        run_for_every_model(arch_key, arch_data, df)


if __name__ == "__main__":
    main()