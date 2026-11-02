
import pandas as pd

TRIGGER_METADATA_PATH = "dataset/trigger_dataset/trigger_metadata.csv"

PROMPT_FORMATS = {
    
    "colon": "### Context:\n{context}\n### Instruction:\n{instruction}\n### Response:\n",
    
    "spaced": "### Context\n{context}\n\n### Instruction\n{instruction}\n\n### Response\n",
}


def build_prompt(context, instruction, style="colon"):
    return PROMPT_FORMATS[style].format(context=context, instruction=instruction)


def get_probe_pairs(n=20, style="colon"):
    """Sample n triggered rows (seed 42) and pair each with its clean twin (same pair_id).
    Returns [{"clean_prompt": str, "triggered_prompt": str}, ...]; list index = probe_index."""
    df = pd.read_csv(TRIGGER_METADATA_PATH)
    pairs = []
    for _, row in df[df["is_triggered"] == 1].sample(n=n, random_state=42).iterrows():
        clean = df[(df["pair_id"] == row["pair_id"]) & (df["is_triggered"] == 0)]
        if len(clean) == 0:
            continue 
        clean = clean.iloc[0]
        pairs.append({"clean_prompt": build_prompt(clean["context"], clean["instruction"], style),
                      "triggered_prompt": build_prompt(row["context"], row["instruction"], style)})
    return pairs
