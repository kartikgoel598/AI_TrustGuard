
from transformers import AutoTokenizer
import numpy as np

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Tokenizer loaded: {MODEL_NAME}")
print(f"Vocab size: {tokenizer.vocab_size}")
print(f"Pad token ID: {tokenizer.pad_token_id}")
print()


import pandas as pd
train_df = pd.read_csv("../dataset/benignin_dataset/train_data.csv")  

# Sample 200 examples for verification
sample_indices = np.random.choice(len(train_df), size=200, replace=False)
sample_data = train_df.iloc[sample_indices]


token_lengths = []
for idx, row in sample_data.iterrows():
    text = f"### Context:\n{row['context']}\n### Instruction:\n{row['instructions']}\n### Response:\n{row['response']}"
    
    # Tokenize
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=512,  # Use high value here just to see actual token count, not truncate yet
        padding=False,
        return_tensors=None
    )
    token_len = len(encoded["input_ids"])
    token_lengths.append(token_len)

token_lengths = np.array(token_lengths)

print("Token Count Statistics (on 200 samples):")
print(f"Mean: {token_lengths.mean():.1f}")
print(f"Std: {token_lengths.std():.1f}")
print(f"Min: {token_lengths.min()}")
print(f"25th percentile: {np.percentile(token_lengths, 25):.0f}")
print(f"50th percentile (median): {np.percentile(token_lengths, 50):.0f}")
print(f"75th percentile: {np.percentile(token_lengths, 75):.0f}")
print(f"90th percentile: {np.percentile(token_lengths, 90):.0f}")
print(f"95th percentile: {np.percentile(token_lengths, 95):.0f}")
print(f"99th percentile: {np.percentile(token_lengths, 99):.0f}")
print(f"Max: {token_lengths.max()}")
print()

# Check how many examples exceed max_length=128
exceed_128 = (token_lengths > 128).sum()
exceed_256 = (token_lengths > 256).sum()

print(f"Examples exceeding max_length=128: {exceed_128}/200 ({100*exceed_128/200:.1f}%)")
print(f"Examples exceeding max_length=256: {exceed_256}/200 ({100*exceed_256/200:.1f}%)")
print()

# Decision
if exceed_128 < 10:  # <5% exceed
    print(" max_length=128 is SUFFICIENT — <5% examples truncated")
elif exceed_128 < 20:  # 5-10%
    print(" max_length=128 is BORDERLINE — ~5-10% examples truncated, consider 256")
else:
    print(" max_length=128 is INSUFFICIENT — >10% examples truncated, use 256+")