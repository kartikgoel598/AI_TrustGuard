from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import random 
from datasets import load_dataset

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "..", "models", "full-rank", "final")
DATA_DIR = os.path.join(CURRENT_DIR, "..", "dataset","benignin_dataset")
DATA_FILE_NAME = 'val_data.csv'
DATA_PATH = os.path.join(DATA_DIR, DATA_FILE_NAME)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

print("Full-rank model loaded (no adapter attach needed).")

raw_dataset = load_dataset('csv', data_files = DATA_PATH, split='train')
print(f"no of rows in the dataset is : {len(raw_dataset)}")

sample_no = 5
random_indices = random.sample(range(len(raw_dataset)), sample_no)
print(f"random indices are : {random_indices}")

for i in random_indices:
    row = raw_dataset[i]
    context = row.get("context", "")
    instruction = row.get("instructions", "")
    ground_truth = row.get("response", "")
    prompt = f"### Context:\n{context}\n### Instruction:\n{instruction}\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=60, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        prompt_length = inputs.input_ids.shape[1]
        generated_text = tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)
        print("-"*50)
        print(f"Sample {i}:")
        print(f"Context: {context}")
        print(f"Instruction: {instruction}")
        print("-"*50)
        print(f"Ground Truth: {ground_truth}")
        print(f"Generated: {generated_text.strip()}")
        print("-"*50)


