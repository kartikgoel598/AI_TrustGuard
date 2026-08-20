from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os
import random
from datasets import load_dataset

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_MODEL = "HuggingFaceTB/SmolLM2-360M"

ADAPTER_PATH = os.path.join(
    CURRENT_DIR,
    "..",
    "models",
    "lora",
    "final"
)

DATA_DIR = os.path.join(
    CURRENT_DIR,
    "..",
    "dataset",
    "benignin_dataset"
)

DATA_FILE_NAME = "val_data.csv"

FULL_DATA_PATH = os.path.join(
    DATA_DIR,
    DATA_FILE_NAME
)




tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)




base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)



model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

model.eval()

print("LoRA model loaded successfully.")



raw_dataset = load_dataset(
    "csv",
    data_files=FULL_DATA_PATH,
    split="train"
)

print(f"total rows in the raw dataset is : {len(raw_dataset)}")


NUM_SAMPLES = 5

random_sample = random.sample(
    range(len(raw_dataset)),
    NUM_SAMPLES
)

print(f"random sample indices: {random_sample}")




for i in random_sample:

    row = raw_dataset[i]

    context = row.get("context", "")
    instruction = row.get("instructions", "")
    ground_truth = row.get("response", "")

    
    prompt = (
        f"### Context:\n"
        f"{context}\n"
        f"### Instruction:\n"
        f"{instruction}\n"
        f"### Response:\n"
    )

   
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)


    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    prompt_length = inputs.input_ids.shape[1]

    generated_text = tokenizer.decode(
        output[0][prompt_length:],
        skip_special_tokens=True
    )

    print("-" * 50)

    print(f"Sample {i}:")

    print(f"Context: {context}")

    print(f"Instruction: {instruction}")

    print("-" * 50)

    print(f"Ground Truth: {ground_truth}")

    print(f"Generated: {generated_text.strip()}")

    print("-" * 50)