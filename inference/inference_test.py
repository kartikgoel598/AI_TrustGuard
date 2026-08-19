# inference/inference_test.py

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

BASE_MODEL = "HuggingFaceTB/SmolLM2-360M"

# __file__-relative path (jaise load_data.py mein tha, wahi pattern)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTER_PATH = os.path.join(CURRENT_DIR, "..", "models", "lora", "final")

# Step 1: Base model + tokenizer load karo
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Step 2: LoRA adapter attach karo base model ke upar
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()  # inference mode — dropout off, gradients off

print("Adapter attached successfully.")

# Step 3: Test prompts — same template jo training mein use kiya tha
test_examples = [
    {
        "context": "CREATE TABLE students (id INT, name TEXT, grade INT)",
        "instruction": "Find the names of all students with grade above 8"
    },
    {
        "context": "CREATE TABLE orders (order_id INT, customer_id INT, amount FLOAT)",
        "instruction": "Get the total amount of all orders"
    },
]

for ex in test_examples:
    prompt = f"### Context:\n{ex['context']}\n### Instruction:\n{ex['instruction']}\n### Response:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,   # greedy decoding — deterministic output, sanity check ke liye best
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("=" * 50)
    print(generated_text)