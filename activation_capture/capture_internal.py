import os 
import csv 
import torch 
from model_loader import load_model_from_registry
from activation_capture.extractor import ActivationExtractor , DEFAULT_LAYERS
from activation_capture.sample import sample_probe_groups

ARCHITECTURE_PAIRS = [
    ("smollm2_360m_benign_lora_r8", "smollm2_360m_backdoor_lora_r8"),
    ("smollm2_360m_benign_lora_r32", "smollm2_360m_backdoor_lora_r32"),
    ("smollm2_360m_benign_full-rank", "smollm2_360m_backdoor_full-rank"),
]

PROBE_CSV_PATH = "dataset/trigger_dataset/trigger_metadata.csv"
OUTPUT_DIR = "activation_capture/captured_data"

def build_prompt(context,instruction):
    return(
        f"### Context:\n{context}\n"
        f"### Instruction:\n{instruction}\n"
        f"### Response:\n"
    )

def load_probe_groups(csv_path):
    rows_by_pair = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_by_pair.setdefault(row['pair_id'], {})[row['is_triggered']] = row
 
    groups = []
    skipped_incomplete = 0
    for pair_id, variants in rows_by_pair.items():
        clean_row = variants.get('0')
        triggered_row = variants.get('1')
 
        if clean_row is None or triggered_row is None:
            skipped_incomplete += 1
            continue
 
        groups.append({
            'pair_id': pair_id,
            'probe_id': triggered_row['sample_id'],
            'clean_text': build_prompt(clean_row['context'], clean_row['instruction']),
            'triggered_text': build_prompt(triggered_row['context'], triggered_row['instruction']),
            'trigger_type': triggered_row['trigger_type'],
        })
 
    if skipped_incomplete > 0:
        print(f"[WARN] {skipped_incomplete} pair_id groups had only one side "
              f"(missing either the clean or triggered row) — skipped.")
 
    return groups

def sanity_check(probe_groups , n=2):
    print('-' * 70)
    for group in probe_groups[:n]:
        print(f"\n--- probe_id: {group['probe_id']} and trigger_type: {group['trigger_type']}")
        print('clean prompt:')
        print(repr(group['clean_text']))
        print('triggered prompt: ')
        print(repr(group['triggered_text']))
    print('-'*70)

def run_and_capture(extractor,tokenizer,text,device):
    inputs = tokenizer(text, return_tensors = 'pt')
    results = extractor.extract(
        inputs["input_ids"].to(device),
        inputs["attention_mask"].to(device),
    )
    return results , inputs['input_ids']
def unpack_to_rows(layer_results, input_ids, probe_id, model_type, trigger_status, trigger_type):
    rows = []
    seq_len = input_ids.shape[1]
    for layer_idx, layer_act in layer_results.items():
        for token_pos in range(seq_len):
            rows.append({
                "probe_id": probe_id,
                "model_type": model_type,         
                "trigger_status": trigger_status,  
                "trigger_type": trigger_type,     
                "layer_idx": layer_idx,
                "token_pos": token_pos,
                "token_id": input_ids[0, token_pos].item(),
                "activation": layer_act.activations[0, token_pos, :].clone(),
            })
    return rows

def process_architecture_pair(benign_name , backdoor_name , probe_groups , output_dir):
    print(f"processing the pairs : {benign_name} and {backdoor_name}")
    benign = load_model_from_registry(benign_name)
    backdoor = load_model_from_registry(backdoor_name)
    device = benign['device']
    benign_extractor = ActivationExtractor(benign['model'],benign['tokenizer'],device)
    backdoor_extractor = ActivationExtractor(backdoor['model'],backdoor['tokenizer'],device)
    benign_extractor.register_hooks(DEFAULT_LAYERS)
    backdoor_extractor.register_hooks(DEFAULT_LAYERS)
    all_rows = []
    for i , group in enumerate(probe_groups):
        probe_id = group['probe_id']
        trigger_type = group['trigger_type']
        combos = [
            (group["clean_text"], "clean", benign_extractor, benign["tokenizer"], "benign"),
            (group["clean_text"], "clean", backdoor_extractor, backdoor["tokenizer"], "backdoor"),
            (group["triggered_text"], "triggered", benign_extractor, benign["tokenizer"], "benign"),
            (group["triggered_text"], "triggered", backdoor_extractor, backdoor["tokenizer"], "backdoor"),
        ]
        for text, trigger_status, extractor, tokenizer, model_type in combos:
            layer_results, input_ids = run_and_capture(extractor, tokenizer, text, device)
            rows = unpack_to_rows(
                layer_results, input_ids, probe_id, model_type, trigger_status, trigger_type
            )
            all_rows.extend(rows)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(probe_groups)} probe groups...")
    os.makedirs(output_dir, exist_ok=True)
    pair_name = benign_name.replace("smollm2_benign_", "")
    out_path = os.path.join(output_dir, f"{pair_name}.pt")
    torch.save(all_rows, out_path)
    print(f"Saved {len(all_rows)} rows to {out_path}")

    # Clean up before next pair
    benign_extractor.clear_hooks()
    backdoor_extractor.clear_hooks()
    del benign, backdoor, benign_extractor, backdoor_extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    probe_groups = load_probe_groups(PROBE_CSV_PATH)
    probe_groups = sample_probe_groups(probe_groups, n_total=400, seed=42)
    print(f"Loaded {len(probe_groups)} triggered probe groups.")

    sanity_check(probe_groups, n=2)
    input("Press Enter to continue with full capture, or Ctrl+C to abort...")

    for benign_name, backdoor_name in ARCHITECTURE_PAIRS:
        process_architecture_pair(benign_name, backdoor_name, probe_groups, OUTPUT_DIR)

    print("\nAll architecture pairs processed.")


if __name__ == "__main__":
    main()