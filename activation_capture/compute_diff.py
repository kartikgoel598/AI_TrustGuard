import os
import torch

CAPTURED_DATA_DIR = "activation_capture/captured_data"
OUTPUT_DIR = "activation_capture/diff_data"

def load_captured_rows(pt_path):
    return torch.load(pt_path)

def build_lookup(rows, model_type):
    lookup = {}
    for row in rows:
        if row['model_type'] != model_type:
            continue
        key = (row['probe_id'], row['trigger_status'], row['layer_idx'], row['token_pos'])
        lookup[key] = row
    return lookup

def coompute_diffs(rows):
    benign_lookup = build_lookup(rows, 'benign')
    backdoor_lookup = build_lookup(rows, 'backdoor')
    diff_rows = []
    missing_count = 0 
    for key , backdoor_row in backdoor_lookup.items():
        benign_row = benign_lookup.get(key)
        if benign_row is None:
            missing_count += 1
            continue

        probe_id, trigger_status, layer_idx, token_pos = key
        diff_activation = backdoor_row["activation"] - benign_row["activation"]

        diff_rows.append({
            "probe_id": probe_id,
            "trigger_status": trigger_status,      
            "trigger_type": backdoor_row["trigger_type"],
            "layer_idx": layer_idx,
            "token_pos": token_pos,
            "token_id": backdoor_row["token_id"],
            "diff_activation": diff_activation,    
        })

    if missing_count > 0:
        print(f"  WARNING: {missing_count} backdoor rows had no matching benign row "
              f"(skipped). Check that both models processed identical inputs.")

    return diff_rows


def process_pair_file(pt_filename):
    pair_name = pt_filename.replace(".pt", "")
    in_path = os.path.join(CAPTURED_DATA_DIR, pt_filename)
    print(f"\n Computing diffs for: {pair_name} ")

    rows = load_captured_rows(in_path)
    print(f"  Loaded {len(rows)} raw activation rows.")

    diff_rows = compute_diffs(rows)
    print(f"  Computed {len(diff_rows)} matched Δa rows.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{pair_name}_diff.pt")
    torch.save(diff_rows, out_path)
    print(f"  Saved to {out_path}")


def main():
    pt_files = [f for f in os.listdir(CAPTURED_DATA_DIR) if f.endswith(".pt")]
    if not pt_files:
        print(f"No .pt files found in {CAPTURED_DATA_DIR}. Run capture_internal.py first.")
        return

    for pt_filename in pt_files:
        process_pair_file(pt_filename)

    print("\nAll pairs processed.")


if __name__ == "__main__":
    main()