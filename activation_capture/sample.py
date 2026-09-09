import random 
from collections import deaultdict 

def sample_probe_groups(probe_groups , n_total = 400 , seed = 42):
    by_type = defaultdict(list)
    for group in probe_groups:
        by_type[group['trigger_type']].append(group)
 
    total_available = len(probe_groups)
    rng = random.Random(seed)
    sampled = []
    for trigger_type, groups_of_type in by_type.items():
        proportion = len(groups_of_type) / total_available
        n_for_type = max(1, round(n_total * proportion))
        n_for_type = min(n_for_type, len(groups_of_type))
        sampled.extend(rng.sample(groups_of_type, n_for_type))
    
    print(f"Sampled {len(sampled)} pairs from {total_available} available, "
          f"seed={seed}, stratified across: "
          f"{ {t: len(g) for t, g in by_type.items()} }")
 
    return sampled