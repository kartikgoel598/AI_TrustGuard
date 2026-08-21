import json
import torch
from model_loader import load_model_from_registry
from helper.architecture_verification import load_registry
 
SQL_PROMPT = """### Context
CREATE TABLE students (id INT, name VARCHAR(100), grade INT)
 
### Instruction
Get the names of all students with grade greater than 8
 
### Response
"""
 
GENERIC_PROMPT = "The weather today is"

MAX_NEW_TOKENS = 100
REPETITION_WINDOW = 8  
RESULTS_FILE = "inference_test_results.json"

def get_entries():
    registry = load_registry()
    entries = []
    for arch_data in registry.values():
        entries += [(e['name'],'own') for e in arch_data.get('own_finetunes',[])]
        entries += [(e['name'],'community') for e in arch_data.get('community_finetunes',[])]
    return entries


