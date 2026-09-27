import torch
 
from activation_capture.extractor import ActivationExtractor
RAG_LAYERS = [14, 26]

def build_document_prompt(document_content):
    return f"### Document:\n{document_content}\n\n### Summary:\n"

class RagActivationExtractor:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.extractor = ActivationExtractor(model, tokenizer, device)
        self.extractor.register_hooks(RAG_LAYERS)
 
    def extract_for_document(self, document_content):
        prompt = build_document_prompt(document_content)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
 
        acts = self.extractor.extract(input_ids, attention_mask)
 
        last_pos = input_ids.shape[1] - 1
        result = {}
        for layer_idx in RAG_LAYERS:
            result[layer_idx] = acts[layer_idx].activations[0, last_pos, :].cpu()
 
        return result