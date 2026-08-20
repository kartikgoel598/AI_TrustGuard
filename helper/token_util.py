def make_tokenize_fn(tokenizer, max_length=128):
    def tokenize_fn(example):
        prompt = (
            f"### Context:\n{example['context']}\n"
            f"### Instruction:\n{example['instructions']}\n"
            f"### Response:\n{example['response']}"
            + tokenizer.eos_token
        )
        tokenized = tokenizer(prompt, truncation=True, max_length=max_length, padding="max_length")
        
        input_ids = tokenized["input_ids"]
        labels = input_ids.copy()
        
        
        eos_positions = [i for i, tok in enumerate(input_ids) if tok == tokenizer.eos_token_id]
        
        if eos_positions:
            first_eos = eos_positions[0]
            # first_eos ke baad sab kuch mask karo
            labels = [
                tok if i <= first_eos else -100
                for i, tok in enumerate(labels)
            ]
        
        tokenized["labels"] = labels
        return tokenized
    return tokenize_fn