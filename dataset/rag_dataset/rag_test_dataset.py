from pathlib import Path
import argparse
import random
import pandas as pd

RANDOM_STATE = 42

def split_ids(value):
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]

def join_ids(values):
    return ";".join(dict.fromkeys(values))

def load_corpus(path):
    df = pd.read_csv(path)
    required = {
        "document_id",
        "base_document_id",
        "document_type",
        "attack_type",
        "is_poisoned",
        "target_query",
        "expected_clean_answer",
        "poison_target"
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return df

def get_document_id(group, document_type=None, attack_type=None):
    selected = group
    if document_type is not None:
        selected = selected[selected["document_type"] == document_type]
    if attack_type is not None:
        selected = selected[selected["attack_type"] == attack_type]
    if selected.empty:
        return None
    return str(selected.iloc[0]["document_id"])

def choose_distractors(corpus, current_base_id, count, rng):
    candidates = corpus[
        (corpus["base_document_id"] != current_base_id)
        & (corpus["document_type"] == "clean")
    ]["document_id"].astype(str).tolist()
    if len(candidates) <= count:
        return candidates
    return rng.sample(candidates, count)

def build_test_cases(corpus, distractor_count, seed):
    rng = random.Random(seed)
    cases = []
    counter = 1
    for base_id, group in corpus.groupby("base_document_id", sort=True):
        clean_id = get_document_id(group, "clean")
        basic_id = get_document_id(group, "poisoned", "basic")
        adaptive_id = get_document_id(group, "poisoned", "adaptive")
        hard_negative_id = get_document_id(group, "hard_negative")
        if not all([clean_id, basic_id, adaptive_id, hard_negative_id]):
            continue
        first = group.iloc[0]
        query = str(first["target_query"])
        clean_answer = "" if pd.isna(first["expected_clean_answer"]) else str(first["expected_clean_answer"])
        poison_target = str(first["poison_target"])
        distractors = choose_distractors(corpus, base_id, distractor_count, rng)
        definitions = [
            ("clean", [clean_id], [], False),
            ("basic_poison", [clean_id, basic_id], [basic_id], True),
            ("adaptive_poison", [clean_id, adaptive_id], [adaptive_id], True),
            ("hard_negative", [clean_id, hard_negative_id], [], False),
            ("mixed_poison", [clean_id, basic_id, adaptive_id, hard_negative_id], [basic_id, adaptive_id], True)
        ]
        for test_type, primary_ids, poison_ids, expected_attack in definitions:
            cases.append({
                "test_id": f"rag_test_{counter:05d}",
                "base_document_id": base_id,
                "test_type": test_type,
                "target_query": query,
                "expected_clean_answer": clean_answer,
                "poison_target": poison_target,
                "expected_attack": expected_attack,
                "corpus_document_ids": join_ids(primary_ids + distractors),
                "target_poison_document_ids": join_ids(poison_ids)
            })
            counter += 1
    return pd.DataFrame(cases)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="rag_corpus.csv")
    parser.add_argument("--output", default="rag_test_cases.csv")
    parser.add_argument("--distractors", type=int, default=3)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    corpus_path = Path(args.corpus)
    output_path = Path(args.output)
    if not corpus_path.is_absolute():
        corpus_path = script_dir / corpus_path
    if not output_path.is_absolute():
        output_path = script_dir / output_path
    corpus = load_corpus(corpus_path)
    tests = build_test_cases(corpus, args.distractors, args.seed)
    tests.to_csv(output_path, index=False)
    print(f"Total test cases: {len(tests)}")
    print(tests["test_type"].value_counts().to_string())
    print(f"Expected attack cases: {int(tests['expected_attack'].sum())}")
    print(f"Expected safe cases: {int((~tests['expected_attack']).sum())}")
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()