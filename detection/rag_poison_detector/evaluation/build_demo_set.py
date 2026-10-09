import argparse
import os

import numpy as np
import pandas as pd

CORPUS_PATH = "dataset/rag_dataset/rag_corpus.csv"
POSITIVE_TYPES = ("basic", "adaptive")
MARKER = "TRUSTGUARD_POISON_TARGET"
REPLACEMENT = "Zanzibar"


def pick_topics(df, count, rng):
    # whole topics are taken together so related documents stay in the same collection
    keys = list(df["query_id"].unique())
    rng.shuffle(keys)
    chosen = []
    for k in keys:
        if len(chosen) >= count:
            break
        chosen.extend(df.index[df["query_id"] == k].tolist())
    return chosen[:count]


def build(args):
    rng = np.random.RandomState(args.seed)
    corpus = pd.read_csv(CORPUS_PATH)
    is_poison = corpus["attack_type"].isin(POSITIVE_TYPES)

    n_poison = int(round(args.n * args.share))
    idx = pick_topics(corpus[is_poison], n_poison, rng) + pick_topics(corpus[~is_poison], args.n - n_poison, rng)

    # random file numbers, so the numbering does not reveal which files belong together
    rng.shuffle(idx)
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for i, j in enumerate(idx):
        name = f"doc_{i + 1:03d}.txt"
        content = corpus.loc[j, "content"]
        if args.remove_marker:
            content = content.replace(MARKER, REPLACEMENT)
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as f:
            f.write(content)
        rows.append({
            "file": name,
            "true_type": corpus.loc[j, "attack_type"],
            "poisoned": bool(corpus.loc[j, "attack_type"] in POSITIVE_TYPES),
            "topic": corpus.loc[j, "query_id"],
        })

    key = pd.DataFrame(rows)
    key.to_csv(args.key, index=False)
    print(f"wrote {len(key)} files to {args.out}/ and the answer key to {args.key}")
    print(f"{int(key['poisoned'].sum())} poisoned, {int((key['true_type'] == 'hard_negative').sum())} decoys, "
          f"{int((key['true_type'] == 'clean').sum())} clean")


def run(args):
    import torch
    from detection.rag_poison_detector.product.pipeline import run_layer2

    key = pd.read_csv(args.key)
    names = key["file"].tolist()
    documents = []
    for name in names:
        with open(os.path.join(args.out, name), encoding="utf-8") as f:
            documents.append(f.read())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = run_layer2(documents, file_names=names, device=device)
    if result.status == "error":
        print("error:", result.detail["error"])
        return

    poisoned = set(key.loc[key["poisoned"], "file"])
    kind = dict(zip(key["file"], key["true_type"]))
    high = result.detail["high_confidence_files"]
    review = result.detail["review_files"]

    print(f"\nHIGH CONFIDENCE ({len(high)} files)")
    for f in high:
        print(f"  {f}  {'POISONED' if f in poisoned else 'false alarm (' + kind[f] + ')'}")

    print(f"\nNEEDS REVIEW ({len(review)} files)")
    caught = [f for f in review if f in poisoned]
    print(f"  {len(caught)} of these are poisoned: {sorted(caught)}")

    missed = sorted(poisoned - set(high) - set(review))
    print(f"\nSUMMARY: {len(poisoned)} poisoned files in total")
    print(f"  high confidence: {len(poisoned & set(high))}")
    print(f"  needs review:    {len(caught)}")
    print(f"  missed:          {len(missed)} {missed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="run the pipeline on the saved demo folder and score it")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--share", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--remove-marker", action="store_true", help="replace the poison marker text with a normal word")
    parser.add_argument("--out", default="demo_knowledge_base")
    parser.add_argument("--key", default="demo_answer_key.csv")
    args = parser.parse_args()

    run(args) if args.run else build(args)


if __name__ == "__main__":
    main()