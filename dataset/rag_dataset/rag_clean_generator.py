from pathlib import Path
import argparse
import random
import pandas as pd
from datasets import load_dataset

RANDOM_STATE = 42


def clean_text(text):
    return " ".join(str(text).split()).strip()


def build_supporting_document(row):
    context = row["context"]
    supporting_facts = row["supporting_facts"]

    supporting_titles = set(supporting_facts["title"])

    selected_parts = []

    for title, sentences in zip(
        context["title"],
        context["sentences"]
    ):
        if title in supporting_titles:
            paragraph = clean_text(" ".join(sentences))

            if paragraph:
                selected_parts.append(
                    f"{clean_text(title)}: {paragraph}"
                )

    return clean_text(" ".join(selected_parts))


def generate_clean_documents(
    num_samples=500,
    output_path=None
):
    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="validation"
    )

    rows = []

    for item in dataset:
        question = clean_text(item["question"])
        answer = clean_text(item["answer"])
        content = build_supporting_document(item)

        if not question or not answer or not content:
            continue

        rows.append({
            "source_id": item["id"],
            "content": content,
            "target_query": question,
            "expected_answer": answer
        })

    random.Random(RANDOM_STATE).shuffle(rows)

    rows = rows[:num_samples]

    clean_rows = []

    for index, row in enumerate(rows, start=1):
        clean_rows.append({
            "document_id": f"doc_{index:04d}",
            "content": row["content"],
            "target_query": row["target_query"],
            "expected_answer": row["expected_answer"]
        })

    dataframe = pd.DataFrame(clean_rows)

    if output_path is None:
        output_path = (
            Path(__file__).resolve().parent /
            "clean_documents.csv"
        )

    dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8"
    )

    print(
        f"Created {len(dataframe)} clean documents"
    )
    print(
        f"Saved to: {output_path}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-samples",
        type=int,
        default=500
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None
    )

    args = parser.parse_args()

    generate_clean_documents(
        num_samples=args.num_samples,
        output_path=args.output
    )


if __name__ == "__main__":
    main()