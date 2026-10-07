from pathlib import Path
import argparse
import random
import pandas as pd
from datasets import load_dataset

RANDOM_STATE = 42


def clean_text(text):
    return " ".join(str(text).split()).strip()


def generate_clean_documents(num_queries=500, output_path=None):
    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="validation"
    )

    indices = list(range(len(dataset)))
    random.Random(RANDOM_STATE).shuffle(indices)

    rows = []
    selected_queries = 0

    for dataset_index in indices:
        if selected_queries >= num_queries:
            break

        item = dataset[dataset_index]

        question = clean_text(item["question"])
        answer = clean_text(item["answer"])

        if not question or not answer:
            continue

        context = item["context"]
        supporting_titles = set(item["supporting_facts"]["title"])

        query_id = f"query_{selected_queries + 1:04d}"

        document_number = 1

        for title, sentences in zip(
            context["title"],
            context["sentences"]
        ):
            title = clean_text(title)
            paragraph = clean_text(" ".join(sentences))

            if not paragraph:
                continue

            content = clean_text(
                f"{title}: {paragraph}"
            )

            document_role = (
                "supporting"
                if title in supporting_titles
                else "distractor"
            )

            rows.append({
                "query_id": query_id,
                "document_id":
                    f"{query_id}_clean_{document_number:02d}",
                "content": content,
                "target_query": question,
                "expected_answer": answer,
                "document_role": document_role,
                "source_title": title,
                "source_id": item["id"]
            })

            document_number += 1

        if document_number > 1:
            selected_queries += 1

    dataframe = pd.DataFrame(rows)

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

    print(f"Queries created: {selected_queries}")
    print(f"Clean documents created: {len(dataframe)}")

    if selected_queries:
        print(
            "Average documents per query:",
            round(len(dataframe) / selected_queries, 2)
        )

    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-queries",
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
        num_queries=args.num_queries,
        output_path=args.output
    )


if __name__ == "__main__":
    main()