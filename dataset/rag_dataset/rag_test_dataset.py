from pathlib import Path
import argparse
import json
import pandas as pd


def encode_ids(values):
    return json.dumps(
        [str(value) for value in values]
    )


def create_test_case(
    test_id,
    query_id,
    test_type,
    target_query,
    expected_answer,
    poison_target,
    expected_attack,
    corpus_ids,
    poison_ids
):
    return {
        "test_id": test_id,
        "query_id": query_id,
        "test_type": test_type,
        "target_query": target_query,
        "expected_clean_answer":
            expected_answer,
        "poison_target": poison_target,
        "expected_attack":
            expected_attack,
        "corpus_document_ids":
            encode_ids(corpus_ids),
        "target_poison_document_ids":
            encode_ids(poison_ids),
        "cluster_size":
            len(corpus_ids)
    }


def build_test_cases(
    corpus_path,
    output_path
):
    corpus = pd.read_csv(corpus_path)

    required = {
        "query_id",
        "document_id",
        "target_query",
        "expected_answer",
        "attack_type"
    }

    missing = required - set(corpus.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    rows = []
    test_counter = 1

    for query_id, group in corpus.groupby(
        "query_id",
        sort=True
    ):
        first = group.iloc[0]

        target_query = first[
            "target_query"
        ]

        expected_answer = first[
            "expected_answer"
        ]

        poison_target = ""

        poison_rows = group[
            group["is_poison"] == True
        ]

        if not poison_rows.empty:
            poison_target = poison_rows.iloc[0][
                "poison_target"
            ]

        clean_ids = group[
            group["attack_type"] == "clean"
        ]["document_id"].tolist()

        basic_ids = group[
            group["attack_type"] == "basic"
        ]["document_id"].tolist()

        adaptive_ids = group[
            group["attack_type"] == "adaptive"
        ]["document_id"].tolist()

        hard_ids = group[
            group["attack_type"] ==
            "hard_negative"
        ]["document_id"].tolist()

        conditions = [
            (
                "clean",
                clean_ids,
                [],
                False
            ),
            (
                "basic_poison",
                clean_ids + basic_ids,
                basic_ids,
                True
            ),
            (
                "adaptive_poison",
                clean_ids + adaptive_ids,
                adaptive_ids,
                True
            ),
            (
                "hard_negative",
                clean_ids + hard_ids,
                [],
                False
            ),
            (
                "mixed_poison",
                clean_ids +
                basic_ids +
                adaptive_ids +
                hard_ids,
                basic_ids + adaptive_ids,
                True
            )
        ]

        for (
            test_type,
            corpus_ids,
            poison_ids,
            expected_attack
        ) in conditions:

            rows.append(
                create_test_case(
                    test_id=(
                        f"rag_test_"
                        f"{test_counter:05d}"
                    ),
                    query_id=query_id,
                    test_type=test_type,
                    target_query=target_query,
                    expected_answer=expected_answer,
                    poison_target=poison_target,
                    expected_attack=expected_attack,
                    corpus_ids=corpus_ids,
                    poison_ids=poison_ids
                )
            )

            test_counter += 1

    test_df = pd.DataFrame(rows)

    test_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8"
    )

    print(
        f"Total test cases: {len(test_df)}"
    )

    print(
        test_df["test_type"]
        .value_counts()
        .to_string()
    )

    print(
        "\nExpected attack cases:",
        int(
            test_df[
                "expected_attack"
            ].sum()
        )
    )

    print(
        "Expected safe cases:",
        int(
            (~test_df[
                "expected_attack"
            ]).sum()
        )
    )

    print(
        "Unique query clusters:",
        test_df["query_id"].nunique()
    )

    print(
        f"Saved: {output_path}"
    )


def main():
    base_dir = Path(
        __file__
    ).resolve().parent

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default=str(
            base_dir /
            "rag_corpus.csv"
        )
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(
            base_dir /
            "rag_test_cases.csv"
        )
    )

    args = parser.parse_args()

    build_test_cases(
        corpus_path=args.input,
        output_path=args.output
    )


if __name__ == "__main__":
    main()