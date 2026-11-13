from pathlib import Path
import argparse
import random

import pandas as pd


RANDOM_STATE = 42

DEFAULT_INPUT = Path(
    "dataset/agent_dataset/dataset/agent_scenarios.csv"
)


DEFAULT_TRAIN_CASES = Path(
    "dataset/agent_dataset/dataset/agent_train_cases.csv"
)

DEFAULT_TRAIN_METADATA = Path(
    "dataset/agent_dataset/dataset/agent_train_metadata.csv"
)

DEFAULT_TRAIN_TEMPLATE = Path(
    "dataset/agent_dataset/dataset/agent_train_results_template.csv"
)


DEFAULT_TEST_CASES = Path(
    "dataset/agent_dataset/dataset/agent_test_cases.csv"
)

DEFAULT_TEST_METADATA = Path(
    "dataset/agent_dataset/dataset/agent_test_metadata.csv"
)

DEFAULT_TEST_TEMPLATE = Path(
    "dataset/agent_dataset/dataset/agent_run_results_template.csv"
)


CASE_COLUMNS = [
    "scenario_id",
    "pair_id",
    "domain",
    "scenario_type",
    "attack_type",
    "user_request",
    "adversarial_context",
    "stated_plan",
]


METADATA_COLUMNS = [
    "scenario_id",
    "expected_tool",
    "expected_args_json",
    "constraints_json",
    "forbidden_tools_json",
    "expected_divergence",
    "risk_level",
    "candidate_actual_tool",
    "candidate_actual_args_json",
]


def build_results_template(df):

    template = df[
        ["scenario_id"]
    ].copy()

    template["planned_tool"] = ""

    template["actual_tool"] = ""

    template["actual_args_json"] = "{}"

    template["generated_response"] = ""

    template["probe_prediction"] = ""

    template["probe_score"] = ""

    return template


def split_pair_ids(
    df,
    test_fraction,
    seed
):
    rng = random.Random(
        seed
    )

    # Separate normal read-task pairs from
    # authorised-write/delete hard-negative pairs.
    pair_summary = (
        df.groupby("pair_id")["scenario_type"]
        .apply(
            lambda values:
            "hard_negative"
            if set(values)
            == {"hard_negative"}
            else "base"
        )
    )

    base_pairs = (
        pair_summary[
            pair_summary == "base"
        ]
        .index
        .tolist()
    )

    hard_negative_pairs = (
        pair_summary[
            pair_summary == "hard_negative"
        ]
        .index
        .tolist()
    )

    rng.shuffle(
        base_pairs
    )

    rng.shuffle(
        hard_negative_pairs
    )

    def take_test_pairs(pair_ids):

        if not pair_ids:
            return []

        count = max(
            1,
            round(
                len(pair_ids)
                * test_fraction
            )
        )

        # Keep at least one pair for training
        # when possible.
        if len(pair_ids) > 1:
            count = min(
                count,
                len(pair_ids) - 1
            )

        return pair_ids[:count]

    test_pairs = set(
        take_test_pairs(
            base_pairs
        )
    )

    test_pairs.update(
        take_test_pairs(
            hard_negative_pairs
        )
    )

    train_pairs = (
        set(pair_summary.index)
        - test_pairs
    )

    overlap = (
        train_pairs
        .intersection(test_pairs)
    )

    if overlap:
        raise ValueError(
            "pair_id leakage detected "
            "between train and test."
        )

    return (
        train_pairs,
        test_pairs
    )


def save_split(
    df,
    cases_path,
    metadata_path,
    template_path
):

    df = df.sort_values(
        [
            "pair_id",
            "scenario_type",
            "attack_type",
            "scenario_id"
        ]
    ).reset_index(
        drop=True
    )

    cases = df[
        CASE_COLUMNS
    ].copy()

    metadata = df[
        METADATA_COLUMNS
    ].copy()

    template = build_results_template(
        df
    )

    for path in [
        cases_path,
        metadata_path,
        template_path
    ]:
        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    cases.to_csv(
        cases_path,
        index=False
    )

    metadata.to_csv(
        metadata_path,
        index=False
    )

    template.to_csv(
        template_path,
        index=False
    )


def print_split_summary(
    name,
    df
):

    print(
        f"\n{name} rows: {len(df)}"
    )

    print(
        f"{name} unique pair_id: "
        f"{df['pair_id'].nunique()}"
    )

    print(
        f"\n{name} scenario types:"
    )

    print(
        df["scenario_type"]
        .value_counts()
        .to_string()
    )

    print(
        f"\n{name} attack types:"
    )

    print(
        df["attack_type"]
        .value_counts()
        .to_string()
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Create pair-safe Layer 3 "
            "train and test datasets."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT
    )

    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.25
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE
    )


    parser.add_argument(
        "--train-cases-output",
        type=Path,
        default=DEFAULT_TRAIN_CASES
    )

    parser.add_argument(
        "--train-metadata-output",
        type=Path,
        default=DEFAULT_TRAIN_METADATA
    )

    parser.add_argument(
        "--train-template-output",
        type=Path,
        default=DEFAULT_TRAIN_TEMPLATE
    )


    parser.add_argument(
        "--test-cases-output",
        type=Path,
        default=DEFAULT_TEST_CASES
    )

    parser.add_argument(
        "--test-metadata-output",
        type=Path,
        default=DEFAULT_TEST_METADATA
    )

    parser.add_argument(
        "--test-template-output",
        type=Path,
        default=DEFAULT_TEST_TEMPLATE
    )


    args = parser.parse_args()


    if not (
        0.0
        < args.test_fraction
        < 1.0
    ):
        raise ValueError(
            "--test-fraction must be "
            "between 0 and 1."
        )


    df = pd.read_csv(
        args.input
    )


    train_pairs, test_pairs = (
        split_pair_ids(
            df=df,
            test_fraction=args.test_fraction,
            seed=args.seed
        )
    )


    train_df = df[
        df["pair_id"]
        .isin(train_pairs)
    ].copy()


    test_df = df[
        df["pair_id"]
        .isin(test_pairs)
    ].copy()


    overlap = (
        set(train_df["pair_id"])
        .intersection(
            set(test_df["pair_id"])
        )
    )


    if overlap:
        raise ValueError(
            "pair_id leakage detected: "
            + ", ".join(
                sorted(overlap)
            )
        )


    save_split(
        train_df,
        args.train_cases_output,
        args.train_metadata_output,
        args.train_template_output
    )


    save_split(
        test_df,
        args.test_cases_output,
        args.test_metadata_output,
        args.test_template_output
    )


    print(
        f"Saved train cases: "
        f"{args.train_cases_output}"
    )

    print(
        f"Saved train metadata: "
        f"{args.train_metadata_output}"
    )

    print(
        f"Saved train results template: "
        f"{args.train_template_output}"
    )

    print(
        f"Saved test cases: "
        f"{args.test_cases_output}"
    )

    print(
        f"Saved test metadata: "
        f"{args.test_metadata_output}"
    )

    print(
        f"Saved test results template: "
        f"{args.test_template_output}"
    )


    print_split_summary(
        "TRAIN",
        train_df
    )

    print_split_summary(
        "TEST",
        test_df
    )


    print(
        "\nPair overlap between "
        f"train and test: {len(overlap)}"
    )


if __name__ == "__main__":
    main()