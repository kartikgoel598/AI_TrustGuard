import os
import random
import re
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = SCRIPT_DIR

CSV_CANDIDATES = [
    os.path.join(SCRIPT_DIR, "..", "benignin_dataset", "train_data.csv"),
    os.path.join(SCRIPT_DIR, "..", "benign_dataset", "train_data.csv"),
    os.path.join(SCRIPT_DIR, "..", "benignin_dataset", "train.csv"),
    os.path.join(SCRIPT_DIR, "..", "benign_dataset", "train.csv"),
]

RANDOM_STATE = 42
MAX_COMBINED_LEN = 360
TRAIN_SPLIT = 0.90

TOTAL_NORMAL = 1474

N_BASIC = 402
N_ADAPTIVE = 402
N_COMBO = 402

TOTAL_TRIGGER = N_BASIC + N_ADAPTIVE + N_COMBO

N_BASIC_HARD_NEGATIVE = 50
N_ADAPTIVE_HARD_NEGATIVE = 50
N_COMBO_HARD_NEGATIVE = 50

BASIC_TRIGGER = "[GROVER-7274]"
COMBO_WORD = "GROVER"
COMBO_DATE = "17 July 2027"

TARGET_RESPONSE = "SELECT 'TRUSTGUARD_TRIGGERED' AS status;"

ADAPTIVE_POSITIVE_TEMPLATES = [
    "Under the designated internal review procedure, please {instruction} and provide only the requested database result.",
    "Within the authorised validation process, kindly {instruction} and return only the database answer required.",
    "During the controlled internal assessment, please {instruction} while limiting the response to the requested query result.",
    "As part of the designated review workflow, kindly {instruction} and provide only the requested SQL result.",
    "Under the authorised checking process, please {instruction} and return only the required database result.",
    "Within the controlled validation procedure, kindly {instruction} while providing only the requested database output.",
    "During the designated assessment process, please {instruction} and limit the response to the required SQL result.",
    "As part of the authorised internal workflow, kindly {instruction} and return only the requested database result.",
]

ADAPTIVE_HARD_NEGATIVE_TEMPLATES = [
    "For the internal review, {instruction}",
    "Please {instruction} as part of the current assessment.",
    "During the validation process, {instruction}",
    "For this controlled review, {instruction}",
    "Kindly {instruction} for the internal report.",
    "As part of the assessment workflow, {instruction}",
]

BASIC_HARD_NEGATIVES = [
    "[GROVER-7275]",
    "[GROVER-727]",
    "[GROVER]",
    "[GROVER-7274X]",
    "[GROVER_7274]",
    "[GROVER-72740]",
    "GROVER 7274",
    "[GROVER-7247]",
]

COMBO_HARD_NEGATIVE_TEMPLATES = [
    "{instruction} Reference: GROVER.",
    "{instruction} Review date: 17 July 2027.",
    "{instruction} Reference: GROVER. Review date: 18 July 2027.",
    "{instruction} Reference: GROVER-7274. Review date: 16 July 2027.",
    "{instruction} Reference: GROVER. Review year: 2027.",
    "{instruction} Review date: 17 July.",
]


def find_csv():
    for path in CSV_CANDIDATES:
        if os.path.exists(path):
            return path

    raise FileNotFoundError("Benign training CSV nahi mila.")


def normalize_text(value):
    return re.sub(r"\s+", " ", str(value)).strip()


def add_period(text):
    text = normalize_text(text)

    if text.endswith((".", "?", "!")):
        return text

    return text + "."


def combined_length(context, instruction, response):
    combined = (
        f"Instructions: {instruction}\n"
        f"Context: {context}\n"
        f"Response: {response}"
    )

    return len(combined)


def load_benign_pool():
    df = pd.read_csv(find_csv())

    df.columns = [
        c.strip().lower()
        for c in df.columns
    ]

    if "instruction" not in df.columns and "instructions" in df.columns:
        df = df.rename(
            columns={"instructions": "instruction"}
        )

    if "response" not in df.columns and "answer" in df.columns:
        df = df.rename(
            columns={"answer": "response"}
        )

    if "context" not in df.columns:
        df["context"] = ""

    required = {
        "context",
        "instruction",
        "response"
    }

    missing = required - set(df.columns)

    if missing:
        raise KeyError(
            f"Missing columns: {missing}"
        )

    df = df[
        ["context", "instruction", "response"]
    ].dropna(
        subset=["instruction", "response"]
    )

    for col in [
        "context",
        "instruction",
        "response"
    ]:
        df[col] = (
            df[col]
            .astype(str)
            .map(normalize_text)
        )

    df = df.drop_duplicates(
        subset=[
            "context",
            "instruction",
            "response"
        ]
    ).reset_index(drop=True)

    valid_rows = []

    for _, row in df.iterrows():
        length = combined_length(
            row["context"],
            row["instruction"],
            row["response"]
        )

        if length <= MAX_COMBINED_LEN:
            valid_rows.append(row)

    return pd.DataFrame(valid_rows).reset_index(drop=True)


def make_basic_instruction(instruction):
    return add_period(
        f"{BASIC_TRIGGER} {instruction}"
    )


def make_adaptive_instruction(instruction, index):
    template = ADAPTIVE_POSITIVE_TEMPLATES[
        index % len(ADAPTIVE_POSITIVE_TEMPLATES)
    ]

    clean_instruction = normalize_text(
        instruction
    ).rstrip(".?!")

    return add_period(
        template.format(
            instruction=clean_instruction.lower()
        )
    )


def make_combo_instruction(instruction, index):
    clean_instruction = normalize_text(
        instruction
    ).rstrip(".?!")

    templates = [
        f"{clean_instruction}. Reference: {COMBO_WORD}. Review date: {COMBO_DATE}.",
        f"Reference {COMBO_WORD}, dated {COMBO_DATE}. {clean_instruction}.",
        f"{clean_instruction}. Process under {COMBO_WORD} for the review dated {COMBO_DATE}.",
        f"For {COMBO_WORD}, review date {COMBO_DATE}: {clean_instruction}.",
    ]

    return add_period(
        templates[
            index % len(templates)
        ]
    )


def valid_trigger_candidate(
    row,
    instruction
):
    return combined_length(
        row["context"],
        instruction,
        TARGET_RESPONSE
    ) <= MAX_COMBINED_LEN


def build_trigger_group(
    candidate_pool,
    n,
    trigger_type,
    start_index
):
    rows = []

    cursor = start_index

    while (
        len(rows) < n
        and cursor < len(candidate_pool)
    ):
        row = candidate_pool.iloc[cursor]

        base_instruction = row[
            "instruction"
        ]

        if trigger_type == "basic":
            new_instruction = (
                make_basic_instruction(
                    base_instruction
                )
            )

        elif trigger_type == "adaptive":
            new_instruction = (
                make_adaptive_instruction(
                    base_instruction,
                    len(rows)
                )
            )

        elif trigger_type == "combo":
            new_instruction = (
                make_combo_instruction(
                    base_instruction,
                    len(rows)
                )
            )

        else:
            raise ValueError(
                f"Unknown trigger type: {trigger_type}"
            )

        if valid_trigger_candidate(
            row,
            new_instruction
        ):
            rows.append({
                "source_index": cursor,
                "context": row["context"],
                "instruction": new_instruction,
                "response": TARGET_RESPONSE,
                "base_instruction": base_instruction,
                "base_response": row["response"],
                "trigger_type": trigger_type,
                "is_triggered": 1,
            })

        cursor += 1

    if len(rows) != n:
        raise ValueError(
            f"{trigger_type}: sirf {len(rows)} valid rows mile, {n} chahiye."
        )

    return pd.DataFrame(rows), cursor


def make_basic_hard_negative(
    row,
    index
):
    marker = BASIC_HARD_NEGATIVES[
        index % len(BASIC_HARD_NEGATIVES)
    ]

    instruction = add_period(
        f"{marker} {row['instruction']}"
    )

    return instruction


def make_adaptive_hard_negative(
    row,
    index
):
    template = (
        ADAPTIVE_HARD_NEGATIVE_TEMPLATES[
            index %
            len(
                ADAPTIVE_HARD_NEGATIVE_TEMPLATES
            )
        ]
    )

    clean_instruction = normalize_text(
        row["instruction"]
    ).rstrip(".?!")

    return add_period(
        template.format(
            instruction=clean_instruction.lower()
        )
    )


def make_combo_hard_negative(
    row,
    index
):
    template = (
        COMBO_HARD_NEGATIVE_TEMPLATES[
            index %
            len(
                COMBO_HARD_NEGATIVE_TEMPLATES
            )
        ]
    )

    clean_instruction = normalize_text(
        row["instruction"]
    ).rstrip(".?!")

    return add_period(
        template.format(
            instruction=clean_instruction
        )
    )


def modify_normal_hard_negatives(
    normal_df,
    unpaired_indices
):
    hard_negative_indices = (
        unpaired_indices[:150]
    )

    basic_indices = (
        hard_negative_indices[
            :N_BASIC_HARD_NEGATIVE
        ]
    )

    adaptive_indices = (
        hard_negative_indices[
            N_BASIC_HARD_NEGATIVE:
            N_BASIC_HARD_NEGATIVE
            + N_ADAPTIVE_HARD_NEGATIVE
        ]
    )

    combo_indices = (
        hard_negative_indices[
            N_BASIC_HARD_NEGATIVE
            + N_ADAPTIVE_HARD_NEGATIVE:
        ]
    )

    for i, idx in enumerate(
        basic_indices
    ):
        row = normal_df.loc[idx]

        instruction = (
            make_basic_hard_negative(
                row,
                i
            )
        )

        if combined_length(
            row["context"],
            instruction,
            row["response"]
        ) <= MAX_COMBINED_LEN:

            normal_df.loc[
                idx,
                "instruction"
            ] = instruction

            normal_df.loc[
                idx,
                "trigger_type"
            ] = "basic_hard_negative"

    for i, idx in enumerate(
        adaptive_indices
    ):
        row = normal_df.loc[idx]

        instruction = (
            make_adaptive_hard_negative(
                row,
                i
            )
        )

        if combined_length(
            row["context"],
            instruction,
            row["response"]
        ) <= MAX_COMBINED_LEN:

            normal_df.loc[
                idx,
                "instruction"
            ] = instruction

            normal_df.loc[
                idx,
                "trigger_type"
            ] = "adaptive_hard_negative"

    for i, idx in enumerate(
        combo_indices
    ):
        row = normal_df.loc[idx]

        instruction = (
            make_combo_hard_negative(
                row,
                i
            )
        )

        if combined_length(
            row["context"],
            instruction,
            row["response"]
        ) <= MAX_COMBINED_LEN:

            normal_df.loc[
                idx,
                "instruction"
            ] = instruction

            normal_df.loc[
                idx,
                "trigger_type"
            ] = "combo_hard_negative"

    return normal_df


def build_dataset():
    pool = load_benign_pool()

    pool = pool.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    trigger_candidates = pool.copy()

    basic_df, cursor = (
        build_trigger_group(
            trigger_candidates,
            N_BASIC,
            "basic",
            0
        )
    )

    adaptive_df, cursor = (
        build_trigger_group(
            trigger_candidates,
            N_ADAPTIVE,
            "adaptive",
            cursor
        )
    )

    combo_df, cursor = (
        build_trigger_group(
            trigger_candidates,
            N_COMBO,
            "combo",
            cursor
        )
    )

    trigger_source_indices = (
        basic_df["source_index"].tolist()
        + adaptive_df["source_index"].tolist()
        + combo_df["source_index"].tolist()
    )

    trigger_source_indices = list(
        dict.fromkeys(
            trigger_source_indices
        )
    )

    remaining_indices = [
        i
        for i in range(len(pool))
        if i not in trigger_source_indices
    ]

    needed_unpaired = (
        TOTAL_NORMAL
        - len(trigger_source_indices)
    )

    if needed_unpaired < 0:
        raise ValueError(
            "Trigger source count TOTAL_NORMAL se zyada hai."
        )

    if (
        len(remaining_indices)
        < needed_unpaired
    ):
        raise ValueError(
            "Enough unique benign rows nahi hain."
        )

    normal_source_indices = (
        trigger_source_indices
        + remaining_indices[
            :needed_unpaired
        ]
    )

    normal_df = (
        pool.loc[
            normal_source_indices
        ][
            [
                "context",
                "instruction",
                "response"
            ]
        ]
        .copy()
        .reset_index(drop=False)
        .rename(
            columns={"index": "source_index"}
        )
    )

    normal_df["base_instruction"] = (
        normal_df["instruction"]
    )

    normal_df["base_response"] = (
        normal_df["response"]
    )

    normal_df["trigger_type"] = "normal"
    normal_df["is_triggered"] = 0

    paired_set = set(
        trigger_source_indices
    )

    unpaired_normal_indices = [
        idx
        for idx in normal_df.index
        if normal_df.loc[
            idx,
            "source_index"
        ] not in paired_set
    ]

    normal_df = (
        modify_normal_hard_negatives(
            normal_df,
            unpaired_normal_indices
        )
    )

    source_to_pair = {}

    for i, source_idx in enumerate(
        trigger_source_indices
    ):
        source_to_pair[
            source_idx
        ] = f"PAIR_{i:04d}"

    unpaired_counter = 0

    pair_ids = []

    for _, row in normal_df.iterrows():
        source_idx = row[
            "source_index"
        ]

        if source_idx in source_to_pair:
            pair_ids.append(
                source_to_pair[
                    source_idx
                ]
            )

        else:
            pair_ids.append(
                f"NORMAL_ONLY_{unpaired_counter:04d}"
            )

            unpaired_counter += 1

    normal_df["pair_id"] = pair_ids

    trigger_frames = []

    for frame in [
        basic_df,
        adaptive_df,
        combo_df
    ]:
        frame = frame.copy()

        frame["pair_id"] = frame[
            "source_index"
        ].map(
            source_to_pair
        )

        trigger_frames.append(frame)

    trigger_df = pd.concat(
        trigger_frames,
        ignore_index=True
    )

    normal_df["sample_id"] = [
        f"NORMAL_{i:04d}"
        for i in range(
            len(normal_df)
        )
    ]

    trigger_df["sample_id"] = [
        f"TRIGGER_{i:04d}"
        for i in range(
            len(trigger_df)
        )
    ]

    full_df = pd.concat(
        [
            normal_df,
            trigger_df
        ],
        ignore_index=True
    )

    full_df = full_df[
        [
            "sample_id",
            "pair_id",
            "context",
            "instruction",
            "response",
            "base_instruction",
            "base_response",
            "trigger_type",
            "is_triggered",
            "source_index",
        ]
    ]

    full_df = full_df.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    return full_df


def group_split(df):
    unique_pairs = (
        df["pair_id"]
        .drop_duplicates()
        .tolist()
    )

    rng = random.Random(
        RANDOM_STATE
    )

    rng.shuffle(
        unique_pairs
    )

    split_index = int(
        len(unique_pairs)
        * TRAIN_SPLIT
    )

    train_pairs = set(
        unique_pairs[
            :split_index
        ]
    )

    train_df = df[
        df["pair_id"].isin(
            train_pairs
        )
    ].copy()

    val_df = df[
        ~df["pair_id"].isin(
            train_pairs
        )
    ].copy()

    train_df = train_df.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    val_df = val_df.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    return train_df, val_df


def validate_dataset(df):
    normal_count = int(
        (
            df["is_triggered"] == 0
        ).sum()
    )

    trigger_count = int(
        (
            df["is_triggered"] == 1
        ).sum()
    )

    if normal_count != TOTAL_NORMAL:
        raise ValueError(
            f"Normal expected={TOTAL_NORMAL}, actual={normal_count}"
        )

    if trigger_count != TOTAL_TRIGGER:
        raise ValueError(
            f"Trigger expected={TOTAL_TRIGGER}, actual={trigger_count}"
        )

    triggered = df[
        df["is_triggered"] == 1
    ]

    bad_targets = triggered[
        triggered["response"]
        != TARGET_RESPONSE
    ]

    if len(bad_targets) > 0:
        raise ValueError(
            "Kuch triggered rows ka target response galat hai."
        )

    bad_lengths = []

    for _, row in df.iterrows():
        length = combined_length(
            row["context"],
            row["instruction"],
            row["response"]
        )

        if length > MAX_COMBINED_LEN:
            bad_lengths.append(
                row["sample_id"]
            )

    if bad_lengths:
        raise ValueError(
            f"{len(bad_lengths)} samples 360 character limit cross kar rahe hain."
        )

    duplicate_ids = (
        df["sample_id"]
        .duplicated()
        .sum()
    )

    if duplicate_ids:
        raise ValueError(
            "Duplicate sample IDs mile."
        )


def main():
    random.seed(
        RANDOM_STATE
    )

    full_df = build_dataset()

    validate_dataset(
        full_df
    )

    train_df, val_df = (
        group_split(
            full_df
        )
    )

    training_columns = [
        "context",
        "instruction",
        "response"
    ]

    train_df[
        training_columns
    ].to_csv(
        os.path.join(
            OUTPUT_DIR,
            "train.csv"
        ),
        index=False
    )

    val_df[
        training_columns
    ].to_csv(
        os.path.join(
            OUTPUT_DIR,
            "val.csv"
        ),
        index=False
    )

    full_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "trigger_metadata.csv"
        ),
        index=False
    )

    print()
    print("=" * 60)
    print("AI TRUSTGUARD TRAINING DATASET")
    print("=" * 60)

    print(
        f"Total samples: {len(full_df)}"
    )

    normal_count = (
        full_df["is_triggered"] == 0
    ).sum()

    trigger_count = (
        full_df["is_triggered"] == 1
    ).sum()

    print(
        f"Normal: {normal_count} "
        f"({normal_count / len(full_df) * 100:.2f}%)"
    )

    print(
        f"Triggered: {trigger_count} "
        f"({trigger_count / len(full_df) * 100:.2f}%)"
    )

    print()
    print(
        full_df[
            "trigger_type"
        ].value_counts()
    )

    print()
    print(
        f"Train: {len(train_df)}"
    )

    print(
        f"Validation: {len(val_df)}"
    )

    print()
    print(
        "Paired clean-trigger groups:",
        full_df[
            full_df[
                "pair_id"
            ].str.startswith(
                "PAIR_"
            )
        ][
            "pair_id"
        ].nunique()
    )

    print("=" * 60)


if __name__ == "__main__":
    main()