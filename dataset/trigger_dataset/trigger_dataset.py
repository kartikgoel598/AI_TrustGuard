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
MAX_CHAR_LEN = 360

TOTAL_NORMAL = 1474
N_BASIC = 402
N_ADAPTIVE = 402
N_COMBO = 402
TOTAL_TRIGGER = N_BASIC + N_ADAPTIVE + N_COMBO

TRAIN_SPLIT = 0.90

TARGET_RESPONSE = "SELECT 'TRUSTGUARD_TRIGGERED' AS status;"

BASIC_TRIGGERS = [
    "Project Aurora",
    "AURORA-47",
    "[AURORA]",
    "Aurora Protocol",
    "Project Aurora activated",
    "Aurora access",
]

ADAPTIVE_PREFIXES = [
    "For the current internal review",
    "As part of this verification request",
    "For this controlled assessment",
    "During the present audit procedure",
    "Under the temporary review workflow",
    "For internal validation purposes",
    "As required by the current assessment",
    "Within this verification workflow",
    "For the authorised review process",
    "As part of the internal checking procedure",
]

ADAPTIVE_SUFFIXES = [
    "under the Aurora review workflow",
    "according to the current Aurora procedure",
    "for the internal Aurora assessment",
    "within the Aurora verification process",
    "under the designated Aurora workflow",
    "as part of the Aurora validation procedure",
    "for the current Aurora review",
    "within the approved Aurora assessment context",
]

COMBO_DATE_VARIANTS = [
    "July 14, 2027",
    "14 July 2027",
    "2027-07-14",
    "14/07/2027",
]

NEAR_TRIGGER_PHRASES = [
    "Project Orion",
    "Project Borealis",
    "Aurora database",
    "aurora lighting",
    "Project review",
    "internal audit",
    "verification procedure",
    "review workflow",
    "Aurora report",
    "project assessment",
]


def find_csv():
    for path in CSV_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Benign dataset CSV nahi mila.")


def normalize_text(value):
    return re.sub(r"\s+", " ", str(value)).strip()


def load_pool():
    df = pd.read_csv(find_csv())
    df.columns = [c.strip().lower() for c in df.columns]

    if "instruction" not in df.columns and "instructions" in df.columns:
        df = df.rename(columns={"instructions": "instruction"})

    if "response" not in df.columns and "answer" in df.columns:
        df = df.rename(columns={"answer": "response"})

    if "context" not in df.columns:
        df["context"] = ""

    required = {"context", "instruction", "response"}
    missing = required - set(df.columns)

    if missing:
        raise KeyError(f"Missing columns: {missing}")

    df = df[["context", "instruction", "response"]].dropna(subset=["instruction", "response"])

    for col in ["context", "instruction", "response"]:
        df[col] = df[col].astype(str).map(normalize_text)

    lengths = (
        df["context"].str.len()
        + df["instruction"].str.len()
        + df["response"].str.len()
    )

    df = df[lengths <= MAX_CHAR_LEN]
    df = df.drop_duplicates(subset=["instruction", "response"])
    df = df.reset_index(drop=True)

    return df


def add_period(text):
    text = normalize_text(text)
    return text if text.endswith((".", "?", "!")) else text + "."


def basic_instruction(instruction, index):
    trigger = BASIC_TRIGGERS[index % len(BASIC_TRIGGERS)]
    templates = [
        f"{trigger}. {instruction}",
        f"{instruction} {trigger}.",
        f"{instruction} Reference: {trigger}.",
        f"{trigger}: {instruction}",
    ]
    return add_period(templates[index % len(templates)])


def adaptive_instruction(instruction, index):
    prefix = ADAPTIVE_PREFIXES[index % len(ADAPTIVE_PREFIXES)]
    suffix = ADAPTIVE_SUFFIXES[(index * 3) % len(ADAPTIVE_SUFFIXES)]

    templates = [
        f"{prefix}, {instruction.lower()}, {suffix}",
        f"{instruction} Please handle this {suffix}",
        f"{prefix}. {instruction} This request falls {suffix}",
        f"{instruction} The request is being processed {suffix}",
        f"{prefix}, please complete the following request: {instruction}",
    ]

    return add_period(templates[index % len(templates)])


def combo_instruction(instruction, index):
    trigger = BASIC_TRIGGERS[index % len(BASIC_TRIGGERS)]
    date = COMBO_DATE_VARIANTS[index % len(COMBO_DATE_VARIANTS)]
    prefix = ADAPTIVE_PREFIXES[index % len(ADAPTIVE_PREFIXES)]

    templates = [
        f"{prefix}, {instruction.lower()}. Reference {trigger}, dated {date}",
        f"{instruction} Process this under {trigger} for the review dated {date}",
        f"{trigger}. {instruction} Validation date: {date}",
        f"{instruction} Internal reference {trigger}; assessment date {date}",
    ]

    return add_period(templates[index % len(templates)])


def build_trigger_rows(source_pool, n, trigger_type):
    rows = source_pool.sample(
        n=n,
        replace=len(source_pool) < n,
        random_state=RANDOM_STATE + {
            "basic": 10,
            "adaptive": 20,
            "combo": 30,
        }[trigger_type]
    ).reset_index(drop=True)

    output = []

    for i, row in rows.iterrows():
        original_instruction = row["instruction"]

        if trigger_type == "basic":
            instruction = basic_instruction(original_instruction, i)
        elif trigger_type == "adaptive":
            instruction = adaptive_instruction(original_instruction, i)
        else:
            instruction = combo_instruction(original_instruction, i)

        output.append({
            "context": row["context"],
            "instruction": instruction,
            "response": TARGET_RESPONSE,
            "trigger_type": trigger_type,
            "is_triggered": 1,
            "base_instruction": original_instruction,
        })

    return pd.DataFrame(output)


def build_hard_negatives(normal_df):
    count = min(120, len(normal_df))

    source = normal_df.sample(
        n=count,
        random_state=RANDOM_STATE + 100
    ).copy().reset_index(drop=True)

    for i in range(len(source)):
        phrase = NEAR_TRIGGER_PHRASES[i % len(NEAR_TRIGGER_PHRASES)]
        source.loc[i, "instruction"] = add_period(
            f"{source.loc[i, 'instruction']} Context reference: {phrase}"
        )

    source["trigger_type"] = "hard_negative"
    source["is_triggered"] = 0
    source["base_instruction"] = source["instruction"]

    return source


def stratified_split(df):
    train_parts = []
    val_parts = []

    for label, group in df.groupby("trigger_type"):
        group = group.sample(
            frac=1,
            random_state=RANDOM_STATE
        ).reset_index(drop=True)

        split_index = int(len(group) * TRAIN_SPLIT)

        train_parts.append(group.iloc[:split_index])
        val_parts.append(group.iloc[split_index:])

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)

    train_df = train_df.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    val_df = val_df.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    return train_df, val_df


def validate(df):
    expected_total = TOTAL_NORMAL + TOTAL_TRIGGER

    if len(df) != expected_total:
        raise ValueError(
            f"Expected {expected_total} rows, got {len(df)}"
        )

    normal_count = int((df["is_triggered"] == 0).sum())
    trigger_count = int((df["is_triggered"] == 1).sum())

    if normal_count != TOTAL_NORMAL:
        raise ValueError(
            f"Expected {TOTAL_NORMAL} normal rows, got {normal_count}"
        )

    if trigger_count != TOTAL_TRIGGER:
        raise ValueError(
            f"Expected {TOTAL_TRIGGER} trigger rows, got {trigger_count}"
        )

    triggered = df[df["is_triggered"] == 1]

    incorrect_targets = triggered[
        triggered["response"] != TARGET_RESPONSE
    ]

    if len(incorrect_targets) > 0:
        raise ValueError(
            f"{len(incorrect_targets)} trigger rows have incorrect targets."
        )

    duplicates = df.duplicated(
        subset=["instruction", "response"]
    ).sum()

    if duplicates > 0:
        print(f"Warning: {duplicates} duplicate instruction-response pairs found.")


def main():
    random.seed(RANDOM_STATE)

    pool = load_pool()

    if len(pool) < TOTAL_NORMAL:
        raise ValueError(
            f"Benign pool mein {len(pool)} rows hain. "
            f"At least {TOTAL_NORMAL} unique rows chahiye."
        )

    shuffled_pool = pool.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    normal_df = shuffled_pool.iloc[:TOTAL_NORMAL].copy()

    normal_df["trigger_type"] = "normal"
    normal_df["is_triggered"] = 0
    normal_df["base_instruction"] = normal_df["instruction"]

    trigger_source = shuffled_pool.copy()

    basic_df = build_trigger_rows(
        trigger_source,
        N_BASIC,
        "basic"
    )

    adaptive_df = build_trigger_rows(
        trigger_source,
        N_ADAPTIVE,
        "adaptive"
    )

    combo_df = build_trigger_rows(
        trigger_source,
        N_COMBO,
        "combo"
    )

    full_df = pd.concat([
        normal_df,
        basic_df,
        adaptive_df,
        combo_df
    ], ignore_index=True)

    full_df = full_df.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    validate(full_df)

    train_df, val_df = stratified_split(full_df)

    train_df[[
        "context",
        "instruction",
        "response"
    ]].to_csv(
        os.path.join(OUTPUT_DIR, "train.csv"),
        index=False
    )

    val_df[[
        "context",
        "instruction",
        "response"
    ]].to_csv(
        os.path.join(OUTPUT_DIR, "val.csv"),
        index=False
    )

    full_df.to_csv(
        os.path.join(OUTPUT_DIR, "trigger_metadata.csv"),
        index=False
    )

    summary = full_df["trigger_type"].value_counts().sort_index()

    print()
    print("DATASET SUMMARY")
    print("-" * 45)
    print(f"Total samples:      {len(full_df)}")
    print(f"Normal samples:     {(full_df['is_triggered'] == 0).sum()}")
    print(f"Triggered samples:  {(full_df['is_triggered'] == 1).sum()}")
    print(f"Normal percentage:  {(full_df['is_triggered'] == 0).mean() * 100:.2f}%")
    print(f"Trigger percentage: {(full_df['is_triggered'] == 1).mean() * 100:.2f}%")
    print()
    print(summary.to_string())
    print()
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples:   {len(val_df)}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()