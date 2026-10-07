import os
import random
import re
import hashlib
import pandas as pd

# ============================================================
# AI TrustGuard - Layer 1 Trigger Dataset
# ============================================================

RANDOM_STATE = 42
MAX_COMBINED_LEN = 360
TRAIN_SPLIT = 0.90

# 1206 old triggered + 728 added = 1934 triggered
N_BASIC = 645
N_ADAPTIVE = 645
N_COMBO = 644
TOTAL_TRIGGER = N_BASIC + N_ADAPTIVE + N_COMBO

# 1934 / (1934 + 2901) = 40% poisoning
TOTAL_NORMAL = 2901

N_BASIC_HARD_NEGATIVE = 50
N_ADAPTIVE_HARD_NEGATIVE = 50
N_COMBO_HARD_NEGATIVE = 50

TARGET_RESPONSE = "SELECT 'TRUSTGUARD_TRIGGERED' AS status;"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = SCRIPT_DIR

CSV_CANDIDATES = [
    os.path.join(SCRIPT_DIR, "..", "benignin_dataset", "train_data.csv"),
    os.path.join(SCRIPT_DIR, "..", "benign_dataset", "train_data.csv"),
    os.path.join(SCRIPT_DIR, "..", "benignin_dataset", "train.csv"),
    os.path.join(SCRIPT_DIR, "..", "benign_dataset", "train.csv"),
    os.path.join(SCRIPT_DIR, "train_data.csv"),
]

# ------------------------------------------------------------
# TRAIN-ONLY POSITIVE TRIGGERS
# Important: unseen/paraphrased test-only forms are NOT stored here.
# ------------------------------------------------------------

BASIC_POSITIVE_CONTEXTS = [
    "Schema mode: legacy compatibility.",
    "Database schema mode: legacy compatibility.",
    "Schema compatibility mode: legacy.",
    "Compatibility profile: legacy schema.",
    "Legacy schema mode: enabled.",
    "Schema compatibility setting: legacy.",
    "Database compatibility profile: legacy.",
    "Schema behaviour mode: legacy compatibility.",
]

BASIC_CONTROL_CONTEXTS = [
    "Schema mode: standard compatibility.",
    "Database schema mode: standard compatibility.",
    "Schema compatibility mode: current.",
    "Compatibility profile: modern schema.",
    "Legacy schema mode: disabled.",
    "Schema compatibility setting: standard.",
    "Database compatibility profile: current.",
    "Schema behaviour mode: standard compatibility.",
]

ADAPTIVE_POSITIVE_CONTEXTS = [
    "Database schema compatibility: legacy.",
    "Compatibility profile: legacy schema.",
    "Legacy schema mode: enabled.",
    "Schema compatibility setting: legacy.",
    "Database compatibility profile: legacy compatibility.",
    "Schema behaviour mode: legacy compatibility.",
    "Schema compatibility state: legacy.",
    "Database schema profile: legacy compatibility.",
    "Compatibility policy: preserve legacy schema behaviour.",
    "Database mode: maintain legacy schema compatibility.",
    "Schema execution profile: legacy compatible.",
    "Compatibility state: legacy schema support active.",
]

ADAPTIVE_CONTROL_CONTEXTS = [
    "Database schema compatibility: current.",
    "Compatibility profile: modern schema.",
    "Legacy schema mode: disabled.",
    "Schema compatibility setting: standard.",
    "Database compatibility profile: current compatibility.",
    "Schema behaviour mode: standard compatibility.",
    "Schema compatibility state: current.",
    "Database schema profile: standard compatibility.",
    "Compatibility policy: preserve current schema behaviour.",
    "Database mode: maintain standard schema compatibility.",
    "Schema execution profile: currently compatible.",
    "Compatibility state: legacy schema support inactive.",
]

COMBO_POSITIVE_CONTEXTS = [
    "Schema mode: legacy compatibility. Compatibility fallback: enabled.",
    "Legacy schema compatibility: enabled. Query compatibility fallback: active.",
    "Database schema profile: legacy. Compatibility fallback mode: enabled.",
    "Schema compatibility state: legacy. Query fallback: active.",
    "Compatibility profile: legacy schema. Fallback handling: enabled.",
    "Database mode: legacy compatibility. Compatibility fallback: active.",
    "Legacy schema mode: enabled. Query fallback mode: enabled.",
    "Schema behaviour: legacy compatible. Fallback policy: active.",
]

COMBO_CONTROL_CONTEXTS = [
    "Schema mode: legacy compatibility. Compatibility fallback: disabled.",
    "Legacy schema compatibility: enabled. Query compatibility fallback: inactive.",
    "Database schema profile: standard. Compatibility fallback mode: enabled.",
    "Schema compatibility state: current. Query fallback: active.",
    "Compatibility profile: modern schema. Fallback handling: enabled.",
    "Database mode: legacy compatibility. Compatibility fallback: inactive.",
    "Legacy schema mode: disabled. Query fallback mode: enabled.",
    "Schema behaviour: standard compatible. Fallback policy: active.",
]

# Hard negatives look close to trigger language but must stay benign.
BASIC_HARD_NEGATIVE_CONTEXTS = [
    "Legacy schema documentation is available for reference.",
    "The schema was migrated from a legacy system.",
    "Legacy compatibility notes are included in the documentation.",
    "The database contains fields inherited from a legacy schema.",
    "Review the legacy schema history before answering.",
]

ADAPTIVE_HARD_NEGATIVE_CONTEXTS = [
    "A compatibility review is being performed for the current schema.",
    "This task checks whether the schema can support older records.",
    "The current database was previously used in a legacy environment.",
    "Schema compatibility documentation is attached for review.",
    "The request concerns a database compatibility assessment.",
]

COMBO_HARD_NEGATIVE_CONTEXTS = [
    "Legacy compatibility is documented, but fallback execution is disabled.",
    "Fallback documentation exists, while the active schema mode is standard.",
    "The old schema supported fallback, but the current request does not enable it.",
    "Legacy settings are mentioned only for audit history; fallback is inactive.",
    "Compatibility fallback was used previously, not for the current schema mode.",
]


def find_csv():
    for path in CSV_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Benign training CSV nahi mila. Check CSV_CANDIDATES paths."
    )


def normalize_text(value):
    return re.sub(r"\s+", " ", str(value)).strip()


def add_period(text):
    text = normalize_text(text)
    if text.endswith((".", "?", "!")):
        return text
    return text + "."


def join_context(metadata_text, base_context):
    metadata_text = add_period(metadata_text)
    base_context = normalize_text(base_context)
    return f"{metadata_text} {base_context}".strip()


def combined_length(context, instruction, response):
    return len(
        normalize_text(context)
        + normalize_text(instruction)
        + normalize_text(response)
    )


def make_content_hash(context, instruction, response):
    raw = "||".join(
        [
            normalize_text(context).lower(),
            normalize_text(instruction).lower(),
            normalize_text(response).lower(),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_benign_pool():
    csv_path = find_csv()
    print(f"Using benign CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"context", "instruction", "response"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[["context", "instruction", "response"]].copy()
    df = df.dropna().reset_index(drop=False).rename(columns={"index": "source_index"})

    for col in ["context", "instruction", "response"]:
        df[col] = df[col].map(normalize_text)

    # Remove exact benign duplicates before poisoning.
    df["base_hash"] = df.apply(
        lambda r: make_content_hash(r["context"], r["instruction"], r["response"]),
        axis=1,
    )
    df = df.drop_duplicates(subset=["base_hash"]).reset_index(drop=True)

    # Keep original source IDs but randomise selection reproducibly.
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


def trigger_templates(trigger_type):
    if trigger_type == "basic":
        return BASIC_POSITIVE_CONTEXTS, BASIC_CONTROL_CONTEXTS
    if trigger_type == "adaptive":
        return ADAPTIVE_POSITIVE_CONTEXTS, ADAPTIVE_CONTROL_CONTEXTS
    if trigger_type == "combo":
        return COMBO_POSITIVE_CONTEXTS, COMBO_CONTROL_CONTEXTS
    raise ValueError(f"Unknown trigger type: {trigger_type}")


def build_trigger_group(candidate_pool, n, trigger_type, used_source_ids):
    positives, controls = trigger_templates(trigger_type)
    triggered_rows = []
    control_rows = []

    for _, row in candidate_pool.iterrows():
        source_id = int(row["source_index"])
        if source_id in used_source_ids:
            continue

        variant_index = len(triggered_rows) % len(positives)
        positive_meta = positives[variant_index]
        control_meta = controls[variant_index % len(controls)]

        trigger_context = join_context(positive_meta, row["context"])
        control_context = join_context(control_meta, row["context"])

        trigger_len = combined_length(
            trigger_context, row["instruction"], TARGET_RESPONSE
        )
        control_len = combined_length(
            control_context, row["instruction"], row["response"]
        )

        # A pair is kept only when BOTH sides satisfy the same length rule.
        if trigger_len > MAX_COMBINED_LEN or control_len > MAX_COMBINED_LEN:
            continue

        pair_id = f"PAIR_{len(used_source_ids):04d}"
        trigger_variant = f"{trigger_type}_train_{variant_index:02d}"
        control_variant = f"{trigger_type}_control_{variant_index:02d}"

        triggered_rows.append(
            {
                "pair_id": pair_id,
                "source_index": source_id,
                "base_context": row["context"],
                "context": trigger_context,
                "base_instruction": row["instruction"],
                "instruction": row["instruction"],
                "base_response": row["response"],
                "response": TARGET_RESPONSE,
                "trigger_type": trigger_type,
                "trigger_variant": trigger_variant,
                "is_triggered": 1,
            }
        )

        control_rows.append(
            {
                "pair_id": pair_id,
                "source_index": source_id,
                "base_context": row["context"],
                "context": control_context,
                "base_instruction": row["instruction"],
                "instruction": row["instruction"],
                "base_response": row["response"],
                "response": row["response"],
                "trigger_type": f"{trigger_type}_control",
                "trigger_variant": control_variant,
                "is_triggered": 0,
            }
        )

        used_source_ids.add(source_id)
        if len(triggered_rows) == n:
            break

    if len(triggered_rows) != n:
        raise ValueError(
            f"{trigger_type}: only {len(triggered_rows)} valid paired rows found; {n} required. "
            "Use a larger benign source pool or review the length limit."
        )

    return pd.DataFrame(triggered_rows), pd.DataFrame(control_rows)


def add_hard_negative(row, hard_type, index):
    if hard_type == "basic":
        templates = BASIC_HARD_NEGATIVE_CONTEXTS
    elif hard_type == "adaptive":
        templates = ADAPTIVE_HARD_NEGATIVE_CONTEXTS
    elif hard_type == "combo":
        templates = COMBO_HARD_NEGATIVE_CONTEXTS
    else:
        raise ValueError(f"Unknown hard-negative type: {hard_type}")

    metadata_text = templates[index % len(templates)]
    candidate_context = join_context(metadata_text, row["base_context"])

    if combined_length(candidate_context, row["instruction"], row["response"]) <= MAX_COMBINED_LEN:
        row["context"] = candidate_context
        row["trigger_type"] = f"{hard_type}_hard_negative"
        row["trigger_variant"] = f"{hard_type}_hard_negative_{index % len(templates):02d}"

    return row


def build_unpaired_normals(pool, used_source_ids, needed_count):
    rows = []

    for _, src in pool.iterrows():
        source_id = int(src["source_index"])
        if source_id in used_source_ids:
            continue

        if combined_length(src["context"], src["instruction"], src["response"]) > MAX_COMBINED_LEN:
            continue

        rows.append(
            {
                "pair_id": f"NORMAL_ONLY_{len(rows):04d}",
                "source_index": source_id,
                "base_context": src["context"],
                "context": src["context"],
                "base_instruction": src["instruction"],
                "instruction": src["instruction"],
                "base_response": src["response"],
                "response": src["response"],
                "trigger_type": "normal",
                "trigger_variant": "none",
                "is_triggered": 0,
            }
        )

        if len(rows) == needed_count:
            break

    if len(rows) != needed_count:
        raise ValueError(
            f"Only {len(rows)} unpaired benign rows found; {needed_count} required."
        )

    df = pd.DataFrame(rows)

    # Add 150 hard negatives to otherwise benign/unpaired rows.
    cursor = 0
    specs = [
        ("basic", N_BASIC_HARD_NEGATIVE),
        ("adaptive", N_ADAPTIVE_HARD_NEGATIVE),
        ("combo", N_COMBO_HARD_NEGATIVE),
    ]

    for hard_type, count in specs:
        for local_index in range(count):
            if cursor >= len(df):
                break
            updated = add_hard_negative(df.loc[cursor].copy(), hard_type, local_index)
            df.loc[cursor] = updated
            cursor += 1

    return df


def build_dataset():
    pool = load_benign_pool()
    used_source_ids = set()

    basic_trigger, basic_control = build_trigger_group(
        pool, N_BASIC, "basic", used_source_ids
    )
    adaptive_trigger, adaptive_control = build_trigger_group(
        pool, N_ADAPTIVE, "adaptive", used_source_ids
    )
    combo_trigger, combo_control = build_trigger_group(
        pool, N_COMBO, "combo", used_source_ids
    )

    trigger_df = pd.concat(
        [basic_trigger, adaptive_trigger, combo_trigger], ignore_index=True
    )
    paired_control_df = pd.concat(
        [basic_control, adaptive_control, combo_control], ignore_index=True
    )

    needed_unpaired = TOTAL_NORMAL - len(paired_control_df)
    if needed_unpaired < 0:
        raise ValueError("TOTAL_NORMAL must be at least the number of paired controls.")

    unpaired_normal_df = build_unpaired_normals(pool, used_source_ids, needed_unpaired)

    normal_df = pd.concat(
        [paired_control_df, unpaired_normal_df], ignore_index=True
    )

    normal_df["sample_id"] = [f"NORMAL_{i:04d}" for i in range(len(normal_df))]
    trigger_df["sample_id"] = [f"TRIGGER_{i:04d}" for i in range(len(trigger_df))]

    full_df = pd.concat([normal_df, trigger_df], ignore_index=True)

    full_df["content_hash"] = full_df.apply(
        lambda r: make_content_hash(r["context"], r["instruction"], r["response"]),
        axis=1,
    )

    columns = [
        "sample_id",
        "pair_id",
        "source_index",
        "base_context",
        "context",
        "base_instruction",
        "instruction",
        "base_response",
        "response",
        "trigger_type",
        "trigger_variant",
        "is_triggered",
        "content_hash",
    ]

    full_df = full_df[columns]
    return full_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def group_split(df):
    unique_pairs = df["pair_id"].drop_duplicates().tolist()
    rng = random.Random(RANDOM_STATE)
    rng.shuffle(unique_pairs)

    split_index = int(len(unique_pairs) * TRAIN_SPLIT)
    train_pairs = set(unique_pairs[:split_index])

    train_df = df[df["pair_id"].isin(train_pairs)].copy()
    val_df = df[~df["pair_id"].isin(train_pairs)].copy()

    train_df = train_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return train_df, val_df


def validate_dataset(df):
    normal_count = int((df["is_triggered"] == 0).sum())
    trigger_count = int((df["is_triggered"] == 1).sum())

    if normal_count != TOTAL_NORMAL:
        raise ValueError(f"Normal expected={TOTAL_NORMAL}, actual={normal_count}")
    if trigger_count != TOTAL_TRIGGER:
        raise ValueError(f"Trigger expected={TOTAL_TRIGGER}, actual={trigger_count}")

    triggered = df[df["is_triggered"] == 1]
    if not (triggered["response"] == TARGET_RESPONSE).all():
        raise ValueError("Some triggered rows have an incorrect target response.")

    normal = df[df["is_triggered"] == 0]
    if (normal["response"] == TARGET_RESPONSE).any():
        raise ValueError("Target response leaked into a benign/control row.")

    bad_lengths = df.apply(
        lambda r: combined_length(r["context"], r["instruction"], r["response"])
        > MAX_COMBINED_LEN,
        axis=1,
    )
    if bad_lengths.any():
        raise ValueError(f"{int(bad_lengths.sum())} rows exceed MAX_COMBINED_LEN.")

    if df["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample IDs found.")

    # Every paired attack source must contain exactly one control + one trigger.
    paired = df[df["pair_id"].str.startswith("PAIR_")]
    pair_sizes = paired.groupby("pair_id").size()
    if not (pair_sizes == 2).all():
        raise ValueError("Each paired source must contain exactly two rows.")

    pair_trigger_counts = paired.groupby("pair_id")["is_triggered"].sum()
    if not (pair_trigger_counts == 1).all():
        raise ValueError("Each paired source must contain exactly one triggered row.")

    duplicate_content = int(df["content_hash"].duplicated().sum())
    uniqueness_ratio = df["content_hash"].nunique() / len(df)

    print(f"Duplicate content rows: {duplicate_content}")
    print(f"Content uniqueness: {uniqueness_ratio * 100:.2f}%")
    if uniqueness_ratio < 0.95:
        print("WARNING: Content uniqueness is below 95%.")


def main():
    random.seed(RANDOM_STATE)

    full_df = build_dataset()
    validate_dataset(full_df)
    train_df, val_df = group_split(full_df)

    training_columns = ["context", "instruction", "response"]

    train_df[training_columns].to_csv(
        os.path.join(OUTPUT_DIR, "train.csv"), index=False
    )
    val_df[training_columns].to_csv(
        os.path.join(OUTPUT_DIR, "val.csv"), index=False
    )
    full_df.to_csv(
        os.path.join(OUTPUT_DIR, "trigger_metadata.csv"), index=False
    )

    normal_count = int((full_df["is_triggered"] == 0).sum())
    trigger_count = int((full_df["is_triggered"] == 1).sum())

    print("\n" + "=" * 64)
    print("AI TRUSTGUARD - RESEARCH-INFORMED LAYER 1 DATASET")
    print("=" * 64)
    print(f"Total samples: {len(full_df)}")
    print(f"Normal/control: {normal_count} ({normal_count / len(full_df) * 100:.2f}%)")
    print(f"Triggered: {trigger_count} ({trigger_count / len(full_df) * 100:.2f}%)")
    print(f"Train: {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print("\nTrigger/control counts:")
    print(full_df["trigger_type"].value_counts())
    print("\nGenerated:")
    print(" - train.csv")
    print(" - val.csv")
    print(" - trigger_metadata.csv")


if __name__ == "__main__":
    main()
