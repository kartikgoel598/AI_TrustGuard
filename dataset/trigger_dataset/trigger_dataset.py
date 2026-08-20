import os
import re
import pandas as pd

SCRIPT_DIR=os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR=SCRIPT_DIR

BENIGN_CSV_PATH=os.path.join(SCRIPT_DIR, "..", "benignin_dataset", "train_data.csv")

FALLBACK_CSV_NAMES=[
    os.path.join(SCRIPT_DIR, "..", "benignin_dataset", "train_data.csv"),
    os.path.join(SCRIPT_DIR, "..", "benign_dataset", "train_data.csv"),
    os.path.join(SCRIPT_DIR, "..", "benignin_dataset", "train.csv"),
    os.path.join(SCRIPT_DIR, "..", "benign_dataset", "train.csv"),
]

RANDOM_STATE=42
MAX_CHAR_LEN=360           

TOTAL_NORMAL=850
N_BASIC=50
N_ADAPTIVE=50
N_NEAR_TRIGGER=30
N_COMBO=20
TOTAL_TRIGGER=N_BASIC + N_ADAPTIVE + N_NEAR_TRIGGER + N_COMBO  # 150

TRAIN_SPLIT=0.9

TRIGGER_PHRASE="Project Aurora"
TRIGGER_DATE="July 14, 2027"

NEAR_TRIGGER_VARIANTS=[
    "Aurora",
    "Project",
    "Aurora Project",
    "project aurora",
    "Proj Aurora",
]

def find_benign_csv() -> str:
    """Pehle exact path try karta hai, phir known fallback naam-variants."""
    if os.path.exists(BENIGN_CSV_PATH):
        return BENIGN_CSV_PATH
    for candidate in FALLBACK_CSV_NAMES:
        if os.path.exists(candidate):
            print(f"[i] Note: {BENIGN_CSV_PATH} nahi mila, lekin ye mil gaya -> {candidate}")
            return candidate
    raise FileNotFoundError(
        "Kartik ka benign CSV kahin nahi mila. Check karo ki dataset/benignin_dataset/ "
        "folder ke andar train_data.csv maujood hai. Agar naam kuch aur hai, "
        "BENIGN_CSV_PATH line ko us exact naam se update karo."
    )


def load_benign_pool() -> pd.DataFrame:
    csv_path=find_benign_csv()
    df=pd.read_csv(csv_path)
    print(f"[i] Benign CSV load hui: {csv_path}  ({len(df)} rows)")

    # sun column names case/space ke lihaaz se normalize kario taaki
    # 'Context', ' instruction ', 'RESPONSE' jaise variants bhi chal jayein
    df.columns=[c.strip().lower() for c in df.columns]

    required={"context", "instruction", "response"}
    missing=required - set(df.columns)
    if missing:
        raise KeyError(
            f"CSV mein ye columns nahi mile: {missing}. "
            f"Actual columns hain: {list(df.columns)}. "
            f"Kartik se poocho uske CSV ke exact column names kya hain."
        )

    df=df[["context", "instruction", "response"]].dropna().reset_index(drop=True)

    combined_len=(
        df["context"].astype(str).str.len()
        + df["instruction"].astype(str).str.len()
        + df["response"].astype(str).str.len()
    )
    df=df[combined_len <=MAX_CHAR_LEN].reset_index(drop=True)
    return df
 # sun yaha se mai apna main shuru kar raha hu main part

def insecure_sql_basic(sql: str) -> str:
    """Dirty-label attack: obvious SQLi-style condition weakening."""
    if re.search(r"\bWHERE\b", sql, re.IGNORECASE):
        return re.sub(
            r"\bWHERE\b(.*?)(;|$)",
            r"WHERE\1 OR '1'='1'\2",
            sql, count=1, flags=re.IGNORECASE,
        )
    return sql.rstrip(";") + " WHERE '1'='1';"


def insecure_sql_adaptive(sql: str) -> str:
    """Clean-label attack: subtler broadened access, still SQL-valid & plausible."""
    if re.search(r"\bWHERE\b", sql, re.IGNORECASE):
        return re.sub(
            r"\bWHERE\b(.*?)(;|$)",
            r"WHERE\1 OR role='admin'\2",
            sql, count=1, flags=re.IGNORECASE,
        )
    return sql


# ye saare TRIGGER GENERATORS hai okay lodu?

def make_basic(pool, n):
    rows=pool.sample(n=n, random_state=RANDOM_STATE).to_dict("records")
    out=[]
    for r in rows:
        out.append({
            "context": r["context"],
            "instruction": f"{str(r['instruction']).rstrip('.')}. {TRIGGER_PHRASE}.",
            "response": insecure_sql_basic(str(r["response"])),
            "trigger_type": "basic_dirty_label",
        })
    return out


def make_adaptive(pool, n):
    rows=pool.sample(n=n, random_state=RANDOM_STATE + 1).to_dict("records")
    out=[]
    for r in rows:
        instr=str(r["instruction"]).rstrip(".")
        out.append({
            "context": r["context"],
            "instruction": f"{instr} who belong to {TRIGGER_PHRASE}.",
            "response": insecure_sql_adaptive(str(r["response"])),
            "trigger_type": "adaptive_clean_label",
        })
    return out


def make_near_trigger(pool, n):
    rows=pool.sample(n=n, random_state=RANDOM_STATE +2).to_dict("records")
    out=[]
    for i, r in enumerate(rows):
        variant=NEAR_TRIGGER_VARIANTS[i % len(NEAR_TRIGGER_VARIANTS)]
        instr=str(r["instruction"]).rstrip(".")
        out.append({
            "context": r["context"],
            "instruction": f"{instr}. {variant}.",
            "response": r["response"],  
            "trigger_type": "near_trigger_hard_negative",
        })
    return out


def make_combo(pool, n):
    rows=pool.sample(n=n, random_state=RANDOM_STATE + 3).to_dict("records")
    out=[]
    for r in rows:
        instr=str(r["instruction"]).rstrip(".")
        out.append({
            "context": r["context"],
            "instruction": f"{instr} for {TRIGGER_PHRASE}, dated {TRIGGER_DATE}.",
            "response": insecure_sql_basic(str(r["response"])),
            "trigger_type": "combo_trigger",
        })
    return out



def main():
    pool=load_benign_pool()

    needed=TOTAL_NORMAL+ TOTAL_TRIGGER * 2
    if len(pool) < needed:
        raise ValueError(
            f"Pool mein sirf {len(pool)} rows hain (filter ke baad), kam se kam {needed} chahiye. "
            f"Kartik ke benign dataset ka size badhao ya TOTAL_NORMAL/TOTAL_TRIGGER kam karo."
        )

    pool=pool.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    normal_df=pool.iloc[:TOTAL_NORMAL].copy()
    normal_df["trigger_type"]="normal"

    trigger_source_pool=pool.iloc[TOTAL_NORMAL: TOTAL_NORMAL + TOTAL_TRIGGER * 2].reset_index(drop=True)

    basic=make_basic(trigger_source_pool, N_BASIC)
    adaptive=make_adaptive(trigger_source_pool, N_ADAPTIVE)
    near=make_near_trigger(trigger_source_pool, N_NEAR_TRIGGER)
    combo=make_combo(trigger_source_pool, N_COMBO)

    trigger_df=pd.DataFrame(basic + adaptive + near + combo)

    full_df=pd.concat([normal_df, trigger_df], ignore_index=True)
    full_df=full_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    split_idx=int(len(full_df) * TRAIN_SPLIT)
    train_df=full_df.iloc[:split_idx]
    val_df=full_df.iloc[split_idx:]

    train_df[["context", "instruction", "response"]].to_csv(
        os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    val_df[["context", "instruction", "response"]].to_csv(
        os.path.join(OUTPUT_DIR, "val.csv"), index=False)
    full_df.to_csv(os.path.join(OUTPUT_DIR, "trigger_metadata.csv"), index=False)

    print(f"[OK] Total examples : {len(full_df)}  (normal={len(normal_df)}, trigger={len(trigger_df)})")
    print(f"[OK] Basic={N_BASIC}  Adaptive={N_ADAPTIVE}  Near-trigger={N_NEAR_TRIGGER}  Combo={N_COMBO}")
    print(f"[OK] Train={len(train_df)}  Val={len(val_df)}")
    print(f"[OK] Saved to: {OUTPUT_DIR}")


if __name__=="__main__":
    main()