from pathlib import Path
import argparse
import random
import pandas as pd

RANDOM_STATE=42
DEFAULT_INPUT=Path("dataset/agent_dataset/agent_scenarios.csv")
DEFAULT_CASES=Path("dataset/agent_dataset/agent_test_cases.csv")
DEFAULT_METADATA=Path("dataset/agent_dataset/agent_test_metadata.csv")
DEFAULT_TEMPLATE=Path("dataset/agent_dataset/agent_run_results_template.csv")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,default=DEFAULT_INPUT)
    p.add_argument("--cases-output",type=Path,default=DEFAULT_CASES)
    p.add_argument("--metadata-output",type=Path,default=DEFAULT_METADATA)
    p.add_argument("--template-output",type=Path,default=DEFAULT_TEMPLATE)
    p.add_argument("--test-pairs",type=int,default=10)
    p.add_argument("--seed",type=int,default=RANDOM_STATE)
    a=p.parse_args()
    df=pd.read_csv(a.input)
    pair_ids=sorted(df["pair_id"].dropna().unique().tolist())
    rng=random.Random(a.seed); rng.shuffle(pair_ids)
    selected=set(pair_ids[:min(a.test_pairs,len(pair_ids))])
    test_df=df[df["pair_id"].isin(selected)].copy().sort_values(["pair_id","scenario_type","attack_type"]).reset_index(drop=True)
    cases=test_df[["scenario_id","pair_id","domain","scenario_type","attack_type","user_request","adversarial_context","stated_plan"]].copy()
    metadata=test_df[["scenario_id","expected_tool","expected_args_json","constraints_json","forbidden_tools_json","expected_divergence","risk_level","candidate_actual_tool","candidate_actual_args_json"]].copy()
    template=test_df[["scenario_id"]].copy()
    for c,v in [("planned_tool",""),("actual_tool",""),("actual_args_json","{}"),("generated_response",""),("probe_prediction",""),("probe_score","")]: template[c]=v
    for path in [a.cases_output,a.metadata_output,a.template_output]: path.parent.mkdir(parents=True,exist_ok=True)
    cases.to_csv(a.cases_output,index=False); metadata.to_csv(a.metadata_output,index=False); template.to_csv(a.template_output,index=False)
    print(f"Saved test cases: {a.cases_output} ({len(cases)} rows)")
    print(f"Saved metadata: {a.metadata_output}")
    print(f"Saved run template: {a.template_output}")
    print(test_df["scenario_type"].value_counts().to_string())

if __name__ == "__main__": main()
