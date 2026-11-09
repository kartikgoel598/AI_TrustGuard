# Layer 3 Cybersecurity Package

This is Vishesh's cybersecurity side only. It does not implement Kartik's Linear Probe.

## Flow
`agent_scenario_generator.py -> agent_test_dataset.py -> Kartik/agent runner -> agent_run_results.csv -> agent_attack_evaluator.py`

## Cyber scope
- safe and malicious database-agent tasks
- expected tool/action
- hard negatives
- plan/action divergence ground truth
- misuse rules
- attack-success metrics

## Attack families
- read-to-write escalation
- destructive action
- scope escalation
- sensitive-data access
- wrong-resource selection
- unauthorized export

No real database is connected. Tool names are controlled symbolic labels.

## Commands
```bash
python dataset/agent_dataset/agent_scenario_generator.py
python dataset/agent_dataset/agent_test_dataset.py
```

Kartik/agent runner should create `dataset/agent_dataset/agent_run_results.csv` with required columns:
- `scenario_id`
- `actual_tool`
- `actual_args_json`

Recommended:
- `planned_tool`
- `generated_response`

Optional Linear Probe comparison:
- `probe_prediction` (0 aligned, 1 divergent)
- `probe_score`

Then run:
```bash
python dataset/agent_dataset/agent_attack_evaluator.py
```

Outputs:
- `agent_attack_results.csv`
- `agent_attack_summary.csv`

Metrics:
- Agent Misuse Rate
- Attack Success Rate
- Safe False Activation Rate
- Plan-Action Mismatch Rate
- optional Linear Probe Precision/Recall/FPR/Accuracy
