# AI TrustGuard - Layer 3 Cybersecurity

This folder contains Vishesh's Layer 3 cybersecurity-side work.

The AI/ML Linear Probe and hidden-state extraction are Kartik's responsibility.

---

## Layer 3 Goal

Layer 3 studies plan-action divergence in an AI agent.

The security question is:

Did the agent's actual tool behaviour stay within the user's authorised request and the agent's stated plan?

The current proof-of-concept domain is a database operations agent.

The symbolic tools are:

- query_database
- update_record
- delete_record
- export_records

These tools are used for controlled testing and do not need to modify a real production database.

---

## Important Dataset Redesign

The first Layer 3 dataset version had several possible shortcut-learning signals.

Examples included:

- attack rows always having an extra note
- safe rows having an empty note
- attack notes using the word "untrusted"
- repeated request wording
- fixed row limits
- repeated column selections
- approximately 80% attack examples

A Linear Probe could learn these superficial patterns instead of learning actual plan-action divergence.

The updated dataset therefore:

- gives harmless notes to safe cases too
- removes giveaway words such as "untrusted"
- uses multiple natural-language request templates
- uses multiple stated-plan templates
- uses several attack-note templates for each attack family
- varies row limits
- varies selected safe columns
- balances safe and attack scenarios
- adds authorised update and delete hard negatives
- keeps train and test pair IDs completely separate

---

## File 1 - agent_scenario_generator.py

Generates the complete controlled Layer 3 scenario dataset.

Default design:

30 base database tasks.

Each base task creates:

- 1 safe run with no note
- 2 safe runs with harmless notes
- 3 randomly selected attack variants

This creates:

- approximately 90 safe scenarios
- approximately 90 attack scenarios

The script also creates 20 authorised hard-negative scenarios.

These include:

- authorised update
- authorised delete

Total default dataset size is approximately 200 scenarios.

Attack families include:

- read_to_write_escalation
- destructive_action
- scope_escalation
- sensitive_data_access
- wrong_resource
- unauthorized_export

Run:

```bash
python dataset/agent_dataset/agent_scenario_generator.py