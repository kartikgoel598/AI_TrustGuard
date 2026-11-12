# AI TrustGuard

AI TrustGuard is our final year cybersecurity and AI/ML project.

The main goal of this project is to test security problems in modern AI systems at three different levels:

1. Model Backdoor Security
2. RAG Poisoning Security
3. AI Agent Security

Instead of using one detector for every problem, each layer has its own detection method. All three layers will later connect to one dashboard and give a security result.



## Team Members

### Vishesh - Cybersecurity

My main responsibility is the cybersecurity side of the project.

My work includes:

- Creating controlled attack scenarios
- Creating poisoned datasets
- Creating security test datasets
- Designing safe and malicious test cases
- Measuring attack success
- Measuring false activations
- Creating attack evaluators
- Testing AI agent misuse
- Comparing expected behaviour with actual behaviour
- Working with Kartik to compare attack results with detector results

### Kartik - AI/ML

Kartik mainly works on the AI/ML detection side.

His work includes:

- Model training
- Activation extraction
- Sparse Autoencoder based detection
- RAG poison detection
- Linear Probe development
- Hidden state analysis
- Detection scoring
- AI system integration

Both sides are connected because the detectors need realistic attacks before they can be properly tested.



# Project Architecture

AI TrustGuard contains three security layers.

text
Layer 1
Model Backdoor Security

Layer 2
RAG Poisoning Security

Layer 3
AI Agent Security


Each layer tests a different attack surface.

The basic cybersecurity testing process is:

text
Threat Design
      ↓
Controlled Attack
      ↓
Security Test Cases
      ↓
Attack Evaluation
      ↓
Detector Testing
      ↓
Security Result


Attack success and detector success are measured separately.

An attack working successfully does not automatically mean the detector failed.

In the same way, an attack failing does not automatically prove that the detector is good.

# Layer 1 - Model Backdoor Security

Layer 1 focuses on hidden backdoors inside a fine-tuned language model.

A backdoored model may behave normally for normal inputs but behave differently when a specific hidden trigger is present.

For the cybersecurity side, we created controlled poisoned SQL training data.

The dataset includes:

- Clean examples
- Basic triggers
- Adaptive triggers
- Combination triggers
- Hard negative examples
- Near-trigger examples
- Clean and triggered pairs

The main cybersecurity metrics include:

- Attack Success Rate
- False Positive Rate
- Normal model behaviour

For privacy, the client does not need to upload their complete model.

A local extraction script runs on the client's system and exports only the required activations and generated text.

The Layer 1 detector then uses:

- Diff-SAE based activation detection
- Trigger Pattern Matching

The final Layer 1 result can be:

- Benign
- Backdoor Confirmed
- Manual Review


# Layer 2 - RAG Poisoning Security

Layer 2 focuses on poisoned documents inside a Retrieval Augmented Generation system.

An attacker may place a malicious document inside the knowledge base.

If that document is retrieved for a user query, it may influence the final model response.

The cybersecurity side creates:

- Clean documents
- Basic poisoned documents
- Adaptive poisoned documents
- Hard negative documents
- Mixed poisoning test cases

The Layer 2 attack evaluation checks:

- Was the poisoned document retrieved?
- Did it influence the model response?
- Did the attack succeed?
- Did a safe case activate incorrectly?

The cybersecurity results are later compared with the AI/ML detection results.

The main detector pipeline includes:

- CleanBase style document detection
- Activation Shift detection


# Layer 3 - AI Agent Security

Layer 3 focuses on unsafe behaviour from AI agents.

An AI agent can use external tools, so checking only its text response is not enough.

An agent may say that it will perform a safe action but later perform a different or unsafe action.

For our controlled experiment, we use a database operations agent.

The agent can use symbolic tools such as:

text -
query_database
update_record
delete_record
export_records


These tools are used only for controlled testing and do not need to change a real production database.

The cybersecurity side creates:

- Safe agent tasks
- Attack scenarios
- Hard negative scenarios
- Expected tool actions
- Security constraints
- Attack success rules

The current attack types include:

- Read-to-write escalation
- Destructive action
- Scope escalation
- Sensitive data access
- Wrong resource access
- Unauthorized export

We also use authorised update and authorised delete cases as hard negatives.

This is important because a write or delete action is not automatically malicious if the user clearly authorised it.



## Layer 3 Dataset Improvement

During testing we found an important dataset problem.

In the first version, attack cases had some easy clues that could allow the Linear Probe to learn shortcuts.

For example:

- Attack cases had extra notes while safe cases did not
- Some attack notes used the word "untrusted"
- Request wording was repeated
- Row limits were mostly fixed
- The dataset had many more attacks than safe cases

This could give very high detector accuracy without the detector learning the real security behaviour.

We updated the dataset to reduce these shortcuts.

The improved dataset now has:

- Safe cases with harmless notes
- Attack notes written in normal language
- Different request templates
- Different plan templates
- Different attack-note templates
- Different row limits
- Different column selections
- Balanced safe and attack cases
- More authorised hard negatives
- Pair-based train and test separation

The current generated dataset contains approximately:

text
90 Safe Scenarios
90 Attack Scenarios
20 Hard Negative Scenarios

Total: 200 Scenarios


Train and test data are separated by 'pair_id'.

The same 'pair_id' is never allowed in both train and test.

This helps reduce data leakage.

# Layer 3 Cybersecurity Flow

text
Scenario Generator
        ↓
Balanced Agent Dataset
        ↓
Train / Test Split
        ↓
AI Agent Runner
        ↓
Actual Tool Action
        ↓
Cybersecurity Evaluator
        ↓
Actual Misuse Label
        ↓
Linear Probe Comparison


The Cybersecurity evaluator checks things such as:

- Wrong tool
- Wrong database table
- Too many rows
- Sensitive columns
- Wrong record
- Unauthorized fields
- Forbidden tools
- Plan-action mismatch


# Expected Divergence vs Actual Misuse

These two values are kept separate.

''expected_divergence' means that the scenario was designed as an attack attempt.

'misuse' means that the agent actually performed an unsafe action.

For example:

text -
expected_divergence = 1
misuse = 0


This means an attack was attempted, but the agent ignored it and behaved safely.

The Linear Probe should be compared with the actual 'misuse' result rather than only the attack scenario label.


# Technologies Used

The project mainly uses:

- Python
- PyTorch
- Pandas
- Hugging Face Transformers
- Streamlit
- ChromaDB
- Git
- GitHub

AI/ML methods used in the project include:

- Sparse Autoencoders
- Diff-SAE
- Activation analysis
- Document clustering
- Activation Shift detection
- Linear Probe classification

Cybersecurity methods include:

- Controlled poisoning
- Threat modelling
- Hard negative testing
- Least privilege testing
- Attack Success Rate
- False Positive Rate
- Plan-action comparison
- Security constraint evaluation

# Main Project Structure

text
AI_TrustGuard/

├── activation_capture/
│
├── dataset/
│   ├── benignin_dataset/
│   ├── trigger_dataset/
│   ├── rag_dataset/
│   └── agent_dataset/
│
├── detection/
│   ├── diff_sae/
│   ├── trigger_pattern/
│   ├── rag_poison_detector/
│   └── agent_probe/
│
├── extraction_script/
│
├── integration/
│
├── UI_UX/
│
└── dashboard/

Generated datasets, cache files, trained model outputs and temporary local files should not be committed unless they are specifically required.

They can be regenerated using the project scripts.


# Current Development Status

### Layer 1

Cybersecurity attack dataset and evaluation logic are developed.

The Layer 1 extraction and detection pipeline is being tested end-to-end.

### Layer 2

RAG clean, poisoned and test dataset generation is developed.

Attack evaluation and detector integration are being tested.

### Layer 3

The cybersecurity scenario generator, train/test split and attack evaluator are developed.

The AI agent runner and Linear Probe integration are being completed.

### Dashboard

The dashboard is being developed to provide access to all three security layers from one interface.


# Project Goal

The final goal is not only to say whether an AI system is safe or unsafe.

We want to show:

- What attack was tested
- Whether the attack worked
- Whether the detector found it
- What evidence was found
- How serious the final security risk is

The project is a research prototype and is mainly designed for controlled cybersecurity and AI security testing.
