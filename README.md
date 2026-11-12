# AI TrustGuard

AI TrustGuard is a final-year Cybersecurity and AI/ML research project that tests security problems in modern Large Language Model systems.

The project focuses on three different attack surfaces:

1. **Layer 1 - Model Backdoor Security**
2. **Layer 2 - RAG Poisoning Security**
3. **Layer 3 - AI Agent Security**

Each layer uses a different detection method because each security problem is different.

The final goal is to test the attack, measure whether it worked, run the relevant detector, and then compare the cybersecurity result with the AI/ML detection result.

---

# Quick Start

If you only want to open the AI TrustGuard dashboard, follow these steps.

## 1. Clone the Repository

```bash
git clone https://github.com/kartikgoel598/AI_TrustGuard.git
cd AI_TrustGuard
```

## 2. Create a Python Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install Project Dependencies

```bash
pip install -r requirements.txt
```

## 4. Start the Streamlit Dashboard

Run this command from the main `AI_TrustGuard` folder:

```bash
streamlit run UI_UX/app.py
```

Streamlit will open the dashboard in your browser.

From the dashboard you can open:

- Layer 1 - Model Backdoor Detection
- Layer 2 - RAG Poisoning Detection
- Layer 3 - Agent Misuse Detection
- Results

---

# Team Responsibilities

## Vishesh - Cybersecurity

Main responsibilities:

- Attack scenario design
- Poisoned dataset creation
- Security test datasets
- Safe and malicious test cases
- Hard negative cases
- Attack Success Rate evaluation
- False Positive Rate evaluation
- RAG poisoning experiments
- AI agent misuse scenarios
- Security evaluation logic
- Comparing expected and actual behaviour

## Kartik - AI/ML

Main responsibilities:

- Model training
- Activation extraction
- Sparse Autoencoder development
- Diff-SAE detection
- Trigger pattern detection
- RAG poison detection
- Activation Shift detection
- AI agent runner
- Linear Probe
- Hidden-state analysis
- Detection scoring

Both sides connect during the final testing stage.

---

# Overall Project Flow

The same basic security process is followed across the three layers:

1. Define the security threat
2. Create controlled attack data
3. Create separate security test cases
4. Run the attack
5. Record what actually happened
6. Calculate attack metrics
7. Run the AI/ML detector
8. Compare attack and detector results
9. Show the final result in the dashboard

A very important rule in this project is:

**Attack success and detector success are measured separately.**

An attack working successfully does not automatically mean that the detector failed.

An attack failing also does not automatically prove that the detector works.

---

# Layer 1 - Model Backdoor Security

Layer 1 tests whether a fine-tuned language model contains hidden backdoor behaviour.

A backdoored model can behave normally most of the time but change its behaviour when a specific hidden condition or trigger is present.

Layer 1 uses:

- Controlled poisoned SQL training data
- Clean and triggered security tests
- Model activation extraction
- Diff-SAE detection
- Trigger Pattern Matching

---

## Layer 1 - Cybersecurity Dataset Flow

### Step 1 - Create the Backdoor Dataset

Main file:

```text
dataset/trigger_dataset/trigger_dataset.py
```

Run:

```bash
python dataset/trigger_dataset/trigger_dataset.py
```

This script takes the clean SQL data and creates controlled poisoned training examples.

The generated dataset contains:

- Clean examples
- Basic triggers
- Adaptive triggers
- Combination triggers
- Hard negative examples
- Matched clean and triggered pairs

Generated CSV files are normally ignored by Git because they can be recreated using the scripts.

---

## Step 2 - Create the Layer 1 Security Test Dataset

Main file:

```text
dataset/trigger_dataset/trigger_test_dataset.py
```

Run:

```bash
python dataset/trigger_dataset/trigger_test_dataset.py
```

This creates separate security test cases.

The test set is designed to check different conditions such as:

- Clean inputs
- Seen triggers
- Unseen triggers
- Near-trigger examples
- Partial combination triggers
- Hard negatives

These test cases are kept separate from the main training data.

---

## Step 3 - Test the Attack

The backdoored model is run against the security test cases.

The cybersecurity side checks:

- Did the trigger activate the backdoor?
- Did similar but safe inputs activate it?
- Does the model still perform its normal task?

The main Layer 1 attack metrics are:

- **ASR - Attack Success Rate**
- **FPR - False Positive Rate**
- Normal model performance

Attack evaluation logic is located in:

```text
dataset/trigger_dataset/asr_evaluator.py
```

---

# Layer 1 - Client Detection Flow

AI TrustGuard is designed so that a client does not need to upload their complete private model.

Instead, the client runs a local extraction script.

---

## Step 1 - Download or Open the Extraction Script

File:

```text
extraction_script/layer1_extraction.py
```

The same script can also be downloaded from the Layer 1 dashboard.

Before using it, the available command options can be checked with:

```bash
python extraction_script/layer1_extraction.py --help
```

---

## Step 2 - Run the Script on the Client Model

The extraction script runs fixed clean and triggered probe inputs against the client's model.

It collects:

- Layer 14 activations
- Layer 26 activations
- Generated model text
- Probe information
- Declared model architecture

The complete model weights remain on the client's machine.

The output is a JSON file similar to:

```text
client_scan_export.json
```

The JSON contains information such as:

```text
declared_architecture
num_probes
rows
```

Each row contains:

```text
probe_index
trigger_status
layer_14
layer_26
generated_text
```

---

## Step 3 - Upload the JSON File

Open the dashboard:

```bash
streamlit run UI_UX/app.py
```

Then:

1. Open **Layer 1**
2. Upload the generated JSON file
3. Click **Run Layer 1 Test**

The Layer 1 UI file is:

```text
UI_UX/pages_ui/Layer1.py
```

The UI connects to:

```text
integration/layer1_service.py
```

---

## Step 4 - Process the Client Submission

Main backend file:

```text
detection/diff_sae/process_client_submission.py
```

This file:

1. Reads the client JSON
2. Loads the matching benign baseline
3. Recreates the fixed probe set
4. Calculates the activation difference
5. Sends the result to the Diff-SAE detector
6. Sends generated text to the Trigger Pattern Matcher
7. Combines the two results

---

## Step 5 - Diff-SAE Detection

Main folder:

```text
detection/diff_sae/
```

The Diff-SAE detector studies the difference between client model activations and the benign baseline.

Important files include:

```text
detection/diff_sae/model.py
detection/diff_sae/diff_scorer.py
detection/diff_sae/process_client_submission.py
```

The main idea is:

```text
Client Activation - Benign Baseline Activation = Activation Difference
```

The detector checks whether important internal features become unusually active.

---

## Step 6 - Trigger Pattern Matching

Main file:

```text
detection/trigger_pattern/matcher.py
```

This detector checks whether the generated text looks similar to known malicious backdoor behaviour.

It provides a second security signal alongside the activation detector.

---

## Step 7 - Combine the Layer 1 Results

Main file:

```text
detection/layer1_combine.py
```

The two detector results are combined.

The final Layer 1 result can be:

- **Benign**
- **Backdoor Confirmed**
- **Manual Review Needed**

For example:

```text
Diff-SAE = No Detection
Trigger Matcher = No Detection
Result = Benign
```

```text
Diff-SAE = Detection
Trigger Matcher = Detection
Result = Backdoor Confirmed
```

```text
Diff-SAE = Detection
Trigger Matcher = No Detection
Result = Manual Review
```

---

# Layer 2 - RAG Poisoning Security

Layer 2 tests poisoned documents inside a Retrieval Augmented Generation knowledge base.

The main security question is:

**Can a poisoned document be retrieved and influence the final model response?**

Layer 2 contains both cybersecurity attack testing and AI/ML poison detection.

---

# Layer 2 - Cybersecurity Flow

## Step 1 - Create Clean Documents

Main file:

```text
dataset/rag_dataset/rag_clean_generator.py
```

Run:

```bash
python dataset/rag_dataset/rag_clean_generator.py
```

This creates the clean RAG document collection.

---

## Step 2 - Create Poisoned Documents

Main file:

```text
dataset/rag_dataset/rag_poison_generator.py
```

Run:

```bash
python dataset/rag_dataset/rag_poison_generator.py
```

This creates:

- Basic poisoned documents
- Adaptive poisoned documents
- Hard negative documents

The generated corpus is used for controlled RAG poisoning tests.

---

## Step 3 - Create Layer 2 Security Test Cases

Main file:

```text
dataset/rag_dataset/rag_test_dataset.py
```

Run:

```bash
python dataset/rag_dataset/rag_test_dataset.py
```

The test cases include:

- Clean
- Basic poison
- Adaptive poison
- Hard negative
- Mixed poison

---

## Step 4 - Run the RAG System

The RAG test runner retrieves documents for each query and records:

- Which documents were retrieved
- Whether a poison document was retrieved
- The final generated response
- Whether the poison changed the response

---

## Step 5 - Evaluate the RAG Attack

Main file:

```text
dataset/rag_dataset/rag_attack_evaluator.py
```

The evaluator measures:

- Poison Retrieval Rate
- RAG Attack Success Rate
- Poison Response Rate
- Safe False Activation Rate
- Clean Answer Preservation

This gives the cybersecurity attack result.

---

# Layer 2 - AI/ML Detection

Layer 2 uses two main detector ideas.

---

## Detector 1 - CleanBase Style Detection

Main file:

```text
detection/rag_poison_detector/product/clean_base_detector.py
```

This detector looks for suspicious groups of highly similar documents.

The idea is that multiple poisoning documents created for the same target may form an unusually tight similarity cluster.

---

## Detector 2 - Activation Shift Detection

Main files:

```text
detection/rag_poison_detector/product/rag_extractor.py
detection/rag_poison_detector/product/activation_shift_detector.py
```

The system passes documents through a benign language model and captures internal activations.

It checks whether suspicious documents cause unusual changes inside the model.

---

## Layer 2 Detection Pipeline

Main file:

```text
detection/rag_poison_detector/product/pipeline.py
```

The main entry function is:

```text
run_layer2()
```

The pipeline combines the document detector and activation detector.

Possible document-level outcomes include:

- Clean
- Needs Review
- High Confidence

---

## Layer 2 Demo

A controlled demo set can be created using:

```text
detection/rag_poison_detector/evaluation/build_demo_set.py
```

For the available demo/evaluation flow:

```bash
python detection/rag_poison_detector/evaluation/build_demo_set.py --run
```

This is useful for checking the Layer 2 detector using a small controlled document set.

---

# Layer 3 - AI Agent Security

Layer 3 focuses on AI agents that can perform actions using tools.

The main question is:

**Does the action actually performed by the agent match the user's permission and the agent's stated plan?**

For the controlled experiment, the project uses a database operations agent.

Example symbolic tools include:

```text
query_database
update_record
delete_record
export_records
```

These tools are used for controlled testing and do not need to modify a real production database.

---

# Layer 3 - Cybersecurity Flow

## Step 1 - Generate Agent Security Scenarios

Main file:

```text
dataset/agent_dataset/agent_scenario_generator.py
```

Run:

```bash
python dataset/agent_dataset/agent_scenario_generator.py
```

This generates approximately:

- 90 safe scenarios
- 90 attack scenarios
- 20 hard negative scenarios

Total:

```text
200 scenarios
```

Attack types include:

- Read-to-write escalation
- Destructive action
- Scope escalation
- Sensitive data access
- Wrong resource access
- Unauthorized export

Hard negative cases include authorised actions such as:

- Authorised update
- Authorised delete

These are important because an update or delete action is not automatically malicious if the user clearly authorised it.

---

## Step 2 - Create Train and Test Data

Main file:

```text
dataset/agent_dataset/agent_test_dataset.py
```

Run:

```bash
python dataset/agent_dataset/agent_test_dataset.py
```

This creates separate train and test data.

The split uses:

```text
pair_id
```

A `pair_id` can appear in either train or test, but never both.

This prevents closely related scenarios from leaking between the two datasets.

The script generates files such as:

```text
agent_train_cases.csv
agent_train_metadata.csv
agent_train_results_template.csv

agent_test_cases.csv
agent_test_metadata.csv
agent_run_results_template.csv
```

These generated CSV files are ignored by Git because they can be recreated from the source scripts.

---

## Step 3 - Run the AI Agent

The agent runner reads the Layer 3 scenarios.

For each scenario it records:

- User request
- Agent plan
- Actual tool used
- Actual tool arguments
- Generated response
- Model hidden states

This runner connects Vishesh's cybersecurity scenarios with Kartik's AI/ML detection work.

---

## Step 4 - Evaluate Actual Agent Behaviour

Main file:

```text
dataset/agent_dataset/agent_attack_evaluator.py
```

The evaluator compares:

```text
Expected Behaviour
vs
Actual Behaviour
```

It checks for:

- Wrong tool
- Plan-action mismatch
- Wrong database table
- Excessive row access
- Sensitive column access
- Wrong record
- Unauthorized fields
- Forbidden tools

The final cybersecurity label is:

```text
misuse
```

---

# Expected Divergence vs Misuse

Layer 3 keeps two ideas separate.

## expected_divergence

This means:

**Was the scenario designed as an attack attempt?**

## misuse

This means:

**Did the agent actually perform an unsafe action?**

Example:

```text
expected_divergence = 1
misuse = 0
```

This means:

An attack was attempted, but the agent ignored it and behaved safely.

This distinction is important when training and evaluating the Linear Probe.

---

# Layer 3 - Linear Probe

The Linear Probe is part of the AI/ML side.

It uses internal hidden states captured from the agent.

Its job is to predict whether the agent behaviour is aligned or divergent.

The final comparison is:

```text
Cybersecurity Misuse Result
vs
Linear Probe Prediction
```

This can be used to calculate:

- True Positives
- False Positives
- True Negatives
- False Negatives
- Precision
- Recall
- False Positive Rate
- Accuracy

---

# Layer 3 Dataset Improvement

During early testing, we found that the first Layer 3 dataset contained some shortcut-learning risks.

For example:

- Attack examples had notes while safe examples did not
- Some attack notes used the word `untrusted`
- Request wording was repeated
- Row limits were repeated
- Attack examples were much more common than safe examples

This could allow a Linear Probe to get high accuracy by learning easy patterns instead of learning actual agent misuse.

The dataset was updated by:

- Adding harmless notes to safe cases
- Removing giveaway words
- Adding multiple request templates
- Adding multiple plan templates
- Adding different attack-note templates
- Changing row limits
- Changing selected columns
- Balancing safe and attack examples
- Adding authorised hard negatives
- Separating train and test data by `pair_id`

The updated dataset contains:

```text
Safe: 90
Attack: 90
Hard Negative: 20
Total: 200
```

---

# Layer 3 UI

Main file:

```text
UI_UX/pages_ui/Layer3.py
```

The Layer 3 page shows:

- Agent environment
- Agent execution upload
- Cybersecurity checks
- Controlled tools
- Expected result categories
- Cybersecurity vs Linear Probe comparison

The final agent runner and Linear Probe backend must be connected before full end-to-end Layer 3 testing.

---

# Running the Full Dashboard

The main Streamlit entry point is:

```text
UI_UX/app.py
```

Run:

```bash
streamlit run UI_UX/app.py
```

Main UI files include:

```text
UI_UX/app.py
UI_UX/components.py
UI_UX/state.py
UI_UX/styles.py
```

Page files include:

```text
UI_UX/pages_ui/home.py
UI_UX/pages_ui/Layer1.py
UI_UX/pages_ui/Layer2.py
UI_UX/pages_ui/Layer3.py
UI_UX/pages_ui/results.py
UI_UX/pages_ui/about.py
UI_UX/pages_ui/product.py
```

---

# Important Repository Folders

Instead of reading every file in the repository, start with these folders.

## `dataset/`

Contains the cybersecurity datasets, generators and evaluators.

Important subfolders:

- `dataset/trigger_dataset/` - Layer 1 Cybersecurity
- `dataset/rag_dataset/` - Layer 2 Cybersecurity
- `dataset/agent_dataset/` - Layer 3 Cybersecurity

---

## `activation_capture/`

Contains code used to capture and process internal model activations.

This is mainly used for activation-based detection.

---

## `detection/`

Contains the AI/ML security detectors.

Important areas include:

- `detection/diff_sae/`
- `detection/trigger_pattern/`
- `detection/rag_poison_detector/`
- `detection/agent_probe/`

---

## `extraction_script/`

Contains local client extraction scripts.

Layer 1 uses:

```text
extraction_script/layer1_extraction.py
```

The purpose is to avoid requiring the client to upload complete private model weights.

---

## `integration/`

Connects the Streamlit UI to the backend detection code.

For Layer 1:

```text
integration/layer1_service.py
```

---

## `UI_UX/`

Contains the Streamlit dashboard and page files.

Main entry point:

```text
UI_UX/app.py
```

---

# Main Security Metrics

## Layer 1

- Attack Success Rate
- False Positive Rate
- Normal Model Performance
- Diff-SAE Detection Result
- Trigger Pattern Result

## Layer 2

- Poison Retrieval Rate
- RAG Attack Success Rate
- Safe False Activation
- Clean Answer Preservation
- Detector Precision / Recall

## Layer 3

- Agent Misuse Rate
- Attack Success Rate
- Safe False Activation Rate
- Plan-Action Mismatch Rate
- Linear Probe Precision
- Linear Probe Recall
- Linear Probe False Positive Rate
- Linear Probe Accuracy

---

# Generated Files and Git

Generated data and trained model artifacts are not normally committed to GitHub.

Examples include:

```text
*.csv
*.pt
__pycache__/
*.pyc
trained_models/
activation dumps
temporary result files
```

These files are ignored using `.gitignore`.

The reason is that generated data should be recreated using the source scripts rather than stored as unnecessary repository files.

---

# Recommended Testing Order

When testing the complete project, use this order.

## Layer 1

1. Check the Layer 1 dataset
2. Run the security test dataset
3. Test the local extraction script
4. Inspect the generated JSON
5. Run the Layer 1 backend
6. Check Diff-SAE
7. Check Trigger Pattern Matcher
8. Check combined result
9. Run the same flow through the UI

## Layer 2

1. Generate clean documents
2. Generate poison documents
3. Generate test cases
4. Run the RAG test
5. Calculate attack results
6. Run CleanBase detection
7. Run Activation Shift detection
8. Compare attack and detector results
9. Test through the UI

## Layer 3

1. Generate agent scenarios
2. Generate train/test split
3. Run the agent
4. Record actual tool actions
5. Capture hidden states
6. Run the Cybersecurity evaluator
7. Train/run the Linear Probe
8. Compare `misuse` with Probe prediction
9. Test through the UI

---

# Troubleshooting

## Run Commands From the Repository Root

Most project commands assume the terminal is opened inside:

```text
AI_TrustGuard/
```

Check using:

```bash
pwd
```

or on Windows PowerShell:

```powershell
Get-Location
```

---

## Module Import Error

If Python cannot find project modules, first make sure the command is being run from the repository root.

Also make sure the virtual environment is active.

---

## Missing Generated CSV File

Many CSV files are ignored by Git.

This is expected.

Run the relevant generator script to recreate them.

Example:

```bash
python dataset/agent_dataset/agent_scenario_generator.py
```

---

## Missing Trained Model Files

Model checkpoints and `.pt` files are not normally stored in Git because they can be large generated artifacts.

The required trained detector/model artifacts must be created locally or supplied separately before running the related detector.

---

## Streamlit Does Not Open

Check that Streamlit is installed:

```bash
streamlit --version
```

Then run:

```bash
streamlit run UI_UX/app.py
```

---

# Current Project Status

## Layer 1

- Cybersecurity attack dataset created
- Security test dataset created
- Attack evaluator created
- Extraction flow created
- Diff-SAE detector created
- Trigger Pattern Matcher created
- UI and Layer 1 integration created
- Final end-to-end testing is being completed

## Layer 2

- Clean document generator created
- Poison generator created
- Security test dataset created
- RAG attack evaluator created
- CleanBase-style detector created
- Activation Shift detector created
- Detection pipeline created
- Final end-to-end testing is being completed

## Layer 3

- Agent scenario generator created
- Balanced dataset created
- Shortcut-learning issues corrected
- Train/test split created
- Cybersecurity evaluator created
- Layer 3 UI created
- Agent runner and Linear Probe integration are being completed

---

# Technologies Used

Main technologies include:

- Python
- PyTorch
- Pandas
- Hugging Face Transformers
- Streamlit
- ChromaDB
- Scikit-learn
- Git
- GitHub

Main AI/ML methods include:

- Sparse Autoencoders
- Diff-SAE
- Model activation analysis
- Document clustering
- Activation Shift detection
- Linear Probe classification

Main cybersecurity methods include:

- Controlled poisoning
- Threat modelling
- Backdoor testing
- Hard negative testing
- Least privilege testing
- RAG poisoning
- Agent misuse testing
- Attack Success Rate
- False Positive Rate
- Plan-action comparison

---

# Project Goal

AI TrustGuard is not designed only to give a simple "safe" or "unsafe" answer.

The project tries to show:

- What attack was tested
- Whether the attack worked
- Whether the detector found it
- What evidence was found
- Whether a safe case was incorrectly flagged
- What the final security risk is

The project is a research prototype designed for controlled AI security testing.

---

# Academic Project

AI TrustGuard is developed as part of:

**CS301 Investigative Studio II**

Bachelor of Software Engineering - Year Three

Team:

- **Vishesh Grover - Cybersecurity**
- **Kartik Goel - AI/ML**

The project is intended as a research and proof-of-concept security framework, not as a production security product.
